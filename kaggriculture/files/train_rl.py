"""
Behaviour cloning  ->  self-play PPO for the Kaggriculture macro policy.

This is a scaled-down version of the 1st-place loop (BC <-> self-play PPO <-> heuristic refinement):

  stage 1  BC   : clone the heuristic planner's daily macro decisions (hires / animals / land / crops)
  stage 2  PPO  : improve the policy by self-play against a pool of opponents
                  (heuristic variants + frozen snapshots of earlier policies)

Only numpy + kaggle_environments are required (no torch).  Everything below the macro level
(task planner, action patch) is the deterministic heuristic in main.py, so the RL problem is small:
30 decisions per game, 7 categorical heads.

    python train_rl.py bc  --games 40 --workers 4 --out policy_bc.npz
    python train_rl.py ppo --init policy_bc.npz --iters 30 --games-per-iter 16 --workers 4 --out policy.npz
    python evaluate.py --me policy --policy policy.npz --opp heuristic --games 8

Then ship:   tar -czf submission.tar.gz main.py policy.npz
(only ship policy.npz if evaluate.py shows it beating the heuristic).
"""
import argparse
import copy
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M  # noqa: E402

REWARD_SCALE = 20000.0


# ------------------------------------------------------------------------------------
# Episode runner (executed inside worker processes)
# ------------------------------------------------------------------------------------
def _wrap(bot, errors):
    def fn(obs, cfg):
        try:
            return bot.act(obs, cfg)
        except Exception:  # keep the episode alive but count the bug
            errors.append(1)
            return {"farmer": ["PASS"], "hands": [], "market": []}
    return fn


def _make_opponent(spec, seed):
    kind = spec["kind"]
    if kind == "starter":
        return "starter"
    if kind == "heuristic":
        return M.Bot(P=spec.get("P"))
    if kind == "policy":
        return M.Bot(policy=M.MLPPolicy({k: np.asarray(v) for k, v in spec["W"].items()}), explore=False, seed=seed)
    raise ValueError(kind)


def run_episode(job):
    """job = dict(seed, learner(dict|None), opp(dict), seat, teacher(bool))  ->  result dict"""
    from kaggle_environments import make
    seed, seat = job["seed"], job["seat"]
    errors = []
    if job.get("teacher"):
        me = M.Bot(record=True, P=job.get("teacher_P"))
    else:
        W = {k: np.asarray(v) for k, v in job["learner"].items()}
        me = M.Bot(policy=M.MLPPolicy(W), explore=job.get("explore", True), record=True, seed=seed)
    opp = _make_opponent(job["opp"], seed + 1)
    me_fn = _wrap(me, errors)
    opp_fn = _wrap(opp, errors) if isinstance(opp, M.Bot) else opp
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([me_fn, opp_fn] if seat == 0 else [opp_fn, me_fn])
    last = env.steps[-1]
    r_me = float(last[seat].reward or 0.0)
    r_op = float(last[1 - seat].reward or 0.0)
    return {"records": me.records, "me": r_me, "opp": r_op, "errors": len(errors), "seed": seed}


def sample_heuristic_variant(rng):
    """Diverse opponents: perturb the heuristic's tunables."""
    return {
        "tile_rent": rng.choice([6.0, 12.0, 20.0]),
        "score_min": rng.choice([0.3, 0.45, 0.8]),
        "melon_cap": rng.choice([0, 60, 130, 250]),
        "animal_cutoff": rng.choice([5, 7, 10]),
        "max_hands": rng.choice([8, 12, 16]),
    }


# ------------------------------------------------------------------------------------
# numpy MLP with manual backprop + Adam
# ------------------------------------------------------------------------------------
HS = M.HEAD_SIZES
OFFS = np.concatenate([[0], np.cumsum(HS)])


def fwd(W, X):
    h1 = np.tanh(X @ W["w1"] + W["b1"])
    h2 = np.tanh(h1 @ W["w2"] + W["b2"])
    logits = h2 @ W["wp"] + W["bp"]
    value = (h2 @ W["wv"] + W["bv"])[:, 0]
    return logits, value, h1, h2


def head_softmax(logits):
    probs = []
    for i, hs in enumerate(HS):
        z = logits[:, OFFS[i]:OFFS[i] + hs]
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        probs.append(e / e.sum(axis=1, keepdims=True))
    return probs


def backward(W, X, h1, h2, dlogits, dvalue):
    g = {}
    g["wp"] = h2.T @ dlogits
    g["bp"] = dlogits.sum(0)
    g["wv"] = h2.T @ dvalue[:, None]
    g["bv"] = dvalue.sum(0, keepdims=True).reshape(1)
    dh2 = dlogits @ W["wp"].T + dvalue[:, None] @ W["wv"].T
    dz2 = dh2 * (1 - h2 ** 2)
    g["w2"] = h1.T @ dz2
    g["b2"] = dz2.sum(0)
    dh1 = dz2 @ W["w2"].T
    dz1 = dh1 * (1 - h1 ** 2)
    g["w1"] = X.T @ dz1
    g["b1"] = dz1.sum(0)
    return g


class Adam:
    def __init__(self, W, lr=3e-4):
        self.m = {k: np.zeros_like(v) for k, v in W.items()}
        self.v = {k: np.zeros_like(v) for k, v in W.items()}
        self.t = 0
        self.lr = lr

    def step(self, W, g, clip=1.0):
        norm = np.sqrt(sum(float((x ** 2).sum()) for x in g.values()))
        s = min(1.0, clip / (norm + 1e-8))
        self.t += 1
        for k in W:
            gk = g[k] * s
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk
            self.v[k] = 0.999 * self.v[k] + 0.001 * gk ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            W[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        return norm


# ------------------------------------------------------------------------------------
# Stage 1: behaviour cloning
# ------------------------------------------------------------------------------------
def collect_teacher(pool, n_games, seed0, opp_mix, rng):
    jobs = []
    for g in range(n_games):
        kind = opp_mix[g % len(opp_mix)]
        if kind == "variant":
            opp = {"kind": "heuristic", "P": sample_heuristic_variant(rng)}
        else:
            opp = {"kind": kind}
        # teacher also sees varied tunables so the dataset is not a single trajectory family
        tp = sample_heuristic_variant(rng) if g % 3 == 2 else None
        jobs.append({"seed": seed0 + g, "seat": g % 2, "opp": opp, "teacher": True, "teacher_P": tp})
    res = pool.map(run_episode, jobs) if pool else [run_episode(j) for j in jobs]
    X, Y = [], []
    for r in res:
        for rec in r["records"]:
            X.append(rec["feats"])
            Y.append(rec["acts"])
    print(f"teacher data: {len(X)} decisions from {n_games} games "
          f"(mean teacher score {np.mean([r['me'] for r in res]):.0f}, errors {sum(r['errors'] for r in res)})",
          flush=True)
    return np.asarray(X, dtype="float64"), np.asarray(Y, dtype="int64")


def bc_train(W, X, Y, epochs=300, lr=2e-3, batch=128, seed=0):
    rng = np.random.RandomState(seed)
    opt = Adam(W, lr)
    n = len(X)
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot, acc = 0.0, np.zeros(len(HS))
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            xb, yb = X[b], Y[b]
            logits, value, h1, h2 = fwd(W, xb)
            probs = head_softmax(logits)
            dlog = np.zeros_like(logits)
            loss = 0.0
            for i, hs in enumerate(HS):
                p = probs[i]
                t = yb[:, i]
                loss += -np.log(p[np.arange(len(b)), t] + 1e-9).mean()
                d = p.copy()
                d[np.arange(len(b)), t] -= 1.0
                dlog[:, OFFS[i]:OFFS[i] + hs] = d / len(b)
                acc[i] += (p.argmax(1) == t).sum()
            g = backward(W, xb, h1, h2, dlog, np.zeros(len(b)))
            opt.step(W, g)
            tot += loss * len(b)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  BC epoch {ep:4d}  loss/head {tot / n / len(HS):.3f}  acc/head {np.round(acc / n, 2)}", flush=True)
    return W


# ------------------------------------------------------------------------------------
# Stage 2: self-play PPO
# ------------------------------------------------------------------------------------
def build_batch(results, gamma=1.0, lam=0.95):
    feats, acts, logp, adv, ret = [], [], [], [], []
    for r in results:
        recs = r["records"]
        if len(recs) < 2:
            continue
        diffs = [rc["money"] - rc["opp_money"] for rc in recs] + [r["me"] - r["opp"]]
        rewards = [(diffs[i + 1] - diffs[i]) / REWARD_SCALE for i in range(len(recs))]
        win = 1.0 if r["me"] > r["opp"] else (-1.0 if r["me"] < r["opp"] else 0.0)
        rewards[-1] += 0.5 * win
        values = [rc["value"] for rc in recs] + [0.0]
        gae, advs = 0.0, [0.0] * len(recs)
        for i in reversed(range(len(recs))):
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            gae = delta + gamma * lam * gae
            advs[i] = gae
        for i, rc in enumerate(recs):
            feats.append(rc["feats"])
            acts.append(rc["acts"])
            logp.append(rc["logp"])
            adv.append(advs[i])
            ret.append(advs[i] + values[i])
    return (np.asarray(feats), np.asarray(acts, dtype="int64"), np.asarray(logp),
            np.asarray(adv), np.asarray(ret))


def ppo_update(W, opt, batch, epochs=4, minibatch=256, clip=0.2, vf=0.5, ent=0.01, seed=0):
    X, A, LP, ADV, RET = batch
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    rng = np.random.RandomState(seed)
    n = len(X)
    stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
    cnt = 0
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, minibatch):
            b = idx[s:s + minibatch]
            xb, ab, lpb, advb, retb = X[b], A[b], LP[b], ADV[b], RET[b]
            logits, value, h1, h2 = fwd(W, xb)
            probs = head_softmax(logits)
            idxs = np.arange(len(b))
            new_lp = np.zeros(len(b))
            for i in range(len(HS)):
                new_lp += np.log(probs[i][idxs, ab[:, i]] + 1e-12)
            ratio = np.exp(np.clip(new_lp - lpb, -20, 20))
            unclipped = ratio * advb
            clipped = np.clip(ratio, 1 - clip, 1 + clip) * advb
            use_unclipped = (unclipped <= clipped)           # gradient flows only through the active branch
            dlogp = np.where(use_unclipped, -advb * ratio, 0.0) / len(b)
            dlog = np.zeros_like(logits)
            ent_tot = 0.0
            for i, hs in enumerate(HS):
                p = probs[i]
                onehot = np.zeros_like(p)
                onehot[idxs, ab[:, i]] = 1.0
                d = (onehot - p) * dlogp[:, None]
                lp = np.log(p + 1e-12)
                H = -(p * lp).sum(1, keepdims=True)
                d += ent * p * (lp + H) / len(b)           # minimise -ent*H
                dlog[:, OFFS[i]:OFFS[i] + hs] = d
                ent_tot += H.mean()
            dvalue = vf * (value - retb) / len(b)
            g = backward(W, xb, h1, h2, dlog, dvalue)
            opt.step(W, g)
            stats["pi"] += float(-np.minimum(unclipped, clipped).mean())
            stats["v"] += float(0.5 * ((value - retb) ** 2).mean())
            stats["ent"] += float(ent_tot / len(HS))
            stats["kl"] += float((lpb - new_lp).mean())
            stats["clipfrac"] += float((~use_unclipped).mean())
            cnt += 1
    return {k: v / max(1, cnt) for k, v in stats.items()}


def eval_policy(pool, W, n_games, seed0, opp="heuristic"):
    jobs = [{"seed": seed0 + g, "seat": g % 2, "learner": W, "explore": False, "opp": {"kind": opp}}
            for g in range(n_games)]
    res = pool.map(run_episode, jobs) if pool else [run_episode(j) for j in jobs]
    wins = sum(r["me"] > r["opp"] for r in res)
    diff = float(np.mean([r["me"] - r["opp"] for r in res]))
    return wins / n_games, diff, float(np.mean([r["me"] for r in res]))


def cmd_bc(args, pool):
    rng = random.Random(args.seed)
    X, Y = collect_teacher(pool, args.games, 5000 + args.seed * 1000, ["starter", "heuristic", "variant"], rng)
    pol = M.MLPPolicy.init(hidden=args.hidden, seed=args.seed)
    W = bc_train(pol.W, X, Y, epochs=args.epochs)
    M.MLPPolicy(W).save(args.out)
    print("saved", args.out)
    wr, diff, me = eval_policy(pool, W, args.eval_games, 9000)
    print(f"BC policy vs heuristic: win-rate {wr:.2f}  mean diff {diff:.0f}  mean score {me:.0f}")


def cmd_ppo(args, pool):
    rng = random.Random(args.seed)
    W = M.MLPPolicy.load(args.init).W if args.init else M.MLPPolicy.init(hidden=args.hidden, seed=args.seed).W
    W = {k: np.array(v, dtype="float64") for k, v in W.items()}
    opt = Adam(W, args.lr)
    snapshots = []
    best = -1e18
    seed_ctr = 100000 + args.seed * 100000
    t0 = time.time()
    for it in range(args.iters):
        jobs = []
        for g in range(args.games_per_iter):
            roll = rng.random()
            if roll < 0.40:
                opp = {"kind": "heuristic"}
            elif roll < 0.65:
                opp = {"kind": "heuristic", "P": sample_heuristic_variant(rng)}
            elif roll < 0.85 and snapshots:
                opp = {"kind": "policy", "W": rng.choice(snapshots)}
            elif roll < 0.92:
                opp = {"kind": "starter"}
            else:
                opp = {"kind": "policy", "W": copy.deepcopy(W)}      # current self
            jobs.append({"seed": seed_ctr, "seat": g % 2, "learner": copy.deepcopy(W), "explore": True, "opp": opp})
            seed_ctr += 1
        res = pool.map(run_episode, jobs) if pool else [run_episode(j) for j in jobs]
        batch = build_batch(res)
        st = ppo_update(W, opt, batch, epochs=args.epochs, seed=it)
        wins = np.mean([r["me"] > r["opp"] for r in res])
        diff = np.mean([r["me"] - r["opp"] for r in res])
        errs = sum(r["errors"] for r in res)
        print(f"iter {it:3d}  win {wins:.2f}  diff {diff:9.0f}  pi {st['pi']:+.3f}  v {st['v']:.3f}  "
              f"ent {st['ent']:.2f}  kl {st['kl']:+.4f}  clip {st['clipfrac']:.2f}  err {errs}  [{time.time()-t0:.0f}s]",
              flush=True)
        if (it + 1) % args.snap_every == 0:
            snapshots.append(copy.deepcopy(W))
            snapshots = snapshots[-8:]
        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            wr, d, me = eval_policy(pool, W, args.eval_games, 9000)
            print(f"   eval vs heuristic: win-rate {wr:.2f}  mean diff {d:.0f}  mean score {me:.0f}", flush=True)
            if d > best:
                best = d
                M.MLPPolicy({k: v.copy() for k, v in W.items()}).save(args.out)
                print(f"   saved {args.out} (best diff {best:.0f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bc", "ppo"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--hidden", type=int, default=96)
        p.add_argument("--out", default="policy.npz")
        p.add_argument("--eval-games", type=int, default=6)
    sub.choices["bc"].add_argument("--games", type=int, default=30)
    sub.choices["bc"].add_argument("--epochs", type=int, default=300)
    pp = sub.choices["ppo"]
    pp.add_argument("--init", default=None)
    pp.add_argument("--iters", type=int, default=30)
    pp.add_argument("--games-per-iter", type=int, default=16)
    pp.add_argument("--epochs", type=int, default=4)
    pp.add_argument("--lr", type=float, default=1e-4)
    pp.add_argument("--snap-every", type=int, default=3)
    pp.add_argument("--eval-every", type=int, default=5)
    args = ap.parse_args()

    if args.workers > 1:
        start_methods = mp.get_all_start_methods()
        start_method = "fork" if "fork" in start_methods else "spawn"
        pool = mp.get_context(start_method).Pool(args.workers)
    else:
        pool = None
    try:
        {"bc": cmd_bc, "ppo": cmd_ppo}[args.cmd](args, pool)
    finally:
        if pool:
            pool.close()


if __name__ == "__main__":
    main()
