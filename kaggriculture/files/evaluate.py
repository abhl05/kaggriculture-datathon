"""
Local evaluation harness.

    python evaluate.py --games 6                       # heuristic vs starter
    python evaluate.py --games 6 --policy policy.npz   # learned macro vs heuristic
    python evaluate.py --games 4 --opp heuristic

Plays each matchup with seats swapped and reports mean final money + win rate.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_environments import make  # noqa: E402

import main as M  # noqa: E402


def wrap(bot):
    def fn(obs, cfg):
        try:
            return bot.act(obs, cfg)
        except Exception as e:  # surface bugs during local testing
            import traceback
            traceback.print_exc()
            raise
    return fn


def make_agent(kind, args, seed):
    if kind == "heuristic":
        return wrap(M.Bot())
    if kind == "policy":
        pol = M.MLPPolicy.load(args.policy)
        return wrap(M.Bot(policy=pol, explore=False, seed=seed))
    return kind  # "starter" | "random" | "pass"


def play(a, b, seed, steps=720):
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=True)
    env.run([a, b])
    last = env.steps[-1]
    return [float(s.reward or 0.0) for s in last], [s.status for s in last]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--me", default="heuristic", help="heuristic | policy")
    ap.add_argument("--opp", default="starter", help="starter | random | pass | heuristic | policy")
    ap.add_argument("--policy", default="policy.npz")
    ap.add_argument("--steps", type=int, default=720)
    args = ap.parse_args()

    wins = draws = 0
    tot_me = tot_opp = 0.0
    t0 = time.time()
    for g in range(args.games):
        seed = 1000 + g
        me = make_agent(args.me, args, seed)
        op = make_agent(args.opp, args, seed + 77)
        if g % 2 == 0:
            r, st = play(me, op, seed, args.steps)
            rm, ro = r[0], r[1]
        else:
            r, st = play(op, me, seed, args.steps)
            rm, ro = r[1], r[0]
        tot_me += rm
        tot_opp += ro
        wins += rm > ro
        draws += rm == ro
        print(f"game {g}: me={rm:9.0f} opp={ro:9.0f}  status={st}  ({time.time()-t0:.0f}s)", flush=True)
    n = args.games
    print(f"\nmean me={tot_me/n:.0f}  mean opp={tot_opp/n:.0f}  wins={wins}/{n} draws={draws}")


if __name__ == "__main__":
    main()
