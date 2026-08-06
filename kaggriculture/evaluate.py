"""
evaluate.py - Measure what you will actually submit.

Training logs report performance under STOCHASTIC sampling. main.py runs
deterministic=True. At ~4.8 nats of policy entropy those are different
agents, and the argmax policy is the one that gets scored on Kaggle.

Also reports variance across seeds -- single-seed results in this game swing
wildly, because market prices are path-dependent and shop unlocks are random.

Usage:
    python evaluate.py --model models/ppo_kaggriculture_final.zip --episodes 10
    python evaluate.py --model models/ppo_kaggriculture_final.zip --plans-only
"""
import argparse
import statistics

import numpy as np
from kaggle_environments import make

import kagg_common as kc
import kagg_control as kcx


BASELINES = {
    "random": "random",
    "starter": "starter",
    "melon": kcx.DayPlan("MELON", 8, True, None, 0, False, 100),
    "wheat": kcx.DayPlan("WHEAT", 6, False, None, 0, False, 25),
    "wheat_land": kcx.DayPlan("WHEAT", 8, True, None, 0, False, 100),
}


def plan_agent(plan):
    def _a(obs):
        return kcx.act(obs, obs["player"], plan)[0]
    return _a


def resolve(opp):
    return opp if isinstance(opp, str) else plan_agent(opp)


def model_agent(model, deterministic=True):
    """Queries the policy once per in-game day, exactly like main.py."""
    cache = {"day": -1, "plan": None}

    def _a(obs):
        player = obs["player"]
        day = int(kcx._g(obs, "day", 0))
        if cache["plan"] is None or cache["day"] != day:
            vec = kc.encode_observation(obs, player)
            mask = kc.get_action_mask(obs, player)
            action, _ = model.predict(
                vec[None, :], action_masks=mask[None, :],
                deterministic=deterministic,
            )
            cache["plan"] = kc.decode_plan(np.asarray(action).reshape(-1))
            cache["day"] = day
        return kcx.act(obs, player, cache["plan"])[0]
    return _a


def duel(agent0, agent1, seed):
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run([agent0, agent1])
    final = env.steps[-1]
    return float(final[0].reward or 0.0), float(final[1].reward or 0.0)


def report(name, banks, opp_banks):
    wins = sum(1 for a, b in zip(banks, opp_banks) if a > b)
    print(f"  {name:<22s} ${statistics.mean(banks):>9,.0f} "
          f"(sd {statistics.pstdev(banks):>7,.0f})  "
          f"opp ${statistics.mean(opp_banks):>9,.0f}  "
          f"win {wins}/{len(banks)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--plans-only", action="store_true",
                   help="Benchmark fixed plans against each other; no model needed.")
    args = p.parse_args()

    seeds = list(range(1, args.episodes + 1))

    if args.plans_only or args.model is None:
        print("Fixed-plan baselines vs 'random':")
        for name, plan in BASELINES.items():
            if isinstance(plan, str):
                continue
            res = [duel(plan_agent(plan), "random", s) for s in seeds]
            report(name, [r[0] for r in res], [r[1] for r in res])
        return

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model, device="cpu")
    model.policy.set_training_mode(False)

    for mode in (True, False):
        label = "DETERMINISTIC" if mode else "STOCHASTIC"
        print(f"\n{label} policy over {args.episodes} seeds:")
        for name, opp in BASELINES.items():
            agent = model_agent(model, deterministic=mode)
            res = [duel(agent, resolve(opp), s) for s in seeds]
            report(f"vs {name}", [r[0] for r in res], [r[1] for r in res])

    # Mirror match: the closest available proxy for a real leaderboard
    # opponent, and the check that matters most before submitting.
    print("\nSelf-play (deterministic vs stochastic):")
    res = [duel(model_agent(model, True), model_agent(model, False), s) for s in seeds]
    report("det vs stoch", [r[0] for r in res], [r[1] for r in res])


if __name__ == "__main__":
    main()