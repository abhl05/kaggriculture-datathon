"""
main.py - Kaggle submission entrypoint (hybrid agent).

    tar -czf submission.tar.gz main.py kagg_common.py kagg_control.py \
        ppo_kaggriculture_final.zip
    kaggle competitions submit kaggriculture -f submission.tar.gz -m "hybrid v5"

The policy is consulted once per in-game day (hour == 0) and its DayPlan is
cached; every turn is executed by the scripted controller. If the model is
missing or fails to load, the controller runs a strong fixed fallback plan
instead of a crash-forfeit.

FIX vs the previous submission: model.predict is now called WITH
action_masks. Previously it was called without, so a policy trained under
masking sampled from a distribution it had never seen at evaluation time --
enough on its own to wreck an otherwise-fine model.
"""
import os
import sys
import traceback

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import kagg_control as kcx  # noqa: E402
import kagg_common as kc    # noqa: E402

_MODEL = None
_LOAD_FAILED = False
_CACHED_PLAN = None
_CACHED_DAY = -1

# Best fixed plan from the strategy sweep; used when no model is available.
FALLBACK_PLAN = kcx.DayPlan(
    crop="MELON", hire_target=8, buy_land=True,
    animal=None, sell_mode=0, fertilize=False, plant_cap=100,
)

_CANDIDATES = ["ppo_kaggriculture_final.zip", "ppo_kaggriculture_final",
               "best_model.zip", "model.zip"]


def _find_model():
    for d in (_THIS_DIR, os.getcwd(), "/kaggle_simulations/agent"):
        for n in _CANDIDATES:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    return None


def _load():
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL
    try:
        from sb3_contrib import MaskablePPO
        path = _find_model()
        if path is None:
            raise FileNotFoundError("no checkpoint found")
        _MODEL = MaskablePPO.load(path, device="cpu")
        _MODEL.policy.set_training_mode(False)
        print(f"[main] loaded {path}", file=sys.stderr)
    except Exception:
        _LOAD_FAILED = True
        traceback.print_exc(file=sys.stderr)
        print("[main] using fallback plan", file=sys.stderr)
    return _MODEL


def _plan_for_day(obs, player):
    """Query the policy once per day; reuse the plan for the other 23 turns."""
    global _CACHED_PLAN, _CACHED_DAY
    day = int(kcx._g(obs, "day", 0))
    if _CACHED_PLAN is not None and _CACHED_DAY == day:
        return _CACHED_PLAN

    plan = FALLBACK_PLAN
    model = _load()
    if model is not None:
        try:
            vec = kc.encode_observation(obs, player)
            mask = kc.get_action_mask(obs, player)
            action, _ = model.predict(
                vec[None, :],
                action_masks=mask[None, :],   # <-- the fix
                deterministic=True,
            )
            plan = kc.decode_plan(np.asarray(action).reshape(-1))
        except Exception:
            traceback.print_exc(file=sys.stderr)
            plan = FALLBACK_PLAN

    _CACHED_PLAN, _CACHED_DAY = plan, day
    return plan


def agent(obs):
    try:
        player = obs["player"]
        plan = _plan_for_day(obs, player)
        action, _counts = kcx.act(obs, player, plan)
        return action
    except Exception:
        traceback.print_exc(file=sys.stderr)
        try:
            action, _ = kcx.act(obs, obs["player"], FALLBACK_PLAN)
            return action
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}


if __name__ == "__main__":
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
    env.run([agent, "random"])
    for i, s in enumerate(env.steps[-1]):
        print(f"Player {i}: reward={s.reward} status={s.status}")