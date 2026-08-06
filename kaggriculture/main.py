"""
main.py - Kaggle submission entrypoint for the Kaggriculture PPO agent.

Submission layout (multi-file, per AGENTS.md):

    tar -czf submission.tar.gz main.py kagg_common.py ppo_kaggriculture_final.zip
    kaggle competitions submit kaggriculture -f submission.tar.gz -m "PPO v1"

`main.py` MUST be at the tar root and define a top-level `agent(obs)` — this
file satisfies that contract. `kagg_common.py` (identical encoding/decoding
logic used by train.py) and the model `.zip` must sit next to it in the same
archive; we locate them relative to this file's own directory so the archive
can be extracted anywhere Kaggle chooses to unpack it.

Robustness: any failure to load the model, or any exception while producing
an action (malformed obs, missing model file, incompatible SB3 version,
etc.), falls back to a cheap deterministic heuristic agent rather than
raising - a crash forfeits the match, a mediocre fallback move does not.
"""
import os
import sys
import traceback

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import kagg_common as kc  # noqa: E402  (must come after sys.path fixup)

# --------------------------------------------------------------------------
# Lazy, cached model load. Kaggle calls `agent(obs)` once per turn for up to
# 720 turns in a single process, so we load the model exactly once and reuse
# it - re-loading per call would blow the per-step time budget.
# --------------------------------------------------------------------------
_MODEL = None
_MODEL_LOAD_FAILED = False

_CANDIDATE_MODEL_NAMES = [
    "ppo_kaggriculture_final.zip",
    "ppo_kaggriculture_final",
    "best_model.zip",
    "model.zip",
]


def _find_model_path():
    search_dirs = [_THIS_DIR, os.getcwd(), "/kaggle_simulations/agent"]
    for d in search_dirs:
        for name in _CANDIDATE_MODEL_NAMES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None


def _load_model():
    global _MODEL, _MODEL_LOAD_FAILED
    if _MODEL is not None or _MODEL_LOAD_FAILED:
        return _MODEL
    try:
        # Imported lazily: if stable_baselines3/torch aren't available in the
        # eval sandbox we still want the module itself to import cleanly so
        # the heuristic fallback can run.
        import torch
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker

        model_path = _find_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"No model checkpoint found near {_THIS_DIR} "
                f"(looked for {_CANDIDATE_MODEL_NAMES})"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = MaskablePPO.load(model_path, device=device)
        _MODEL.policy.set_training_mode(False)
        print(f"[main.py] Loaded PPO model from {model_path} on {device}", file=sys.stderr)
    except Exception:
        _MODEL_LOAD_FAILED = True
        traceback.print_exc(file=sys.stderr)
        print("[main.py] Falling back to heuristic agent for the rest of the game.", file=sys.stderr)
    return _MODEL


# --------------------------------------------------------------------------
# Heuristic fallback agent (also doubles as a sane "turn 0" no-op-free move
# if the model ever produces something degenerate). Mirrors the wheat-loop
# example from AGENTS.md, generalized slightly: keep buying/planting/
# watering/harvesting wheat, and sell whatever lands in the shed.
# --------------------------------------------------------------------------
def _heuristic_agent(obs):
    try:
        player = obs["player"]
        me = obs["farms"][player]
        private = obs["private"]
        fx, fy = me["farmer"]
        tile = me["tiles"][fy][fx]

        market = []
        if private["seeds"].get("WHEAT", 0) == 0 and me["money"] >= 10:
            market.append(["BUY_SEED", "WHEAT", 5])
        wheat_in_shed = private["shed"].get("WHEAT", 0)
        if wheat_in_shed > 0:
            market.append(["SELL", "WHEAT", wheat_in_shed])

        if tile is None and private["seeds"].get("WHEAT", 0) > 0:
            return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": market}
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop_age = obs["day"] - tile["planted_day"]
            if crop_age >= 2:
                return {"farmer": ["HARVEST"], "hands": [], "market": market}
            if not tile.get("watered_today", False):
                return {"farmer": ["WATER"], "hands": [], "market": market}

        return {"farmer": ["PASS"], "hands": [], "market": market}
    except Exception:
        # Last-resort absolute-safe no-op.
        return {"farmer": ["PASS"], "hands": [], "market": []}


# --------------------------------------------------------------------------
# Top-level entrypoint the kaggle_environments runner calls each turn.
# --------------------------------------------------------------------------
def agent(obs):
    model = _load_model()
    if model is None:
        return _heuristic_agent(obs)

    try:
        player = obs["player"]
        obs_vec = kc.encode_observation(obs, player)
        # model.predict expects a batch dimension; deterministic=True for
        # stable, reproducible play (no exploration noise at inference time).
        action, _state = model.predict(obs_vec[None, :], deterministic=True)
        action = np.asarray(action).reshape(-1)
        action_dict = kc.decode_action(action, obs, player)
        return action_dict
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return _heuristic_agent(obs)


# Allow quick local smoke-testing: `python main.py`
if __name__ == "__main__":
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"episodeSteps": 100}, debug=True)
    env.run([agent, "random"])
    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")