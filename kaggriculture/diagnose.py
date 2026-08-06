"""
diagnose.py - Call the policy OUTSIDE kaggle_environments so the traceback
is visible. The env swallows agent exceptions when debug=False, which is
why evaluate.py silently reported a flat $3,000.

    python diagnose.py --model models/ppo_kagg_80000_steps.zip
"""
import argparse
import traceback

import numpy as np
from kaggle_environments import make

import kagg_common as kc
import kagg_control as kcx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    args = p.parse_args()

    print("1. loading model ...")
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model, device="cpu")
    print("   OK")
    print(f"   policy obs_space   : {model.observation_space}")
    print(f"   policy action_space: {model.action_space}")
    print(f"   expected OBS_DIM   : {kc.OBS_DIM}")
    print(f"   expected nvec      : {kc.action_nvec()}  (mask dim {kc.MASK_DIM})")

    print("\n2. building a live observation ...")
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
    t = env.train([None, "random"])
    obs = t.reset()
    vec = kc.encode_observation(obs, 0)
    mask = kc.get_action_mask(obs, 0)
    print(f"   vec  {vec.shape} {vec.dtype}  finite={np.isfinite(vec).all()}")
    print(f"   mask {mask.shape} {mask.dtype}  n_legal={int(mask.sum())}")

    print("\n3. predict WITH masks ...")
    try:
        a, _ = model.predict(vec[None, :], action_masks=mask[None, :],
                             deterministic=True)
        print(f"   OK -> {np.asarray(a).reshape(-1)}  plan={kc.decode_plan(np.asarray(a).reshape(-1))}")
    except Exception:
        traceback.print_exc()

    print("\n4. predict WITHOUT masks (isolates the masking call) ...")
    try:
        a, _ = model.predict(vec[None, :], deterministic=True)
        print(f"   OK -> {np.asarray(a).reshape(-1)}")
    except Exception:
        traceback.print_exc()

    print("\n5. unbatched obs (some sb3 versions prefer this) ...")
    try:
        a, _ = model.predict(vec, action_masks=mask, deterministic=True)
        print(f"   OK -> {np.asarray(a).reshape(-1)}")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()