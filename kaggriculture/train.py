"""
train.py - Day-level MaskablePPO for Kaggriculture (v5, hybrid).

One RL step = one in-game day. The scripted controller in kagg_control.py
runs the 24 turns in between, so the policy only ever decides strategy.

Key differences from v4:
  * 30 steps/episode instead of 720 -> gamma=0.99 spans the whole season.
  * Reward is unshaped bank delta + terminal margin. No per-step bonuses to
    farm, no terminal spike 500x larger than every other step (that was what
    made value_loss oscillate between 0.1 and 1364).
  * Opponents are strong fixed plans (the sweep winners), not a wheat bot,
    plus optional self-play against frozen checkpoints.
  * Episode counters live in the env, so the curriculum advances per-env
    rather than being silently divided by n_envs.
"""
import argparse
import multiprocessing
import os
import time
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from kaggle_environments import make as kaggle_make

import kagg_common as kc
import kagg_control as kcx


# ==========================================================================
# Scripted opponents
# ==========================================================================
def fixed_plan_agent(plan):
    def _agent(obs):
        action, _ = kcx.act(obs, obs["player"], plan)
        return action
    return _agent


OPPONENT_PLANS = {
    # Strongest single-plan baselines found by sweeping fixed strategies.
    "melon": kcx.DayPlan("MELON", 8, True, None, 0, False, 100),
    "wheat": kcx.DayPlan("WHEAT", 6, False, None, 0, False, 25),
    "wheat_land": kcx.DayPlan("WHEAT", 8, True, None, 0, False, 100),
}


def make_opponent(name):
    if name in OPPONENT_PLANS:
        return fixed_plan_agent(OPPONENT_PLANS[name])
    return name  # "random" / "pass" / "starter" resolve inside kaggle_environments


# ==========================================================================
# Environment
# ==========================================================================
class KaggricultureDayEnv(gym.Env):
    """One gym step == one in-game day (24 engine turns)."""

    metadata = {"render_modes": []}

    def __init__(self, opponent="melon", episode_steps=720, player=0, seed=None,
                 curriculum=True):
        super().__init__()
        self.opponent_name = opponent
        self.episode_steps = episode_steps
        self.turns_per_day = 24
        self.player = player
        self.curriculum = curriculum

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(kc.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(kc.action_nvec())

        self._env = None
        self._trainer = None
        self._raw_obs = None
        self._episode_count = 0
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    def _pick_opponent(self):
        if not self.curriculum:
            return self.opponent_name
        # Ramp difficulty, then mix so the policy doesn't overfit one style.
        if self._episode_count < 50:
            return "random"
        if self._episode_count < 200:
            return "wheat"
        return str(self._rng.choice(["melon", "wheat_land", "wheat"]))

    def _new_trainer(self):
        opp = make_opponent(self._pick_opponent())
        self._env = kaggle_make(
            "kaggriculture",
            configuration={"episodeSteps": self.episode_steps},
            debug=False,
        )
        pair = [None, opp] if self.player == 0 else [opp, None]
        self._trainer = self._env.train(pair)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._new_trainer()
        self._raw_obs = self._trainer.reset()
        return kc.encode_observation(self._raw_obs, self.player), {}

    def _money(self, obs):
        farms = kcx._g(obs, "farms", [{}, {}])
        return float(kcx._g(farms[self.player] if self.player < len(farms) else {},
                            "money", 0.0))

    def step(self, action):
        plan = kc.decode_plan(action)
        prev_money = self._money(self._raw_obs)

        done = False
        info = {}
        # Hold the plan fixed and let the controller run out the in-game day.
        for _ in range(self.turns_per_day):
            act_dict, _counts = kcx.act(self._raw_obs, self.player, plan)
            result = self._trainer.step(act_dict)
            if len(result) == 4:
                raw_obs, _r, done, info = result
            else:
                raw_obs, _r, term, trunc, info = result
                done = bool(term or trunc)
            self._raw_obs = raw_obs
            if done:
                break

        curr_money = self._money(self._raw_obs)
        outcome = kc.episode_outcome_info(self._raw_obs, self.player)
        reward = kc.compute_reward(
            prev_money, curr_money, done,
            outcome["final_money"], outcome["opp_final_money"],
        )

        info = dict(info) if info else {}
        if done:
            info.update(outcome)
            self._episode_count += 1

        obs_vec = kc.encode_observation(self._raw_obs, self.player)
        return obs_vec, reward, bool(done), False, info

    def close(self):
        self._env = None
        self._trainer = None


def mask_fn(env):
    return kc.get_action_mask(env._raw_obs, env.player)


def make_env(rank, opponent, episode_steps, seed, curriculum):
    def _init():
        e = KaggricultureDayEnv(
            opponent=opponent, episode_steps=episode_steps,
            player=0, seed=seed + rank, curriculum=curriculum,
        )
        return ActionMasker(e, mask_fn)
    return _init


# ==========================================================================
# Logging
# ==========================================================================
class ProgressCallback(BaseCallback):
    def __init__(self, window=50, verbose=1):
        super().__init__(verbose)
        self.money = deque(maxlen=window)
        self.opp_money = deque(maxlen=window)
        self.wins = deque(maxlen=window)
        self.rews = deque(maxlen=window)
        self._t0 = None
        self._s0 = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "final_money" in info:
                self.money.append(info["final_money"])
                self.opp_money.append(info["opp_final_money"])
                self.wins.append(info["win"])
            if "episode" in info:
                self.rews.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self):
        now = time.time()
        fps = None
        if self._t0 is not None:
            fps = (self.num_timesteps - self._s0) / max(now - self._t0, 1e-9)
        self._t0, self._s0 = now, self.num_timesteps

        def m(buf):
            return float(np.mean(buf)) if len(buf) else float("nan")

        print("-" * 78)
        print(f"steps={self.num_timesteps:>9,d}" + (f"  fps={fps:5.1f}" if fps else ""))
        print(f"  bank      : ${m(self.money):>10,.0f}   opp ${m(self.opp_money):>10,.0f}")
        print(f"  win_rate  : {m(self.wins):.3f}   ep_rew {m(self.rews):8.2f}   (n={len(self.wins)})")
        lg = self.model.logger.name_to_value
        print(f"  value_loss={lg.get('train/value_loss', float('nan')):.4f}  "
              f"expl_var={lg.get('train/explained_variance', float('nan')):.3f}  "
              f"entropy={lg.get('train/entropy_loss', float('nan')):.3f}", flush=True)


# ==========================================================================
# Main
# ==========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=12)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--episode-steps", type=int, default=720)
    p.add_argument("--opponent", type=str, default="melon")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=25_000)
    p.add_argument("--models-dir", type=str, default="models")
    p.add_argument("--tb-dir", type=str, default="tb_logs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.tb_dir, exist_ok=True)

    print(f"device={args.device}  n_envs={args.n_envs}")
    print(f"OBS_DIM={kc.OBS_DIM}  action_nvec={kc.action_nvec()}")
    print(f"~{args.total_timesteps // 30:,} episodes at 30 steps/episode")

    # The policy net is small on purpose: 64 inputs, 7 heads. Compute here is
    # bound by the game engine on CPU, not by the network, so device matters
    # far less than n_envs.
    vec = SubprocVecEnv(
        [make_env(i, args.opponent, args.episode_steps, args.seed,
                  not args.no_curriculum) for i in range(args.n_envs)],
        start_method="spawn",
    )
    vec = VecMonitor(vec)

    if args.resume:
        model = MaskablePPO.load(args.resume, env=vec, device=args.device)
    else:
        model = MaskablePPO(
            "MlpPolicy", vec,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]),
                               activation_fn=torch.nn.Tanh),
            tensorboard_log=args.tb_dir,
            device=args.device,
            verbose=0,
            seed=args.seed,
        )

    cbs = CallbackList([
        CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=args.models_dir, name_prefix="ppo_kagg",
        ),
        ProgressCallback(),
    ])

    model.learn(total_timesteps=args.total_timesteps, callback=cbs,
                tb_log_name="ppo_kagg", reset_num_timesteps=args.resume is None,
                progress_bar=True)

    final = os.path.join(args.models_dir, "ppo_kaggriculture_final")
    model.save(final)
    print(f"saved {final}.zip")
    vec.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()