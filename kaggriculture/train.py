"""
train.py - PPO training pipeline for Kaggriculture.

Usage:
    python train.py --total-timesteps 5000000 --n-envs 32 --opponent random

Requires:
    pip install -U kaggle-environments stable-baselines3[extra] gymnasium torch

Hardware target: single RTX 5060 Ti, 16GB VRAM. The bottleneck for this game
is NOT GPU compute (the policy net is small relative to 16GB) - it's CPU-side
environment stepping (kaggle_environments is pure Python). We therefore lean
on many parallel `SubprocVecEnv` workers (one per CPU core) feeding a modestly
sized MLP so the GPU is busy doing large-batch backprop between rollouts
instead of sitting idle waiting on single-threaded env.step() calls.
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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from kaggle_environments import make as kaggle_make

import kagg_common as kc


# ==============================================================================
# 1. Gymnasium wrapper around the Kaggriculture kaggle_environments trainer
# ==============================================================================
class KaggricultureEnv(gym.Env):
    """Single-agent gym.Env view of the two-player Kaggriculture game.

    We always train from the point of view of `player` (default 0), with the
    OTHER seat driven by a fixed built-in opponent ("random" or "starter")
    via kaggle_environments' `env.train([None, opponent])` helper - this is
    the standard pattern kaggle_environments uses for all of its two-player
    games (see connectx, etc). `opponent` can also be a *list* of opponent
    names; a fresh one is chosen uniformly at random each `reset()`, which is
    a cheap form of opponent diversity / curriculum.
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent="random", episode_steps=720, player=0, seed=None):
        super().__init__()
        self.opponents = [opponent] if isinstance(opponent, str) else list(opponent)
        self.episode_steps = episode_steps
        self.player = player
        self._np_random_seed = seed

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(kc.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(kc.action_nvec())

        self._kaggle_env = None
        self._trainer = None
        self._raw_obs = None
        self._t = 0
        self._rng = np.random.default_rng(seed)

    def _new_trainer(self):
        opp = self.opponents[self._rng.integers(0, len(self.opponents))]
        self._kaggle_env = kaggle_make(
            "kaggriculture", configuration={"episodeSteps": self.episode_steps}, debug=False
        )
        self._trainer = self._kaggle_env.train([None, opp] if self.player == 0 else [opp, None])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._new_trainer()
        self._raw_obs = self._trainer.reset()
        self._t = 0
        obs_vec = kc.encode_observation(self._raw_obs, self.player)
        return obs_vec, {}

    def step(self, action):
        prev_obs = self._raw_obs
        action_dict = kc.decode_action(action, prev_obs, self.player)

        result = self._trainer.step(action_dict)
        # kaggle_environments trainers return the legacy 4-tuple (obs, reward, done, info)
        if len(result) == 4:
            raw_obs, _env_reward, done, info = result
        else:
            raw_obs, _env_reward, terminated, truncated, info = result
            done = terminated or truncated
        self._raw_obs = raw_obs
        self._t += 1

        reward = kc.compute_reward(prev_obs, raw_obs, self.player)
        truncated = False
        terminated = bool(done)

        info = dict(info) if info else {}
        if terminated or truncated:
            info.update(kc.episode_outcome_info(raw_obs, self.player))

        obs_vec = kc.encode_observation(raw_obs, self.player)
        return obs_vec, reward, terminated, truncated, info

    def close(self):
        self._kaggle_env = None
        self._trainer = None


def mask_fn(env: gym.Env) -> np.ndarray:
    return kc.get_action_mask(env._raw_obs, env.player)

def make_env(rank, opponents, episode_steps, seed, log_dir):
    def _init():
        env = KaggricultureEnv(
            opponent=opponents, episode_steps=episode_steps, player=0, seed=seed + rank
        )
        env = ActionMasker(env, mask_fn) # <-- ADD THIS WRAPPER
        env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}"))
        return env
    set_random_seed(seed)
    return _init


# ==============================================================================
# 2. Extremely verbose logging callback
# ==============================================================================
class VerboseTrainingCallback(BaseCallback):
    """Prints rich per-rollout diagnostics: reward stats, bank balances,
    win-rate vs the built-in opponent, episode length, FPS, and learning-rate
    / clip-range schedule values. Pulls the extra `final_money` / `win` keys
    straight out of the VecEnv `infos` list at episode boundaries (Monitor
    preserves custom info keys alongside its own `episode` key)."""

    def __init__(self, print_freq_rollouts=1, window=100, verbose=1):
        super().__init__(verbose)
        self.print_freq_rollouts = print_freq_rollouts
        self.rollout_count = 0
        self.window = window
        self.money_buf = deque(maxlen=window)
        self.opp_money_buf = deque(maxlen=window)
        self.win_buf = deque(maxlen=window)
        self.ep_reward_buf = deque(maxlen=window)
        self.ep_len_buf = deque(maxlen=window)
        self._last_time = None
        self._last_steps = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "final_money" in info:
                self.money_buf.append(info["final_money"])
                self.opp_money_buf.append(info["opp_final_money"])
                self.win_buf.append(info["win"])
            if "episode" in info:
                self.ep_reward_buf.append(info["episode"]["r"])
                self.ep_len_buf.append(info["episode"]["l"])
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        if self.rollout_count % self.print_freq_rollouts != 0:
            return

        now = time.time()
        steps = self.num_timesteps
        fps = None
        if self._last_time is not None:
            dt = now - self._last_time
            fps = (steps - self._last_steps) / max(dt, 1e-9)
        self._last_time, self._last_steps = now, steps

        def _stat(buf):
            return (float(np.mean(buf)), float(np.std(buf))) if len(buf) else (float("nan"), float("nan"))

        r_mean, r_std = _stat(self.ep_reward_buf)
        l_mean, _ = _stat(self.ep_len_buf)
        money_mean, money_std = _stat(self.money_buf)
        opp_money_mean, _ = _stat(self.opp_money_buf)
        win_rate = float(np.mean(self.win_buf)) if len(self.win_buf) else float("nan")

        print("=" * 96)
        print(f"[rollout {self.rollout_count:5d}] timesteps={steps:>10,d}"
              + (f"  fps={fps:6.1f}" if fps else ""))
        print(f"  episode_reward   : mean={r_mean:9.2f}  std={r_std:8.2f}  (n={len(self.ep_reward_buf)})")
        print(f"  episode_length   : mean={l_mean:9.1f}")
        print(f"  final_bank($)    : mean={money_mean:9.1f}  std={money_std:8.1f}  vs opp mean={opp_money_mean:9.1f}")
        print(f"  win_rate         : {win_rate:6.3f}  (n={len(self.win_buf)}, over last {self.window} episodes)")
        if len(self.model.ep_info_buffer):
            print(f"  sb3 ep_info_buf  : mean_reward={np.mean([e['r'] for e in self.model.ep_info_buffer]):9.2f}")
        print(f"  policy loss      : {self.model.logger.name_to_value.get('train/policy_gradient_loss', float('nan')):.5f}")
        print(f"  value loss       : {self.model.logger.name_to_value.get('train/value_loss', float('nan')):.5f}")
        print(f"  entropy loss     : {self.model.logger.name_to_value.get('train/entropy_loss', float('nan')):.5f}")
        print(f"  approx_kl        : {self.model.logger.name_to_value.get('train/approx_kl', float('nan')):.5f}")
        print(f"  clip_fraction    : {self.model.logger.name_to_value.get('train/clip_fraction', float('nan')):.5f}")
        print(f"  learning_rate    : {self.model.logger.name_to_value.get('train/learning_rate', float('nan')):.2e}")
        print("=" * 96, flush=True)


# ==============================================================================
# 3. Main training entrypoint
# ==============================================================================
def build_model(vec_env, args):
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
        activation_fn=torch.nn.ReLU,
    )
    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=args.n_steps,                 # per-env rollout length
        batch_size=args.batch_size,           # minibatch size for SGD (fits comfortably in 16GB)
        n_epochs=10,
        gamma=0.997,                          # long horizon (720 steps/episode) -> high gamma
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,                        # encourage exploration over a huge action space
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.tb_dir,
        device=args.device,
        verbose=1,
        seed=args.seed,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=max(4, multiprocessing.cpu_count() - 1))
    parser.add_argument("--n-steps", type=int, default=256)   # per-env steps per rollout -> n_envs*256 samples/update
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--opponent", type=str, nargs="+", default=["random", "starter"])
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--tb-dir", type=str, default="tb_logs")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="path to a .zip checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.tb_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {args.device}, n_envs={args.n_envs}, n_steps={args.n_steps}, "
          f"batch_size={args.batch_size}, total_timesteps={args.total_timesteps:,}")
    print(f"Observation dim: {kc.OBS_DIM}, action nvec: {kc.action_nvec()}")

    env_fns = [
        make_env(i, args.opponent, args.episode_steps, args.seed, args.log_dir)
        for i in range(args.n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecMonitor(vec_env)  # aggregate episode stats across workers for SB3's logger

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = MaskablePPO.load(args.resume, env=vec_env, device=args.device)
    else:
        model = build_model(vec_env, args)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_every // args.n_envs, 1),
        save_path=args.models_dir,
        name_prefix="ppo_kaggriculture",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    verbose_cb = VerboseTrainingCallback(print_freq_rollouts=1, window=100)
    callbacks = CallbackList([checkpoint_cb, verbose_cb])

    print("Starting training. TensorBoard: tensorboard --logdir tb_logs")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_kaggriculture",
        reset_num_timesteps=args.resume is None,
        progress_bar=True,
    )

    final_path = os.path.join(args.models_dir, "ppo_kaggriculture_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")
    vec_env.close()


if __name__ == "__main__":
    # SubprocVecEnv + CUDA needs the 'spawn' start method to avoid CUDA-fork issues.
    multiprocessing.set_start_method("spawn", force=True)
    main()