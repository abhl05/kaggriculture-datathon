"""
train_v4.py - COMPLETE REWORK

Lessons learned from v3:
1. Money-only reward is too sparse - agent can't discover profitable strategies
2. Random opponent also loses money, creating a "do nothing = 50% win" local optimum
3. Value loss explodes because returns have huge variance (-150 to +100)
4. Agent needs intermediate signals to learn the full profit cycle

This version:
1. Adds back CAREFUL sub-goal bonuses (much smaller, only for productive actions)
2. Adds inventory-based reward (harvested product has value even if unsold)
3. Uses a greedy heuristic opponent (makes money, forces agent to compete)
4. Reward clipping to stabilize value function
5. Higher entropy for better exploration
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
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from kaggle_environments import make as kaggle_make

import kagg_common as kc


# ==============================================================================
# GREEDY HEURISTIC OPPONENT
# ==============================================================================

def greedy_opponent_agent(obs):
    """
    A simple greedy opponent that:
    1. Buys wheat seeds if none left
    2. Plants wheat on empty tiles
    3. Waters plants
    4. Harvests when ready
    5. Sells everything in shed

    This opponent should make modest profit and force our agent to compete.
    """
    try:
        player = obs["player"]
        me = obs["farms"][player]
        private = obs["private"]
        fx, fy = me["farmer"]
        tile = me["tiles"][fy][fx]

        market = []
        # Buy seeds if low
        if private["seeds"].get("WHEAT", 0) < 5 and me["money"] >= 10:
            market.append(["BUY_SEED", "WHEAT", 5])

        # Sell everything
        for item, qty in private.get("shed", {}).items():
            if qty > 0 and item in kc.PRODUCTS + ["FERTILIZER"]:
                market.append(["SELL", item, qty])

        # Priority: harvest > water > plant > move to empty
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop_age = obs["day"] - tile.get("planted_day", 0)
            if tile.get("yield_units", 0) > 0:
                return {"farmer": ["HARVEST"], "hands": [], "market": market}
            if not tile.get("watered_today", False):
                return {"farmer": ["WATER"], "hands": [], "market": market}
            # Move to find something to do
            return {"farmer": ["EAST"], "hands": [], "market": market}

        if tile is None and private["seeds"].get("WHEAT", 0) > 0:
            return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": market}

        # Wander looking for empty tiles or plants
        moves = [["NORTH"], ["SOUTH"], ["EAST"], ["WEST"]]
        return {"farmer": moves[obs["day"] % 4], "hands": [], "market": market}
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}


# ==============================================================================
# FIXED REWARD FUNCTION v4
# ==============================================================================

def compute_reward(prev_obs, curr_obs, player):
    """
    v4 Reward function: Balanced approach

    Problem with v3 (pure money): Too sparse, agent can't discover profit cycle
    Problem with v2 (big bonuses): Agent games the bonuses, ignores profit

    v4 Solution:
    1. Money delta is PRIMARY (60% of signal)
    2. Small sub-goal bonuses only for COMPLETING productive actions
    3. Inventory value tracking (product in shed = partial credit)
    4. Win bonus at end-game
    5. Reward clipping to stabilize value function
    """
    prev_farms = kc._g(prev_obs, "farms", [{}, {}])
    curr_farms = kc._g(curr_obs, "farms", [{}, {}])
    prev_private = kc._g(prev_obs, "private", {}) or {}
    curr_private = kc._g(curr_obs, "private", {}) or {}

    prev_money = kc._g(prev_farms[player] if player < len(prev_farms) else {}, "money", 0.0)
    curr_money = kc._g(curr_farms[player] if player < len(curr_farms) else {}, "money", 0.0)

    reward = 0.0

    # 1. MONEY DELTA (primary signal, 60% weight)
    money_delta = curr_money - prev_money
    reward += money_delta / 20.0

    # 2. INVENTORY VALUE CHANGE (product in shed has value)
    # This gives intermediate reward for harvesting even before selling
    prev_shed = kc._g(prev_private, "shed", {}) or {}
    curr_shed = kc._g(curr_private, "shed", {}) or {}

    # Approximate market prices (from kagg_common.py normalization)
    prices = {"WHEAT": 25, "CARROT": 40, "TOMATO": 60, "STRAWBERRY": 100, 
              "MELON": 200, "EGG": 30, "MILK": 50, "WOOL": 80, "FERTILIZER": 50}

    prev_inv_value = sum(prev_shed.get(k, 0) * prices.get(k, 0) for k in prices)
    curr_inv_value = sum(curr_shed.get(k, 0) * prices.get(k, 0) for k in prices)
    inv_delta = curr_inv_value - prev_inv_value
    reward += inv_delta / 100.0  # $100 inventory change = +1.0 reward

    # 3. SMALL sub-goal bonuses (only for completing productive actions)
    prev_tiles = kc._g(prev_farms[player] if player < len(prev_farms) else {}, "tiles", []) or []
    curr_tiles = kc._g(curr_farms[player] if player < len(curr_farms) else {}, "tiles", []) or []

    h = min(len(prev_tiles), len(curr_tiles), kc.BOARD_SIZE)
    for y in range(h):
        prow, crow = prev_tiles[y], curr_tiles[y]
        w = min(len(prow), len(crow), kc.BOARD_SIZE)
        for x in range(w):
            pt, ct = prow[x], crow[x]
            p_kind = kc._g(pt, "kind") if isinstance(pt, dict) else pt
            c_kind = kc._g(ct, "kind") if isinstance(ct, dict) else ct

            # Small bonus for harvesting (but ONLY if yield was actually collected)
            if p_kind == "PLANT" and kc._g(pt, "yield_units", 0) > 0:
                if c_kind != "PLANT" or kc._g(ct, "yield_units", 0) < kc._g(pt, "yield_units", 0):
                    reward += 0.5  # Small bonus (was +10.0 in v2)

            # Tiny bonus for watering (prevents withering = saves money)
            if c_kind == "PLANT" and p_kind == "PLANT":
                if not kc._g(pt, "watered_today", False) and kc._g(ct, "watered_today", False):
                    reward += 0.1  # Tiny (was +1.0 in v2)

    # 4. Profit amplification
    if money_delta > 0:
        reward += 0.5  # +0.5 for any money gain
    if money_delta > 100:
        reward += 1.0

    # 5. Loss penalty
    if money_delta < -100:
        reward -= 0.5

    # 6. Survival bonus (smaller)
    reward += 0.02

    # 7. END-GAME bonus
    day = kc._g(curr_obs, "day", 0)
    hour = kc._g(curr_obs, "hour", 0)
    if day >= 29 and hour >= 20:
        opp_money = kc._g(curr_farms[1 - player] if (1 - player) < len(curr_farms) else {}, "money", 0.0)

        if curr_money > opp_money:
            reward += 50.0
        elif curr_money > opp_money * 0.8:
            reward += 10.0

        reward += curr_money / 1000.0

    # 8. REWARD CLIPPING (critical for stable value function)
    reward = np.clip(reward, -10.0, 10.0)

    return float(reward)


# ==============================================================================
# CURRICULUM ENVIRONMENT
# ==============================================================================

class CurriculumKaggricultureEnv(gym.Env):
    """Environment with curriculum and opponent selection."""

    metadata = {"render_modes": []}

    def __init__(self, opponent="greedy", episode_steps=720, player=0, seed=None):
        super().__init__()
        self.opponent = opponent
        self.base_episode_steps = episode_steps
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
        self._episode_count = 0

    def get_opponent(self):
        """Progress opponents: random → greedy → self-play"""
        if self._episode_count < 100:
            return "random"
        elif self._episode_count < 500:
            return "greedy"
        else:
            # Could use previous model checkpoint here
            return "greedy"

    def _new_trainer(self):
        opp = self.get_opponent()
        self._kaggle_env = kaggle_make(
            "kaggriculture", configuration={"episodeSteps": self.base_episode_steps}, debug=False
        )

        if opp == "greedy":
            self._trainer = self._kaggle_env.train([None, greedy_opponent_agent] if self.player == 0 else [greedy_opponent_agent, None])
        else:
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
        if len(result) == 4:
            raw_obs, _env_reward, done, info = result
        else:
            raw_obs, _env_reward, terminated, truncated, info = result
            done = terminated or truncated
        self._raw_obs = raw_obs
        self._t += 1

        reward = compute_reward(prev_obs, raw_obs, self.player)

        farms = kc._g(raw_obs, "farms", [{}, {}])
        curr_money = kc._g(farms[self.player] if self.player < len(farms) else {}, "money", 0.0)

        truncated = False
        terminated = bool(done)

        info = dict(info) if info else {}
        info["current_money"] = float(curr_money)

        if terminated or truncated:
            info.update(kc.episode_outcome_info(raw_obs, self.player))
            self._episode_count += 1

        obs_vec = kc.encode_observation(raw_obs, self.player)
        return obs_vec, reward, terminated, truncated, info

    def close(self):
        self._kaggle_env = None
        self._trainer = None


def mask_fn(env: gym.Env) -> np.ndarray:
    return kc.get_action_mask(env._raw_obs, env.player)


def make_env(rank, opponent, episode_steps, seed):
    def _init():
        env = CurriculumKaggricultureEnv(
            opponent=opponent, episode_steps=episode_steps, player=0, seed=seed + rank
        )
        env = ActionMasker(env, mask_fn)
        return env
    return _init


# ==============================================================================
# LOGGING CALLBACK
# ==============================================================================

class VerboseTrainingCallback(BaseCallback):
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
        print(f"  win_rate         : {win_rate:6.3f}  (n={len(self.win_buf)})")
        if len(self.model.ep_info_buffer):
            recent_rewards = [e["r"] for e in self.model.ep_info_buffer]
            print(f"  sb3 ep_info_buf  : mean={np.mean(recent_rewards):9.2f}  min={np.min(recent_rewards):9.2f}  max={np.max(recent_rewards):9.2f}")
        print(f"  policy loss      : {self.model.logger.name_to_value.get('train/policy_gradient_loss', float('nan')):.5f}")
        print(f"  value loss       : {self.model.logger.name_to_value.get('train/value_loss', float('nan')):.5f}")
        print(f"  entropy loss     : {self.model.logger.name_to_value.get('train/entropy_loss', float('nan')):.5f}")
        print(f"  approx_kl        : {self.model.logger.name_to_value.get('train/approx_kl', float('nan')):.5f}")
        print(f"  clip_fraction    : {self.model.logger.name_to_value.get('train/clip_fraction', float('nan')):.5f}")
        print(f"  explained_var    : {self.model.logger.name_to_value.get('train/explained_variance', float('nan')):.5f}")
        print("=" * 96, flush=True)


# ==============================================================================
# MAIN
# ==============================================================================

def build_model(vec_env, args):
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )
    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_vf=None,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.tb_dir,
        device=args.device,
        verbose=1,
        seed=args.seed,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05)  # HIGHER for exploration
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--opponent", type=str, default="greedy")
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--tb-dir", type=str, default="tb_logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.tb_dir, exist_ok=True)

    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {args.device}, n_envs={args.n_envs}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"OBS_DIM={kc.OBS_DIM}, action_nvec={kc.action_nvec()}")
    print("\nKEY FIXES IN V4:")
    print("  1. Inventory value reward (credit for harvested product)")
    print("  2. Small sub-goal bonuses (0.1-0.5, not 1-10)")
    print("  3. Greedy opponent (forces competition, breaks 'do nothing' optimum)")
    print("  4. Reward clipping [-10, 10] (stabilizes value function)")
    print("  5. Higher entropy (0.05) for better exploration")
    print("  6. Opponent curriculum: random → greedy")

    env_fns = [
        make_env(i, args.opponent, args.episode_steps, args.seed)
        for i in range(args.n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecMonitor(vec_env)

    if args.resume:
        print(f"Resuming from: {args.resume}")
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

    print("\nStarting training...")
    print("=" * 96)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_kaggriculture",
        reset_num_timesteps=args.resume is None,
        progress_bar=True,
    )

    final_path = os.path.join(args.models_dir, "ppo_kaggriculture_final")
    model.save(final_path)
    print(f"\nTraining complete! Final model: {final_path}.zip")
    vec_env.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()