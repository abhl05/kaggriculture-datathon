"""
kagg_common.py - Day-level RL interface (v5, hybrid).

Replaces the turn-level encoding. The policy now acts ONCE PER IN-GAME DAY
and emits a DayPlan; kagg_control.py executes it over the following 24
turns. Consequences:

* Episode length drops 720 -> 30 steps. gamma=0.99 now covers the whole
  season (0.99^30 = 0.74) instead of ~4 in-game days, so day-10 melon
  payoffs are finally visible to the critic.
* OBS_DIM drops 5699 -> 64. No flattened 10x10x28 grid; the policy sees
  aggregate farm composition, which is all a strategic decision needs.
* Action space is 7 small independent heads instead of 15 heads over 28
  unit-ops, and every action is meaningful (no PASS-spam, no self-inflicted
  bankruptcy via repeated HIRE).
* Reward is the day's bank delta plus a terminal margin term. No sub-goal
  bonuses: the scripted layer already guarantees competent execution, so
  shaping can only be gamed, not exploited usefully.
"""
from __future__ import annotations

import numpy as np

import kagg_control as kcx
from kagg_control import DayPlan, _g

# --------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------
CROP_CHOICES = [None, "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
HIRE_CHOICES = [0, 2, 4, 6, 8, 10]
LAND_CHOICES = [False, True]
ANIMAL_CHOICES = [None, "GOOSE", "COW", "SHEEP"]
SELL_CHOICES = [0, 1, 2, 3]
FERT_CHOICES = [False, True]
CAP_CHOICES = [0, 10, 25, 100]

ACTION_NVEC = [
    len(CROP_CHOICES), len(HIRE_CHOICES), len(LAND_CHOICES),
    len(ANIMAL_CHOICES), len(SELL_CHOICES), len(FERT_CHOICES),
    len(CAP_CHOICES),
]
MASK_DIM = sum(ACTION_NVEC)

PRODUCTS = kcx.PRODUCTS
SELLABLE = kcx.SELLABLE
CROPS = kcx.CROPS
ANIMALS = kcx.ANIMALS

OBS_DIM = 64

MONEY_NORM = 20000.0
TILE_NORM = 100.0
SHED_NORM = 100.0


def action_nvec():
    return list(ACTION_NVEC)


def decode_plan(raw):
    """MultiDiscrete vector -> DayPlan."""
    raw = [int(v) for v in np.asarray(raw).reshape(-1)]
    while len(raw) < len(ACTION_NVEC):
        raw.append(0)

    def pick(choices, i):
        idx = raw[i]
        return choices[idx] if 0 <= idx < len(choices) else choices[0]

    return DayPlan(
        crop=pick(CROP_CHOICES, 0),
        hire_target=pick(HIRE_CHOICES, 1),
        buy_land=pick(LAND_CHOICES, 2),
        animal=pick(ANIMAL_CHOICES, 3),
        sell_mode=pick(SELL_CHOICES, 4),
        fertilize=pick(FERT_CHOICES, 5),
        plant_cap=pick(CAP_CHOICES, 6),
    )


# --------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------
def encode_observation(obs, player):
    farms = _g(obs, "farms", [{}, {}])
    me = farms[player] if player < len(farms) else {}
    them = farms[1 - player] if (1 - player) < len(farms) else {}
    private = _g(obs, "private", {}) or {}
    market = _g(obs, "market", {}) or {}
    town = _g(obs, "town", {}) or {}

    day = int(_g(obs, "day", 0))
    _, my_counts = kcx.scan_farm(me, day)
    _, their_counts = kcx.scan_farm(them, day)

    v = []
    # Time (2)
    v.append(day / 30.0)
    v.append(max(0.0, (30 - day)) / 30.0)

    # Money (3)
    my_money = float(_g(me, "money", 0.0))
    opp_money = float(_g(them, "money", 0.0))
    v.append(my_money / MONEY_NORM)
    v.append(opp_money / MONEY_NORM)
    v.append(np.clip((my_money - opp_money) / MONEY_NORM, -2.0, 2.0))

    # My farm composition (8)
    v.append(my_counts["empty"] / TILE_NORM)
    v.append(my_counts["weed"] / TILE_NORM)
    v.append(my_counts["plants"] / TILE_NORM)
    v.append(my_counts["animals"] / TILE_NORM)
    v.append(my_counts["ripe"] / TILE_NORM)
    v.append(my_counts["empty_coop"] / TILE_NORM)
    v.append(my_counts["empty_pasture"] / TILE_NORM)
    total_tiles = 25 * len(_g(me, "unlocked_quadrants", ["NW"]) or ["NW"])
    v.append(total_tiles / TILE_NORM)

    # Per-crop tile counts (5)
    for c in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]:
        v.append(my_counts["per_crop"].get(c, 0) / TILE_NORM)

    # Land / labour (6)
    quads = _g(me, "unlocked_quadrants", ["NW"]) or ["NW"]
    for q in kcx.QUADRANTS:
        v.append(1.0 if q in quads else 0.0)
    v.append(len(_g(me, "hands", []) or []) / 10.0)
    n_extra = len(quads) - 1
    next_land = kcx.LAND_PRICES[n_extra] if n_extra < len(kcx.LAND_PRICES) else 0
    v.append(1.0 if (next_land and my_money >= next_land) else 0.0)

    # Seeds (5)
    seeds = _g(private, "seeds", {}) or {}
    for c in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]:
        v.append(min(int(_g(seeds, c, 0)), 200) / SHED_NORM)

    # Shed contents (9)
    shed = _g(private, "shed", {}) or {}
    for it in SELLABLE:
        v.append(min(int(_g(shed, it, 0)), 200) / SHED_NORM)

    # Market prices, normalised against each product's own base so the
    # policy sees "how glutted is this good" rather than raw dollars.
    prices = _g(market, "prices", {}) or {}
    for it in SELLABLE:
        base = kcx.BASE_PRICE.get(it, 1)
        v.append(np.clip(float(_g(prices, it, base)) / base, 0.0, 3.0))

    # Opponent summary (4)
    v.append(their_counts["plants"] / TILE_NORM)
    v.append(their_counts["animals"] / TILE_NORM)
    v.append(len(_g(them, "unlocked_quadrants", ["NW"]) or ["NW"]) / 4.0)
    v.append(their_counts["empty"] / TILE_NORM)

    # Town demand (8)
    shops = _g(town, "unlocked_shops", []) or []
    for s in ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
              "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"]:
        v.append(1.0 if s in shops else 0.0)

    vec = np.asarray(v, dtype=np.float32)
    if vec.shape[0] < OBS_DIM:
        vec = np.concatenate([vec, np.zeros(OBS_DIM - vec.shape[0], dtype=np.float32)])
    return vec[:OBS_DIM]


# --------------------------------------------------------------------------
# Action mask
# --------------------------------------------------------------------------
def get_action_mask(obs, player):
    """Flat boolean mask, concatenated per MultiDiscrete head.

    Only genuinely impossible choices are masked. Strategy stays with the
    policy -- masking on 'is this a good idea' would bake in the very
    priors we want it to learn.
    """
    farms = _g(obs, "farms", [{}, {}])
    me = farms[player] if player < len(farms) else {}
    money = float(_g(me, "money", 0.0))
    day = int(_g(obs, "day", 0))

    crop_m = np.ones(len(CROP_CHOICES), dtype=bool)
    for i, c in enumerate(CROP_CHOICES):
        if c is None:
            continue
        # Can't sow what we can't buy a single seed of.
        if money < CROPS[c]["seed"]:
            crop_m[i] = False
        # Nothing planted this late will mature before the season ends.
        remaining = 29 - day
        if remaining < kcx.HARVEST_AGE.get(c, CROPS[c]["max_yield_day"]):
            crop_m[i] = False
    if not crop_m.any():
        crop_m[0] = True

    hire_m = np.ones(len(HIRE_CHOICES), dtype=bool)

    land_m = np.ones(len(LAND_CHOICES), dtype=bool)
    quads = _g(me, "unlocked_quadrants", ["NW"]) or ["NW"]
    n_extra = len(quads) - 1
    if n_extra >= len(kcx.LAND_PRICES) or money < kcx.LAND_PRICES[n_extra]:
        land_m[1] = False

    animal_m = np.ones(len(ANIMAL_CHOICES), dtype=bool)
    for i, a in enumerate(ANIMAL_CHOICES):
        if a is None:
            continue
        if money < ANIMALS[a]["cost"]:
            animal_m[i] = False
        # An animal bought too late never reaches first yield.
        if (29 - day) < ANIMALS[a]["cost"] // 100:
            pass
    animal_m[0] = True

    sell_m = np.ones(len(SELL_CHOICES), dtype=bool)
    if day >= 28:
        # Holding inventory into the final scoring is strictly worthless.
        sell_m[:] = False
        sell_m[0] = True

    fert_m = np.ones(len(FERT_CHOICES), dtype=bool)
    if money < 500:
        fert_m[1] = False

    cap_m = np.ones(len(CAP_CHOICES), dtype=bool)

    return np.concatenate([crop_m, hire_m, land_m, animal_m, sell_m, fert_m, cap_m])


# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------
def compute_reward(prev_money, curr_money, done, my_final, opp_final):
    """Day bank delta, plus a terminal margin term.

    Deliberately unshaped. Summed over an episode the delta term telescopes
    to (final - starting) bank, which IS the objective; the terminal term
    adds the competitive component the leaderboard actually scores.
    """
    reward = (curr_money - prev_money) / 1000.0
    if done:
        margin = (my_final - opp_final) / 1000.0
        reward += float(np.clip(margin, -10.0, 10.0))
        reward += 5.0 if my_final > opp_final else (0.0 if my_final == opp_final else -5.0)
    return float(reward)


def episode_outcome_info(obs, player):
    farms = _g(obs, "farms", [{}, {}])
    me = float(_g(farms[player] if player < len(farms) else {}, "money", 0.0))
    them = float(_g(farms[1 - player] if (1 - player) < len(farms) else {}, "money", 0.0))
    return {
        "final_money": me,
        "opp_final_money": them,
        "win": 1.0 if me > them else (0.5 if me == them else 0.0),
    }