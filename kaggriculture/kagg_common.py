"""
kagg_common.py
==============
Shared logic for the Kaggriculture PPO agent. This module is imported by BOTH
`train.py` (during training) and `main.py` (at Kaggle submission time), so the
observation encoding and action decoding used to produce the model are
*guaranteed* to be identical to the ones used to consume its output.

For Kaggle submission, bundle this file alongside main.py:

    tar -czf submission.tar.gz main.py kagg_common.py <model>.zip

------------------------------------------------------------------------------
DESIGN OVERVIEW
------------------------------------------------------------------------------
1. Observation -> fixed-length float32 vector ("encode_observation")
   - Per-tile features (one-hot tile kind, crop/animal one-hot, normalized
     age/yield/water/feed counters) for a BOARD_SIZE x BOARD_SIZE grid, for
     BOTH players (own farm fully, opponent's *public* farm - its tiles are
     visible per the rules, but not its shed/seeds/inventory).
   - A block of scalar features: day/hour, money, quadrants owned, hires,
     positions, market prices/inventory, town shops unlocked, own seeds,
     own shed contents, and items currently carried by farmer + hands.

2. Model action (MultiDiscrete vector) -> game action dict ("decode_action")
   - One categorical "unit action" slot for the farmer and each of up to
     MAX_HANDS hired hands, drawn from a fixed catalog of 27 primitive
     ops (movement / plant / animal / terrain / shed ops).
   - MARKET_SLOTS categorical "market macro-action" slots (BUY_SEED,
     BUY_ANIMAL, BUY_PRODUCT, SELL-all-in-shed per item, HIRE, BUY_LAND,
     NOOP), each resolved against the *current* obs (e.g. "sell all wheat
     in the shed") so the model doesn't need to predict exact quantities.

Invalid/no-op actions are safe: per AGENTS.md, the game engine silently
no-ops illegal actions (e.g. PLANT on a locked tile, SELL with 0 in shed),
so we do not need action masking for correctness - only for sample
efficiency, which is a documented future improvement (see train.py notes).
"""

from __future__ import annotations
import numpy as np

# ------------------------------------------------------------------------
# Game constants (from README.md "Object Types" / "Observation Format")
# ------------------------------------------------------------------------
BOARD_SIZE = 10          # default boardSize
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
# Everything that can appear in `private.shed` (harvestable products + fertilizer + animals)
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]
SHED_ITEMS = PRODUCTS + ["FERTILIZER", "GOOSE", "COW", "SHEEP"]
MARKET_RESOURCES = PRODUCTS + ["FERTILIZER"]           # keys present in market.inventory / market.prices
QUADRANTS = ["NW", "NE", "SW", "SE"]
SHOPS = ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
         "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"]

MAX_HANDS = 6            # action-space cap on controllable hired hands (fib hire cost makes >6 rare/expensive)
# Macro market-order slots issued per turn (<= maxMarketOrdersPerTurn = 10).
# The catalog alone has 9 distinct SELL entries (8 PRODUCTS + FERTILIZER)
# plus HIRE/BUY_LAND/BUY_SEED/BUY_ANIMAL/BUY_PRODUCT -- a turn where the
# agent wants to liquidate a full shed AND restock seeds AND hire needs
# more than 6 slots to express. Bumped to 8 (leaves 2 slots of headroom
# under the env's hard cap of 10; duplicate/invalid slots beyond the shed's
# actual contents just no-op via decode_market_slot, so there's no harm in
# giving the policy more room than it typically needs).
MARKET_SLOTS = 8
TILE_FEATS = 28          # float features encoded per tile (see encode_tile)
SCALAR_FEATS = 99        # float features encoded in the scalar block (see encode_scalars)
OBS_DIM = 2 * BOARD_SIZE * BOARD_SIZE * TILE_FEATS + SCALAR_FEATS

# Normalization constants (rough game-scale denominators, not exact bounds -
# PPO with a Box observation space tolerates values mildly outside [-1, 1]).
MONEY_NORM = 20000.0
PRICE_NORM = 300.0
INV_NORM = 10000.0
SHED_NORM = 100.0        # shedCapacity default
SEED_NORM = 50.0
DAY_NORM = 30.0
HOUR_NORM = 24.0
POS_NORM = float(BOARD_SIZE - 1)
YIELD_NORM = 10.0
AGE_NORM = 30.0
UNWATERED_NORM = 2.0


def _g(d, key, default=None):
    """Safe get that works for plain dicts AND kaggle_environments' Struct
    (both support Mapping-style __getitem__ / .get)."""
    if d is None:
        return default
    try:
        val = d.get(key, default)
    except AttributeError:
        try:
            val = d[key]
        except (KeyError, IndexError, TypeError):
            val = default
    return default if val is None else val


def _one_hot(index, n):
    v = [0.0] * n
    if 0 <= index < n:
        v[index] = 1.0
    return v


# ------------------------------------------------------------------------
# Tile encoding
# ------------------------------------------------------------------------
def encode_tile(tile, current_day):
    """Encode a single tile dict/None/"LOCKED" into a TILE_FEATS-length list.

    Layout (28 floats):
      [0]  is_empty            [1]  is_locked          [2]  is_weed
      [3]  is_plant            [4]  is_coop_empty      [5]  is_coop_occupied
      [6]  is_pasture_empty    [7]  is_pasture_occupied
      [8:13]  crop one-hot (WHEAT, CARROT, TOMATO, STRAWBERRY, MELON)
      [13] plant_age_norm      [14] watered_today      [15] consec_unwatered_norm
      [16] plant_yield_norm    [17] fertilized_active
      [18:21] animal one-hot (GOOSE, COW, SHEEP)
      [21] fed_today           [22] consec_unfed_norm  [23] cared_today
      [24] animal_yield_norm   [25] fertilizer_available [26] pending_care_bonus_norm
      [27] reserved (0.0)
    """
    f = [0.0] * TILE_FEATS

    if tile is None:
        f[0] = 1.0
        return f
    if tile == "LOCKED":
        f[1] = 1.0
        return f

    kind = _g(tile, "kind")
    if kind == "WEED":
        f[2] = 1.0
        return f

    if kind == "PLANT":
        f[3] = 1.0
        crop = _g(tile, "crop")
        if crop in CROPS:
            f[8:13] = _one_hot(CROPS.index(crop), 5)
        planted_day = _g(tile, "planted_day", current_day)
        f[13] = np.clip((current_day - planted_day) / AGE_NORM, 0.0, 2.0)
        f[14] = 1.0 if _g(tile, "watered_today", False) else 0.0
        f[15] = np.clip(_g(tile, "consecutive_unwatered", 0) / UNWATERED_NORM, 0.0, 2.0)
        f[16] = np.clip(_g(tile, "yield_units", 0) / YIELD_NORM, 0.0, 2.0)
        fert_until = _g(tile, "fertilized_until_day", -1)
        f[17] = 1.0 if fert_until is not None and fert_until >= current_day else 0.0
        return f

    if kind in ("COOP", "PASTURE"):
        occupied = _g(tile, "animal") is not None
        if kind == "COOP":
            f[4] = 0.0 if occupied else 1.0
            f[5] = 1.0 if occupied else 0.0
        else:
            f[6] = 0.0 if occupied else 1.0
            f[7] = 1.0 if occupied else 0.0
        if occupied:
            animal = _g(tile, "animal")
            if animal in ANIMALS:
                f[18:21] = _one_hot(ANIMALS.index(animal), 3)
            f[21] = 1.0 if _g(tile, "fed_today", False) else 0.0
            f[22] = np.clip(_g(tile, "consecutive_unfed", 0) / UNWATERED_NORM, 0.0, 2.0)
            f[23] = 1.0 if _g(tile, "cared_today", False) else 0.0
            f[24] = np.clip(_g(tile, "yield_units", 0) / YIELD_NORM, 0.0, 2.0)
            f[25] = 1.0 if _g(tile, "fertilizer_available", False) else 0.0
            f[26] = np.clip(_g(tile, "pending_care_bonus", 0) / YIELD_NORM, 0.0, 2.0)
        return f

    # Unknown tile type (forward-compat): leave as all-zero (treated ~empty).
    return f


def encode_farm_tiles(farm, current_day):
    """Flatten a farm's BOARD_SIZE x BOARD_SIZE `tiles` grid into a flat
    float32 vector of length BOARD_SIZE*BOARD_SIZE*TILE_FEATS. Pads with
    "LOCKED"-like zeros / truncates defensively if the actual grid size
    differs from BOARD_SIZE (keeps the observation space shape fixed)."""
    tiles = _g(farm, "tiles", [])
    out = np.zeros((BOARD_SIZE, BOARD_SIZE, TILE_FEATS), dtype=np.float32)
    h = min(len(tiles), BOARD_SIZE)
    for y in range(h):
        row = tiles[y]
        w = min(len(row), BOARD_SIZE)
        for x in range(w):
            out[y, x, :] = encode_tile(row[x], current_day)
    return out.reshape(-1)


def encode_scalars(obs, player, max_hands=MAX_HANDS):
    """Encode all non-grid features into a flat SCALAR_FEATS-length vector."""
    opp = 1 - player
    farms = _g(obs, "farms", [{}, {}])
    me = farms[player] if player < len(farms) else {}
    them = farms[opp] if opp < len(farms) else {}
    private = _g(obs, "private", {}) or {}
    market = _g(obs, "market", {}) or {}
    town = _g(obs, "town", {}) or {}

    day = _g(obs, "day", 0)
    hour = _g(obs, "hour", 0)

    vec = []
    vec.append(day / DAY_NORM)
    vec.append(hour / HOUR_NORM)

    vec.append(_g(me, "money", 0.0) / MONEY_NORM)
    vec.append(_g(them, "money", 0.0) / MONEY_NORM)

    my_quads = _g(me, "unlocked_quadrants", []) or []
    their_quads = _g(them, "unlocked_quadrants", []) or []
    vec.extend([1.0 if q in my_quads else 0.0 for q in QUADRANTS])
    vec.extend([1.0 if q in their_quads else 0.0 for q in QUADRANTS])

    vec.append(_g(me, "hires_today", 0) / 8.0)
    vec.append(_g(them, "hires_today", 0) / 8.0)

    my_farmer = _g(me, "farmer", [0, 0]) or [0, 0]
    their_farmer = _g(them, "farmer", [0, 0]) or [0, 0]
    vec.extend([my_farmer[0] / POS_NORM, my_farmer[1] / POS_NORM])
    vec.extend([their_farmer[0] / POS_NORM, their_farmer[1] / POS_NORM])

    my_hands = _g(me, "hands", []) or []
    their_hands = _g(them, "hands", []) or []
    for i in range(max_hands):
        if i < len(my_hands):
            vec.extend([my_hands[i][0] / POS_NORM, my_hands[i][1] / POS_NORM])
        else:
            vec.extend([-1.0, -1.0])
    for i in range(max_hands):
        if i < len(their_hands):
            vec.extend([their_hands[i][0] / POS_NORM, their_hands[i][1] / POS_NORM])
        else:
            vec.extend([-1.0, -1.0])
    vec.append(len(my_hands) / 8.0)
    vec.append(len(their_hands) / 8.0)

    m_inv = _g(market, "inventory", {}) or {}
    m_price = _g(market, "prices", {}) or {}
    for r in MARKET_RESOURCES:
        vec.append(_g(m_inv, r, INV_NORM) / INV_NORM)
    for r in MARKET_RESOURCES:
        vec.append(_g(m_price, r, 0) / PRICE_NORM)

    shops = _g(town, "unlocked_shops", []) or []
    vec.extend([1.0 if s in shops else 0.0 for s in SHOPS])

    seeds = _g(private, "seeds", {}) or {}
    for c in CROPS:
        vec.append(_g(seeds, c, 0) / SEED_NORM)

    shed = _g(private, "shed", {}) or {}
    for it in SHED_ITEMS:
        vec.append(_g(shed, it, 0) / SHED_NORM)

    carried = {it: 0 for it in SHED_ITEMS}
    for inv in (_g(private, "inventories", []) or []):
        inv = inv or {}
        for it in SHED_ITEMS:
            carried[it] += _g(inv, it, 0)
    for it in SHED_ITEMS:
        vec.append(carried[it] / SHED_NORM)

    assert len(vec) == SCALAR_FEATS, f"scalar feature length drift: {len(vec)} != {SCALAR_FEATS}"
    return np.asarray(vec, dtype=np.float32)


def encode_observation(obs, player):
    """Top-level: raw game `obs` (dict-like) -> fixed-length float32 vector."""
    day = _g(obs, "day", 0)
    farms = _g(obs, "farms", [{}, {}])
    me = farms[player] if player < len(farms) else {}
    them = farms[1 - player] if (1 - player) < len(farms) else {}
    my_tiles = encode_farm_tiles(me, day)
    their_tiles = encode_farm_tiles(them, day)
    scalars = encode_scalars(obs, player)
    return np.concatenate([my_tiles, their_tiles, scalars]).astype(np.float32)


# ------------------------------------------------------------------------
# Unit action catalog (farmer / hand primitive ops)
# ------------------------------------------------------------------------
UNIT_ACTIONS = [
    ["PASS"], ["NORTH"], ["SOUTH"], ["EAST"], ["WEST"],                      # 0-4
    ["PLANT", "WHEAT"], ["PLANT", "CARROT"], ["PLANT", "TOMATO"],            # 5-7
    ["PLANT", "STRAWBERRY"], ["PLANT", "MELON"],                             # 8-9
    ["WATER"], ["HARVEST"], ["FERTILIZE"],                                   # 10-12
    ["BUILD_COOP"], ["BUILD_PASTURE"],                                       # 13-14
    ["FEED"], ["COLLECT_FERTILIZER"], ["CARE"],                              # 15-17
    ["DIG"],                                                                 # 18
    ["PICKUP", "WHEAT", 10], ["PICKUP", "FERTILIZER", 10],                   # 19-20
    ["PICKUP", "GOOSE", 1], ["PICKUP", "COW", 1], ["PICKUP", "SHEEP", 1],    # 21-23
    ["PLACE", "GOOSE"], ["PLACE", "COW"], ["PLACE", "SHEEP"],                # 24-26
    ["DROP"],                                                                # 27
]
N_UNIT_ACTIONS = len(UNIT_ACTIONS)


def decode_unit_action(idx):
    idx = int(idx)
    if not (0 <= idx < N_UNIT_ACTIONS):
        return ["PASS"]
    return list(UNIT_ACTIONS[idx])


# ------------------------------------------------------------------------
# Market macro-action catalog. Each entry is resolved lazily against the
# current obs so the policy doesn't have to output exact quantities.
# ------------------------------------------------------------------------
_SEED_BUY_QTY = {"WHEAT": 5, "CARROT": 5, "TOMATO": 2, "STRAWBERRY": 2, "MELON": 1}


def _market_noop(obs, player):
    return None


def _make_buy_seed(crop):
    def f(obs, player):
        return ["BUY_SEED", crop, _SEED_BUY_QTY[crop]]
    return f


def _make_buy_animal(animal):
    def f(obs, player):
        return ["BUY_ANIMAL", animal, 1]
    return f


def _make_buy_product(item, qty):
    def f(obs, player):
        return ["BUY_PRODUCT", item, qty]
    return f


def _make_sell_all(item):
    def f(obs, player):
        private = _g(obs, "private", {}) or {}
        shed = _g(private, "shed", {}) or {}
        n = int(_g(shed, item, 0))
        if n <= 0:
            return None
        return ["SELL", item, n]
    return f


def _market_hire(obs, player):
    return ["HIRE"]


def _market_buy_land(obs, player):
    return ["BUY_LAND"]


MARKET_CATALOG = [
    _market_noop,
    *[_make_buy_seed(c) for c in CROPS],                    # BUY_SEED x5
    *[_make_buy_animal(a) for a in ANIMALS],                 # BUY_ANIMAL x3
    _make_buy_product("WHEAT", 10), _make_buy_product("FERTILIZER", 5),  # BUY_PRODUCT x2
    *[_make_sell_all(p) for p in PRODUCTS],                  # SELL-all x8
    _make_sell_all("FERTILIZER"),                            # SELL-all fertilizer
    _market_hire,
    _market_buy_land,
]
N_MARKET_ACTIONS = len(MARKET_CATALOG)


def decode_market_slot(idx, obs, player):
    idx = int(idx)
    if not (0 <= idx < N_MARKET_ACTIONS):
        return None
    return MARKET_CATALOG[idx](obs, player)


# ------------------------------------------------------------------------
# Full action space / decode
# ------------------------------------------------------------------------
def action_nvec(max_hands=MAX_HANDS, market_slots=MARKET_SLOTS):
    """Returns the nvec list for a gymnasium.spaces.MultiDiscrete action space:
    [farmer, hand_1, ..., hand_max_hands, market_1, ..., market_market_slots]"""
    return [N_UNIT_ACTIONS] * (1 + max_hands) + [N_MARKET_ACTIONS] * market_slots


def decode_action(raw, obs, player, max_hands=MAX_HANDS, market_slots=MARKET_SLOTS):
    """raw: flat int array/list of length (1+max_hands+market_slots), i.e. a
    sampled MultiDiscrete action. obs: the raw game observation the action is
    being taken *from* (needed to resolve e.g. "sell all wheat" and to know
    how many hands actually exist today). Returns the game action dict."""
    raw = list(raw)
    farmer_idx = raw[0]
    hand_idxs = raw[1:1 + max_hands]
    market_idxs = raw[1 + max_hands:1 + max_hands + market_slots]

    farms = _g(obs, "farms", [{}, {}])
    me = farms[player] if player < len(farms) else {}
    n_hands_actual = len(_g(me, "hands", []) or [])

    farmer_op = decode_unit_action(farmer_idx)
    hands_ops = [decode_unit_action(hand_idxs[i]) for i in range(min(n_hands_actual, max_hands))]

    market_orders = []
    for idx in market_idxs:
        order = decode_market_slot(idx, obs, player)
        if order is not None:
            market_orders.append(order)
    market_orders = market_orders[:10]  # maxMarketOrdersPerTurn

    return {"farmer": farmer_op, "hands": hands_ops, "market": market_orders}


# ------------------------------------------------------------------------
# Dense reward shaping
# ------------------------------------------------------------------------
# Fallback costs, used only if market prices are somehow missing from obs.
SEED_PRICES = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
PRODUCE_PRICES = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200}

# The real win condition only counts bank cash at the final turn -- unsold
# shed/seed inventory is worth exactly 0 then. Below this many days
# remaining, we linearly taper how much unsold inventory "counts" toward
# the dense reward's net-worth term, so the training signal converges
# toward the real payout structure as the season winds down instead of
# rewarding hoarding all the way to turn 720.
LIQUIDATION_TAPER_DAYS = 3.0


def calculate_net_worth(obs, player):
    farms = _g(obs, "farms", [{}, {}])
    p_farm = farms[player] if player < len(farms) else {}
    cash = _g(p_farm, "money", 0.0)

    private = _g(obs, "private", {})
    seeds = _g(private, "seeds", {})
    shed = _g(private, "shed", {})

    # Value unplanted seeds at their purchase cost (a fair proxy -- you can't
    # sell seeds back, so their "value" really is just what you paid).
    seed_val = sum(count * SEED_PRICES.get(crop, 10) for crop, count in seeds.items())

    # Value harvested produce at the CURRENT live market price, not a fixed
    # base price -- using the base price regardless of actual market
    # conditions lets the policy get credit for holding a shed full of
    # melon/strawberry that a bulk SELL would actually crash to the $1
    # floor (see the price function in README.md), which is a soft
    # reward-hacking incentive to hoard oversupplied goods.
    market = _g(obs, "market", {}) or {}
    m_price = _g(market, "prices", {}) or {}
    shed_val = sum(
        count * _g(m_price, item, PRODUCE_PRICES.get(item, 20))
        for item, count in shed.items()
    )

    day = _g(obs, "day", 0)
    remaining_days = max(0.0, DAY_NORM - day)  # DAY_NORM == SEASON_DAYS == 30
    liquidity_factor = min(1.0, remaining_days / LIQUIDATION_TAPER_DAYS)

    return cash + liquidity_factor * (seed_val + shed_val)

def compute_reward(prev_obs, curr_obs, player):
    # 1. Primary Signal: Delta in Total Net Worth (Cash + Assets)
    prev_nw = calculate_net_worth(prev_obs, player)
    curr_nw = calculate_net_worth(curr_obs, player)
    
    # Scaled down so a $100 gain gives +1.0 reward
    reward = (curr_nw - prev_nw) / 100.0  

    prev_farms = _g(prev_obs, "farms", [{}, {}])
    curr_farms = _g(curr_obs, "farms", [{}, {}])
    
    prev_tiles = _g(prev_farms[player] if player < len(prev_farms) else {}, "tiles", []) or []
    curr_tiles = _g(curr_farms[player] if player < len(curr_farms) else {}, "tiles", []) or []

    bonus = 0.0
    h = min(len(prev_tiles), len(curr_tiles), BOARD_SIZE)
    for y in range(h):
        prow, crow = prev_tiles[y], curr_tiles[y]
        w = min(len(prow), len(crow), BOARD_SIZE)
        for x in range(w):
            pt, ct = prow[x], crow[x]
            p_kind = _g(pt, "kind") if isinstance(pt, dict) else pt
            c_kind = _g(ct, "kind") if isinstance(ct, dict) else ct

            # Bonus for planting
            if c_kind == "PLANT" and p_kind != "PLANT":
                bonus += 0.05  
            # Bonus for watering thirsty crops
            elif c_kind == "PLANT" and p_kind == "PLANT":
                if (not _g(pt, "watered_today", False)) and _g(ct, "watered_today", False):
                    bonus += 0.02  
            
            # Explicit HARVEST Bonus
            if p_kind == "PLANT" and _g(pt, "yield_units", 0) > 0:
                if c_kind != "PLANT" or _g(ct, "yield_units", 0) < _g(pt, "yield_units", 0):
                    bonus += 0.15  

            # Animal Husbandry
            if c_kind in ("COOP", "PASTURE"):
                c_animal = _g(ct, "animal")
                p_animal = _g(pt, "animal") if isinstance(pt, dict) else None
                if c_animal is not None and p_animal is not None:
                    if (not _g(pt, "fed_today", False)) and _g(ct, "fed_today", False):
                        bonus += 0.02
                    if (not _g(pt, "cared_today", False)) and _g(ct, "cared_today", False):
                        bonus += 0.01

            # Penalties
            if c_kind == "WEED" and p_kind == "PLANT":
                bonus -= 0.3  # Crop death penalty
            if c_kind is None and p_kind in ("COOP", "PASTURE") and _g(pt, "animal") is not None:
                bonus -= 0.5  # Escaped animal penalty

    return float(reward + bonus)

def episode_outcome_info(obs, player):
    """Small dict of end-of-episode diagnostics for logging (win/loss/money)."""
    farms = _g(obs, "farms", [{}, {}])
    me = _g(farms[player] if player < len(farms) else {}, "money", 0.0)
    them = _g(farms[1 - player] if (1 - player) < len(farms) else {}, "money", 0.0)
    win = 1.0 if me > them else (0.5 if me == them else 0.0)
    return {"final_money": float(me), "opp_final_money": float(them), "win": win}

def get_action_mask(obs, player, max_hands=MAX_HANDS, market_slots=MARKET_SLOTS):
    """Generates a boolean mask tuple for the MultiDiscrete action space.

    IMPORTANT: hand slots beyond the number of hands the player *actually
    has this turn* are forced to PASS-only. `decode_action` already drops
    the sampled action for nonexistent hands (`hands_ops = [... for i in
    range(min(n_hands_actual, max_hands))]`), so leaving those slots fully
    open (as before) meant the network was sampling, scoring, and getting
    PPO-updated on a real 28-way categorical for units that don't exist and
    whose "action" never touches the environment -- pure gradient/entropy
    noise on however many of the `max_hands` slots aren't filled, which for
    most of a typical episode (hires are Fibonacci-cost-scaled and thus
    slow to accumulate) is most of them. Locking those slots to PASS-only
    makes their log-prob ~log(1)~0 and their entropy contribution ~0, so
    they stop injecting noise into the objective.
    """
    farms = _g(obs, "farms", [{}, {}])
    me = farms[player] if player < len(farms) else {}
    money = _g(me, "money", 0.0)
    n_hands_actual = len(_g(me, "hands", []) or [])

    private = _g(obs, "private", {}) or {}
    shed = _g(private, "shed", {}) or {}

    # Unit actions (Farmer + Hands): Allow all primitive ops (game safely no-ops invalid ones)
    unit_mask = np.ones(N_UNIT_ACTIONS, dtype=np.bool_)
    pass_only_mask = np.zeros(N_UNIT_ACTIONS, dtype=np.bool_)
    pass_only_mask[0] = True  # UNIT_ACTIONS[0] == ["PASS"]

    # Market actions
    market_mask = np.ones(N_MARKET_ACTIONS, dtype=np.bool_)

    # Prevent buying actions if out of money (cost floor is ~10)
    if money < 500.0:
        market_mask[1:11] = False  # Disable BUY_SEED(1-5), BUY_ANIMAL(6-8), BUY_PRODUCT(9-10)
        market_mask[20] = False    # Disable HIRE
        market_mask[21] = False    # Disable BUY_LAND

    # Prevent SELL actions if the shed has 0 of that item
    # SELL_ALL for PRODUCTS starts at index 11
    for i, item in enumerate(PRODUCTS):
        if _g(shed, item, 0) <= 0:
            market_mask[11 + i] = False

    # SELL_ALL for FERTILIZER is at index 19
    if _g(shed, "FERTILIZER", 0) <= 0:
        market_mask[19] = False

    hand_masks = [
        unit_mask if i < n_hands_actual else pass_only_mask
        for i in range(max_hands)
    ]

    # MaskablePPO requires a SINGLE, flat 1D numpy array for MultiDiscrete spaces
    return np.concatenate([unit_mask] + hand_masks + [market_mask] * market_slots)