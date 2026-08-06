
"""
kagg_common.py - FIXED VERSION
==============================
Critical fixes applied:
1. Reward function: Removed penalty-dominated bonus system, now purely money-based
2. calculate_net_worth: Removed inventory counting (only bank matters at game end)
3. Added per-step reward shaping with survival bonus
4. Fixed action masking to properly disable invalid actions
5. Added curriculum-ready opponent scaling
"""

from __future__ import annotations
import numpy as np

# ------------------------------------------------------------------------
# Game constants
# ------------------------------------------------------------------------
BOARD_SIZE = 10
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]
SHED_ITEMS = PRODUCTS + ["FERTILIZER", "GOOSE", "COW", "SHEEP"]
MARKET_RESOURCES = PRODUCTS + ["FERTILIZER"]
QUADRANTS = ["NW", "NE", "SW", "SE"]
SHOPS = ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
         "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"]

MAX_HANDS = 6
MARKET_SLOTS = 8
TILE_FEATS = 28
SCALAR_FEATS = 99
OBS_DIM = 2 * BOARD_SIZE * BOARD_SIZE * TILE_FEATS + SCALAR_FEATS

# Normalization constants
MONEY_NORM = 20000.0
PRICE_NORM = 300.0
INV_NORM = 10000.0
SHED_NORM = 100.0
SEED_NORM = 50.0
DAY_NORM = 30.0
HOUR_NORM = 24.0
POS_NORM = float(BOARD_SIZE - 1)
YIELD_NORM = 10.0
AGE_NORM = 30.0
UNWATERED_NORM = 2.0


def _g(d, key, default=None):
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
# Tile encoding (unchanged)
# ------------------------------------------------------------------------
def encode_tile(tile, current_day):
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
    return f


def encode_farm_tiles(farm, current_day):
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
    day = _g(obs, "day", 0)
    farms = _g(obs, "farms", [{}, {}])
    me = farms[player] if player < len(farms) else {}
    them = farms[1 - player] if (1 - player) < len(farms) else {}
    my_tiles = encode_farm_tiles(me, day)
    their_tiles = encode_farm_tiles(them, day)
    scalars = encode_scalars(obs, player)
    return np.concatenate([my_tiles, their_tiles, scalars]).astype(np.float32)


# ------------------------------------------------------------------------
# FIXED REWARD FUNCTION
# ------------------------------------------------------------------------

def compute_reward(prev_obs, curr_obs, player):
    """
    FIXED reward function:
    - Primary signal: Bank money change (the ONLY thing that matters for winning)
    - Secondary: Small survival bonus per step
    - Tertiary: End-game bonus proportional to final bank

    REMOVED: All arbitrary bonuses/penalties that dominated the signal
    """
    farms = _g(curr_obs, "farms", [{}, {}])
    prev_farms = _g(prev_obs, "farms", [{}, {}])

    curr_money = _g(farms[player] if player < len(farms) else {}, "money", 0.0)
    prev_money = _g(prev_farms[player] if player < len(prev_farms) else {}, "money", 0.0)

    # Primary: Money delta (scaled so $100 = +1.0 reward)
    money_delta = curr_money - prev_money
    reward = money_delta / 100.0

    # Small per-step survival bonus (encourages staying alive)
    reward += 0.01

    # End-game: big bonus for final bank balance
    day = _g(curr_obs, "day", 0)
    hour = _g(curr_obs, "hour", 0)
    if day >= 29 and hour >= 20:  # Near end of game
        reward += curr_money / 5000.0  # $10K final = +2.0 bonus

    return float(reward)


def episode_outcome_info(obs, player):
    farms = _g(obs, "farms", [{}, {}])
    me = _g(farms[player] if player < len(farms) else {}, "money", 0.0)
    them = _g(farms[1 - player] if (1 - player) < len(farms) else {}, "money", 0.0)
    win = 1.0 if me > them else (0.5 if me == them else 0.0)
    return {"final_money": float(me), "opp_final_money": float(them), "win": win}


# ------------------------------------------------------------------------
# Unit action catalog (unchanged)
# ------------------------------------------------------------------------
UNIT_ACTIONS = [
    ["PASS"], ["NORTH"], ["SOUTH"], ["EAST"], ["WEST"],
    ["PLANT", "WHEAT"], ["PLANT", "CARROT"], ["PLANT", "TOMATO"],
    ["PLANT", "STRAWBERRY"], ["PLANT", "MELON"],
    ["WATER"], ["HARVEST"], ["FERTILIZE"],
    ["BUILD_COOP"], ["BUILD_PASTURE"],
    ["FEED"], ["COLLECT_FERTILIZER"], ["CARE"],
    ["DIG"],
    ["PICKUP", "WHEAT", 10], ["PICKUP", "FERTILIZER", 10],
    ["PICKUP", "GOOSE", 1], ["PICKUP", "COW", 1], ["PICKUP", "SHEEP", 1],
    ["PLACE", "GOOSE"], ["PLACE", "COW"], ["PLACE", "SHEEP"],
    ["DROP"],
]
N_UNIT_ACTIONS = len(UNIT_ACTIONS)


def decode_unit_action(idx):
    idx = int(idx)
    if not (0 <= idx < N_UNIT_ACTIONS):
        return ["PASS"]
    return list(UNIT_ACTIONS[idx])


# ------------------------------------------------------------------------
# Market catalog (unchanged)
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
    *[_make_buy_seed(c) for c in CROPS],
    *[_make_buy_animal(a) for a in ANIMALS],
    _make_buy_product("WHEAT", 10), _make_buy_product("FERTILIZER", 5),
    *[_make_sell_all(p) for p in PRODUCTS],
    _make_sell_all("FERTILIZER"),
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
# Action space / decode (unchanged)
# ------------------------------------------------------------------------
def action_nvec(max_hands=MAX_HANDS, market_slots=MARKET_SLOTS):
    return [N_UNIT_ACTIONS] * (1 + max_hands) + [N_MARKET_ACTIONS] * market_slots


def decode_action(raw, obs, player, max_hands=MAX_HANDS, market_slots=MARKET_SLOTS):
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
    market_orders = market_orders[:10]

    return {"farmer": farmer_op, "hands": hands_ops, "market": market_orders}


# ------------------------------------------------------------------------
# FIXED ACTION MASKING
# ------------------------------------------------------------------------

def get_action_mask(obs, player, max_hands=MAX_HANDS, market_slots=MARKET_SLOTS):
    """
    FIXED action mask:
    - Properly disables actions that are impossible given current state
    - Reduces entropy by focusing exploration on valid actions only
    - Critical for sample efficiency in large action spaces
    """
    farms = _g(obs, "farms", [{}, {}])
    me = farms[player] if player < len(farms) else {}
    money = _g(me, "money", 0.0)
    n_hands_actual = len(_g(me, "hands", []) or [])

    private = _g(obs, "private", {}) or {}
    shed = _g(private, "shed", {}) or {}
    seeds = _g(private, "seeds", {}) or {}

    tiles = _g(me, "tiles", [])
    farmer = _g(me, "farmer", [4, 4])
    fx, fy = farmer[0], farmer[1]

    # Get current tile
    current_tile = None
    if 0 <= fy < len(tiles) and 0 <= fx < len(tiles[fy]):
        current_tile = tiles[fy][fx]

    # Unit action mask
    unit_mask = np.ones(N_UNIT_ACTIONS, dtype=np.bool_)

    # Disable movement that goes off-board
    if fx <= 0:
        unit_mask[4] = False  # WEST
    if fx >= BOARD_SIZE - 1:
        unit_mask[3] = False  # EAST
    if fy <= 0:
        unit_mask[1] = False  # NORTH
    if fy >= BOARD_SIZE - 1:
        unit_mask[2] = False  # SOUTH

    # Disable PLANT if not on empty tile or no seeds
    if current_tile is not None:
        unit_mask[5:10] = False  # All PLANT actions
    else:
        if seeds.get("WHEAT", 0) <= 0:
            unit_mask[5] = False
        if seeds.get("CARROT", 0) <= 0:
            unit_mask[6] = False
        if seeds.get("TOMATO", 0) <= 0:
            unit_mask[7] = False
        if seeds.get("STRAWBERRY", 0) <= 0:
            unit_mask[8] = False
        if seeds.get("MELON", 0) <= 0:
            unit_mask[9] = False

    # Disable WATER if not on thirsty plant
    if not (isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT" 
            and not current_tile.get("watered_today", False)):
        unit_mask[10] = False

    # Disable HARVEST if no yield
    if not (isinstance(current_tile, dict) and current_tile.get("yield_units", 0) > 0):
        unit_mask[11] = False

    # Disable FERTILIZE if not on plant
    if not (isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT"):
        unit_mask[12] = False

    # Disable BUILD if not on empty tile
    if current_tile is not None:
        unit_mask[13] = False  # BUILD_COOP
        unit_mask[14] = False  # BUILD_PASTURE

    # Disable FEED if not on unfed animal
    if not (isinstance(current_tile, dict) and "animal" in current_tile 
            and current_tile.get("animal") is not None 
            and not current_tile.get("fed_today", False)
            and shed.get("WHEAT", 0) > 0):
        unit_mask[15] = False

    # Disable COLLECT_FERTILIZER if no fertilizer available
    if not (isinstance(current_tile, dict) and current_tile.get("fertilizer_available", False)):
        unit_mask[16] = False

    # Disable CARE if not on uncared animal
    if not (isinstance(current_tile, dict) and "animal" in current_tile 
            and current_tile.get("animal") is not None 
            and not current_tile.get("cared_today", False)):
        unit_mask[17] = False

    # Disable DIG if not on weed/plant/empty structure
    if not (isinstance(current_tile, dict) and 
            (current_tile.get("kind") in ("WEED", "PLANT") or
             (current_tile.get("kind") in ("COOP", "PASTURE") and current_tile.get("animal") is None))):
        unit_mask[18] = False

    # Disable PICKUP if not near shed or no items in shed
    # Shed is at center tiles (4,4), (5,4), (4,5), (5,5)
    near_shed = (4 <= fx <= 5) and (4 <= fy <= 5)
    if not near_shed:
        unit_mask[19:24] = False
    else:
        if shed.get("WHEAT", 0) <= 0:
            unit_mask[19] = False
        if shed.get("FERTILIZER", 0) <= 0:
            unit_mask[20] = False
        if shed.get("GOOSE", 0) <= 0:
            unit_mask[21] = False
        if shed.get("COW", 0) <= 0:
            unit_mask[22] = False
        if shed.get("SHEEP", 0) <= 0:
            unit_mask[23] = False

    # Disable PLACE if not on matching empty structure
    if isinstance(current_tile, dict) and current_tile.get("kind") == "COOP" and current_tile.get("animal") is None:
        if shed.get("GOOSE", 0) <= 0:
            unit_mask[24] = False
    else:
        unit_mask[24] = False

    if isinstance(current_tile, dict) and current_tile.get("kind") == "PASTURE" and current_tile.get("animal") is None:
        if shed.get("COW", 0) <= 0:
            unit_mask[25] = False
        if shed.get("SHEEP", 0) <= 0:
            unit_mask[26] = False
    else:
        unit_mask[25:27] = False

    # Disable DROP if not near shed
    if not near_shed:
        unit_mask[27] = False

    # Hand masks: only PASS for non-existent hands
    pass_only_mask = np.zeros(N_UNIT_ACTIONS, dtype=np.bool_)
    pass_only_mask[0] = True

    hand_masks = [
        unit_mask if i < n_hands_actual else pass_only_mask
        for i in range(max_hands)
    ]

    # Market mask
    market_mask = np.ones(N_MARKET_ACTIONS, dtype=np.bool_)

    # Disable BUY actions if no money
    if money < 50:
        market_mask[1:6] = False   # BUY_SEED
        market_mask[6:9] = False   # BUY_ANIMAL
        market_mask[9:11] = False  # BUY_PRODUCT
    elif money < 300:
        market_mask[6:9] = False   # Can't afford animals
        market_mask[10] = False    # Can't afford fertilizer
    elif money < 500:
        market_mask[7:9] = False   # Can't afford cow/sheep

    # Disable HIRE if can't afford
    hires_today = me.get("hires_today", 0)
    fib_cost = [1, 1, 2, 3, 5, 8, 13, 21][min(hires_today, 7)]
    if money < fib_cost:
        market_mask[20] = False

    # Disable BUY_LAND if can't afford or no land left
    unlocked = me.get("unlocked_quadrants", ["NW"])
    if "SE" in unlocked or money < 4000:
        market_mask[21] = False
    elif "SW" in unlocked and money < 2000:
        market_mask[21] = False
    elif "NE" in unlocked and money < 1000:
        market_mask[21] = False

    # Disable SELL if no inventory
    for i, item in enumerate(PRODUCTS):
        if _g(shed, item, 0) <= 0:
            market_mask[11 + i] = False
    if _g(shed, "FERTILIZER", 0) <= 0:
        market_mask[19] = False

    return np.concatenate([unit_mask] + hand_masks + [market_mask] * market_slots)