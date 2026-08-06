"""
kagg_control.py - Scripted turn-level controller.

This is the "muscle" layer of the hybrid agent. It knows nothing about
strategy: it receives a DayPlan (what to plant, how many hands to hire,
whether to buy land / animals, when to sell) and executes it competently at
the turn level -- pathfinding, watering, feeding, harvest timing, and
market order construction.

The RL policy chooses the DayPlan once per day. Everything below is
deterministic, so the policy never has to learn navigation or crop
bookkeeping.

Design notes
------------
* Jobs are (priority, tile) pairs derived from farm state each turn.
  Lower priority number = more urgent. Survival jobs (WATER, FEED) outrank
  growth jobs (PLANT, HARVEST) which outrank upkeep (CARE, DIG).
* Units are assigned greedily: each unit takes the nearest unclaimed job of
  the best available priority. With cheap labour this is close enough to
  optimal; a Hungarian assignment gains very little here.
* FEED consumes wheat from the *acting unit's inventory* (verified against
  kaggriculture.py `_apply_unit_action`), so units that are assigned feeding
  duty must first walk to the shed and PICKUP wheat.
"""
from __future__ import annotations

BOARD_SIZE = 10
TURNS_PER_DAY = 24
SHED_CAPACITY = 100

CROPS = {
    "WHEAT":      {"seed": 10, "max_yield_day": 4,  "ongoing": False, "base": 25},
    "CARROT":     {"seed": 20, "max_yield_day": 3,  "ongoing": False, "base": 35},
    "TOMATO":     {"seed": 50, "max_yield_day": 8,  "ongoing": True,  "base": 60},
    "STRAWBERRY": {"seed": 100, "max_yield_day": 10, "ongoing": True, "base": 120},
    "MELON":      {"seed": 80, "max_yield_day": 12, "ongoing": False, "base": 250},
}

# Age (in days) at which a one-time crop stops gaining yield from watering.
# Harvesting earlier throws away units; later starts decay.
HARVEST_AGE = {"WHEAT": 4, "CARROT": 3, "MELON": 10}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "product": "EGG",  "interval": 1},
    "COW":   {"cost": 400, "structure": "PASTURE", "product": "MILK", "interval": 2},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "product": "WOOL", "interval": 3},
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]
SELLABLE = PRODUCTS + ["FERTILIZER"]
BASE_PRICE = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
    "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
}

QUADRANTS = ["NW", "NE", "SW", "SE"]
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]

SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]

# Job priorities (lower = more urgent).
P_WATER = 0
P_FEED = 1
P_HARVEST = 2
P_PLACE = 3
P_PLANT = 4
P_BUILD = 5
P_FERTILIZE = 6
P_CARE = 7
P_COLLECT = 8
P_DIG = 9


def _g(d, key, default=None):
    """Tolerant getter: kaggle observations are Struct (dict subclass)."""
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


def quadrant_of(x, y, board_size=BOARD_SIZE):
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


class DayPlan:
    """High-level decisions the RL policy makes once per in-game day."""

    __slots__ = ("crop", "hire_target", "buy_land", "animal", "sell_mode",
                 "fertilize", "plant_cap")

    def __init__(self, crop="WHEAT", hire_target=4, buy_land=False, animal=None,
                 sell_mode=0, fertilize=False, plant_cap=25):
        self.crop = crop              # crop to sow on empty tiles, or None
        self.hire_target = hire_target  # hands to have on payroll today
        self.buy_land = buy_land      # attempt BUY_LAND this morning
        self.animal = animal          # animal to invest in, or None
        self.sell_mode = sell_mode    # 0 dump, 1 >=60% base, 2 >=90% base, 3 hold
        self.fertilize = fertilize    # buy + apply fertilizer
        self.plant_cap = plant_cap    # max tiles to sow today

    def __repr__(self):
        return (f"DayPlan(crop={self.crop}, hire={self.hire_target}, "
                f"land={self.buy_land}, animal={self.animal}, "
                f"sell={self.sell_mode}, fert={self.fertilize}, "
                f"cap={self.plant_cap})")


# --------------------------------------------------------------------------
# Farm inspection
# --------------------------------------------------------------------------
def scan_farm(farm, day):
    """One pass over the grid; returns job list and summary counts."""
    tiles = _g(farm, "tiles", []) or []
    unlocked = set(_g(farm, "unlocked_quadrants", ["NW"]) or ["NW"])

    jobs = []          # (priority, x, y, op_kind)
    counts = {
        "empty": 0, "weed": 0, "plants": 0, "animals": 0,
        "empty_coop": 0, "empty_pasture": 0, "ripe": 0,
    }
    per_crop = {c: 0 for c in CROPS}

    for y in range(min(len(tiles), BOARD_SIZE)):
        row = tiles[y]
        for x in range(min(len(row), BOARD_SIZE)):
            if quadrant_of(x, y) not in unlocked:
                continue
            t = row[x]

            if t is None:
                counts["empty"] += 1
                jobs.append((P_PLANT, x, y, "PLANT"))
                continue
            if t == "LOCKED" or not isinstance(t, dict):
                continue

            kind = _g(t, "kind")

            if kind == "WEED":
                counts["weed"] += 1
                jobs.append((P_DIG, x, y, "DIG"))
                continue

            if kind == "PLANT":
                counts["plants"] += 1
                crop = _g(t, "crop", "WHEAT")
                per_crop[crop] = per_crop.get(crop, 0) + 1
                age = day - _g(t, "planted_day", day)
                cd = CROPS.get(crop, CROPS["WHEAT"])

                if not _g(t, "watered_today", False):
                    jobs.append((P_WATER, x, y, "WATER"))

                yield_units = _g(t, "yield_units", 0)
                if yield_units > 0:
                    if cd["ongoing"]:
                        # Ongoing crops: take produce as soon as it appears,
                        # the tile keeps producing regardless.
                        jobs.append((P_HARVEST, x, y, "HARVEST"))
                        counts["ripe"] += 1
                    elif age >= HARVEST_AGE.get(crop, cd["max_yield_day"]):
                        # One-time crops: wait for peak, then take it before
                        # decay starts one day after max_yield_day.
                        jobs.append((P_HARVEST, x, y, "HARVEST"))
                        counts["ripe"] += 1
                continue

            if kind in ("COOP", "PASTURE"):
                animal = _g(t, "animal")
                if animal is None:
                    if kind == "COOP":
                        counts["empty_coop"] += 1
                    else:
                        counts["empty_pasture"] += 1
                    jobs.append((P_PLACE, x, y, "PLACE_" + kind))
                    continue

                counts["animals"] += 1
                if not _g(t, "fed_today", False):
                    jobs.append((P_FEED, x, y, "FEED"))
                if _g(t, "yield_units", 0) > 0:
                    jobs.append((P_HARVEST, x, y, "HARVEST"))
                    counts["ripe"] += 1
                if not _g(t, "cared_today", False):
                    jobs.append((P_CARE, x, y, "CARE"))
                if _g(t, "fertilizer_available", False):
                    jobs.append((P_COLLECT, x, y, "COLLECT_FERTILIZER"))

    counts["per_crop"] = per_crop
    return jobs, counts


# --------------------------------------------------------------------------
# Movement
# --------------------------------------------------------------------------
def step_toward(pos, target):
    """One orthogonal step. Locked tiles are passable so no obstacle logic."""
    px, py = pos
    tx, ty = target
    dx, dy = tx - px, ty - py
    if abs(dx) >= abs(dy):
        if dx > 0:
            return ["EAST"]
        if dx < 0:
            return ["WEST"]
    if dy > 0:
        return ["SOUTH"]
    if dy < 0:
        return ["NORTH"]
    return None  # already there


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_shed_tile(pos, unlocked):
    """Closest shed-access tile inside an unlocked quadrant."""
    cands = [t for t in SHED_TILES if quadrant_of(t[0], t[1]) in unlocked]
    if not cands:
        cands = SHED_TILES
    return min(cands, key=lambda t: manhattan(pos, t))


# --------------------------------------------------------------------------
# Unit assignment
# --------------------------------------------------------------------------
def assign_units(obs, player, plan, jobs, counts):
    """Return a list of ops, one per unit (main farmer first)."""
    farms = _g(obs, "farms", [{}, {}])
    me = farms[player] if player < len(farms) else {}
    private = _g(obs, "private", {}) or {}

    unlocked = set(_g(me, "unlocked_quadrants", ["NW"]) or ["NW"])
    farmer = list(_g(me, "farmer", [4, 4]) or [4, 4])
    hands = [list(h) for h in (_g(me, "hands", []) or [])]
    positions = [farmer] + hands

    inventories = _g(private, "inventories", []) or []
    seeds = _g(private, "seeds", {}) or {}
    shed = _g(private, "shed", {}) or {}

    n_units = len(positions)
    ops = [["PASS"] for _ in range(n_units)]

    # How much can we actually sow right now?
    seed_budget = 0
    if plan.crop:
        seed_budget = min(int(_g(seeds, plan.crop, 0)), plan.plant_cap)

    # Wheat needed on-hand for feeding. FEED takes from unit inventory.
    n_feed_jobs = sum(1 for j in jobs if j[3] == "FEED")

    # Animals waiting for a tenant, and animals available in the shed.
    shed_animals = {a: int(_g(shed, a, 0)) for a in ANIMALS}

    available = sorted(jobs, key=lambda j: j[0])
    claimed = set()

    def unit_wheat(i):
        if i < len(inventories):
            return int(_g(inventories[i] or {}, "WHEAT", 0))
        return 0

    for i, pos in enumerate(positions):
        best = None
        best_cost = None

        for job in available:
            prio, jx, jy, kind = job
            if (jx, jy) in claimed:
                continue

            # Feasibility filters -----------------------------------------
            if kind == "PLANT":
                if seed_budget <= 0:
                    continue
            elif kind == "FEED":
                if unit_wheat(i) <= 0:
                    continue
            elif kind == "PLACE_COOP":
                if shed_animals.get("GOOSE", 0) <= 0:
                    continue
            elif kind == "PLACE_PASTURE":
                if shed_animals.get("COW", 0) <= 0 and shed_animals.get("SHEEP", 0) <= 0:
                    continue

            cost = manhattan(pos, (jx, jy))
            # Priority dominates distance; distance breaks ties within a tier.
            score = (prio, cost)
            if best_cost is None or score < best_cost:
                best_cost = score
                best = job

        # A unit with no feasible job but feeding work outstanding fetches
        # wheat from the shed so it can feed next turn.
        if best is None and n_feed_jobs > 0 and unit_wheat(i) == 0 and int(_g(shed, "WHEAT", 0)) > 0:
            target = nearest_shed_tile(pos, unlocked)
            if tuple(pos) == target:
                ops[i] = ["PICKUP", "WHEAT", min(5, int(_g(shed, "WHEAT", 0)))]
            else:
                mv = step_toward(pos, target)
                ops[i] = mv if mv else ["PASS"]
            continue

        if best is None:
            ops[i] = ["PASS"]
            continue

        prio, jx, jy, kind = best
        if tuple(pos) != (jx, jy):
            mv = step_toward(pos, (jx, jy))
            ops[i] = mv if mv else ["PASS"]
            # Don't claim: another unit standing closer should still take it,
            # but reserve so units don't all converge on one tile.
            claimed.add((jx, jy))
            continue

        claimed.add((jx, jy))
        if kind == "PLANT":
            ops[i] = ["PLANT", plan.crop]
            seed_budget -= 1
        elif kind == "PLACE_COOP":
            ops[i] = ["PLACE", "GOOSE"]
            shed_animals["GOOSE"] -= 1
        elif kind == "PLACE_PASTURE":
            pick = "COW" if shed_animals.get("COW", 0) > 0 else "SHEEP"
            ops[i] = ["PLACE", pick]
            shed_animals[pick] -= 1
        elif kind == "FEED":
            ops[i] = ["FEED"]
        else:
            ops[i] = [kind]

    # Spend genuinely idle units on building animal housing if the plan
    # calls for livestock and there is nowhere to put it.
    if plan.animal:
        want_struct = ANIMALS[plan.animal]["structure"]
        have_empty = counts["empty_coop"] if want_struct == "COOP" else counts["empty_pasture"]
        pending = shed_animals.get(plan.animal, 0)
        if pending > have_empty:
            for i, pos in enumerate(positions):
                if ops[i] != ["PASS"]:
                    continue
                tiles = _g(me, "tiles", []) or []
                x, y = pos
                on_empty = (0 <= y < len(tiles) and 0 <= x < len(tiles[y])
                            and tiles[y][x] is None
                            and quadrant_of(x, y) in unlocked)
                if on_empty:
                    ops[i] = ["BUILD_COOP"] if want_struct == "COOP" else ["BUILD_PASTURE"]
                    break

    return ops[0], ops[1:]


# --------------------------------------------------------------------------
# Market orders
# --------------------------------------------------------------------------
def build_market_orders(obs, player, plan, counts, max_orders=10):
    farms = _g(obs, "farms", [{}, {}])
    me = farms[player] if player < len(farms) else {}
    private = _g(obs, "private", {}) or {}
    market = _g(obs, "market", {}) or {}
    prices = _g(market, "prices", {}) or {}

    day = int(_g(obs, "day", 0))
    hour = int(_g(obs, "hour", 0))
    money = float(_g(me, "money", 0.0))
    shed = _g(private, "shed", {}) or {}
    seeds = _g(private, "seeds", {}) or {}
    hires_today = int(_g(me, "hires_today", 0))
    n_hands = len(_g(me, "hands", []) or [])

    orders = []

    # -- Morning block: hire, land, capital goods ---------------------------
    if hour == 0:
        # Labour is the cheapest resource in the game: fib(n) with the default
        # multiplier means 8 hands cost 54 coins for a full day of 24 turns.
        want = max(0, plan.hire_target - n_hands)
        budget = money
        for k in range(want):
            cost = _fib(hires_today + k)
            if budget < cost:
                break
            orders.append(["HIRE"])
            budget -= cost

        if plan.buy_land:
            n_extra = len(_g(me, "unlocked_quadrants", ["NW"]) or ["NW"]) - 1
            if 0 <= n_extra < len(LAND_PRICES) and budget >= LAND_PRICES[n_extra]:
                orders.append(["BUY_LAND"])
                budget -= LAND_PRICES[n_extra]

        if plan.animal:
            a = ANIMALS[plan.animal]
            if budget >= a["cost"]:
                orders.append(["BUY_ANIMAL", plan.animal, 1])
                budget -= a["cost"]

    # -- Seed restock -------------------------------------------------------
    if plan.crop:
        have = int(_g(seeds, plan.crop, 0))
        need = max(0, min(plan.plant_cap, counts["empty"]) - have)
        if need > 0:
            cost_each = CROPS[plan.crop]["seed"]
            afford = int(money // cost_each)
            n = max(0, min(need, afford))
            if n > 0:
                orders.append(["BUY_SEED", plan.crop, n])

    # -- Feed wheat for livestock ------------------------------------------
    if counts["animals"] > 0:
        wheat_on_hand = int(_g(shed, "WHEAT", 0))
        need_wheat = counts["animals"] * 3
        if wheat_on_hand < need_wheat and money > 500:
            orders.append(["BUY_PRODUCT", "WHEAT", need_wheat - wheat_on_hand])

    # -- Fertilizer ---------------------------------------------------------
    if plan.fertilize and money > 1000:
        if int(_g(shed, "FERTILIZER", 0)) < 5:
            orders.append(["BUY_PRODUCT", "FERTILIZER", 5])

    # -- Sells --------------------------------------------------------------
    shed_total = sum(int(v) for k, v in shed.items() if k in SELLABLE)
    last_day = day >= 29
    force = last_day or shed_total > 0.8 * SHED_CAPACITY

    for item in SELLABLE:
        qty = int(_g(shed, item, 0))
        if qty <= 0:
            continue
        # Never hold livestock we intend to place; only sell surplus produce.
        price = float(_g(prices, item, BASE_PRICE.get(item, 1)))
        base = BASE_PRICE.get(item, 1)

        if force or plan.sell_mode == 0:
            ok = True
        elif plan.sell_mode == 1:
            ok = price >= 0.60 * base
        elif plan.sell_mode == 2:
            ok = price >= 0.90 * base
        else:
            ok = False

        if ok:
            orders.append(["SELL", item, qty])

    return orders[:max_orders]


def _fib(n):
    a, b = 1, 1
    for _ in range(int(n)):
        a, b = b, a + b
    return a


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def act(obs, player, plan):
    """Turn a DayPlan into a concrete per-turn action dict."""
    farms = _g(obs, "farms", [{}, {}])
    me = farms[player] if player < len(farms) else {}
    day = int(_g(obs, "day", 0))

    jobs, counts = scan_farm(me, day)
    farmer_op, hand_ops = assign_units(obs, player, plan, jobs, counts)
    orders = build_market_orders(obs, player, plan, counts)

    return {"farmer": farmer_op, "hands": hand_ops, "market": orders}, counts