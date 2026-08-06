"""
Kaggriculture Tier-1 agent.

Strategy:
  - Every turn, pick the single most profitable crop to be planting *right now*
    given live market prices (base price * yield/tile/day, minus amortized
    seed cost, gated so we never start a crop that can't finish before day 30).
  - Route the farmer + every hired hand with a greedy nearest-job assignment:
    HARVEST > WATER/DIG > PLANT, recomputed fresh each turn (so it self-heals
    if a job disappears or a new one appears).
  - Sell shed inventory every turn, but throttle per-item quantity so we don't
    crater premium (melon/strawberry/milk/wool) prices in one shot -- except
    in the final two days, when unsold shed inventory is worthless anyway, so
    we dump everything.
  - Hire extra hands when there's more work queued up than units to do it and
    we can comfortably afford the (Fibonacci-scaled) hire cost.
  - Buy the next quadrant once the current land is mostly built out and there's
    enough season left for the investment to pay off.
"""

SEASON_DAYS = 30
BOARD_SIZE = 10

CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "yield_per_day": 0.80},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "yield_per_day": 0.75},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 11, "yield_per_day": 0.33},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 16, "yield_per_day": 0.24},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "yield_per_day": 0.55},
}

# Per-turn sell throttle, roughly scaled to how hard each resource's price
# craters on oversupply (see the Price Function table -- melon/strawberry/
# milk/wool have above_target > 1 and crash to the $1 floor fast).
SELL_CAP_PER_TURN = {
    "WHEAT": 15, "CARROT": 15, "TOMATO": 8, "STRAWBERRY": 4,
    "MELON": 4, "EGG": 8, "MILK": 3, "WOOL": 3, "FERTILIZER": 10,
}

LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = {"NE": 1000, "SW": 2000, "SE": 4000}


def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def is_plant(tile):
    return isinstance(tile, dict) and tile.get("kind") == "PLANT"


def is_weed(tile):
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def is_animal_tile(tile):
    return isinstance(tile, dict) and "animal" in tile


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_towards(cur, target):
    cx, cy = cur
    tx, ty = target
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return "PASS"
    # Close the larger axis first.
    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    return "SOUTH" if dy > 0 else "NORTH"


def best_crop(prices, day, money):
    """Pick the crop with the best (price * yield/tile/day - amortized seed
    cost) among crops we can afford and that can still finish this season."""
    remaining = SEASON_DAYS - day
    best, best_score = None, float("-inf")
    for crop, cd in CROPS.items():
        if cd["seed"] > money:
            continue
        grow = cd["max_yield_day"]
        if remaining < grow + 1:
            continue  # wouldn't finish in time -- don't start it
        price = prices.get(crop, 1)
        score = price * cd["yield_per_day"] - cd["seed"] / grow
        if score > best_score:
            best_score, best = score, crop
    return best


def quadrant_utilization(tiles):
    """Fraction of *unlocked* tiles that are doing something productive
    (plant, weed pending removal doesn't count as productive, structure)."""
    total, used = 0, 0
    for row in tiles:
        for t in row:
            if t == "LOCKED":
                continue
            total += 1
            if t is not None and not is_weed(t):
                used += 1
    return used / total if total else 0.0


def find_jobs(tiles, day, plantable_seeds, target_crop):
    """Return [(priority, (x, y), job_str), ...], priority 0 = most urgent."""
    jobs = []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if tile == "LOCKED":
                continue
            if is_plant(tile):
                cd = CROPS[tile["crop"]]
                ready = (
                    tile["yield_units"] > 0
                    and day - tile["planted_day"] >= cd["first_yield_day"]
                )
                if ready:
                    jobs.append((0, (x, y), "HARVEST"))
                elif not tile["watered_today"]:
                    jobs.append((1, (x, y), "WATER"))
            elif is_animal_tile(tile):
                if tile["yield_units"] > 0:
                    jobs.append((0, (x, y), "HARVEST"))
            elif is_weed(tile):
                jobs.append((1, (x, y), "DIG"))
            elif tile is None:
                if plantable_seeds > 0 and target_crop:
                    jobs.append((2, (x, y), "PLANT:" + target_crop))
    jobs.sort(key=lambda j: j[0])
    return jobs


def assign_and_act(unit_positions, jobs):
    """Greedy nearest-unclaimed-job assignment. Units standing on a job do it
    immediately; otherwise they take one step toward the nearest free job."""
    claimed = set()
    actions = []
    for pos in unit_positions:
        pos = tuple(pos)
        here_job = None
        for _, jpos, job in jobs:
            if jpos == pos and jpos not in claimed:
                here_job = (jpos, job)
                break
        if here_job:
            jpos, job = here_job
            claimed.add(jpos)
            if job == "HARVEST":
                actions.append(["HARVEST"])
            elif job == "WATER":
                actions.append(["WATER"])
            elif job == "DIG":
                actions.append(["DIG"])
            else:  # "PLANT:<crop>"
                actions.append(["PLANT", job.split(":", 1)[1]])
            continue

        best_job, best_dist = None, float("inf")
        for _, jpos, job in jobs:
            if jpos in claimed:
                continue
            d = manhattan(pos, jpos)
            if d < best_dist:
                best_dist, best_job = d, (jpos, job)
        if best_job:
            jpos, _job = best_job
            claimed.add(jpos)
            actions.append([step_towards(pos, jpos)])
        else:
            actions.append(["PASS"])
    return actions


def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    market = obs["market"]
    day = obs.get("day", 0)

    tiles = me["tiles"]
    money = me["money"]
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    prices = market.get("prices", {})
    remaining_days = SEASON_DAYS - day
    end_game = remaining_days <= 2

    market_orders = []

    # 1) Sell shed inventory (throttled, except in the final stretch).
    for item, count in shed.items():
        if count <= 0:
            continue
        cap = count if end_game else SELL_CAP_PER_TURN.get(item, 10)
        qty = min(count, cap)
        if qty > 0:
            market_orders.append(["SELL", item, qty])

    # 2) Pick a target crop and top up seed stock if it's cheap to do so.
    target_crop = None if end_game else best_crop(prices, day, money)
    if target_crop and seeds.get(target_crop, 0) < 3 and money >= CROPS[target_crop]["seed"]:
        market_orders.append(["BUY_SEED", target_crop, 1])

    # 3) Expand land once the current farm is mostly built out and there's
    #    enough season left to make the investment worthwhile.
    unlocked = me.get("unlocked_quadrants", ["NW"])
    next_quad = next((q for q in LAND_ORDER if q not in unlocked), None)
    if (
        next_quad
        and remaining_days > 8
        and quadrant_utilization(tiles) > 0.75
        and money >= LAND_PRICES[next_quad] + 500
    ):
        market_orders.append(["BUY_LAND"])

    # 4) Compute this turn's job list once (used for both hiring math and
    #    unit routing).
    plantable = seeds.get(target_crop, 0) if target_crop else 0
    jobs = find_jobs(tiles, day, plantable, target_crop)

    # 5) Hire another hand if there's a real backlog and it's cheap relative
    #    to our bank, but not so late in the day/season that it can't pay back.
    num_units = 1 + len(me.get("hands", []))
    hires_today = me.get("hires_today", 0)
    next_hire_cost = _fib(hires_today)
    if (
        len(jobs) > num_units + 2
        and remaining_days > 3
        and hires_today < 4
        and money >= next_hire_cost + 300
    ):
        market_orders.append(["HIRE"])

    market_orders = market_orders[:10]  # maxMarketOrdersPerTurn default

    # 6) Route farmer + hands.
    unit_positions = [tuple(me["farmer"])] + [tuple(h) for h in me.get("hands", [])]
    unit_actions = assign_and_act(unit_positions, jobs)

    return {
        "farmer": unit_actions[0],
        "hands": unit_actions[1:],
        "market": market_orders,
    }