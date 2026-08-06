
"""
Kaggriculture Tier-2 Agent with Animal Support.

Strategy:
  - Phase 1 (Days 0-3): Plant wheat for quick cash and feed stockpile.
  - Phase 2 (Days 4-30): Build animal infrastructure (COW > SHEEP > GOOSE).
  - Animals need daily: FEED (wheat), CARE (2-4x yield multiplier), HARVEST.
  - Fertilizer from animals collected and sold for extra revenue.
  - Remaining land for high-value crops (melon, strawberry) in late game.
  - Sell with throttling to prevent market price crashes.

Animal Economics (with daily CARE):
  - COW:  $215/day profit, pays back in ~2 days, harvest every 2 days
  - SHEEP: $242/day profit, pays back in ~2 days, harvest every 3 days
  - GOOSE:  $75/day profit, pays back in ~4 days, harvest daily
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

ANIMALS = {
    "GOOSE": {"structure": "COOP",     "product": "EGG",  "cost": 300, "build_cost": 1,
              "first_yield_day": 4, "interval": 1, "max_held": 4},
    "COW":   {"structure": "PASTURE",  "product": "MILK", "cost": 400, "build_cost": 1,
              "first_yield_day": 8, "interval": 2, "max_held": 6},
    "SHEEP": {"structure": "PASTURE",  "product": "WOOL", "cost": 500, "build_cost": 1,
              "first_yield_day": 6, "interval": 3, "max_held": 6},
}

# Priority: COW = best ROI, SHEEP = highest total profit, GOOSE = early cash
ANIMAL_PRIORITY = ["COW", "SHEEP", "GOOSE"]

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


def is_structure(tile):
    return isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE")


def is_empty_structure(tile):
    return is_structure(tile) and tile.get("animal") is None


def is_occupied_structure(tile):
    return is_structure(tile) and tile.get("animal") is not None


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_towards(cur, target):
    cx, cy = cur
    tx, ty = target
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return "PASS"
    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    return "SOUTH" if dy > 0 else "NORTH"


def best_crop(prices, day, money, remaining_days):
    best, best_score = None, float("-inf")
    for crop, cd in CROPS.items():
        if cd["seed"] > money:
            continue
        grow = cd["max_yield_day"]
        if remaining_days < grow + 1:
            continue
        price = prices.get(crop, 1)
        score = price * cd["yield_per_day"] - cd["seed"] / grow
        if score > best_score:
            best_score, best = score, crop
    return best


def count_animals(tiles):
    counts = {"GOOSE": 0, "COW": 0, "SHEEP": 0}
    for row in tiles:
        for t in row:
            if is_occupied_structure(t):
                a = t.get("animal")
                if a in counts:
                    counts[a] += 1
    return counts


def count_structures(tiles):
    counts = {"COOP": 0, "PASTURE": 0}
    for row in tiles:
        for t in row:
            if is_structure(t):
                counts[t["kind"]] += 1
    return counts


def find_empty_structures(tiles, struct_type):
    """Find empty structures of a specific type."""
    result = []
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if is_empty_structure(t) and t["kind"] == struct_type:
                result.append((x, y))
    return result


def find_empty_unlocked_tiles(tiles):
    empty = []
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if t is None:
                empty.append((x, y))
    return empty


def quadrant_utilization(tiles):
    total, used = 0, 0
    for row in tiles:
        for t in row:
            if t == "LOCKED":
                continue
            total += 1
            if t is not None and not is_weed(t):
                used += 1
    return used / total if total else 0.0


def find_jobs(tiles, day, plantable_seeds, target_crop, shed):
    """Find all jobs, sorted by priority (0 = most urgent)."""
    jobs = []
    wheat_in_shed = shed.get("WHEAT", 0)

    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if t == "LOCKED":
                continue

            # === PLANTS ===
            if is_plant(t):
                cd = CROPS[t["crop"]]
                ready = (t["yield_units"] > 0
                         and day - t["planted_day"] >= cd["first_yield_day"])
                if ready:
                    jobs.append((0, (x, y), "HARVEST"))
                elif not t["watered_today"]:
                    jobs.append((1, (x, y), "WATER"))

            # === OCCUPIED STRUCTURES (animals) ===
            elif is_occupied_structure(t):
                animal = t["animal"]

                # Harvest products before max_held cap
                if t["yield_units"] > 0:
                    jobs.append((0, (x, y), "HARVEST"))

                # Feed animals (need wheat in shed)
                if not t.get("fed_today", False) and wheat_in_shed > 0:
                    jobs.append((1, (x, y), "FEED"))

                # Care for animals (critical for bonus yield)
                if not t.get("cared_today", False):
                    jobs.append((1, (x, y), "CARE"))

                # Collect fertilizer
                if t.get("fertilizer_available", False):
                    jobs.append((2, (x, y), "COLLECT_FERTILIZER"))

            # === EMPTY STRUCTURES (place animals from shed) ===
            elif is_empty_structure(t):
                struct = t["kind"]
                for aname, acfg in ANIMALS.items():
                    if acfg["structure"] == struct and shed.get(aname, 0) > 0:
                        jobs.append((2, (x, y), "PLACE:" + aname))
                        break

            # === WEEDS ===
            elif is_weed(t):
                jobs.append((1, (x, y), "DIG"))

            # === EMPTY TILES (plant crops) ===
            elif t is None:
                if plantable_seeds > 0 and target_crop:
                    jobs.append((3, (x, y), "PLANT:" + target_crop))

    jobs.sort(key=lambda j: j[0])
    return jobs


def assign_and_act(unit_positions, jobs, tiles, shed):
    """Greedy nearest-unclaimed-job assignment."""
    claimed = set()
    actions = []
    wheat_in_shed = shed.get("WHEAT", 0)

    for pos in unit_positions:
        pos = tuple(pos)

        # Check if standing on a valid job
        here_job = None
        for _, jpos, job in jobs:
            if jpos == pos and jpos not in claimed:
                tile = tiles[jpos[1]][jpos[0]]
                if is_occupied_structure(tile) and job == "FEED" and wheat_in_shed <= 0:
                    continue
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
            elif job == "FEED":
                actions.append(["FEED"])
            elif job == "CARE":
                actions.append(["CARE"])
            elif job == "COLLECT_FERTILIZER":
                actions.append(["COLLECT_FERTILIZER"])
            elif job.startswith("PLANT:"):
                actions.append(["PLANT", job.split(":", 1)[1]])
            elif job.startswith("PLACE:"):
                actions.append(["PLACE", job.split(":", 1)[1]])
            continue

        # Move toward nearest valid job
        best_job, best_dist = None, float("inf")
        for _, jpos, job in jobs:
            if jpos in claimed:
                continue
            tile = tiles[jpos[1]][jpos[0]]
            if is_occupied_structure(tile) and job == "FEED" and wheat_in_shed <= 0:
                continue
            d = manhattan(pos, jpos)
            if d < best_dist:
                best_dist, best_job = d, (jpos, job)

        if best_job:
            jpos, _ = best_job
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

    # Count animals and structures
    animal_counts = count_animals(tiles)
    structure_counts = count_structures(tiles)
    total_animals = sum(animal_counts.values())

    # === 1) SELL SHED INVENTORY ===
    for item, count in shed.items():
        if count <= 0:
            continue
        cap = count if end_game else SELL_CAP_PER_TURN.get(item, 10)
        qty = min(count, cap)
        if qty > 0:
            market_orders.append(["SELL", item, qty])

    # === 2) WHEAT MANAGEMENT FOR ANIMAL FEED ===
    wheat_in_shed = shed.get("WHEAT", 0)
    wheat_price = prices.get("WHEAT", 25)
    wheat_needed = total_animals + 3  # Buffer for new animals

    # Emergency wheat purchase if animals would starve
    if wheat_in_shed < total_animals and total_animals > 0 and not end_game:
        buy_qty = total_animals - wheat_in_shed + 3
        if money >= wheat_price * buy_qty + 100:
            market_orders.append(["BUY_PRODUCT", "WHEAT", buy_qty])

    # === 3) ANIMAL ACQUISITION STRATEGY ===
    target_animal = None
    need_build = False

    if not end_game and day >= 2:
        for aname in ANIMAL_PRIORITY:
            acfg = ANIMALS[aname]
            min_days = acfg["first_yield_day"] + 3
            if remaining_days < min_days:
                continue

            # Check for empty structures of right type
            empty_structs = find_empty_structures(tiles, acfg["structure"])

            if empty_structs:
                # Buy animal if we don't have one in shed
                if money >= acfg["cost"] + 200 and shed.get(aname, 0) == 0:
                    target_animal = aname
                    break
            else:
                # Need to build new structure
                empty_tiles = find_empty_unlocked_tiles(tiles)
                total_cost = acfg["cost"] + acfg["build_cost"]
                if money >= total_cost + 400 and len(empty_tiles) > 2:
                    target_animal = aname
                    need_build = True
                    break

    # Buy animal
    if target_animal and shed.get(target_animal, 0) == 0:
        acfg = ANIMALS[target_animal]
        if money >= acfg["cost"]:
            market_orders.append(["BUY_ANIMAL", target_animal, 1])

    # === 4) CROP STRATEGY ===
    # Early game: wheat for cash and feed
    # Later: wheat if low on feed, otherwise best crop
    if day <= 3:
        target_crop = "WHEAT"
    else:
        if total_animals >= 2 and wheat_in_shed < total_animals * 2:
            target_crop = "WHEAT"
        else:
            target_crop = None if end_game else best_crop(prices, day, money, remaining_days)

    if target_crop and seeds.get(target_crop, 0) < 3:
        if money >= CROPS[target_crop]["seed"]:
            market_orders.append(["BUY_SEED", target_crop, 1])

    # === 5) LAND EXPANSION ===
    unlocked = me.get("unlocked_quadrants", ["NW"])
    next_quad = next((q for q in LAND_ORDER if q not in unlocked), None)
    if (
        next_quad
        and remaining_days > 8
        and quadrant_utilization(tiles) > 0.70
        and money >= LAND_PRICES[next_quad] + 500
    ):
        market_orders.append(["BUY_LAND"])

    # === 6) COMPUTE JOBS ===
    plantable = seeds.get(target_crop, 0) if target_crop else 0
    jobs = find_jobs(tiles, day, plantable, target_crop, shed)

    # Add BUILD jobs if needed
    if need_build and target_animal:
        acfg = ANIMALS[target_animal]
        empty_tiles = find_empty_unlocked_tiles(tiles)
        center = (BOARD_SIZE // 2, BOARD_SIZE // 2)
        empty_tiles.sort(key=lambda p: manhattan(p, center))
        for ex, ey in empty_tiles:
            if tiles[ey][ex] is None:
                build_op = "BUILD_COOP" if acfg["structure"] == "COOP" else "BUILD_PASTURE"
                jobs.append((2, (ex, ey), build_op))
                break

    jobs.sort(key=lambda j: j[0])

    # === 7) HIRING ===
    num_units = 1 + len(me.get("hands", []))
    hires_today = me.get("hires_today", 0)
    next_hire_cost = _fib(hires_today)

    # Hire when workload exceeds workers
    animal_work = total_animals * 3
    plant_work = sum(1 for row in tiles for t in row if is_plant(t))
    if (
        len(jobs) > num_units + 1
        and remaining_days > 3
        and hires_today < 6
        and money >= next_hire_cost + 200
    ):
        market_orders.append(["HIRE"])

    market_orders = market_orders[:10]

    # === 8) ROUTE UNITS ===
    unit_positions = [tuple(me["farmer"])] + [tuple(h) for h in me.get("hands", [])]
    unit_actions = assign_and_act(unit_positions, jobs, tiles, shed)

    return {
        "farmer": unit_actions[0],
        "hands": unit_actions[1:],
        "market": market_orders,
    }