"""
Kaggriculture agent  --  hierarchical "macro policy + heuristic planner + action patch".

Architecture (mirrors the structure of the 1st-place write-up, scaled down):

  1. MACRO LEVEL (once per in-game day, hour 0)
       decides: how many hands to hire, how many animals of each type to buy,
       whether to buy land, and which crops to put on the free tiles.
       - default: value-based heuristic (price-curve aware, pipeline aware)
       - optional: small MLP (policy.npz, numpy-only inference) trained with
         behaviour cloning from the heuristic and then self-play PPO (train_rl.py)

  2. PLANNER / EXECUTOR (every turn)
       turns the macro decision into per-unit actions: task generation per tile,
       greedy assignment of tasks to farmer/hands, supply trips to the shed.

  3. ACTION PATCH (every turn)
       rule-based clean-up: caps PLANT demand by seed stock, validates ops against
       the observed state, trims market orders to the per-turn cap, final-day
       liquidation.

Submission: put this file (and optionally policy.npz) in the archive root:
    tar -czf submission.tar.gz main.py policy.npz
"""
import math
import os
import random

# --------------------------------------------------------------------------------------
# Game constants (mirrors kaggle_environments/envs/kaggriculture/kaggriculture.py)
# --------------------------------------------------------------------------------------
BOARD = 10
HALF = 5
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}

CROPS = {
    "WHEAT":      {"seed": 10,  "first": 2,  "maxday": 4,  "ongoing": False, "maxy": 6},
    "CARROT":     {"seed": 20,  "first": 2,  "maxday": 3,  "ongoing": False, "maxy": 4},
    "TOMATO":     {"seed": 50,  "first": 8,  "maxday": 8,  "ongoing": True,  "maxy": 4},
    "STRAWBERRY": {"seed": 100, "first": 10, "maxday": 10, "ongoing": True,  "maxy": 4},
    "MELON":      {"seed": 80,  "first": 10, "maxday": 12, "ongoing": False, "maxy": 6},
}
CROP_NAMES = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = {
    "GOOSE": {"cost": 300, "struct": "COOP",    "first": 4, "interval": 1, "product": "EGG"},
    "COW":   {"cost": 400, "struct": "PASTURE", "first": 8, "interval": 2, "product": "MILK"},
    "SHEEP": {"cost": 500, "struct": "PASTURE", "first": 6, "interval": 3, "product": "WOOL"},
}
ANIMAL_NAMES = ["GOOSE", "COW", "SHEEP"]
# average units/day once producing, assuming FEED + CARE every day (care bonus banks +1/day)
ANIMAL_RATE = {"GOOSE": 2.0, "COW": 1.5, "SHEEP": 4.0 / 3.0}
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

# age at which a one-time crop reaches its (watered, unfertilised) full yield, and that yield
CROP_READY_AGE = {"WHEAT": 4, "CARROT": 3, "MELON": 10}
CROP_UNITS = {"WHEAT": 4, "CARROT": 3, "MELON": 6}
# ages at which ongoing crops produce one unit each (unfertilised)
ONGOING_AGES = {"TOMATO": [8, 9, 10, 11], "STRAWBERRY": [10, 12, 14, 16]}

MP = {
    "WHEAT":      {"base": 25,  "I0": 10000, "T": 400, "bf": "sqrt",   "bt": 0.80, "af": "log",    "at": 0.20},
    "CARROT":     {"base": 35,  "I0": 10000, "T": 450, "bf": "hinge",  "bt": 1.00, "af": "sqrt",   "at": 0.70},
    "TOMATO":     {"base": 60,  "I0": 10000, "T": 200, "bf": "hinge",  "bt": 0.40, "af": "sqrt",   "at": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "bf": "sqrt",   "bt": 0.70, "af": "linear", "at": 1.60},
    "MELON":      {"base": 250, "I0": 10000, "T": 300, "bf": "log",    "bt": 0.20, "af": "sq",     "at": 3.60},
    "EGG":        {"base": 50,  "I0": 10000, "T": 332, "bf": "hinge",  "bt": 0.40, "af": "log",    "at": 0.20},
    "MILK":       {"base": 160, "I0": 10000, "T": 122, "bf": "sqrt",   "bt": 0.60, "af": "linear", "at": 1.60},
    "WOOL":       {"base": 200, "I0": 10000, "T": 105, "bf": "log",    "bt": 0.20, "af": "sq",     "at": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "bf": "linear", "bt": 0.40, "af": "linear", "at": 0.40},
}
_ENV_KEYS = {"below_func": "bf", "below_target": "bt", "above_func": "af", "above_target": "at"}
HINGE_GAIN = 8.0


def _shape(f, x, T=None):
    x = max(0.0, x)
    if f == "linear":
        return x
    if f == "sq":
        return x * x
    if f == "sqrt":
        return math.sqrt(x)
    if f == "log":
        return math.log(1.0 + x)
    if f == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def _norm_params(params):
    """The env may hand us params with env-style keys; normalise to short keys."""
    if not params:
        return MP
    out = {}
    for k, p in params.items():
        q = {}
        for kk, vv in p.items():
            q[_ENV_KEYS.get(kk, kk)] = vv
        out[k] = q
    return out


def price_at(item, inv, params):
    p = params[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inv < I0:
        amp = p["bt"] * base / _shape(p["bf"], T, T)
        pr = base + amp * _shape(p["bf"], I0 - inv, T)
    else:
        amp = p["at"] * base / _shape(p["af"], T, T)
        pr = base - amp * _shape(p["af"], inv - I0, T)
    return max(1.0, pr)


def avg_price(item, inv0, n, params):
    """Average price when selling n units starting at market inventory inv0 (Simpson rule)."""
    a = price_at(item, inv0, params)
    if n <= 0:
        return a
    b = price_at(item, inv0 + n / 2.0, params)
    c = price_at(item, inv0 + n, params)
    return (a + 4.0 * b + c) / 6.0


def quad_of(x, y):
    return ("N" if y < HALF else "S") + ("W" if x < HALF else "E")


def shed_dist(x, y):
    return min(abs(x - sx) + abs(y - sy) for sx, sy in SHED_TILES)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --------------------------------------------------------------------------------------
# Default tunables (heuristic macro).  train_rl.py / evaluate.py can override via Bot(P=...)
# --------------------------------------------------------------------------------------
DEFAULT_P = {
    "tile_rent": 12.0,         # $/tile-day shadow price of land when ranking options
    "score_min": 0.45,         # minimum (net profit)/(cash+rent) to buy an option
    "labor_per_animal_day": 6.0,
    "cash_reserve": 150.0,     # money kept back (feed, hires)
    "animal_cutoff": 7,        # do not buy animals with fewer than this many days left
    "crop_cutoff_pad": 0,      # extra safety days for crops
    "actions_per_unit": 13.0,  # effective actions a unit can do per day (after walking)
    "max_hands": 16,
    "pipe_weight": 0.8,        # how much of the existing pipeline counts against marginal prices
    "land_min_days": 9,
    "land_score": 0.9,
    "melon_cap": 130,          # max melons (pipeline incl. opponent) the heuristic will plan
}


# --------------------------------------------------------------------------------------
# Observation parsing
# --------------------------------------------------------------------------------------
class V:
    """Parsed per-turn view of the observation."""


def _count_farm(farm):
    """Return (animals_by_type, empty_structs, plants list, free, weeds) for a public farm dict."""
    animals = {a: 0 for a in ANIMAL_NAMES}
    empty = {"COOP": [], "PASTURE": []}
    plants = []
    free, weeds = [], []
    tiles = farm["tiles"]
    for y in range(BOARD):
        for x in range(BOARD):
            t = tiles[y][x]
            if t is None:
                free.append((x, y))
            elif t == "LOCKED":
                continue
            else:
                k = t.get("kind")
                if k == "WEED":
                    weeds.append((x, y))
                elif k == "PLANT":
                    plants.append((x, y, t))
                elif k in ("COOP", "PASTURE"):
                    if t.get("animal"):
                        animals[t["animal"]] += 1
                    else:
                        empty[k].append((x, y))
    return animals, empty, plants, free, weeds


def parse_obs(obs, cfg):
    v = V()
    v.p = obs["player"]
    v.day = obs["day"]
    v.hour = obs["hour"]
    v.tpd = int(cfg.get("turnsPerDay", 24)) if cfg else 24
    v.step = obs.get("step", v.day * v.tpd + v.hour) if hasattr(obs, "get") else v.day * v.tpd + v.hour
    v.ep_steps = int(cfg.get("episodeSteps", 720)) if cfg else 720
    v.last_step = v.ep_steps - 2                    # last step on which actions still count
    v.last_day = (v.ep_steps - 1) // v.tpd          # index of final in-game day
    v.rem_steps = v.last_step - v.step
    v.me = obs["farms"][v.p]
    v.opp = obs["farms"][1 - v.p]
    priv = obs["private"]
    v.shed = priv.get("shed", {})
    v.seeds = priv.get("seeds", {})
    v.invs = priv.get("inventories", [{}])
    mk = obs["market"]
    v.prices = mk["prices"]
    v.minv = mk["inventory"]
    v.params = _norm_params(mk.get("params"))
    v.shops = list(obs["town"].get("unlocked_shops", [])) if obs.get("town") else []
    v.money = float(v.me["money"])
    v.tiles = v.me["tiles"]
    v.units = [tuple(v.me["farmer"])] + [tuple(h) for h in v.me["hands"]]
    while len(v.invs) < len(v.units):
        v.invs.append({})
    v.animals, v.empty, v.plants, v.free, v.weeds = _count_farm(v.me)
    v.o_animals, v.o_empty, v.o_plants, v.o_free, v.o_weeds = _count_farm(v.opp)
    v.nquad = len(v.me["unlocked_quadrants"])
    v.days_left = v.last_day - v.day               # days after today
    return v


# --------------------------------------------------------------------------------------
# Economic model used by the macro heuristic
# --------------------------------------------------------------------------------------
def animal_future_units(kind, days_left, placed_age=None):
    """(product units, fertilizer units) an animal bought today yields (harvestable by the end)."""
    a = ANIMALS[kind]
    prod_days = max(0, days_left - a["first"] + 1)
    return prod_days * ANIMAL_RATE[kind], max(0, days_left)


def crop_future(crop, days_left):
    """(units, occupancy_days) of a crop planted today, harvestable by the final day; (0,0) if infeasible."""
    cd = CROPS[crop]
    if not cd["ongoing"]:
        ready = CROP_READY_AGE[crop]
        if days_left < ready:
            return 0.0, 0
        return float(CROP_UNITS[crop]), ready + 1
    ages = [a for a in ONGOING_AGES[crop] if a <= days_left]
    if not ages:
        return 0.0, 0
    return float(len(ages)), ages[-1] + 1


def pipeline(v):
    """Expected future units per product from existing assets of BOTH players."""
    pipe = {p: 0.0 for p in PRODUCTS}
    for farm_animals, plants in ((v.animals, v.plants), (v.o_animals, v.o_plants)):
        for kind, n in farm_animals.items():
            if n:
                u, f = animal_future_units(kind, v.days_left + 1)
                pipe[ANIMALS[kind]["product"]] += n * u * 0.9
                pipe["FERTILIZER"] += n * f * 0.9
        for (_x, _y, t) in plants:
            crop = t["crop"]
            cd = CROPS[crop]
            age = v.day - t["planted_day"]
            if not cd["ongoing"]:
                pipe[crop] += max(t.get("yield_units", 1), CROP_UNITS[crop])
            else:
                rem = [a for a in ONGOING_AGES[crop] if a >= age and a <= age + v.days_left]
                pipe[crop] += t.get("yield_units", 0) + len(rem)
    return pipe


def hands_needed(n_animals, n_plants, n_new, P):
    actions = 5.0 * n_animals + 1.6 * n_plants + 3.0 * n_new
    return int(min(P["max_hands"], max(0, math.ceil(actions / P["actions_per_unit"]) - 1)))


def heuristic_macro(v, P):
    """Greedy value-based allocation of cash and free tiles over animals / crops / land."""
    params = v.params
    days_left = v.days_left
    macro = {"hands": 0, "animals": {a: 0 for a in ANIMAL_NAMES}, "land": 0,
             "crops": {c: 0 for c in CROP_NAMES}}
    pipe = pipeline(v)
    pw = P["pipe_weight"]

    sell_val = 0.0
    for p in PRODUCTS:
        if p == "WHEAT":
            continue
        n = v.shed.get(p, 0)
        if n:
            sell_val += n * v.prices.get(p, 1) * 0.85
    cash = v.money + sell_val - P["cash_reserve"]

    n_animals = sum(v.animals.values())
    empty_slots = len(v.empty["COOP"]) + len(v.empty["PASTURE"])
    slots = len(v.free) + len(v.weeds) + empty_slots
    wheat_price = v.prices.get("WHEAT", 25)
    last_score = 0.0
    total_melon = pipe["MELON"]

    for _ in range(80):
        if slots <= 0 or cash < 10:
            break
        best, best_sc = None, P["score_min"]
        # ---- animals
        if days_left >= P["animal_cutoff"]:
            for kind in ANIMAL_NAMES:
                a = ANIMALS[kind]
                if cash < a["cost"] + wheat_price * 2:
                    continue
                units, fert = animal_future_units(kind, days_left)
                prod = a["product"]
                rev = units * avg_price(prod, v.minv.get(prod, 10000) + pw * pipe[prod], units, params)
                rev += fert * avg_price("FERTILIZER", v.minv.get("FERTILIZER", 10000) + pw * pipe["FERTILIZER"],
                                        fert, params)
                feed = (days_left + 1) * wheat_price
                labor = (days_left + 1) * P["labor_per_animal_day"]
                net = rev - a["cost"] - feed - labor
                denom = a["cost"] + feed + P["tile_rent"] * (days_left + 1)
                sc = net / denom
                if sc > best_sc:
                    best, best_sc = ("A", kind, a["cost"] + feed * 0.15), sc
        # ---- crops
        for crop in CROP_NAMES:
            units, occ = crop_future(crop, days_left - P["crop_cutoff_pad"])
            if units <= 0:
                continue
            seed = CROPS[crop]["seed"]
            if cash < seed:
                continue
            if crop == "MELON" and total_melon >= P["melon_cap"]:
                continue
            rev = units * avg_price(crop, v.minv.get(crop, 10000) + pw * pipe[crop], units, params)
            net = rev - seed
            denom = seed + P["tile_rent"] * (occ + 1)
            sc = net / denom
            if sc > best_sc:
                best, best_sc = ("C", crop, seed), sc
        if best is None:
            break
        kind, name, cost = best
        cash -= cost
        slots -= 1
        last_score = best_sc
        if kind == "A":
            macro["animals"][name] += 1
            units, fert = animal_future_units(name, days_left)
            pipe[ANIMALS[name]["product"]] += units
            pipe["FERTILIZER"] += fert
            n_animals += 1
        else:
            macro["crops"][name] += 1
            units, _ = crop_future(name, days_left)
            pipe[name] += units
            if name == "MELON":
                total_melon += units

    n_new = sum(macro["animals"].values()) + sum(macro["crops"].values())
    n_plants = len(v.plants) + sum(macro["crops"].values())
    macro["hands"] = hands_needed(n_animals, n_plants, n_new, P)

    # land: only when slot-starved and the marginal option is still strong
    nxt = v.nquad - 1
    if (nxt < len(LAND_PRICES) and slots <= 1 and last_score >= P["land_score"]
            and days_left >= P["land_min_days"] and cash >= LAND_PRICES[nxt] + 300):
        macro["land"] = 1
    return macro


# --------------------------------------------------------------------------------------
# Optional learned macro policy (numpy-only inference)
# --------------------------------------------------------------------------------------
HEAD_SIZES = [17, 13, 5, 5, 2, 6, 5]       # hands, goose, cow, sheep, land, crop, crop_frac
CROP_CHOICES = [None, "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
FEAT_DIM = 74


def features(v):
    """Fixed-size feature vector for the macro policy."""
    f = []
    f.append(v.day / 29.0)
    f.append(v.days_left / 29.0)
    f.append(math.log10(max(1.0, v.money)) / 5.0)
    f.append(v.nquad / 4.0)
    f.append(len(v.free) / 100.0)
    f.append(len(v.weeds) / 25.0)
    f.append((len(v.empty["COOP"])) / 25.0)
    f.append((len(v.empty["PASTURE"])) / 25.0)
    for a in ANIMAL_NAMES:
        f.append(v.animals[a] / 40.0)
    cnt = {c: 0 for c in CROP_NAMES}
    for (_x, _y, t) in v.plants:
        cnt[t["crop"]] += 1
    for c in CROP_NAMES:
        f.append(cnt[c] / 40.0)
    for p in PRODUCTS:
        f.append(min(v.shed.get(p, 0), 100) / 100.0)
    for p in PRODUCTS:
        f.append(v.prices.get(p, 1) / MP[p]["base"] / 2.0)
    for p in PRODUCTS:
        T = MP[p]["T"]
        f.append(max(-3.0, min(3.0, (v.minv.get(p, 10000) - 10000) / T)) / 3.0)
    f.append(math.log10(max(1.0, float(v.opp["money"]))) / 5.0)
    f.append((v.money - float(v.opp["money"])) / 20000.0)
    f.append(len(v.opp["unlocked_quadrants"]) / 4.0)
    for a in ANIMAL_NAMES:
        f.append(v.o_animals[a] / 40.0)
    ocnt = {c: 0 for c in CROP_NAMES}
    for (_x, _y, t) in v.o_plants:
        ocnt[t["crop"]] += 1
    for c in CROP_NAMES:
        f.append(ocnt[c] / 40.0)
    f.append(len(v.shops) / 8.0)
    pipe = pipeline(v)
    for p in PRODUCTS:
        f.append(min(pipe[p], 600.0) / 600.0)
    f.append(len(v.units) / 20.0)
    f.append(sum(v.invs[0].values()) / 50.0 if v.invs else 0.0)
    while len(f) < FEAT_DIM:
        f.append(0.0)
    return f[:FEAT_DIM]


class MLPPolicy:
    """tanh MLP trunk -> multi-head categorical logits + value head. numpy only."""

    def __init__(self, W):
        self.W = W

    @staticmethod
    def init(hidden=96, seed=0):
        import numpy as np
        rng = np.random.RandomState(seed)
        tot = sum(HEAD_SIZES)
        def lin(i, o, s=1.0):
            return (rng.randn(i, o) * s / math.sqrt(i)).astype("float64"), np.zeros(o)
        W = {}
        W["w1"], W["b1"] = lin(FEAT_DIM, hidden)
        W["w2"], W["b2"] = lin(hidden, hidden)
        W["wp"], W["bp"] = lin(hidden, tot, 0.01)
        W["wv"], W["bv"] = lin(hidden, 1, 1.0)
        return MLPPolicy(W)

    @staticmethod
    def load(path):
        import numpy as np
        d = np.load(path)
        return MLPPolicy({k: d[k] for k in d.files})

    def save(self, path):
        import numpy as np
        np.savez(path, **self.W)

    def forward(self, x):
        import numpy as np
        W = self.W
        h1 = np.tanh(x @ W["w1"] + W["b1"])
        h2 = np.tanh(h1 @ W["w2"] + W["b2"])
        logits = h2 @ W["wp"] + W["bp"]
        value = float((h2 @ W["wv"] + W["bv"])[0])
        return logits, value, (h1, h2)

    def act(self, feats, explore, rng):
        import numpy as np
        x = np.asarray(feats, dtype="float64")
        logits, value, _ = self.forward(x)
        acts, logp, o = [], 0.0, 0
        for hs in HEAD_SIZES:
            z = logits[o:o + hs]
            z = z - z.max()
            pr = np.exp(z)
            pr /= pr.sum()
            if explore:
                a = int(rng.choice(hs, p=pr))
            else:
                a = int(pr.argmax())
            logp += float(math.log(max(1e-12, pr[a])))
            acts.append(a)
            o += hs
        return acts, logp, value


def macro_from_heads(acts, v):
    hands, goose, cow, sheep, land, crop_i, frac_i = acts
    macro = {"hands": int(hands), "animals": {"GOOSE": int(goose), "COW": int(cow), "SHEEP": int(sheep)},
             "land": int(land), "crops": {c: 0 for c in CROP_NAMES}}
    macro["_crop_choice"] = CROP_CHOICES[crop_i]
    macro["_crop_frac"] = FRACS[frac_i]
    return macro


def heads_from_macro(m, v):
    """Teacher labels (for behaviour cloning) from a heuristic macro dict."""
    hands = min(HEAD_SIZES[0] - 1, m["hands"])
    goose = min(HEAD_SIZES[1] - 1, m["animals"]["GOOSE"])
    cow = min(HEAD_SIZES[2] - 1, m["animals"]["COW"])
    sheep = min(HEAD_SIZES[3] - 1, m["animals"]["SHEEP"])
    land = int(m["land"])
    tot_c = sum(m["crops"].values())
    if tot_c > 0:
        best = max(m["crops"], key=lambda c: m["crops"][c])
        crop_i = CROP_CHOICES.index(best)
    else:
        crop_i = 0
    n_anim = sum(m["animals"].values())
    free_after = max(1, len(v.free) + len(v.weeds) - n_anim)
    ratio = min(1.0, tot_c / float(free_after)) if tot_c else 0.0
    frac_i = min(range(len(FRACS)), key=lambda i: abs(FRACS[i] - ratio))
    return [hands, goose, cow, sheep, land, crop_i, frac_i]


# --------------------------------------------------------------------------------------
# The Bot
# --------------------------------------------------------------------------------------
class Bot:
    def __init__(self, policy=None, explore=False, P=None, seed=None, record=False):
        self.P = dict(DEFAULT_P)
        if P:
            self.P.update(P)
        self.policy = policy
        self.explore = explore
        self.rng = __import__("numpy").random.RandomState(seed) if (policy is not None) else random.Random(seed)
        self.record = record
        self.records = []              # (feats, acts, logp, value, day) when record=True
        self.reset()

    # ---- episode state ---------------------------------------------------------
    def reset(self):
        self.cur_day = -1
        self.macro = None
        self.queue = []
        self.turn_in_day = 0
        self.placed0 = {a: 0 for a in ANIMAL_NAMES}
        self.last_step = -1

    # ---- entry point -----------------------------------------------------------
    def act(self, obs, cfg=None):
        v = parse_obs(obs, cfg)
        if v.step == 0 or v.step < self.last_step:
            self.reset()
        self.last_step = v.step
        if v.day != self.cur_day:
            self._start_day(v)
        self.turn_in_day = v.hour

        farmer, hands, drops = self._plan_units(v)
        orders = self._plan_market(v, drops)
        action = {"farmer": farmer, "hands": hands, "market": orders}
        return self._patch(v, action)

    # ---- macro -----------------------------------------------------------------
    def _start_day(self, v):
        self.cur_day = v.day
        self.placed0 = dict(v.animals)
        if self.policy is not None:
            feats = features(v)
            acts, logp, value = self.policy.act(feats, self.explore, self.rng)
            if self.record:
                self.records.append({"feats": feats, "acts": acts, "logp": logp, "value": value,
                                     "day": v.day, "money": v.money, "opp_money": float(v.opp["money"])})
            macro = macro_from_heads(acts, v)
            macro = self._materialise_crops(v, macro)
        else:
            macro = heuristic_macro(v, self.P)
        macro = self._clip_macro(v, macro)
        if self.record and self.policy is None:
            # teacher labels for behaviour cloning (what the heuristic actually executes)
            self.records.append({"feats": features(v), "acts": heads_from_macro(macro, v), "day": v.day,
                                 "money": v.money, "opp_money": float(v.opp["money"])})
        self.macro = macro
        self.queue = self._build_queue(v, macro)

    def _materialise_crops(self, v, macro):
        """Turn (crop choice, fraction) heads into per-crop tile counts."""
        crop = macro.pop("_crop_choice", None)
        frac = macro.pop("_crop_frac", 0.0)
        if crop:
            n_anim = sum(macro["animals"].values())
            free_after = max(0, len(v.free) + len(v.weeds) - n_anim)
            macro["crops"][crop] = int(round(frac * free_after))
        return macro

    def _clip_macro(self, v, macro):
        """Make a macro decision feasible: cash, free tiles, end-of-season cut-offs."""
        P = self.P
        days_left = v.days_left
        pipe_cash = v.money + 0.85 * sum(v.shed.get(p, 0) * v.prices.get(p, 1)
                                         for p in PRODUCTS if p != "WHEAT")
        cash = pipe_cash - 60.0
        slots = len(v.free) + len(v.weeds) + len(v.empty["COOP"]) + len(v.empty["PASTURE"])
        # land first (it adds slots)
        nxt = v.nquad - 1
        if macro.get("land") and nxt < len(LAND_PRICES) and days_left >= 6 and cash >= LAND_PRICES[nxt]:
            cash -= LAND_PRICES[nxt]
            slots += 25
        else:
            macro["land"] = 0
        # animals
        for kind in ANIMAL_NAMES:
            n = macro["animals"][kind]
            if days_left < 5:
                n = 0
            c = ANIMALS[kind]["cost"] + 40
            n = max(0, min(n, int(cash // c), slots))
            macro["animals"][kind] = n
            cash -= n * c
            slots -= n
        # crops
        for crop in CROP_NAMES:
            n = macro["crops"][crop]
            units, occ = crop_future(crop, days_left)
            if units <= 0:
                n = 0
            seed = CROPS[crop]["seed"]
            n = max(0, min(n, int(cash // seed), slots))
            macro["crops"][crop] = n
            cash -= n * seed
            slots -= n
        # hands
        n_anim = sum(v.animals.values()) + sum(macro["animals"].values())
        need = hands_needed(n_anim, len(v.plants) + sum(macro["crops"].values()),
                            sum(macro["animals"].values()), P)
        if self.policy is None:
            macro["hands"] = min(macro["hands"], P["max_hands"])
        else:
            macro["hands"] = int(max(0, min(macro["hands"], P["max_hands"] + 2)))
        # affordable hire count (fib costs): keep the bill under 6% of cash
        bill, k, a, b = 0.0, 0, 1, 1
        marg_cap = min(150.0, max(30.0, 0.05 * v.money))
        while k < macro["hands"] and a <= marg_cap and bill + a <= max(60.0, 0.4 * v.money):
            bill += a
            a, b = b, a + b
            k += 1
        macro["hands"] = k if v.days_left >= 0 else 0
        if v.day == v.last_day:
            macro["hands"] = min(macro["hands"], max(0, need))
        return macro

    def _build_queue(self, v, macro):
        q = []
        hires = ["HIRE"] * macro["hands"]
        q.append(("H", hires[:5]))
        rest_hires = hires[5:]
        buys = []
        for kind in ANIMAL_NAMES:
            n = macro["animals"][kind]
            if n > 0:
                buys.append(["BUY_ANIMAL", kind, n])
        if macro["land"]:
            buys.insert(0, ["BUY_LAND"])
        for crop in CROP_NAMES:
            n = macro["crops"][crop] - v.seeds.get(crop, 0)
            if n > 0:
                buys.append(["BUY_SEED", crop, n])
        # wheat for feeding (existing + new animals fed from tomorrow; also today for existing)
        n_anim = sum(v.animals.values()) + sum(macro["animals"].values())
        have = v.shed.get("WHEAT", 0) + sum(i.get("WHEAT", 0) for i in v.invs)
        need = n_anim - have
        if v.day >= v.last_day:
            need = 0
        if need > 0:
            buys.append(["BUY_PRODUCT", "WHEAT", int(need)])
        orders = [[h] for h in q[0][1]]
        # (hand orders first, then the rest of hires, then buys)
        orders += [[h] for h in rest_hires]
        orders += buys
        return orders

    # ---- unit planning ---------------------------------------------------------
    def _plan_units(self, v):
        P = self.P
        n_units = len(v.units)
        invs = [dict(i) for i in v.invs[:n_units]]
        tiles = v.tiles
        day = v.day
        final_day = (day >= v.last_day)
        end_phase = v.rem_steps <= 8

        shed_wheat = v.shed.get("WHEAT", 0)
        shed_animals = {a: v.shed.get(a, 0) for a in ANIMAL_NAMES}
        seeds_left = dict(v.seeds)
        # animals waiting to be placed (shed + inventories + still-to-buy today)
        waiting = {a: shed_animals[a] + sum(i.get(a, 0) for i in invs) for a in ANIMAL_NAMES}
        if self.macro and v.hour <= 3:
            for a in ANIMAL_NAMES:
                bought_so_far = waiting[a] + max(0, v.animals[a] - self.placed0.get(a, 0))
                waiting[a] += max(0, self.macro["animals"][a] - bought_so_far)

        # ---- tile tasks
        tasks = {}   # (x,y) -> (op list, priority bonus, need item or None)
        empty_struct = {"COOP": list(v.empty["COOP"]), "PASTURE": list(v.empty["PASTURE"])}
        need_struct = {"COOP": waiting["GOOSE"], "PASTURE": waiting["COW"] + waiting["SHEEP"]}
        free_sorted = sorted(v.free, key=lambda p: (shed_dist(*p), p))
        build_tiles = []
        for st in ("COOP", "PASTURE"):
            deficit = max(0, need_struct[st] - len(empty_struct[st]))
            for _ in range(deficit):
                if free_sorted:
                    build_tiles.append((free_sorted.pop(0), st))
        for (pos, st) in build_tiles:
            tasks[pos] = (["BUILD_COOP" if st == "COOP" else "BUILD_PASTURE"], 8, None)
        # place animals on empty structures
        place_kind = {"COOP": ["GOOSE"], "PASTURE": ["COW", "SHEEP"]}
        avail_place = dict(waiting)
        for st in ("COOP", "PASTURE"):
            for pos in empty_struct[st]:
                for a in place_kind[st]:
                    if avail_place[a] > 0:
                        tasks[pos] = (["PLACE", a], 10, a)
                        avail_place[a] -= 1
                        break
        # weeds
        for pos in v.weeds:
            tasks[pos] = (["DIG"], 2, None)
        # crops on remaining free tiles (farthest first so animals keep the near tiles)
        crop_tiles = sorted(free_sorted, key=lambda p: (-shed_dist(*p), p))
        plantable = [c for c in CROP_NAMES if seeds_left.get(c, 0) > 0 and
                     crop_future(c, v.days_left)[0] > 0 and not final_day]
        for pos in crop_tiles:
            crop = None
            for c in sorted(plantable, key=lambda c: -seeds_left[c]):
                if seeds_left[c] > 0:
                    crop = c
                    break
            if crop is None:
                break
            seeds_left[crop] -= 1
            tasks[pos] = (["PLANT", crop], 5, None)
        # plants / animals on occupied tiles
        for (x, y, t) in v.plants:
            crop = t["crop"]
            cd = CROPS[crop]
            age = day - t["planted_day"]
            yu = t.get("yield_units", 0)
            if cd["ongoing"]:
                ready = yu > 0
            else:
                ready = age >= CROP_READY_AGE.get(crop, cd["maxday"]) or yu >= cd["maxy"] or \
                    (age >= cd["first"] and v.rem_steps <= 30)
            if ready and age >= cd["first"]:
                tasks[(x, y)] = (["HARVEST"], 9, None)
                continue
            if not t.get("watered_today"):
                must = t.get("consecutive_unwatered", 0) >= 1
                win_start = (cd["maxday"] + 1) // 2
                in_win = (not cd["ongoing"]) and win_start <= age <= cd["maxday"]
                if (must or in_win) and not final_day:
                    tasks[(x, y)] = (["WATER"], 11 if must else 6, None)
        for st in ("COOP", "PASTURE"):
            pass
        tiles_animals = []
        for y in range(BOARD):
            for x in range(BOARD):
                t = tiles[y][x]
                if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and t.get("animal"):
                    tiles_animals.append((x, y, t))
        for (x, y, t) in tiles_animals:
            if t.get("yield_units", 0) > 0:
                tasks[(x, y)] = (["HARVEST"], 9, None)
            elif t.get("fertilizer_available"):
                tasks[(x, y)] = (["COLLECT_FERTILIZER"], 7, None)
            elif not final_day and not t.get("fed_today"):
                tasks[(x, y)] = (["FEED"], 12, "WHEAT")
            elif not final_day and not t.get("cared_today"):
                tasks[(x, y)] = (["CARE"], 7, None)
            elif t.get("fertilizer_available"):
                tasks[(x, y)] = (["COLLECT_FERTILIZER"], 7, None)
        # animal feed has priority over harvest on the same tile when unfed (survival)
        for (x, y, t) in tiles_animals:
            if not final_day and not t.get("fed_today") and (x, y) in tasks and tasks[(x, y)][0][0] != "FEED":
                if t.get("consecutive_unfed", 0) >= 1:
                    tasks[(x, y)] = (["FEED"], 14, "WHEAT")

        # ---- assign
        claimed = set()
        actions = [None] * n_units
        drops = []         # list of dicts of items dropped this turn per unit
        feed_demand = sum(1 for p, tk in tasks.items() if tk[2] == "WHEAT")
        wheat_carried = sum(i.get("WHEAT", 0) for i in invs)
        wheat_in_shed_left = shed_wheat
        place_demand = {a: sum(1 for tk in tasks.values() if tk[2] == a) for a in ANIMAL_NAMES}

        order = list(range(n_units))
        # units standing on a tile with a task go first
        order.sort(key=lambda i: 0 if (v.units[i] in tasks) else 1)

        def needs_ok(unit_i, need):
            if need is None:
                return True
            return invs[unit_i].get(need, 0) > 0

        for ui in order:
            pos = v.units[ui]
            inv = invs[ui]
            act = None
            harvest_items = {k: n for k, n in inv.items() if k not in ("WHEAT",) + tuple(ANIMAL_NAMES)}
            # final phase: dump everything
            if end_phase and inv:
                if pos in SHED_TILES:
                    act = ["DROP"]
                else:
                    act = self._step_towards(pos, self._nearest_shed(pos))
            if act is None:
                # on a task tile and able to do it
                if pos in tasks and pos not in claimed and needs_ok(ui, tasks[pos][2]):
                    act = list(tasks[pos][0])
                    claimed.add(pos)
                    if act[0] == "FEED":
                        pass
            if act is None:
                # choose target; tasks that need an item the unit lacks are reachable via a shed trip
                stock = {"WHEAT": wheat_in_shed_left}
                for a_ in ANIMAL_NAMES:
                    stock[a_] = shed_animals[a_]
                sh = self._nearest_shed(pos)
                best, best_sc, best_trip = None, 1e9, None
                for tpos, (op, bonus, need) in tasks.items():
                    if tpos in claimed:
                        continue
                    trip = None
                    if need is None or inv.get(need, 0) > 0:
                        d = manhattan(pos, tpos)
                    else:
                        if stock.get(need, 0) <= 0:
                            continue
                        d = manhattan(pos, sh) + manhattan(sh, tpos)
                        trip = need
                    sc = d - bonus
                    if v.hour >= 17 and op[0] in ("FEED", "WATER", "HARVEST", "PLACE"):
                        sc -= 25
                    if sc < best_sc:
                        best, best_sc, best_trip = tpos, sc, trip
                if best is not None:
                    claimed.add(best)
                    if best_trip is None:
                        act = self._step_towards(pos, best)
                    elif pos in SHED_TILES:
                        item = best_trip
                        total_need = sum(1 for tk in tasks.values() if tk[2] == item)
                        n = max(1, math.ceil(total_need / max(1, n_units))) + (1 if item == "WHEAT" else 0)
                        n = int(min(n, stock[item]))
                        act = ["PICKUP", item, n]
                        if item == "WHEAT":
                            wheat_in_shed_left -= n
                        else:
                            shed_animals[item] -= n
                    else:
                        act = self._step_towards(pos, sh)
            if act is None and harvest_items:
                # nothing else to do: drop the harvest
                if pos in SHED_TILES:
                    keep_w = inv.get("WHEAT", 0) > 0 and feed_demand > 0
                    if not keep_w and not any(inv.get(a, 0) for a in ANIMAL_NAMES):
                        act = ["DROP"]
                    else:
                        it = next(iter(harvest_items))
                        act = ["PLACE", it, int(harvest_items[it])]
                else:
                    act = self._step_towards(pos, self._nearest_shed(pos))
            if act is None:
                act = ["PASS"]
            # heavy inventories get dumped opportunistically when standing at the shed
            if (act[0] not in ("DROP", "PLACE", "PICKUP") and pos in SHED_TILES and
                    sum(harvest_items.values()) >= 10 and not any(inv.get(a, 0) for a in ANIMAL_NAMES)
                    and inv.get("WHEAT", 0) == 0 and act[0] in ("PASS",)):
                act = ["DROP"]
            actions[ui] = act
            if act[0] == "DROP":
                drops.append(dict(inv))
            elif act[0] == "PLACE" and act[1] not in ANIMAL_NAMES and pos in SHED_TILES:
                drops.append({act[1]: int(act[2])})

        return actions[0], actions[1:], drops

    # ---- helpers ---------------------------------------------------------------
    @staticmethod
    def _nearest_shed(pos):
        return min(SHED_TILES, key=lambda s: manhattan(pos, s))

    @staticmethod
    def _step_towards(pos, target):
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        if dx == 0 and dy == 0:
            return ["PASS"]
        if abs(dx) >= abs(dy):
            return ["EAST" if dx > 0 else "WEST"]
        return ["SOUTH" if dy > 0 else "NORTH"]

    # ---- market ----------------------------------------------------------------
    def _plan_market(self, v, drops):
        final = v.day >= v.last_day
        end_phase = v.rem_steps <= 8
        n_anim = sum(v.animals.values())
        if self.macro:
            n_anim += sum(self.macro["animals"].values()) if v.hour <= 3 else 0
        reserve_wheat = 0 if final else n_anim
        incoming = {}
        for d in drops:
            for k, n in d.items():
                incoming[k] = incoming.get(k, 0) + n
        orders = []
        shed_total = sum(v.shed.values())
        for p in PRODUCTS:
            have = v.shed.get(p, 0) + incoming.get(p, 0)
            if p == "WHEAT":
                have -= reserve_wheat
            if have <= 0:
                continue
            price = v.prices.get(p, 1)
            hold = (price <= 2 and not (final or end_phase) and shed_total < 60)
            if hold:
                continue
            orders.append(["SELL", p, int(have)])
        # pop queued buy/hire orders
        slots = 10 - len(orders)
        buys = []
        while self.queue and len(buys) < max(0, slots):
            buys.append(self.queue.pop(0))
        # hires first (cheap, time critical), then sells, then buys
        hires = [o for o in buys if o[0] == "HIRE"]
        rest = [o for o in buys if o[0] != "HIRE"]
        if final or v.day >= v.last_day:
            rest = [o for o in rest if o[0] not in ("BUY_ANIMAL", "BUY_SEED", "BUY_LAND", "BUY_PRODUCT")]
        return hires + orders + rest

    # ---- action patch ------------------------------------------------------------
    def _patch(self, v, action):
        """Rule-based clean-up of the planned action."""
        n_units = len(v.units)
        hands = list(action["hands"])
        # one hands entry per existing hand
        while len(hands) < n_units - 1:
            hands.append(["PASS"])
        hands = hands[: n_units - 1]
        acts = [action["farmer"]] + hands
        # PLANT demand must not exceed seed stock (env drops ALL plants of that crop otherwise)
        demand = {}
        for i, a in enumerate(acts):
            if a and a[0] == "PLANT":
                demand[a[1]] = demand.get(a[1], 0) + 1
        for crop, n in demand.items():
            if n > v.seeds.get(crop, 0):
                keep = v.seeds.get(crop, 0)
                for i, a in enumerate(acts):
                    if a and a[0] == "PLANT" and a[1] == crop:
                        if keep > 0:
                            keep -= 1
                        else:
                            acts[i] = ["PASS"]
        # state validation for single-tile ops
        for i, a in enumerate(acts):
            x, y = v.units[i]
            t = v.tiles[y][x]
            op = a[0]
            if op == "PLANT" and t is not None:
                acts[i] = ["PASS"]
            elif op in ("HARVEST", "WATER", "FERTILIZE") and not isinstance(t, dict):
                acts[i] = ["PASS"]
            elif op == "FEED" and (not isinstance(t, dict) or t.get("fed_today") or v.invs[i].get("WHEAT", 0) <= 0):
                acts[i] = ["PASS"]
        market = [o for o in action["market"] if o][:10]
        return {"farmer": acts[0], "hands": acts[1:], "market": market}


# --------------------------------------------------------------------------------------
# Kaggle entry point
# --------------------------------------------------------------------------------------
_BOT = None


def _load_policy():
    here = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    cands = []
    if here:
        cands.append(os.path.join(here, "policy.npz"))
    cands += ["/kaggle_simulations/agent/policy.npz", "policy.npz"]
    for pth in cands:
        if os.path.exists(pth):
            try:
                return MLPPolicy.load(pth)
            except Exception:
                return None
    return None


def agent(obs, config=None):
    global _BOT
    if _BOT is None:
        _BOT = Bot(policy=_load_policy() if os.environ.get("KAGG_USE_POLICY", "1") == "1" else None)
    try:
        return _BOT.act(obs, config)
    except Exception:
        # never crash the episode: fall back to a harmless action
        return {"farmer": ["PASS"], "hands": [], "market": []}
