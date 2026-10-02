"""
data_generator.py
-----------------
Generates realistic synthetic data for the UpayPulse AI prototype.
No real personal data is used. All data is seeded for reproducibility.

Entities:
  - 10 MFS Agents   : Dinajpur Sadar zone with cash/e-float state
  - 5  Merchants    : Local businesses offering digital-payment discounts
  - 50 Users        : With cash-out intent scores
"""

import numpy as np
import pandas as pd

# ── Seed for reproducibility ──────────────────────────────────────────────────
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ── Geographic bounding box: Dinajpur Sadar, Bangladesh ───────────────────────
LAT_CENTER, LON_CENTER = 25.6279, 88.6336
LAT_SPREAD, LON_SPREAD = 0.025, 0.030   # ~3 km radius jitter


def _jitter_coords(n, lat_c, lon_c, lat_s, lon_s):
    """Return n (lat, lon) pairs scattered around a centre point."""
    lats = rng.normal(lat_c, lat_s, n)
    lons = rng.normal(lon_c, lon_s, n)
    return lats, lons


# ─────────────────────────────────────────────────────────────────────────────
# 1. AGENT DATA
# ─────────────────────────────────────────────────────────────────────────────
AGENT_NAMES = [
    "Rahim Uddin", "Salma Begum", "Karim Hossain", "Nasrin Akter",
    "Jamal Mia", "Fatema Khatun", "Babul Sheikh", "Reza Mahmud",
    "Kohinoor Bibi", "Arif Chowdhury"
]

def generate_agents():
    """
    Generate 10 synthetic MFS agents in Dinajpur Sadar.

    Columns
    -------
    agent_id           : Unique identifier
    name               : Agent display name
    latitude / longitude
    current_cash       : BDT cash in hand  (5k - 80k)
    current_efloat     : BDT e-float balance (5k - 80k)
    predicted_demand_4h: Estimated cash-out demand next 4 h (10k - 90k)
    reliability_score  : Historical reliability  (0.50 - 1.00)
    """
    n = 10
    lats, lons = _jitter_coords(n, LAT_CENTER, LON_CENTER, LAT_SPREAD, LON_SPREAD)

    current_cash        = rng.integers(5_000,  80_000, n).astype(float)
    current_efloat      = rng.integers(5_000,  80_000, n).astype(float)
    predicted_demand_4h = rng.integers(10_000, 90_000, n).astype(float)
    reliability_score   = np.round(rng.uniform(0.50, 1.00, n), 2)

    df = pd.DataFrame({
        "agent_id"           : [f"AGT-{i+1:03d}" for i in range(n)],
        "name"               : AGENT_NAMES,
        "latitude"           : np.round(lats, 6),
        "longitude"          : np.round(lons, 6),
        "current_cash"       : current_cash,
        "current_efloat"     : current_efloat,
        "predicted_demand_4h": predicted_demand_4h,
        "reliability_score"  : reliability_score,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. MERCHANT DATA
# ─────────────────────────────────────────────────────────────────────────────
MERCHANT_DATA = [
    ("MRC-001", "Bismillah General Store",  "Grocery",     True,  5),
    ("MRC-002", "Dinajpur Pharmacy Plus",   "Pharmacy",    True,  8),
    ("MRC-003", "City Tea & Snacks",        "Food",        True,  5),
    ("MRC-004", "New Style Clothing",       "Apparel",     False, 0),
    ("MRC-005", "Khan Electronics Hub",     "Electronics", True, 10),
]

def generate_merchants():
    """
    Generate 5 local merchants with optional instant digital-payment offers.

    Columns
    -------
    merchant_id   : Unique identifier
    name          : Business name
    category      : Business category
    latitude / longitude
    active_offer  : Whether a discount offer is currently live
    discount_pct  : Discount percentage for digital payment (0 if no offer)
    """
    n = len(MERCHANT_DATA)
    lats, lons = _jitter_coords(n, LAT_CENTER, LON_CENTER, LAT_SPREAD * 0.8, LON_SPREAD * 0.8)

    records = []
    for i, (mid, name, cat, offer, disc) in enumerate(MERCHANT_DATA):
        records.append({
            "merchant_id" : mid,
            "name"        : name,
            "category"    : cat,
            "latitude"    : round(float(lats[i]), 6),
            "longitude"   : round(float(lons[i]), 6),
            "active_offer": offer,
            "discount_pct": disc,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 3. USER DATA
# ─────────────────────────────────────────────────────────────────────────────
def generate_users():
    """
    Generate 50 synthetic MFS users located in Dinajpur Sadar.

    Columns
    -------
    user_id          : Unique identifier
    latitude / longitude
    cashout_intent   : Probability (0-1) the user wants to withdraw cash
    typical_txn_bdt  : Typical transaction amount in BDT (200 - 5000)
    """
    n = 50
    lats, lons = _jitter_coords(n, LAT_CENTER, LON_CENTER, LAT_SPREAD * 1.2, LON_SPREAD * 1.2)

    df = pd.DataFrame({
        "user_id"        : [f"USR-{i+1:04d}" for i in range(n)],
        "latitude"       : np.round(lats, 6),
        "longitude"      : np.round(lons, 6),
        "cashout_intent" : np.round(rng.beta(2, 3, n), 3),   # realistic skewed dist
        "typical_txn_bdt": rng.integers(200, 5_000, n).astype(float),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. PUBLIC LOADER  (cached singleton used by app.py & ml_engine.py)
# ─────────────────────────────────────────────────────────────────────────────
_cache = {}

def load_all():
    """
    Return (agents_df, merchants_df, users_df).
    Data is generated once and cached in-process.
    """
    if not _cache:
        _cache["agents"]    = generate_agents()
        _cache["merchants"] = generate_merchants()
        _cache["users"]     = generate_users()
    return _cache["agents"].copy(), _cache["merchants"].copy(), _cache["users"].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agents, merchants, users = load_all()
    print("=== AGENTS ===")
    print(agents.to_string(index=False))
    print("\n=== MERCHANTS ===")
    print(merchants.to_string(index=False))
    print("\n=== USERS (first 5) ===")
    print(users.head().to_string(index=False))
    print(f"\nTotal users: {len(users)}")
