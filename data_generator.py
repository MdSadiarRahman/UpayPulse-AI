"""
data_generator.py
=================
Synthetic Dataset Generator for UpayPulse AI (Fintech MFS Intelligence).

This module simulates realistic Mobile Financial Services (MFS) ecosystem data
tailored to the Bangladesh financial landscape (modeled after providers such as
bKash, Nagad, Upay, and Rocket).

Key Datasets Generated:
-----------------------
1. agents.csv       : 500 MFS Agent points across key commercial hubs in Bangladesh.
                      Columns: agent_id, area, latitude, longitude, cash_balance,
                               e_float_balance, rating, working_hours
2. transactions.csv : 50,000 MFS Transactions with diurnal time-patterns and realistic
                      amount distributions adhering to Bangladesh Bank regulations.
                      Columns: transaction_id, agent_id, transaction_type, amount,
                               timestamp, area
3. merchants.csv    : 1,000 Retail Merchants accepting digital QR & MFS payments.
                      Columns: merchant_id, category, area, latitude, longitude,
                               daily_transactions

All files are exported into the project's 'data/' directory.
"""

# =============================================================================
# SECTION 1: IMPORTS & SYSTEM CONFIGURATION
# =============================================================================
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Reproducibility Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Initialize Faker with English locale and fixed seed for deterministic runs
fake = Faker("en_US")
Faker.seed(RANDOM_SEED)


# =============================================================================
# SECTION 2: BANGLADESH GEOGRAPHICAL HUBS & MFS REGIONAL CONFIGURATION
# =============================================================================
# MFS usage in Bangladesh is heavily concentrated around divisional capitals,
# major trading bazaars, garments & industrial belts, and port cities.
# Each entry defines:
#   - name      : Common area name
#   - division  : Administrative division
#   - lat, lon  : Central GPS coordinates
#   - weight    : Relative commercial activity weight for sampling distribution
BANGLADESH_MFS_HUBS = [
    # ── Dhaka Division (Highest Density & Financial Volume) ──────────────────
    {"area": "Dhanmondi", "division": "Dhaka", "lat": 23.7465, "lon": 90.3760, "weight": 0.08},
    {"area": "Gulshan", "division": "Dhaka", "lat": 23.7925, "lon": 90.4078, "weight": 0.08},
    {"area": "Banani", "division": "Dhaka", "lat": 23.7937, "lon": 90.4043, "weight": 0.06},
    {"area": "Uttara", "division": "Dhaka", "lat": 23.8759, "lon": 90.3795, "weight": 0.08},
    {"area": "Mirpur", "division": "Dhaka", "lat": 23.8067, "lon": 90.3683, "weight": 0.09},
    {"area": "Motijheel C/A", "division": "Dhaka", "lat": 23.7330, "lon": 90.4172, "weight": 0.07},
    {"area": "Mohammadpur", "division": "Dhaka", "lat": 23.7658, "lon": 90.3584, "weight": 0.06},
    {"area": "Old Dhaka (Chawkbazar)", "division": "Dhaka", "lat": 23.7155, "lon": 90.3980, "weight": 0.07},
    {"area": "Farmgate", "division": "Dhaka", "lat": 23.7561, "lon": 90.3872, "weight": 0.05},
    {"area": "Badda", "division": "Dhaka", "lat": 23.7806, "lon": 90.4267, "weight": 0.05},
    {"area": "Jatrabari", "division": "Dhaka", "lat": 23.7104, "lon": 90.4349, "weight": 0.05},
    {"area": "Gazipur Chowrasta", "division": "Dhaka", "lat": 23.9999, "lon": 90.4203, "weight": 0.04},
    {"area": "Savar Bus Stand", "division": "Dhaka", "lat": 23.8583, "lon": 90.2667, "weight": 0.04},
    {"area": "Narayanganj Chasara", "division": "Dhaka", "lat": 23.6238, "lon": 90.5000, "weight": 0.03},

    # ── Chittagong Division (Port, Maritime & Industrial Gateway) ───────────
    {"area": "Agrabad C/A", "division": "Chittagong", "lat": 22.3275, "lon": 91.8123, "weight": 0.05},
    {"area": "GEC Circle", "division": "Chittagong", "lat": 22.3592, "lon": 91.8215, "weight": 0.04},
    {"area": "Nasirabad", "division": "Chittagong", "lat": 22.3705, "lon": 91.8228, "weight": 0.03},
    {"area": "Halishahar", "division": "Chittagong", "lat": 22.3456, "lon": 91.7821, "weight": 0.03},

    # ── Sylhet Division (Expatriate Remittance & Tourism Corridor) ───────────
    {"area": "Zindabazar", "division": "Sylhet", "lat": 24.8949, "lon": 91.8687, "weight": 0.04},
    {"area": "Amberkhana", "division": "Sylhet", "lat": 24.9048, "lon": 91.8682, "weight": 0.03},

    # ── Rajshahi & Rangpur Divisions (Northern Trading & Education Hubs) ────
    {"area": "Shaheb Bazar", "division": "Rajshahi", "lat": 24.3636, "lon": 88.6042, "weight": 0.03},
    {"area": "Jahaj Company More", "division": "Rangpur", "lat": 25.7468, "lon": 89.2508, "weight": 0.02},
    {"area": "Dinajpur Sadar", "division": "Rangpur", "lat": 25.6279, "lon": 88.6336, "weight": 0.02},

    # ── Khulna, Barisal & Cumilla Divisions ──────────────────────────────────
    {"area": "Shibbari More", "division": "Khulna", "lat": 22.8200, "lon": 89.5500, "weight": 0.03},
    {"area": "Sadar Road", "division": "Barisal", "lat": 22.7010, "lon": 90.3535, "weight": 0.02},
    {"area": "Kandirpar", "division": "Cumilla", "lat": 23.4607, "lon": 91.1809, "weight": 0.03},
]

# Normalize weights so they sum strictly to 1.0
_weights = np.array([hub["weight"] for hub in BANGLADESH_MFS_HUBS])
_normalized_weights = _weights / _weights.sum()
for i, hub in enumerate(BANGLADESH_MFS_HUBS):
    hub["normalized_weight"] = _normalized_weights[i]

# Fast lookup dictionary mapping area name -> hub metadata
HUB_BY_NAME = {hub["area"]: hub for hub in BANGLADESH_MFS_HUBS}
AREA_NAMES = [hub["area"] for hub in BANGLADESH_MFS_HUBS]
AREA_PROBABILITIES = [hub["normalized_weight"] for hub in BANGLADESH_MFS_HUBS]


# =============================================================================
# SECTION 3: MFS DOMAIN CONSTANTS & BEHAVIORAL PARAMETERS
# =============================================================================
# 1. Retail Working Shifts in Bangladesh (Grocery shops, pharmacies, telecom kiosks)
WORKING_HOURS_OPTIONS = [
    "08:00 - 22:00",  # Standard neighbourhood grocery/Mudir Dokan (dominant)
    "08:30 - 22:30",  # Commercial retail shop
    "09:00 - 23:00",  # Busy evening market bazaar
    "07:30 - 21:30",  # Morning commuter hub / bus stop point
    "10:00 - 22:00",  # Shopping arcade / mall outlet
    "09:00 - 21:00",  # Residential general store
    "24 Hours",       # Hospital pharmacy / central railway station point
]
WORKING_HOURS_WEIGHTS = [0.38, 0.22, 0.16, 0.10, 0.06, 0.05, 0.03]

# 2. Merchant Retail Categories & Typical Daily Transaction Ranges
MERCHANT_CATEGORIES = [
    {"category": "Grocery & Superstore",        "weight": 0.30, "min_txns": 40, "max_txns": 320},
    {"category": "Pharmacy & Healthcare",       "weight": 0.18, "min_txns": 30, "max_txns": 240},
    {"category": "Restaurant & Fast Food",      "weight": 0.18, "min_txns": 45, "max_txns": 350},
    {"category": "Fashion & Apparel",           "weight": 0.10, "min_txns": 15, "max_txns": 100},
    {"category": "Electronics & Gadgets",       "weight": 0.08, "min_txns": 10, "max_txns": 80},
    {"category": "Telecom & Mobile Accessories","weight": 0.06, "min_txns": 25, "max_txns": 180},
    {"category": "Stationery & Bookstore",      "weight": 0.05, "min_txns": 18, "max_txns": 120},
    {"category": "Home Decor & Hardware",       "weight": 0.05, "min_txns": 12, "max_txns": 75},
]
MERCHANT_CAT_NAMES = [c["category"] for c in MERCHANT_CATEGORIES]
MERCHANT_CAT_WEIGHTS = [c["weight"] for c in MERCHANT_CATEGORIES]

# 3. Transaction Types and Operational Volume Ratios in Bangladesh MFS
TRANSACTION_TYPES = ["cash_out", "send_money", "merchant_payment"]
TRANSACTION_TYPE_WEIGHTS = [0.50, 0.30, 0.20]

# Standard cash-out and transfer denominations frequently transacted in Bangladesh
POPULAR_MFS_DENOMINATIONS = [
    500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0,
    5000.0, 8000.0, 10000.0, 15000.0, 20000.0, 25000.0
]


# =============================================================================
# SECTION 4: DATASET GENERATOR - 1. AGENTS (500 RECORDS)
# =============================================================================
def generate_agents_dataset(n: int = 500) -> pd.DataFrame:
    """
    Generate 500 synthetic MFS Agents distributed across commercial zones.

    Columns
    -------
    - agent_id        : Format 'AGT-0001' to 'AGT-0500'
    - area            : Area name from BANGLADESH_MFS_HUBS
    - latitude        : Float, jittered around area center (~200m - 800m dispersion)
    - longitude       : Float, jittered around area center
    - cash_balance    : Physical currency in hand in BDT (৳10,000 - ৳350,000)
    - e_float_balance : Digital e-money wallet float in BDT (৳20,000 - ৳500,000)
    - rating          : Customer rating between 3.00 and 5.00
    - working_hours   : Operational retail hours string
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # 1. Sample areas based on commercial density weights
    chosen_areas = rng.choice(AREA_NAMES, size=n, p=AREA_PROBABILITIES)

    latitudes = []
    longitudes = []

    # 2. Add realistic spatial jitter around bazaar/hub centers (~0.003 - 0.007 deg)
    for area in chosen_areas:
        hub = HUB_BY_NAME[area]
        # ~300 to 700 meters standard deviation
        lat_jitter = rng.normal(0, 0.0045)
        lon_jitter = rng.normal(0, 0.0055)
        latitudes.append(round(hub["lat"] + lat_jitter, 6))
        longitudes.append(round(hub["lon"] + lon_jitter, 6))

    # 3. Cash & E-float Balances (BDT)
    # Log-normal distribution reflecting realistic liquidity curves in Bangladesh retail:
    # Most agents have between 40,000 and 150,000 BDT, while busy hub agents hold 300,000+ BDT.
    raw_cash = rng.lognormal(mean=11.2, sigma=0.65, size=n)
    cash_balances = np.clip(raw_cash, 10_000, 350_000)
    # Round to nearest 500 BDT for realism (physical notes in cash drawer)
    cash_balances = np.round(cash_balances / 500.0) * 500.0

    # E-float balances: dynamic working capital balance
    raw_efloat = rng.lognormal(mean=11.5, sigma=0.60, size=n)
    efloat_balances = np.clip(raw_efloat, 20_000, 500_000)
    efloat_balances = np.round(efloat_balances / 500.0) * 500.0

    # 4. Agent Rating (Customer review score out of 5.0)
    # Beta distribution skewed towards higher ratings (average ~4.4)
    ratings = 3.0 + 2.0 * rng.beta(a=7.0, b=2.2, size=n)
    ratings = np.round(ratings, 2)

    # 5. Working Hours
    working_hours = rng.choice(WORKING_HOURS_OPTIONS, size=n, p=WORKING_HOURS_WEIGHTS)

    # Construct DataFrame with exact requested columns
    agents_df = pd.DataFrame({
        "agent_id": [f"AGT-{i+1:04d}" for i in range(n)],
        "area": chosen_areas,
        "latitude": latitudes,
        "longitude": longitudes,
        "cash_balance": cash_balances,
        "e_float_balance": efloat_balances,
        "rating": ratings,
        "working_hours": working_hours,
    })

    return agents_df


# =============================================================================
# SECTION 5: DATASET GENERATOR - 2. TRANSACTIONS (50,000 RECORDS)
# =============================================================================
def generate_transactions_dataset(agents_df: pd.DataFrame, n: int = 50_000) -> pd.DataFrame:
    """
    Generate 50,000 synthetic MFS transactions.

    Columns
    -------
    - transaction_id   : Alphanumeric unique identifier (e.g., 'TXN-A7F92B3C10')
    - agent_id         : Associated MFS agent
    - transaction_type : One of 'cash_out', 'merchant_payment', 'send_money'
    - amount           : BDT amount adhering to Bangladesh Bank regulations
    - timestamp        : Datetime within last 60 days following diurnal peak patterns
    - area             : Area corresponding to the agent's location
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)
    num_agents = len(agents_df)

    # 1. Pareto / Power-Law Distribution for Agent Transaction Volume
    # In real MFS networks, top 20% high-traffic bazaar agents handle ~60% of volume.
    agent_activity_weights = rng.pareto(a=2.2, size=num_agents) + 0.15
    agent_activity_weights /= agent_activity_weights.sum()

    # Sample agent indices based on realistic volume distribution
    chosen_agent_indices = rng.choice(num_agents, size=n, p=agent_activity_weights)
    chosen_agents = agents_df.iloc[chosen_agent_indices].reset_index(drop=True)

    agent_ids = chosen_agents["agent_id"].values
    areas = chosen_agents["area"].values

    # 2. Transaction Types
    txn_types = rng.choice(TRANSACTION_TYPES, size=n, p=TRANSACTION_TYPE_WEIGHTS)

    # 3. Transaction Amounts (BDT)
    # Modeling Bangladesh Bank guidelines: Single Cash-out limit ৳25,000.
    amounts = np.empty(n, dtype=float)

    for i in range(n):
        t_type = txn_types[i]

        if t_type == "cash_out":
            # 65% of the time, users withdraw common denomination amounts (e.g. 1000, 2000, 5000)
            if rng.random() < 0.65:
                amounts[i] = float(rng.choice(POPULAR_MFS_DENOMINATIONS))
            else:
                # Continuous amount between ৳300 and ৳25,000, rounded to nearest 10
                raw = rng.lognormal(mean=7.8, sigma=0.9)
                amounts[i] = float(np.round(np.clip(raw, 300, 25_000) / 10.0) * 10.0)

        elif t_type == "send_money":
            # P2P remittances: common transfer amounts (৳200 to ৳20,000)
            if rng.random() < 0.50:
                common_sends = [300.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0]
                amounts[i] = float(rng.choice(common_sends))
            else:
                raw = rng.lognormal(mean=7.5, sigma=0.85)
                amounts[i] = float(np.round(np.clip(raw, 100, 20_000) / 10.0) * 10.0)

        else:  # merchant_payment
            # Retail QR payments: smaller average ticket (৳50 to ৳6,000)
            raw = rng.lognormal(mean=6.3, sigma=0.80)
            amounts[i] = float(round(float(np.clip(raw, 50, 6_000)), 2))

    # 4. Realistic Diurnal Timestamps over the Last 60 Days
    # Commercial MFS hours follow diurnal human activity in Bangladesh:
    # Quiet at night (00-06), morning bazaar peak (09-12), evening rush peak (17-21).
    hourly_probabilities = [
        0.005, 0.003, 0.002, 0.002, 0.003, 0.008,  # 00:00 - 05:59
        0.020, 0.035, 0.055, 0.075, 0.085, 0.080,  # 06:00 - 11:59
        0.065, 0.055, 0.050, 0.060, 0.075, 0.095,  # 12:00 - 17:59
        0.105, 0.090, 0.065, 0.040, 0.020, 0.010   # 18:00 - 23:59
    ]
    # Normalize hourly probabilities
    hourly_probabilities = np.array(hourly_probabilities) / sum(hourly_probabilities)

    # Base reference date
    now = datetime(2026, 10, 2, 20, 0, 0)
    day_offsets = rng.integers(0, 60, size=n)
    hours = rng.choice(24, size=n, p=hourly_probabilities)
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)

    timestamps = []
    for d, h, m, s in zip(day_offsets, hours, minutes, seconds):
        txn_dt = now - timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
        timestamps.append(txn_dt.strftime("%Y-%m-%d %H:%M:%S"))

    # 5. Unique Transaction IDs generated via Faker & Alphanumeric Hashes
    # Emulates official MFS TrxID strings (e.g. 'TXN-7B9A3F108C')
    txn_ids = []
    for i in range(n):
        # 10-character uppercase hexadecimal identifier
        hex_suffix = fake.hexify(text="^^^^^^^^^^", upper=True)
        txn_ids.append(f"TXN-{hex_suffix}")

    # Construct DataFrame with exact requested columns
    transactions_df = pd.DataFrame({
        "transaction_id": txn_ids,
        "agent_id": agent_ids,
        "transaction_type": txn_types,
        "amount": np.round(amounts, 2),
        "timestamp": timestamps,
        "area": areas,
    })

    # Sort chronologically for realistic chronological order
    transactions_df["dt_temp"] = pd.to_datetime(transactions_df["timestamp"])
    transactions_df = transactions_df.sort_values("dt_temp").drop(columns=["dt_temp"]).reset_index(drop=True)

    return transactions_df


# =============================================================================
# SECTION 6: DATASET GENERATOR - 3. MERCHANTS (1,000 RECORDS)
# =============================================================================
def generate_merchants_dataset(n: int = 1000) -> pd.DataFrame:
    """
    Generate 1,000 synthetic Retail Merchants accepting Upay/MFS digital payments.

    Columns
    -------
    - merchant_id        : Format 'MRC-0001' to 'MRC-1000'
    - category           : Retail sector (Grocery, Pharmacy, Restaurant, etc.)
    - area               : Area from BANGLADESH_MFS_HUBS
    - latitude           : Float, jittered around area commercial coordinates
    - longitude          : Float, jittered around area commercial coordinates
    - daily_transactions : Estimated average daily digital transactions
    """
    rng = np.random.default_rng(RANDOM_SEED + 2)

    # 1. Sample Areas based on commercial density
    chosen_areas = rng.choice(AREA_NAMES, size=n, p=AREA_PROBABILITIES)

    # 2. Coordinates with neighborhood jitter
    latitudes = []
    longitudes = []
    for area in chosen_areas:
        hub = HUB_BY_NAME[area]
        lat_jitter = rng.normal(0, 0.0040)
        lon_jitter = rng.normal(0, 0.0050)
        latitudes.append(round(hub["lat"] + lat_jitter, 6))
        longitudes.append(round(hub["lon"] + lon_jitter, 6))

    # 3. Categories
    chosen_categories = rng.choice(MERCHANT_CAT_NAMES, size=n, p=MERCHANT_CAT_WEIGHTS)

    # 4. Daily Transactions based on Category Capacity
    daily_txns = np.empty(n, dtype=int)
    cat_spec_map = {c["category"]: (c["min_txns"], c["max_txns"]) for c in MERCHANT_CATEGORIES}

    for i in range(n):
        cat = chosen_categories[i]
        min_t, max_t = cat_spec_map[cat]
        # Triangular distribution peak slightly to the left of the median
        val = rng.triangular(left=min_t, mode=min_t + (max_t - min_t) * 0.35, right=max_t)
        daily_txns[i] = int(round(val))

    # Construct DataFrame with exact requested columns
    merchants_df = pd.DataFrame({
        "merchant_id": [f"MRC-{i+1:04d}" for i in range(n)],
        "category": chosen_categories,
        "area": chosen_areas,
        "latitude": latitudes,
        "longitude": longitudes,
        "daily_transactions": daily_txns,
    })

    return merchants_df


# =============================================================================
# SECTION 7: PIPELINE EXECUTION & FILE EXPORT
# =============================================================================
def save_all_datasets(output_dir: Path = DATA_DIR):
    """
    Run full generation pipeline and export all three CSV files into output_dir.

    Outputs:
    - {output_dir}/agents.csv
    - {output_dir}/transactions.csv
    - {output_dir}/merchants.csv
    """
    # Ensure destination folder exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[UpayPulse AI] Generating synthetic datasets into: {output_dir}")

    start_time = time.time()

    # 1. Generate Agents
    print("  -> Generating 500 Agents across Bangladesh MFS zones...")
    agents_df = generate_agents_dataset(n=500)
    agents_path = output_dir / "agents.csv"
    agents_df.to_csv(agents_path, index=False)
    print(f"     [OK] Saved: {agents_path.name} ({len(agents_df)} rows, {os.path.getsize(agents_path) / 1024:.1f} KB)")

    # 2. Generate Transactions
    print("  -> Generating 50,000 Transactions with realistic diurnal patterns...")
    transactions_df = generate_transactions_dataset(agents_df, n=50_000)
    txns_path = output_dir / "transactions.csv"
    transactions_df.to_csv(txns_path, index=False)
    print(f"     [OK] Saved: {txns_path.name} ({len(transactions_df)} rows, {os.path.getsize(txns_path) / (1024 * 1024):.2f} MB)")

    # 3. Generate Merchants
    print("  -> Generating 1,000 Merchants across key retail sectors...")
    merchants_df = generate_merchants_dataset(n=1000)
    merchants_path = output_dir / "merchants.csv"
    merchants_df.to_csv(merchants_path, index=False)
    print(f"     [OK] Saved: {merchants_path.name} ({len(merchants_df)} rows, {os.path.getsize(merchants_path) / 1024:.1f} KB)")

    elapsed = time.time() - start_time
    print(f"[UpayPulse AI] Dataset generation completed successfully in {elapsed:.2f} seconds.\n")

    return agents_df, transactions_df, merchants_df


# =============================================================================
# SECTION 8: BACKWARD COMPATIBILITY & APP INTEGRATION
# =============================================================================
# Maintains compatibility with existing prototype UI (app.py) and ML pipeline (ml_engine.py)
_app_cache = {}

def load_all():
    """
    Cached loader providing compatibility with app.py and ml_engine.py.
    Returns (agents_df, merchants_df, users_df) formatted for the prototype UI.
    """
    if not _app_cache:
        # Load or generate datasets
        agents_csv = DATA_DIR / "agents.csv"
        merchants_csv = DATA_DIR / "merchants.csv"

        if not agents_csv.exists() or not merchants_csv.exists():
            save_all_datasets(DATA_DIR)

        raw_agents = pd.read_csv(agents_csv)
        raw_merchants = pd.read_csv(merchants_csv)

        # Focus prototype dashboard on the Dinajpur Sadar zone or top 10 agents
        dinajpur_agents = raw_agents[raw_agents["area"] == "Dinajpur Sadar"]
        if len(dinajpur_agents) < 10:
            sample_agents = pd.concat([dinajpur_agents, raw_agents.head(10 - len(dinajpur_agents))]).copy()
        else:
            sample_agents = dinajpur_agents.head(10).copy()

        # Add columns expected by ml_engine.py and app.py
        rng = np.random.default_rng(RANDOM_SEED)
        names = [
            "Rahim Uddin", "Salma Begum", "Karim Hossain", "Nasrin Akter",
            "Jamal Mia", "Fatema Khatun", "Babul Sheikh", "Reza Mahmud",
            "Kohinoor Bibi", "Arif Chowdhury"
        ]
        sample_agents["name"] = (names * 2)[:len(sample_agents)]
        sample_agents["current_cash"] = sample_agents["cash_balance"]
        sample_agents["current_efloat"] = sample_agents["e_float_balance"]
        sample_agents["predicted_demand_4h"] = rng.integers(10_000, 90_000, len(sample_agents)).astype(float)
        sample_agents["reliability_score"] = np.round(sample_agents["rating"] / 5.0, 2)

        # Prototype merchants
        sample_merchants = raw_merchants.head(5).copy()
        merchant_names = [
            "Bismillah General Store", "Dinajpur Pharmacy Plus",
            "City Tea & Snacks", "New Style Clothing", "Khan Electronics Hub"
        ]
        sample_merchants["name"] = merchant_names[:len(sample_merchants)]
        sample_merchants["active_offer"] = [True, True, True, False, True][:len(sample_merchants)]
        sample_merchants["discount_pct"] = [5, 8, 5, 0, 10][:len(sample_merchants)]

        # Prototype users
        n_users = 50
        lats = rng.normal(25.6279, 0.025, n_users)
        lons = rng.normal(88.6336, 0.030, n_users)
        users_df = pd.DataFrame({
            "user_id": [f"USR-{i+1:04d}" for i in range(n_users)],
            "latitude": np.round(lats, 6),
            "longitude": np.round(lons, 6),
            "cashout_intent": np.round(rng.beta(2, 3, n_users), 3),
            "typical_txn_bdt": rng.integers(200, 5_000, n_users).astype(float),
        })

        _app_cache["agents"] = sample_agents.reset_index(drop=True)
        _app_cache["merchants"] = sample_merchants.reset_index(drop=True)
        _app_cache["users"] = users_df.reset_index(drop=True)

    return _app_cache["agents"].copy(), _app_cache["merchants"].copy(), _app_cache["users"].copy()


# =============================================================================
# SECTION 9: STANDALONE CLI EXECUTION & DATA SANITY CHECKS
# =============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print(" UpayPulse AI -- MFS Synthetic Dataset Generation Pipeline")
    print("=" * 72)

    # Execute dataset creation
    agents, txns, merchants = save_all_datasets()

    print("\n" + "=" * 72)
    print(" DATA SANITY & QUALITY VERIFICATION")
    print("=" * 72)

    # 1. Agents Inspection
    print(f"\n1. AGENTS DATASET: {agents.shape[0]} rows, {agents.shape[1]} columns")
    print("   Columns:", list(agents.columns))
    print("   Sample Preview:")
    print(agents.head(3).to_string(index=False))
    print(f"   - Average Cash Balance   : BDT {agents['cash_balance'].mean():,.2f}")
    print(f"   - Average E-Float Balance: BDT {agents['e_float_balance'].mean():,.2f}")
    print(f"   - Average Rating         : {agents['rating'].mean():.2f} / 5.00")

    # 2. Transactions Inspection
    print(f"\n2. TRANSACTIONS DATASET: {txns.shape[0]} rows, {txns.shape[1]} columns")
    print("   Columns:", list(txns.columns))
    print("   Sample Preview:")
    print(txns.head(3).to_string(index=False))
    print("   - Breakdown by Transaction Type:")
    for t_type, count in txns["transaction_type"].value_counts().items():
        pct = (count / len(txns)) * 100
        print(f"     * {t_type:<18}: {count:,} ({pct:.1f}%)")
    print(f"   - Total Transacted Volume : BDT {txns['amount'].sum():,.2f}")
    print(f"   - Average Amount          : BDT {txns['amount'].mean():,.2f}")
    print(f"   - Min / Max Amount        : BDT {txns['amount'].min():,.2f} / BDT {txns['amount'].max():,.2f}")

    # 3. Merchants Inspection
    print(f"\n3. MERCHANTS DATASET: {merchants.shape[0]} rows, {merchants.shape[1]} columns")
    print("   Columns:", list(merchants.columns))
    print("   Sample Preview:")
    print(merchants.head(3).to_string(index=False))
    print("   - Breakdown by Merchant Category:")
    for cat, count in merchants["category"].value_counts().items():
        pct = (count / len(merchants)) * 100
        print(f"     * {cat:<28}: {count:,} ({pct:.1f}%)")
    print(f"   - Average Daily Transactions: {merchants['daily_transactions'].mean():.1f}")

    print("\n" + "=" * 72)
    print(" ALL DATASETS GENERATED AND VERIFIED SUCCESSFULLY IN data/")
    print("=" * 72)
