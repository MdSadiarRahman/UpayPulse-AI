"""
ml_engine.py
------------
Core AI & Algorithm logic for UpayPulse AI.

Modules
-------
1. ShortageCalculator   - Computes expected cash shortage per agent
2. RiskClassifier       - Labels each agent HIGH / SAFE
3. RebalanceRecommender - Scores and ranks surplus agents as rebalancing partners
4. MerchantMatcher      - Estimates user diversion from cash-out to digital payment
"""

import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Haversine distance (km)
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Return great-circle distance in kilometres between two lat/lon points.
    """
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# 1. SHORTAGE CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
def calculate_shortage(agents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `expected_shortage` column to agents DataFrame.

    Formula
    -------
    expected_shortage = max(0, predicted_demand_4h - current_cash)

    Returns a copy of the DataFrame with added columns:
      - expected_shortage : BDT shortfall amount
      - surplus_cash      : Cash available above predicted demand (0 if shortage)
    """
    df = agents_df.copy()
    df["expected_shortage"] = np.maximum(
        0, df["predicted_demand_4h"] - df["current_cash"]
    )
    df["surplus_cash"] = np.maximum(
        0, df["current_cash"] - df["predicted_demand_4h"]
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. RISK CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
def classify_risk(agents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `risk_level` column: "HIGH" if expected_shortage > 0, else "SAFE".

    Requires `expected_shortage` column (run calculate_shortage first).
    """
    df = agents_df.copy()
    if "expected_shortage" not in df.columns:
        df = calculate_shortage(df)
    df["risk_level"] = df["expected_shortage"].apply(
        lambda x: "HIGH" if x > 0 else "SAFE"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. REBALANCE RECOMMENDER
# ─────────────────────────────────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "surplus"     : 0.45,
    "distance"    : 0.35,
    "reliability" : 0.20,
}

def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Normalize a Series to [0, 1]. Returns 0.5 if all values are equal."""
    rng_ = series.max() - series.min()
    if rng_ == 0:
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / rng_


def recommend_rebalancing(
    agents_df: pd.DataFrame,
    top_k: int = 2
) -> dict:
    """
    For each HIGH-risk agent, find the top_k nearest surplus agents and
    compute a composite rebalancing score.

    Score formula
    -------------
    Score = (0.45 * Surplus_Norm) + (0.35 * DistanceInverse_Norm) + (0.20 * Reliability)

    Parameters
    ----------
    agents_df : DataFrame with shortage, surplus_cash, risk_level columns
    top_k     : Number of recommended partners to return per risky agent

    Returns
    -------
    dict keyed by agent_id of HIGH-risk agents, each value is a list of dicts:
      {
        "partner_id"        : str,
        "partner_name"      : str,
        "distance_km"       : float,
        "score"             : float,
        "suggested_transfer": float  (BDT)
      }
    """
    df = agents_df.copy()
    if "risk_level" not in df.columns:
        df = classify_risk(df)

    high_risk = df[df["risk_level"] == "HIGH"]
    surplus   = df[df["surplus_cash"] > 0]

    recommendations = {}

    for _, risky_agent in high_risk.iterrows():
        if surplus.empty:
            recommendations[risky_agent["agent_id"]] = []
            continue

        candidates = surplus.copy()

        # ── Distance component ──────────────────────────────────────────
        candidates["distance_km"] = candidates.apply(
            lambda r: haversine_km(
                risky_agent["latitude"], risky_agent["longitude"],
                r["latitude"], r["longitude"]
            ),
            axis=1
        )

        # Inverse distance (closer = higher score component)
        candidates["distance_inv"] = 1.0 / (candidates["distance_km"] + 0.001)

        # ── Normalize all three components ──────────────────────────────
        candidates["surplus_norm"]   = _min_max_normalize(candidates["surplus_cash"])
        candidates["dist_norm"]      = _min_max_normalize(candidates["distance_inv"])
        candidates["rel_norm"]       = candidates["reliability_score"]  # already 0-1

        # ── Composite score ─────────────────────────────────────────────
        candidates["score"] = (
            SCORE_WEIGHTS["surplus"]     * candidates["surplus_norm"]   +
            SCORE_WEIGHTS["distance"]    * candidates["dist_norm"]      +
            SCORE_WEIGHTS["reliability"] * candidates["rel_norm"]
        )

        top_partners = candidates.nlargest(top_k, "score")

        # Suggested transfer = min(partner surplus, risky agent shortage)
        shortage = risky_agent["expected_shortage"]
        recs = []
        for _, partner in top_partners.iterrows():
            suggested = round(min(partner["surplus_cash"] * 0.5, shortage), 2)
            recs.append({
                "partner_id"        : partner["agent_id"],
                "partner_name"      : partner["name"],
                "distance_km"       : round(partner["distance_km"], 2),
                "score"             : round(partner["score"], 4),
                "suggested_transfer": suggested,
            })
        recommendations[risky_agent["agent_id"]] = recs

    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# 4. MERCHANT MATCHER  (Cash-out diversion estimator)
# ─────────────────────────────────────────────────────────────────────────────
DIVERSION_DISCOUNT_THRESHOLD = 5   # min discount % to trigger diversion intent
DIVERSION_INTENT_CUTOFF      = 0.4 # users above this threshold are eligible

def estimate_merchant_diversion(
    users_df    : pd.DataFrame,
    merchants_df: pd.DataFrame,
    discount_pct: float = 5.0
) -> dict:
    """
    Estimate how many users can be diverted from cash-out to digital payment
    by offering a minimum discount_pct at nearby merchants.

    Logic
    -----
    1. Filter users with cashout_intent >= DIVERSION_INTENT_CUTOFF (at-risk)
    2. Filter merchants with active_offer == True and discount_pct >= threshold
    3. Eligible users who receive the offer are diverted with probability
       proportional to (discount_pct / 10), capped at 0.85

    Returns
    -------
    dict with:
      total_users          : int
      at_risk_users        : int  (high cashout intent)
      diverted_users       : int  (estimated diversion)
      diversion_rate_pct   : float
      total_cash_saved_bdt : float (approximate BDT not withdrawn)
      active_merchants     : list of merchant names with live offers
    """
    # Users who were planning to cash out
    at_risk = users_df[users_df["cashout_intent"] >= DIVERSION_INTENT_CUTOFF]

    # Active merchants with sufficient discount
    active_m = merchants_df[
        (merchants_df["active_offer"] == True) &
        (merchants_df["discount_pct"] >= DIVERSION_DISCOUNT_THRESHOLD)
    ]

    if active_m.empty or at_risk.empty:
        return {
            "total_users": len(users_df),
            "at_risk_users": len(at_risk),
            "diverted_users": 0,
            "diversion_rate_pct": 0.0,
            "total_cash_saved_bdt": 0.0,
            "active_merchants": [],
        }

    # Diversion probability scales with offered discount
    max_discount = max(active_m["discount_pct"].max(), discount_pct)
    diversion_prob = min(max_discount / 10.0, 0.85)

    diverted_count = int(len(at_risk) * diversion_prob)
    avg_txn        = at_risk["typical_txn_bdt"].mean()
    cash_saved     = round(diverted_count * avg_txn, 2)

    return {
        "total_users"         : len(users_df),
        "at_risk_users"       : len(at_risk),
        "diverted_users"      : diverted_count,
        "diversion_rate_pct"  : round(diverted_count / len(at_risk) * 100, 1),
        "total_cash_saved_bdt": cash_saved,
        "active_merchants"    : active_m["name"].tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run full pipeline and return enriched agents + recommendations
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(agents_df, merchants_df, users_df):
    """
    Execute the complete ML pipeline in one call.

    Returns
    -------
    enriched_agents  : DataFrame with shortage, surplus, risk_level columns
    recommendations  : dict of rebalancing suggestions per high-risk agent
    diversion_stats  : dict with merchant diversion summary
    """
    enriched = calculate_shortage(agents_df)
    enriched = classify_risk(enriched)
    recs     = recommend_rebalancing(enriched)
    div      = estimate_merchant_diversion(users_df, merchants_df)
    return enriched, recs, div


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import load_all
    agents, merchants, users = load_all()
    enriched, recs, div = run_pipeline(agents, merchants, users)

    print("=== ENRICHED AGENTS ===")
    cols = ["agent_id", "name", "current_cash", "predicted_demand_4h",
            "expected_shortage", "surplus_cash", "risk_level"]
    print(enriched[cols].to_string(index=False))

    print("\n=== REBALANCING RECOMMENDATIONS ===")
    for agent_id, partners in recs.items():
        agent_name = enriched.loc[enriched["agent_id"] == agent_id, "name"].values[0]
        print(f"\n  {agent_id} ({agent_name}) needs rebalancing:")
        for p in partners:
            print(f"    -> Partner: {p['partner_name']} | "
                  f"Score: {p['score']} | "
                  f"Distance: {p['distance_km']} km | "
                  f"Suggested Transfer: BDT {p['suggested_transfer']:,.0f}")

    print("\n=== MERCHANT DIVERSION STATS ===")
    for k, v in div.items():
        print(f"  {k}: {v}")
