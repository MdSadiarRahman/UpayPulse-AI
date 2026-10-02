import datetime
import math
import pandas as pd

# Constants
MIN_CASH_BUFFER = 50000.0  # Safe cash balance threshold before considering an agent as having "surplus"

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 2 * R * math.asin(math.sqrt(a))

def is_agent_working(working_hours_str: str, current_time: datetime.time = None) -> bool:
    """Check if agent is currently working based on their hours string."""
    if pd.isna(working_hours_str):
        return False
    if "24 Hours" in working_hours_str:
        return True
    
    if current_time is None:
        current_time = datetime.datetime.now().time()
        
    try:
        # Expected format: "09:00 - 23:00"
        parts = working_hours_str.split(" - ")
        if len(parts) == 2:
            start_t = datetime.datetime.strptime(parts[0].strip(), "%H:%M").time()
            end_t = datetime.datetime.strptime(parts[1].strip(), "%H:%M").time()
            if start_t <= end_t:
                return start_t <= current_time <= end_t
            else:
                # Night shift crossing midnight
                return current_time >= start_t or current_time <= end_t
    except Exception:
        pass
    return False

def get_liquidity_recommendations(
    target_agent_id: str, 
    agents_df: pd.DataFrame, 
    top_n: int = 5,
    current_time: datetime.time = None
) -> list:
    """
    Recommend nearby agents who can provide liquidity.
    
    Returns a list of dicts with recommendation details.
    """
    if target_agent_id not in agents_df['agent_id'].values:
        raise ValueError(f"Agent {target_agent_id} not found in dataset.")
        
    target_row = agents_df[agents_df['agent_id'] == target_agent_id].iloc[0]
    target_lat = target_row['latitude']
    target_lon = target_row['longitude']
    
    candidates = agents_df[agents_df['agent_id'] != target_agent_id].copy()
    
    # Calculate features for candidates
    candidates['distance_km'] = candidates.apply(
        lambda r: haversine_km(target_lat, target_lon, r['latitude'], r['longitude']), axis=1
    )
    candidates['surplus_cash'] = candidates['cash_balance'] - MIN_CASH_BUFFER
    candidates['is_working'] = candidates['working_hours'].apply(lambda x: is_agent_working(x, current_time))
    
    # Filter valid candidates (must be working, have surplus cash, and be relatively close < 10km)
    valid_candidates = candidates[
        (candidates['is_working'] == True) & 
        (candidates['surplus_cash'] > 0) &
        (candidates['distance_km'] <= 10.0)
    ].copy()
    
    if valid_candidates.empty:
        return []

    # Score calculation
    # Normalize values for fair scoring
    max_surplus = valid_candidates['surplus_cash'].max() or 1
    min_dist = valid_candidates['distance_km'].min()
    max_dist = valid_candidates['distance_km'].max()
    dist_range = (max_dist - min_dist) if max_dist > min_dist else 1

    valid_candidates['surplus_score'] = valid_candidates['surplus_cash'] / max_surplus
    valid_candidates['distance_score'] = 1 - ((valid_candidates['distance_km'] - min_dist) / dist_range)
    valid_candidates['rating_score'] = valid_candidates['rating'] / 5.0
    
    # Weights
    W_DIST = 0.4
    W_SURPLUS = 0.4
    W_RATING = 0.2
    
    valid_candidates['final_score'] = (
        (valid_candidates['distance_score'] * W_DIST) +
        (valid_candidates['surplus_score'] * W_SURPLUS) +
        (valid_candidates['rating_score'] * W_RATING)
    )
    
    # Sort and take top N
    top_candidates = valid_candidates.sort_values(by='final_score', ascending=False).head(top_n)
    
    recommendations = []
    for _, row in top_candidates.iterrows():
        dist = round(row['distance_km'], 2)
        surplus = int(row['surplus_cash'])
        rating = row['rating']
        
        # Reliability textual interpretation
        if rating >= 4.5:
            reliability = "high"
        elif rating >= 3.5:
            reliability = "medium"
        else:
            reliability = "low"
            
        explanation = (
            f"Selected because:\n"
            f"- {dist} km away\n"
            f"- surplus cash {surplus}\n"
            f"- reliability {reliability}"
        )
        
        recommendations.append({
            "agent_id": row['agent_id'],
            "area": row['area'],
            "distance_km": dist,
            "surplus_cash": surplus,
            "rating": rating,
            "working_status": "Active Now",
            "score": round(row['final_score'], 3),
            "explanation": explanation
        })
        
    return recommendations

if __name__ == "__main__":
    # Test the module
    df = pd.read_csv('data/agents.csv')
    
    # Simulate a target agent who has a shortage
    # E.g., AGT-0005 has a low cash balance
    target = "AGT-0005" 
    print(f"Finding liquidity for: {target}")
    
    # Fix time to afternoon for testing to ensure agents are "working"
    test_time = datetime.time(14, 0)
    
    recs = get_liquidity_recommendations(target, df, top_n=5, current_time=test_time)
    
    for i, r in enumerate(recs, 1):
        print(f"\n--- Recommendation {i} ---")
        print(f"Agent ID: {r['agent_id']} ({r['area']})")
        print(r['explanation'])
