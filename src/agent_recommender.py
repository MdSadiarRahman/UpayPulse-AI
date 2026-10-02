import pandas as pd
import numpy as np
from datetime import datetime
import math
import os

class AgentRecommender:
    def __init__(self, data_path="data/agents.csv"):
        self.data_path = data_path
        
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance in km between two lat/lon points."""
        R = 6371.0 # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance
        
    def _parse_availability(self, working_hours, current_time=None):
        """Calculate availability score based on working hours."""
        if current_time is None:
            current_time = datetime.now().time()
            
        working_hours = str(working_hours).strip()
        if "24" in working_hours.lower():
            return 1.0
            
        try:
            parts = working_hours.split('-')
            if len(parts) == 2:
                start_str = parts[0].strip()
                end_str = parts[1].strip()
                start_time = datetime.strptime(start_str, "%H:%M").time()
                end_time = datetime.strptime(end_str, "%H:%M").time()
                
                # Check if current time is within working hours
                if start_time <= current_time <= end_time:
                    return 1.0
                elif start_time > end_time: # Crosses midnight
                    if current_time >= start_time or current_time <= end_time:
                        return 1.0
        except:
            pass
            
        return 0.0 # Closed or unparseable

    def get_recommendations(self, target_agent_id, top_n=5):
        if not os.path.exists(self.data_path):
            return {"error": f"Data file {self.data_path} not found."}
            
        agents_df = pd.read_csv(self.data_path)
        
        if target_agent_id not in agents_df['agent_id'].values:
            return {"error": f"Agent {target_agent_id} not found."}
            
        target_agent = agents_df[agents_df['agent_id'] == target_agent_id].iloc[0]
        target_lat = target_agent['latitude']
        target_lon = target_agent['longitude']
        
        # Filter out the target agent itself
        candidates = agents_df[agents_df['agent_id'] != target_agent_id].copy()
        
        # 1. Calculate Distance
        candidates['distance_km'] = candidates.apply(
            lambda row: self._haversine_distance(target_lat, target_lon, row['latitude'], row['longitude']), 
            axis=1
        )
        
        # Normalize distance score (closer is better)
        # Using 1 / (1 + distance_km) so 0km -> 1.0, 1km -> 0.5, 9km -> 0.1
        candidates['distance_score'] = 1.0 / (1.0 + candidates['distance_km'])
        
        # 2. Surplus score
        # Using cash_balance. Normalize to 0-1
        max_cash = candidates['cash_balance'].max() if not candidates.empty else 1
        candidates['surplus_score'] = candidates['cash_balance'] / max_cash if max_cash > 0 else 0
        
        # 3. Reliability score
        # Rating is out of 5.0
        candidates['reliability_score'] = candidates['rating'] / 5.0
        
        # 4. Operating availability
        current_time = datetime.now().time()
        candidates['availability_score'] = candidates['working_hours'].apply(lambda x: self._parse_availability(x, current_time))
        
        # Calculate Final Partner Score
        candidates['partner_score'] = (
            0.45 * candidates['surplus_score'] + 
            0.30 * candidates['distance_score'] + 
            0.15 * candidates['reliability_score'] + 
            0.10 * candidates['availability_score']
        )
        
        # Sort and get top N
        top_candidates = candidates.sort_values(by='partner_score', ascending=False).head(top_n)
        
        recommendations = []
        for _, row in top_candidates.iterrows():
            # Generate explanation
            reasons = []
            if row['distance_km'] < 2.0:
                reasons.append("Very close location")
            if row['cash_balance'] > 50000:
                reasons.append("High cash surplus available")
            if row['rating'] >= 4.5:
                reasons.append("Highly reliable agent")
            if row['availability_score'] > 0:
                reasons.append("Currently open")
                
            rec = {
                "agent_id": row['agent_id'],
                "distance_km": round(row['distance_km'], 2),
                "cash_balance": row['cash_balance'],
                "rating": row['rating'],
                "score": round(row['partner_score'], 4),
                "explanation": " | ".join(reasons) if reasons else "Good overall partner score."
            }
            recommendations.append(rec)
            
        return {
            "target_agent": target_agent_id,
            "recommendations": recommendations
        }

if __name__ == "__main__":
    recommender = AgentRecommender()
    print("Testing recommendation engine:")
    res = recommender.get_recommendations("AGT-0005", top_n=2)
    print(f"\nAgent {res['target_agent']} needs liquidity support.")
    for i, r in enumerate(res['recommendations'], 1):
        print(f"\nRecommended partner {i}:")
        print(f"Agent {r['agent_id']}")
        print(f"Distance: {r['distance_km']} km")
        print(f"Available cash: ৳{r['cash_balance']:,.0f}")
        print(f"Reliability: {'High' if r['rating'] >= 4.0 else 'Medium'}")
        print(f"Reason: {r['explanation']}")
