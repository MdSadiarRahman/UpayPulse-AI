import pandas as pd
import numpy as np
import math
import os

class CustomerOfferEngine:
    def __init__(self, merchants_path="data/merchants.csv"):
        self.merchants_path = merchants_path
        
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance in km between two lat/lon points."""
        R = 6371.0 # Earth radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
        
    def generate_offer(self, customer_id, customer_lat, customer_lon, preferred_category=None):
        if not os.path.exists(self.merchants_path):
            return {"error": f"Data file {self.merchants_path} not found."}
            
        merchants_df = pd.read_csv(self.merchants_path)
        
        # Calculate distances
        merchants_df['distance_km'] = merchants_df.apply(
            lambda row: self._haversine_distance(customer_lat, customer_lon, row['latitude'], row['longitude']), 
            axis=1
        )
        
        # Filter by distance (< 5km)
        nearby = merchants_df[merchants_df['distance_km'] < 5.0].copy()
        
        if nearby.empty:
            return {"error": "No nearby merchants found."}
            
        # Consider category interest (if specified, boost those merchants)
        if preferred_category:
            nearby['category_score'] = nearby['category'].apply(
                lambda x: 1.0 if preferred_category.lower() in str(x).lower() else 0.2
            )
        else:
            nearby['category_score'] = 0.5
            
        # Distance score (closer is better)
        nearby['distance_score'] = 1.0 / (1.0 + nearby['distance_km'])
        
        # Total score (simulating customer behavior and offer fatigue avoidance by boosting relevant category)
        nearby['score'] = (0.6 * nearby['category_score']) + (0.4 * nearby['distance_score'])
        
        # Sort and pick the best one
        best_merchant = nearby.sort_values(by='score', ascending=False).iloc[0]
        category = str(best_merchant['category'])
        distance_m = int(best_merchant['distance_km'] * 1000)
        
        # Offer logic based on category and spending patterns
        if "Pharmacy" in category or "Healthcare" in category:
            offer_type = "৳25 discount on medicine purchase"
            reason = "Customer frequently purchases medicine."
            expected_benefit = "Improved customer retention & health engagement."
        elif "Restaurant" in category or "Fast Food" in category:
            offer_type = "Free delivery or 10% off on orders over ৳300"
            reason = "Customer orders food during peak hours."
            expected_benefit = "Higher order volume & cart value."
        elif "Grocery" in category or "Supermarket" in category:
            offer_type = "10% Cashback on weekly groceries"
            reason = "Customer has consistent weekly grocery spend pattern."
            expected_benefit = "Increased basket size & loyal spending."
        else:
            offer_type = "5% Cashback on next transaction"
            reason = "General engagement strategy based on proximity."
            expected_benefit = "Increased general platform usage & discovery."
            
        return {
            "customer_id": customer_id,
            "recommended_merchant": {
                "merchant_id": best_merchant['merchant_id'],
                "category": category,
                "distance": f"{distance_m} meter",
                "offer": offer_type,
                "reason": reason,
                "expected_benefit": expected_benefit
            }
        }

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    engine = CustomerOfferEngine()
    print("Testing Customer Offer Recommendation Engine:\n")
    # Using a dummy location near Gulshan
    res = engine.generate_offer("CUST-9921", 23.79, 90.41, preferred_category="Pharmacy")
    if "error" in res:
        print(res["error"])
    else:
        rec = res['recommended_merchant']
        print(f"Nearby offer:\n{rec['merchant_id']} ({rec['category']})\n")
        print(f"Distance:\n{rec['distance']}\n")
        print(f"Offer:\n{rec['offer']}\n")
        print(f"Reason:\n{rec['reason']}\n")
        print(f"Expected Benefit:\n{rec['expected_benefit']}")
