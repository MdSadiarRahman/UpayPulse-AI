import pandas as pd
import os
import random

class MerchantGrowthEngine:
    def __init__(self, merchants_path="data/merchants.csv", txns_path="data/transactions.csv"):
        self.merchants_path = merchants_path
        self.txns_path = txns_path
        
    def generate_recommendation(self, merchant_id):
        if not os.path.exists(self.merchants_path):
            return {"error": f"Data file {self.merchants_path} not found."}
            
        merchants_df = pd.read_csv(self.merchants_path)
        
        if merchant_id not in merchants_df['merchant_id'].values:
            return {"error": f"Merchant {merchant_id} not found."}
            
        merchant = merchants_df[merchants_df['merchant_id'] == merchant_id].iloc[0]
        category = str(merchant['category'])
        daily_txns = merchant['daily_transactions']
        
        # Base recommendations by category
        # NOTE: ৳ (Taka, U+09F3) is encoded as HTML entity &#2547; so it renders
        # correctly when these strings are embedded inside Streamlit HTML templates.
        if "Pharmacy" in category or "Healthcare" in category:
            offer_type = "&#2547;30 discount on &#2547;500 payment"
            best_time = "5 PM - 8 PM"
            target_segment = "Regular Patients &amp; Elderly"
            reason = "Low afternoon transactions. High nearby healthcare activity in the evening."
            impact = f"+{int(daily_txns * 0.15)} transactions"
        elif "Restaurant" in category or "Fast Food" in category:
            offer_type = "10% Cashback on &#2547;300+ bill"
            best_time = "1 PM - 3 PM (Lunch) &amp; 7 PM - 10 PM (Dinner)"
            target_segment = "Students &amp; Young Professionals"
            reason = "Capitalize on peak meal times with aggressive cashback to beat local competition."
            impact = f"+{int(daily_txns * 0.25)} transactions"
        elif "Grocery" in category or "Supermarket" in category:
            offer_type = "5% Cashback on &#2547;1000+ basket"
            best_time = "10 AM - 1 PM"
            target_segment = "Families &amp; Housemakers"
            reason = "Encourage bulk weekly purchases during off-peak morning hours."
            impact = f"+{int(daily_txns * 0.20)} transactions"
        else:
            offer_type = "&#2547;20 Cashback on minimum &#2547;200 spend"
            best_time = "4 PM - 7 PM"
            target_segment = "General Shoppers"
            reason = "Boost overall engagement in average shopping hours."
            impact = f"+{max(5, int(daily_txns * 0.10))} transactions"
            
        return {
            "merchant_id": merchant_id,
            "category": category,
            "recommendation": {
                "offer": offer_type,
                "best_time": best_time,
                "target_segment": target_segment,
                "reason": reason,
                "expected_impact": impact
            }
        }

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    engine = MerchantGrowthEngine()
    print("Testing Merchant Growth Recommendation Engine:\n")
    res = engine.generate_recommendation("MRC-0001")
    if "error" in res:
        print(res["error"])
    else:
        print(f"Merchant: {res['merchant_id']} ({res['category']})\n")
        print("Recommendation:\n")
        print(f"Offer: {res['recommendation']['offer']}")
        print(f"Best Time: {res['recommendation']['best_time']}")
        print(f"Target: {res['recommendation']['target_segment']}")
        print(f"Reason: {res['recommendation']['reason']}")
        print(f"Expected Impact: {res['recommendation']['expected_impact']}")
