import pandas as pd
import datetime

def recommend_merchant_offers(merchants_df: pd.DataFrame, transactions_df: pd.DataFrame) -> list:
    """
    Generate rule-based offer recommendations for merchants based on transaction data.
    Returns a list of dicts with recommendations.
    """
    
    # 1. Pre-process transactions to find area-level insights
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
    transactions_df['hour'] = transactions_df['timestamp'].dt.hour
    
    # Area-wise peak hours
    # Group by area and hour, find the most active hour
    area_hourly_activity = transactions_df.groupby(['area', 'hour']).size().reset_index(name='count')
    idx = area_hourly_activity.groupby('area')['count'].idxmax()
    area_peak_hours = area_hourly_activity.loc[idx].set_index('area')['hour'].to_dict()
    
    # Area-wise cash-out vs merchant payment ratio
    area_txn_type = transactions_df.groupby(['area', 'transaction_type']).size().unstack(fill_value=0)
    
    recommendations = []
    
    for _, merchant in merchants_df.iterrows():
        merchant_id = merchant['merchant_id']
        area = merchant['area']
        category = merchant['category']
        daily_txns = merchant['daily_transactions']
        
        # Determine Best Time
        peak_hour = area_peak_hours.get(area, 18)  # default to 6 PM if area not found
        best_time_str = f"{peak_hour}:00 - {peak_hour+2}:00"
        
        # Determine Target Customer Group and Explanation Part 1
        cash_out_vol = area_txn_type.loc[area, 'cash_out'] if area in area_txn_type.index and 'cash_out' in area_txn_type.columns else 0
        merch_pay_vol = area_txn_type.loc[area, 'merchant_payment'] if area in area_txn_type.index and 'merchant_payment' in area_txn_type.columns else 0
        
        if cash_out_vol > merch_pay_vol * 1.5:
            target_group = "High Cash-Out Users (Diversion Target)"
            target_reason = f"the area {area} has high cash-out volume. We want to convert cash-outs to digital payments at {category}."
        else:
            target_group = "Frequent Digital Shoppers (Loyalty Target)"
            target_reason = f"the area {area} already prefers digital payments. This offer builds loyalty for {category}."
            
        # Determine Discount Percentage and Explanation Part 2
        if daily_txns < 100:
            discount = "10%"
            discount_reason = f"this merchant has low daily traffic ({daily_txns} txns), needing a higher incentive to drive footfall."
        elif daily_txns < 250:
            discount = "7%"
            discount_reason = f"this merchant has moderate traffic ({daily_txns} txns). A balanced discount maintains volume."
        else:
            discount = "5%"
            discount_reason = f"this merchant already has high traffic ({daily_txns} txns). A standard 5% discount is enough for retention."
            
        explanation = (
            f"Recommended because {target_reason} "
            f"Additionally, {discount_reason} "
            f"The best time is {best_time_str} since it aligns with the local peak activity hour ({peak_hour}:00) in {area}."
        )
        
        recommendations.append({
            "merchant_id": merchant_id,
            "category": category,
            "area": area,
            "recommended_discount": discount,
            "best_time": best_time_str,
            "target_customer_group": target_group,
            "explanation": explanation
        })
        
    return recommendations

if __name__ == "__main__":
    # Test the recommendation engine
    merchants = pd.read_csv('data/merchants.csv')
    transactions = pd.read_csv('data/transactions.csv')
    
    # Just test on the first 5 merchants for a quick output
    sample_merchants = merchants.head(5)
    recs = recommend_merchant_offers(sample_merchants, transactions)
    
    print("=== MERCHANT OFFER RECOMMENDATIONS ===\n")
    for r in recs:
        print(f"Merchant ID: {r['merchant_id']} ({r['category']}, {r['area']})")
        print(f"Target Group: {r['target_customer_group']}")
        print(f"Discount: {r['recommended_discount']}")
        print(f"Best Time: {r['best_time']}")
        print(f"Explanation: {r['explanation']}")
        print("-" * 50)
