import json

class MerchantGrowthAgent:
    """
    AI Merchant Growth Agent for UpayPulse AI.
    
    Helps merchants increase digital payments by analyzing their sales history, 
    customer activity, and local demand to suggest targeted offers.
    """

    def __init__(self):
        pass

    def generate_growth_strategy(
        self, 
        merchant_id: str,
        sales_history: dict,
        customer_activity: dict,
        local_demand: dict
    ) -> dict:
        """
        Analyze merchant data to generate an actionable growth strategy.
        
        Args:
            merchant_id (str): ID of the merchant.
            sales_history (dict): Summary of past sales (e.g., avg_ticket_size, peak_days).
            customer_activity (dict): Summary of customer behavior (e.g., frequent_buyers, demographics).
            local_demand (dict): Current trends in the area (e.g., high_cash_out, festive_season).
            
        Returns:
            dict: Structured recommendation containing offer, timing, target, impact, and reasoning.
        """
        
        # In a real-world scenario, this logic would utilize an ML model or complex rules.
        # Here we use rule-based heuristics to demonstrate the AI layer's logic.
        
        # 1. Determine Target Customer
        target_customer = "Local Students & Young Professionals"
        if customer_activity.get("frequent_buyers_age_group") == "30-50":
            target_customer = "Families & Daily Shoppers"

        # 2. Determine Best Timing
        best_timing = "Weekend Evenings (5 PM - 9 PM)"
        if sales_history.get("peak_days") == "Weekdays":
            best_timing = "Weekday Lunch Hours (1 PM - 3 PM)"
            
        # 3. Determine Best Offer
        best_offer = "10% Cashback on Digital Payments"
        if local_demand.get("trend") == "high_cash_out":
            # If people are cashing out, incentivize keeping money digital
            best_offer = "Pay directly via App and Save BDT 50 on purchases over BDT 500"
            
        # 4. Estimate Expected Impact
        expected_impact = "Increases digital transactions by ~15-20%"
        
        # 5. Explain the Reasoning
        reasoning = (
            f"Based on local demand showing '{local_demand.get('trend', 'normal')}', "
            f"this offer converts potential cash-outs into direct sales. "
            f"Timing aligns with your peak historical sales periods ({sales_history.get('peak_days')}), "
            f"targeting {target_customer} who frequent your store."
        )

        strategy = {
            "merchant_id": merchant_id,
            "Best_Offer": best_offer,
            "Best_Timing": best_timing,
            "Target_Customer": target_customer,
            "Expected_Impact": expected_impact,
            "Explanation": reasoning
        }
        
        return strategy

    def format_strategy(self, strategy: dict) -> str:
        """
        Format the strategy dictionary into a readable text report.
        """
        formatted_text = (
            f"📈 Growth Strategy for Merchant {strategy['merchant_id']}\n"
            f"==================================================\n"
            f"1. Best Offer: {strategy['Best_Offer']}\n"
            f"2. Best Timing: {strategy['Best_Timing']}\n"
            f"3. Target Customer: {strategy['Target_Customer']}\n"
            f"4. Expected Impact: {strategy['Expected_Impact']}\n\n"
            f"💡 Why this recommendation?\n"
            f"{strategy['Explanation']}\n"
            f"==================================================\n"
        )
        return formatted_text

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Test the Agent
    merchant_agent = MerchantGrowthAgent()
    
    mock_sales = {"avg_ticket_size": 450, "peak_days": "Weekends"}
    mock_customer = {"frequent_buyers_age_group": "18-25"}
    mock_demand = {"trend": "high_cash_out"}
    
    strategy = merchant_agent.generate_growth_strategy(
        merchant_id="MRC-001",
        sales_history=mock_sales,
        customer_activity=mock_customer,
        local_demand=mock_demand
    )
    
    print(merchant_agent.format_strategy(strategy))
