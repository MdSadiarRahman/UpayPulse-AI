class CustomerAssistanceAgent:
    """
    AI Customer Assistance Agent for UpayPulse AI.
    
    Provides location-aware merchant offer recommendations to customers 
    in a friendly Bengali tone. Prioritizes user privacy by requiring 
    explicit location consent before providing recommendations.
    """

    def __init__(self):
        pass

    def get_recommendation(
        self, 
        has_location_consent: bool,
        customer_location_zone: str,
        customer_category_interest: str,
        merchant_data: list,
        offer_data: list
    ) -> str:
        """
        Generates a friendly Bengali recommendation based on customer preferences 
        and nearby merchant offers.
        
        Args:
            has_location_consent (bool): True if user allowed location tracking.
            customer_location_zone (str): The area the customer is in (e.g. "Gulshan").
            customer_category_interest (str): The preferred category (e.g. "Restaurant & Fast Food").
            merchant_data (list): List of merchant dicts.
            offer_data (list): List of available offers mapping to merchants.
            
        Returns:
            str: The formatted Bengali recommendation text.
        """
        
        # 1. Privacy / Consent Check
        if not has_location_consent:
            return "আপনার লোকেশন পারমিশন বন্ধ আছে। আপনার আশেপাশের সেরা অফারগুলো দেখতে অনুগ্রহ করে লোকেশন অ্যাক্সেস অন করুন।"
            
        # 2. Filter merchants by zone and category
        nearby_merchants = [
            m for m in merchant_data 
            if m.get("area") == customer_location_zone and m.get("category") == customer_category_interest
        ]
        
        if not nearby_merchants:
            return f"দুঃখিত, এই মুহূর্তে {customer_location_zone} এলাকায় {customer_category_interest} এর উপর কোনো বিশেষ অফার নেই। অন্য কোনো ক্যাটাগরি চেক করতে পারেন!"
            
        # 3. Match offers and select the best one
        # For simplicity, we just pick the first matching one
        selected_merchant = nearby_merchants[0]
        merchant_id = selected_merchant.get("merchant_id")
        
        selected_offer = next((o for o in offer_data if o.get("merchant_id") == merchant_id), None)
        
        if not selected_offer:
            return f"আপনার কাছাকাছি {selected_merchant.get('name', 'একটি মার্চেন্ট')} আছে, তবে বর্তমানে সেখানে কোনো ডিসকাউন্ট নেই।"
            
        # 4. Extract details
        merchant_name = selected_merchant.get("name", "মার্চেন্ট")
        distance = selected_merchant.get("distance_km", "কাছাকাছি")
        discount = selected_offer.get("discount", "0%")
        benefit = selected_offer.get("benefit_text", "ক্যাশ-আউটের বদলে পেমেন্ট করে টাকা বাঁচান")

        # 5. Formulate friendly Bengali message
        bangla_msg = (
            f"👋 হ্যালো! আপনি যেহেতু {customer_category_interest} পছন্দ করেন, আপনার জন্য একটি দারুণ অফার আছে!\n\n"
            f"🏪 মার্চেন্ট: {merchant_name}\n"
            f"📍 দূরত্ব: মাত্র {distance} কি.মি. দূরে ({customer_location_zone})\n"
            f"🎁 অফার: {discount} ডিসকাউন্ট\n"
            f"💡 সুবিধা: {benefit}!\n\n"
            f"আজই ভিজিট করুন এবং ডিজিটাল পেমেন্টের মাধ্যমে আপনার সঞ্চয় বাড়ান!"
        )
        
        return bangla_msg

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    # Test the Agent
    customer_agent = CustomerAssistanceAgent()
    
    mock_merchants = [
        {"merchant_id": "MRC-001", "name": "KFC", "category": "Restaurant & Fast Food", "area": "Gulshan", "distance_km": 0.8},
        {"merchant_id": "MRC-002", "name": "Shwapno", "category": "Grocery & Superstore", "area": "Banani", "distance_km": 1.2}
    ]
    
    mock_offers = [
        {"merchant_id": "MRC-001", "discount": "10%", "benefit_text": "ক্যাশ-আউটের ঝামেলা এড়িয়ে সরাসরি পে করুন আর ১০% ক্যাশব্যাক পান"},
        {"merchant_id": "MRC-002", "discount": "5%", "benefit_text": "মুদি বাজারে ৫% ছাড়"}
    ]
    
    print("--- Test 1: No Consent ---")
    print(customer_agent.get_recommendation(
        has_location_consent=False,
        customer_location_zone="Gulshan",
        customer_category_interest="Restaurant & Fast Food",
        merchant_data=mock_merchants,
        offer_data=mock_offers
    ))
    
    print("\n--- Test 2: Valid Match ---")
    print(customer_agent.get_recommendation(
        has_location_consent=True,
        customer_location_zone="Gulshan",
        customer_category_interest="Restaurant & Fast Food",
        merchant_data=mock_merchants,
        offer_data=mock_offers
    ))
