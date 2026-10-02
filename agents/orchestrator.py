import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.liquidity_predictor import LiquidityPredictor
from src.merchant_engine import MerchantGrowthEngine
from src.customer_offer import CustomerOfferEngine

class OrchestratorAgent:
    """
    AI Orchestrator Agent for UpayPulse AI.
    
    Receives user queries and intelligently routes them to the correct 
    specialized agent (Risk, Customer, or Merchant) engine.
    """

    def __init__(self):
        self.risk_engine = LiquidityPredictor()
        self.merchant_engine = MerchantGrowthEngine()
        self.customer_engine = CustomerOfferEngine()
        
    def determine_intent(self, query: str) -> str:
        """
        Simple intent detection based on keywords. 
        """
        query_lower = query.lower()
        
        # Risk / Liquidity Intent
        if any(word in query_lower for word in ["cash", "shortage", "risk", "rebalance", "liquidity", "agent", "টাকা কম", "টাকা", "ঝুঁকি"]):
            return "risk"
            
        # Merchant Growth Intent
        elif any(word in query_lower for word in ["merchant", "increase sales", "business", "growth", "মার্চেন্ট", "বিক্রি", "shop", "দোকান"]):
            return "merchant"
            
        # Customer Assistance Intent
        elif any(word in query_lower for word in ["offer", "discount", "where", "buy", "customer", "অফার", "ডিসকাউন্ট"]):
            return "customer"
            
        else:
            return "unknown"

    def handle_query(self, query: str, lang: str = "en") -> str:
        intent = self.determine_intent(query)
        
        if intent == "risk":
            # Call LiquidityPredictor (using AGT-0435 as default for example queries)
            agent_id = "AGT-0435"
            try:
                res = self.risk_engine.predict_risk(agent_id)
                if "error" in res:
                    return res["error"]
                prob = int(res['risk_probability'] * 100)
                
                if lang == "bn":
                    return f"Routed to: Risk Agent\n\n{agent_id} এর liquidity risk বেশি।\n\nRisk:\n{prob}%\n\nRecommendation:\nNearby agent থেকে liquidity support নেওয়া উচিত।"
                else:
                    return f"Routed to: Risk Agent\n\nAgent {agent_id} has high liquidity risk.\n\nRisk:\n{prob}%\n\nRecommendation:\nYou should get liquidity support from a nearby agent."
            except Exception as e:
                return f"Risk Engine Error: {str(e)}"
            
        elif intent == "merchant":
            # Call Merchant Engine (using MRC-0001 as default)
            merchant_id = "MRC-0001"
            try:
                res = self.merchant_engine.generate_recommendation(merchant_id)
                if "error" in res:
                    return res["error"]
                rec = res['recommendation']
                
                if lang == "bn":
                    return f"Routed to: Merchant Agent\n\nMerchant:\n{merchant_id}\n\nRecommendation:\n\nOffer:\n{rec['offer']}\n\nReason:\n{rec['reason']}\n\nExpected:\n{rec['expected_impact']}"
                else:
                    return f"Routed to: Merchant Agent\n\nMerchant:\n{merchant_id}\n\nRecommendation:\n\nOffer:\n{rec['offer']}\n\nReason:\n{rec['reason']}\n\nExpected:\n{rec['expected_impact']}"
            except Exception as e:
                return f"Merchant Engine Error: {str(e)}"
            
        elif intent == "customer":
            # Call Customer Engine (using dummy location data)
            try:
                res = self.customer_engine.generate_offer("CUST-001", 23.79, 90.41, preferred_category="Pharmacy")
                if "error" in res:
                    return res["error"]
                rec = res['recommended_merchant']
                
                if lang == "bn":
                    return f"Routed to: Customer Agent\n\nNearby offer:\n{rec['merchant_id']} ({rec['category']})\n\nDistance:\n{rec['distance']}\n\nOffer:\n{rec['offer']}\n\nReason:\n{rec['reason']}"
                else:
                    return f"Routed to: Customer Agent\n\nNearby offer:\n{rec['merchant_id']} ({rec['category']})\n\nDistance:\n{rec['distance']}\n\nOffer:\n{rec['offer']}\n\nReason:\n{rec['reason']}"
            except Exception as e:
                return f"Customer Engine Error: {str(e)}"
            
        else:
            if lang == "bn":
                return "আমি বুঝতে পারিনি। দয়া করে এজেন্ট ঝুঁকি, মার্চেন্ট অফার বা গ্রাহক অফার সম্পর্কে জিজ্ঞাসা করুন।"
            else:
                return "I am sorry, I couldn't understand your request. Please ask about agent cash shortages, merchant sales, or customer offers."

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    orchestrator = OrchestratorAgent()
    queries = [
        ("Which agent needs cash?", "en"),
        ("How can my shop increase sales?", "en"),
        ("Where can I get discount?", "en"),
        ("কোন এজেন্টের টাকা কম?", "bn")
    ]
    for q, l in queries:
        print(f"User Question: \"{q}\"")
        print("-" * 40)
        print(orchestrator.handle_query(q, lang=l))
        print("=" * 60, "\n")
