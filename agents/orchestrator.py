import sys
# Make sure we can import from the agents folder if this is run directly
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.risk_agent import RiskAgent
from agents.customer_agent import CustomerAssistanceAgent
from agents.merchant_agent import MerchantGrowthAgent

class OrchestratorAgent:
    """
    AI Orchestrator Agent for UpayPulse AI.
    
    Receives user queries and intelligently routes them to the correct 
    specialized agent (Risk, Customer, or Merchant).
    """

    def __init__(self):
        self.risk_agent = RiskAgent()
        self.customer_agent = CustomerAssistanceAgent()
        self.merchant_agent = MerchantGrowthAgent()
        
    def determine_intent(self, query: str) -> str:
        """
        Simple intent detection based on keywords. 
        In a production system, this could be an LLM call or advanced NLP.
        """
        query_lower = query.lower()
        
        # Risk / Liquidity Intent
        if any(word in query_lower for word in ["cash", "shortage", "risk", "rebalance", "liquidity", "agent", "টাকা কম", "টাকা"]):
            return "risk"
            
        # Merchant Growth Intent
        elif any(word in query_lower for word in ["merchant", "increase sales", "business", "growth", "মার্চেন্ট", "বিক্রি"]):
            return "merchant"
            
        # Customer Assistance Intent
        elif any(word in query_lower for word in ["offer", "discount", "where", "buy", "customer", "অফার"]):
            return "customer"
            
        else:
            return "unknown"

    def handle_query(self, query: str, lang: str = "en") -> str:
        """
        Routes the query to the appropriate agent, executes it, and returns the combined answer.
        The lang parameter determines if the response should be in English (en) or Bangla (bn).
        """
        intent = self.determine_intent(query)
        
        if intent == "risk":
            # Mock risk context for example
            agent_id = "Agent A102"
            prob = 82
            if lang == "bn":
                response = f"{agent_id} এর টাকা সংকটের ঝুঁকি বেশি।\nসম্ভাবনা: {prob}%"
            else:
                response = f"{agent_id} has high liquidity shortage risk.\nProbability: {prob}%"
            
            return f"Routed to: Risk Agent\n\n{response}"
            
        elif intent == "merchant":
            # Mock merchant context
            if lang == "bn":
                response = "MRC-001 মার্চেন্টের জন্য সুপারিশ: সপ্তাহান্তে ১০% ডিসকাউন্ট অফার দিন। এতে বিক্রি বাড়বে।"
            else:
                response = "Recommendation for MRC-001: Offer 10% discount on weekends to increase sales."
            
            return f"Routed to: Merchant Agent\n\n{response}"
            
        elif intent == "customer":
            # Mock customer context
            if lang == "bn":
                response = "KFC (গুলশান)-এ একটি অফার আছে। ক্যাশ-আউটের ঝামেলা এড়িয়ে সরাসরি পে করুন আর ১০% ক্যাশব্যাক পান!"
            else:
                response = "There is an offer at KFC (Gulshan). Avoid cash-out hassle, pay directly and get 10% cashback!"
            
            return f"Routed to: Customer Agent\n\n{response}"
            
        else:
            if lang == "bn":
                return "আমি বুঝতে পারিনি। দয়া করে এজেন্ট ঝুঁকি, মার্চেন্ট অফার বা গ্রাহক অফার সম্পর্কে জিজ্ঞাসা করুন।"
            else:
                return "I am sorry, I couldn't understand your request. Please ask about agent cash shortages, merchant sales, or customer offers."


if __name__ == "__main__":
    # Ensure stdout handles unicode/Bengali characters
    sys.stdout.reconfigure(encoding='utf-8')
    
    orchestrator = OrchestratorAgent()
    
    queries = [
        "Which agent needs cash?",
        "Where can I get offer?",
        "How can merchant increase sales?"
    ]
    
    for q in queries:
        print(f"User Question: \"{q}\"")
        print("-" * 40)
        print(orchestrator.handle_query(q))
        print("=" * 60, "\n")
