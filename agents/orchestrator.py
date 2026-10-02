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
        if any(word in query_lower for word in ["cash", "shortage", "risk", "rebalance", "liquidity", "agent"]):
            return "risk"
            
        # Merchant Growth Intent
        elif any(word in query_lower for word in ["merchant", "increase sales", "business", "growth"]):
            return "merchant"
            
        # Customer Assistance Intent
        elif any(word in query_lower for word in ["offer", "discount", "where", "buy", "customer"]):
            return "customer"
            
        else:
            return "unknown"

    def handle_query(self, query: str) -> str:
        """
        Routes the query to the appropriate agent, executes it, and returns the combined answer.
        """
        intent = self.determine_intent(query)
        
        if intent == "risk":
            # Call Risk Agent with some mock context
            mock_input = {
                "agent_id": "AGT-0005",
                "cash_balance": 35000.0,
                "risk_probability": 0.82,
                "nearby_agents": [{"agent_id": "AGT-0217", "distance_km": 0.65, "surplus": 292500}]
            }
            summary = self.risk_agent.analyze_risk(**mock_input)
            response = self.risk_agent.format_summary(summary)
            return f"Routed to: Risk Agent\n\n{response}"
            
        elif intent == "merchant":
            # Call Merchant Agent with mock context
            strategy = self.merchant_agent.generate_growth_strategy(
                merchant_id="MRC-001",
                sales_history={"avg_ticket_size": 450, "peak_days": "Weekends"},
                customer_activity={"frequent_buyers_age_group": "18-25"},
                local_demand={"trend": "high_cash_out"}
            )
            response = self.merchant_agent.format_strategy(strategy)
            return f"Routed to: Merchant Agent\n\n{response}"
            
        elif intent == "customer":
            # Call Customer Agent with mock context
            mock_merchants = [{"merchant_id": "MRC-001", "name": "KFC", "category": "Restaurant & Fast Food", "area": "Gulshan", "distance_km": 0.8}]
            mock_offers = [{"merchant_id": "MRC-001", "discount": "10%", "benefit_text": "ক্যাশ-আউটের ঝামেলা এড়িয়ে সরাসরি পে করুন আর ১০% ক্যাশব্যাক পান"}]
            
            response = self.customer_agent.get_recommendation(
                has_location_consent=True,
                customer_location_zone="Gulshan",
                customer_category_interest="Restaurant & Fast Food",
                merchant_data=mock_merchants,
                offer_data=mock_offers
            )
            return f"Routed to: Customer Agent\n\n{response}"
            
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
