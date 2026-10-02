import datetime

class RiskAgent:
    """
    AI Risk Agent for UpayPulse AI.
    
    Acts as an intelligent layer on top of the ML predictions.
    Reads liquidity prediction outputs and formulates human-readable 
    explanations, reasoning, and recommended actions without making 
    autonomous financial decisions.
    """

    def __init__(self):
        # We can store thresholds or config here
        self.high_risk_threshold = 0.70
        self.critical_balance_threshold = 50000

    def analyze_risk(
        self, 
        agent_id: str, 
        cash_balance: float, 
        risk_probability: float, 
        nearby_agents: list
    ) -> dict:
        """
        Analyze the agent's risk profile based on inputs and return a structured summary.
        """
        
        # 1. Determine Problem Statement
        if risk_probability > self.high_risk_threshold:
            problem = f"Agent {agent_id} is highly likely to face a cash shortage in the next 4 hours (Risk: {risk_probability*100:.1f}%)."
        else:
            problem = f"Agent {agent_id} has stable liquidity at the moment (Risk: {risk_probability*100:.1f}%)."

        # 2. Formulate Reasons
        reasons = []
        if cash_balance < self.critical_balance_threshold:
            reasons.append(f"Cash balance is critically low at BDT {cash_balance:,.0f}.")
        
        day_of_week = datetime.datetime.now().strftime("%A")
        reasons.append(f"Historical demand patterns for {day_of_week} indicate increased cash-out behavior.")
        
        if not reasons:
            reasons.append("Demand is expected to remain within current balance capacity.")

        # 3. Recommend Actions (Human-in-the-loop)
        actions = []
        if risk_probability > self.high_risk_threshold:
            # Check for nearby agents to suggest rebalancing
            if nearby_agents:
                top_nearby = nearby_agents[0]
                suggested_transfer = 20000 # Default suggestion, could be calculated dynamically
                actions.append(f"Rebalance BDT {suggested_transfer:,.0f} from Agent {top_nearby['agent_id']} ({top_nearby['distance_km']} km away).")
            else:
                actions.append("No nearby surplus agents available. Initiate bank withdrawal.")
            
            actions.append("Notify operator regarding critical shortage risk.")
            actions.append("Show nearby merchant offers to customers to divert cash-out into digital payment.")
        else:
            actions.append("No immediate action required. Continue monitoring.")

        # Construct final output dictionary
        summary = {
            "agent_id": agent_id,
            "analysis": {
                "Problem": problem,
                "Reason": reasons,
                "Recommended_Action": actions
            }
        }
        
        return summary

    def format_summary(self, summary_dict: dict) -> str:
        """
        Format the summary dictionary into a clean string for reporting.
        """
        analysis = summary_dict["analysis"]
        
        reasons_str = "\n".join([f"- {r}" for r in analysis["Reason"]])
        actions_str = "\n".join([f"{i+1}. {a}" for i, a in enumerate(analysis["Recommended_Action"])])
        
        formatted_text = (
            f"Risk Summary for {summary_dict['agent_id']}:\n"
            f"=========================================\n"
            f"Analysis:\n"
            f"{analysis['Problem']}\n\n"
            f"Reason:\n"
            f"{reasons_str}\n\n"
            f"Recommended Action:\n"
            f"{actions_str}\n"
            f"=========================================\n"
            f"Note: This is a recommendation only. Please confirm before executing any financial transfers."
        )
        return formatted_text


if __name__ == "__main__":
    # Test the Risk Agent
    risk_agent = RiskAgent()
    
    # Mock ML Output data
    mock_input = {
        "agent_id": "AGT-0005",
        "cash_balance": 35000.0,
        "risk_probability": 0.82,
        "nearby_agents": [
            {"agent_id": "AGT-0217", "distance_km": 0.65, "surplus": 292500},
            {"agent_id": "AGT-0469", "distance_km": 0.52, "surplus": 218500}
        ]
    }
    
    # Process
    summary_dict = risk_agent.analyze_risk(**mock_input)
    
    # Output Result
    print(risk_agent.format_summary(summary_dict))
