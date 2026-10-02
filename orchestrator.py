class RiskAgent:
    def process(self, input_data):
        return f"[Risk Agent] Evaluated liquidity risk for {input_data}"

class MerchantAgent:
    def process(self, input_data):
        return f"[Merchant Agent] Generated offers for {input_data}"

class CustomerAgent:
    def process(self, input_data):
        return f"[Customer Agent] Analyzed behavior for {input_data}"

class MLModels:
    def predict(self, model_type, data):
        return f"[ML Model] Running {model_type} prediction on {data}"

class Database:
    def query(self, query_str):
        return f"[Database] Executing: {query_str}"

class OrchestratorAgent:
    def __init__(self):
        self.risk_agent = RiskAgent()
        self.merchant_agent = MerchantAgent()
        self.customer_agent = CustomerAgent()
        self.ml_models = MLModels()
        self.db = Database()

    def handle_request(self, user_request, target="risk"):
        print(f"User Request: {user_request}")
        
        # 1. Fetch data
        db_data = self.db.query(f"SELECT * FROM {target}_data")
        
        # 2. ML Prediction (Simulated)
        ml_prediction = self.ml_models.predict(target, db_data)
        
        # 3. Route to specific Agent
        if target == "risk":
            response = self.risk_agent.process(ml_prediction)
        elif target == "merchant":
            response = self.merchant_agent.process(ml_prediction)
        elif target == "customer":
            response = self.customer_agent.process(ml_prediction)
        else:
            response = "Unknown request type."
            
        print(f"Final Output: {response}\n")
        return response

if __name__ == "__main__":
    orchestrator = OrchestratorAgent()
    orchestrator.handle_request("Check Agent AGT-005 liquidity risk", target="risk")
    orchestrator.handle_request("Give me offers for Merchant MRC-02", target="merchant")
