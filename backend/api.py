import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Ensure we can import from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import OrchestratorAgent
from agents.risk_agent import RiskAgent
from agents.merchant_agent import MerchantGrowthAgent

app = FastAPI(
    title="UpayPulse AI Backend API",
    description="API to interact with the UpayPulse Multi-Agent Architecture",
    version="1.0.0"
)

# Initialize the Agents
orchestrator = OrchestratorAgent()
risk_agent = RiskAgent()
merchant_agent = MerchantGrowthAgent()

# --- Response Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    status: str

class AgentRiskResponse(BaseModel):
    agent_id: str
    risk_score: float
    shortage_prediction: float
    explanation: dict

class NearbyAgentRecommendation(BaseModel):
    partner_id: str
    distance_km: float
    available_liquidity: float
    suggested_transfer: float

class NearbyAgentsResponse(BaseModel):
    agent_id: str
    recommended_agents: List[NearbyAgentRecommendation]

class MerchantOfferResponse(BaseModel):
    merchant_id: str
    recommended_offer: str
    timing: str
    expected_impact: str
    reasoning: str

# --- Endpoints ---

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"message": "Welcome to UpayPulse AI Backend API"}

@app.post("/ask", response_model=QueryResponse)
def ask_orchestrator(request: QueryRequest):
    """
    Send a natural language query to the Orchestrator Agent.
    Routes intelligently to Risk, Customer, or Merchant agents.
    """
    try:
        response_text = orchestrator.handle_query(request.query)
        return QueryResponse(response=response_text, status="success")
    except Exception as e:
        return QueryResponse(response=str(e), status="error")

@app.get("/agent-risk/{agent_id}", response_model=AgentRiskResponse)
def get_agent_risk(agent_id: str):
    """
    1. Liquidity prediction model & Risk Agent
    Returns risk score, shortage prediction, and AI explanation.
    """
    # Mocking Database / ML Output Retrieval
    mock_cash = 35000.0
    mock_risk_prob = 0.82
    mock_shortage = 15000.0
    
    # Call Risk Agent to generate explanation
    summary = risk_agent.analyze_risk(
        agent_id=agent_id,
        cash_balance=mock_cash,
        risk_probability=mock_risk_prob,
        nearby_agents=[]
    )
    
    return AgentRiskResponse(
        agent_id=agent_id,
        risk_score=mock_risk_prob,
        shortage_prediction=mock_shortage,
        explanation=summary["analysis"]
    )

@app.get("/nearby-agents/{agent_id}", response_model=NearbyAgentsResponse)
def get_nearby_agents(agent_id: str):
    """
    2. Agent recommendation engine
    Returns recommended nearby agents for rebalancing liquidity.
    """
    # Mocking RebalanceRecommender / Database Retrieval
    # In production, this calls ml_engine.recommend_rebalancing()
    recommendations = [
        NearbyAgentRecommendation(
            partner_id="AGT-0217",
            distance_km=0.65,
            available_liquidity=292500.0,
            suggested_transfer=20000.0
        ),
        NearbyAgentRecommendation(
            partner_id="AGT-0469",
            distance_km=1.20,
            available_liquidity=150000.0,
            suggested_transfer=15000.0
        )
    ]
    
    return NearbyAgentsResponse(
        agent_id=agent_id,
        recommended_agents=recommendations
    )

@app.get("/merchant-offer/{merchant_id}", response_model=MerchantOfferResponse)
def get_merchant_offer(merchant_id: str):
    """
    3. Merchant offer engine & Merchant Growth Agent
    Returns recommended digital payment offer, timing, and impact.
    """
    # Mocking Database Retrieval
    sales_history = {"avg_ticket_size": 450, "peak_days": "Weekends"}
    customer_activity = {"frequent_buyers_age_group": "18-25"}
    local_demand = {"trend": "high_cash_out"}
    
    # Call Merchant Agent to generate strategy
    strategy = merchant_agent.generate_growth_strategy(
        merchant_id=merchant_id,
        sales_history=sales_history,
        customer_activity=customer_activity,
        local_demand=local_demand
    )
    
    return MerchantOfferResponse(
        merchant_id=merchant_id,
        recommended_offer=strategy["Best_Offer"],
        timing=strategy["Best_Timing"],
        expected_impact=strategy["Expected_Impact"],
        reasoning=strategy["Explanation"]
    )

# To run the server locally:
# uvicorn backend.api:app --reload
