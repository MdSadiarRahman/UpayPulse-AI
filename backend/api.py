import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

# Ensure we can import from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import OrchestratorAgent
from src.liquidity_predictor import LiquidityPredictor
from src.agent_recommender import AgentRecommender
from src.merchant_engine import MerchantGrowthEngine

app = FastAPI(
    title="UpayPulse AI Backend API",
    description="API to interact with the UpayPulse Multi-Agent Architecture",
    version="1.0.0"
)

# Initialize the Agents and Engines
orchestrator = OrchestratorAgent()
liquidity_predictor = LiquidityPredictor()
agent_recommender = AgentRecommender()
merchant_engine = MerchantGrowthEngine()

# --- Response Models ---
class ChatRequest(BaseModel):
    query: str
    lang: Optional[str] = "en"

class ChatResponse(BaseModel):
    response: str
    status: str

class AgentRiskResponse(BaseModel):
    agent_id: str
    risk_level: str
    risk_score: float
    shortage_prediction: float
    explanation: dict
    shap_reasons: Optional[List[str]] = []

class NearbyAgentRecommendation(BaseModel):
    partner_id: str
    distance_km: float
    available_liquidity: float
    rating: float
    score: float
    explanation: str

class NearbyAgentsResponse(BaseModel):
    agent_id: str
    recommended_agents: List[NearbyAgentRecommendation]

class MerchantOfferRecommendation(BaseModel):
    offer: str
    best_time: str
    target_segment: str
    reason: str
    expected_impact: str

class MerchantOfferResponse(BaseModel):
    merchant_id: str
    category: str
    recommendation: MerchantOfferRecommendation

# --- Endpoints ---

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"message": "Welcome to UpayPulse AI Backend API"}

@app.post("/chat", response_model=ChatResponse)
def ask_orchestrator(request: ChatRequest):
    """
    Send a natural language query to the AI Assistant.
    Routes intelligently to Risk, Customer, or Merchant agents.
    """
    try:
        response_text = orchestrator.handle_query(request.query, lang=request.lang)
        return ChatResponse(response=response_text, status="success")
    except Exception as e:
        return ChatResponse(response=str(e), status="error")

@app.get("/agent-risk/{agent_id}", response_model=AgentRiskResponse)
def get_agent_risk(agent_id: str):
    """
    Returns risk score, shortage prediction, and SHAP explainability.
    """
    try:
        ml_res = liquidity_predictor.predict_risk(agent_id)
        if "error" in ml_res:
            raise HTTPException(status_code=404, detail=ml_res["error"])
            
        return AgentRiskResponse(
            agent_id=ml_res["agent_id"],
            risk_level=ml_res.get("risk_level", "UNKNOWN"),
            risk_score=ml_res["risk_probability"],
            shortage_prediction=ml_res["expected_shortage_amount"],
            explanation={"analysis": " | ".join(ml_res["main_reasons"])},
            shap_reasons=ml_res.get("shap_reasons", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/liquidity-partners/{agent_id}", response_model=NearbyAgentsResponse)
def get_liquidity_partners(agent_id: str):
    """
    Returns recommended nearby agents for rebalancing liquidity.
    """
    try:
        res = agent_recommender.get_recommendations(agent_id)
        if "error" in res:
            raise HTTPException(status_code=404, detail=res["error"])
            
        recommendations = []
        for r in res["recommendations"]:
            recommendations.append(NearbyAgentRecommendation(
                partner_id=r["agent_id"],
                distance_km=r["distance_km"],
                available_liquidity=r["cash_balance"],
                rating=r["rating"],
                score=r["score"],
                explanation=r["explanation"]
            ))
            
        return NearbyAgentsResponse(
            agent_id=res["target_agent"],
            recommended_agents=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/merchant-recommendation/{merchant_id}", response_model=MerchantOfferResponse)
def get_merchant_offer(merchant_id: str):
    """
    Returns recommended digital payment offer, timing, and impact.
    """
    try:
        res = merchant_engine.generate_recommendation(merchant_id)
        if "error" in res:
            raise HTTPException(status_code=404, detail=res["error"])
            
        return MerchantOfferResponse(
            merchant_id=res["merchant_id"],
            category=res["category"],
            recommendation=MerchantOfferRecommendation(**res["recommendation"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the server locally:
# uvicorn backend.api:app --reload
