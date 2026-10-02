import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure we can import from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import OrchestratorAgent

app = FastAPI(
    title="UpayPulse AI Backend API",
    description="API to interact with the UpayPulse Multi-Agent Architecture",
    version="1.0.0"
)

# Initialize the Orchestrator
orchestrator = OrchestratorAgent()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    status: str

@app.get("/")
def read_root():
    return {"message": "Welcome to UpayPulse AI Backend API"}

@app.post("/ask", response_model=QueryResponse)
def ask_orchestrator(request: QueryRequest):
    """
    Send a natural language query to the Orchestrator Agent.
    The Orchestrator will route it to the correct specialized agent 
    (Risk, Customer, or Merchant) and return the response.
    """
    try:
        response_text = orchestrator.handle_query(request.query)
        return QueryResponse(response=response_text, status="success")
    except Exception as e:
        return QueryResponse(response=str(e), status="error")

# Run using: uvicorn backend.api:app --reload
