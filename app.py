"""
API Integration
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import json
from haystack import Pipeline

from components.action_extractor_component import ActionExtractorComponent
from components.layer_extractor_component import LayerExtractorComponent
from components.location_extractor_component import LocationExtractorComponent
from components.master_agent_component import (
    MasterAgentComponent,
    AgentExecutorComponent,
    ResultAggregatorComponent,
)
from components.text_response_component import TextResponseComponent



app = FastAPI(title="Chat with your map API")
CSV_PATH = "./data/csv datei.csv"
class QueryRequest(BaseModel):
  query: str
  include_reasoning: bool = False
  language: str = "German"

class QueryResponse(BaseModel):
  success: bool
  text_response: str
  data: Dict[str, Any]
  metadata: Dict[str, Any]

def initialize_pipeline():
  """Create and configure the master agent pipeline"""
  pipeline = Pipeline()

  # Create fresh instances of specialized agents
  action_agent = ActionExtractorComponent()
  layer_agent = LayerExtractorComponent(csv_path=CSV_PATH)
  location_agent = LocationExtractorComponent()

  master_agent = MasterAgentComponent()
  agent_executor = AgentExecutorComponent(
    action_agent=action_agent,
    layer_agent=layer_agent,
    location_agent=location_agent
  )

  result_aggregator = ResultAggregatorComponent()
  text_responder = TextResponseComponent()

  # Add components
  pipeline.add_component("master_agent", master_agent)
  pipeline.add_component("agent_executor", agent_executor)
  pipeline.add_component("result_aggregator", result_aggregator)
  pipeline.add_component("text_responder", text_responder)

  # Connect components
  # Connect components
  pipeline.connect("master_agent.execution_plan", "agent_executor.execution_plan")
  pipeline.connect("agent_executor.results", "result_aggregator.results")
  pipeline.connect("master_agent.execution_plan", "result_aggregator.execution_plan")
  pipeline.connect("result_aggregator.final_json", "text_responder.final_json")
  pipeline.connect("agent_executor.execution_log", "text_responder.execution_log")

  return pipeline

master_pipeline = initialize_pipeline()

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
  try:
    result = master_pipeline.run(
      {
      "master_agent": {"query": request.query},
      "result_aggregator": {"raw_query": request.query},
      "text_responder": {"query": request.query}
      },
      include_outputs_from=["master_agent", "agent_executor", "result_aggregator", "text_responder"]
    )

    return QueryResponse(
      success=True,
      text_response=result["text_responder"]["text_response"],
      data=result["result_aggregator"]["final_json"],
      metadata={
        "num_agent_calls": len(result["agent_executor"]["execution_log"]),
        "reasoning": result["master_agent"]["plan_json"]["reasoning"] if request.include_reasoning else None
      }
    )
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def healt_check():
  """Health check endpoint."""
  return {
      "status": "healthy",
      "service": "Chat with your map API"
  }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Chat with your map API",
        "endpoints": {"query": "/api/query", "health": "/api/health", "docs": "/docs"},
    }
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8082)
