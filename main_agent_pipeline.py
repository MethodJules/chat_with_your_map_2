"""
Main script for the Master Agent Pipeline
THIS IS A STANDALONE TESTING SCRIPT - DO NOT IMPORT FROM THIS FILE
"""

import json
from haystack import Pipeline

# Import your existing components
from components.action_extractor_component import ActionExtractorComponent
from components.layer_extractor_component import LayerExtractorComponent
from components.location_extractor_component import LocationExtractorComponent

# Import the new master agent components
from components.master_agent_component import (
    MasterAgentComponent,
    AgentExecutorComponent,
    ResultAggregatorComponent,
)
from components.text_response_component import TextResponseComponent

# --- Configuration ---

# USER_QUERY = "Zeige mir alle Parkplätze und Bahnhalt sowie Fahrradsparkplätze Stellen in Altona und auch in Wandsbek."
USER_QUERY = "Zeige mir alle Fahrradstationen in Altona"
CSV_PATH = "./data/csv datei.csv"

# --- Initialize Components ---

print("Initializing the master agent system...")

# Create instances of the specialized agents (your existing components)
action_agent = ActionExtractorComponent()
layer_agent = LayerExtractorComponent(csv_path=CSV_PATH)
location_agent = LocationExtractorComponent()

# Create the master agent and execution components
master_agent = MasterAgentComponent()  # or "gpt-4o" for better reasoning
agent_executor = AgentExecutorComponent(
    action_agent=action_agent, layer_agent=layer_agent, location_agent=location_agent
)
result_aggregator = ResultAggregatorComponent()
text_responder = TextResponseComponent()

# ---- Build Pipeline ----
pipeline = Pipeline()

# Add components
pipeline.add_component("master_agent", master_agent)
pipeline.add_component("agent_executor", agent_executor)
pipeline.add_component("result_aggregator", result_aggregator)
pipeline.add_component("text_responder", text_responder)

# Connect components
pipeline.connect("master_agent.execution_plan", "agent_executor.execution_plan")
pipeline.connect("agent_executor.results", "result_aggregator.results")
pipeline.connect("master_agent.execution_plan", "result_aggregator.execution_plan")
pipeline.connect("result_aggregator.final_json", "text_responder.final_json")
pipeline.connect("agent_executor.execution_log", "text_responder.execution_log")

# --- Execute pipeline ---
print(f"\n{'=' * 60}")
print(f"Processing query: '{USER_QUERY}'")
print(f"{'=' * 60}\n")

result = pipeline.run(
    {
        "master_agent": {"query": USER_QUERY},
        "result_aggregator": {"raw_query": USER_QUERY},
        "text_responder": {"query": USER_QUERY}
    },
    include_outputs_from=["master_agent", "agent_executor", "result_aggregator", "text_responder"]
)

# --- Display Results ---
print(result)
print("\n" + "=" * 60)
print("MASTER AGENT ANALYSIS")
print("=" * 60)
print(f"Reasoning: {result['master_agent']['plan_json']['reasoning']}")
print("\nPlanned Agent Calls:")
for i, call in enumerate(result["master_agent"]["plan_json"]["agent_calls"], 1):
    print(f"  {i}. {call['agent_type'].upper()} Agent")
    print(f"     Sub-query: '{call['sub_query']}'")
    print(f"     Priority: {call['priority']}")

# print("\n" + "=" * 60)
# print("EXECUTION LOG")
# print("=" * 60)
# for log_entry in result["agent_executor"]["execution_log"]:
#     print(f"  • {log_entry}")

print("\n" + "="*60)
print("USER-FRIENDLY RESPONSE")
print("="*60)
print(f"\n{result['text_responder']['text_response']}\n")

print("\n" + "=" * 60)
print("FINAL CONSOLIDATED OUTPUT")
print("=" * 60)
final_output = result["result_aggregator"]["final_json"]
print(json.dumps(final_output, indent=2, ensure_ascii=False))
print("=" * 60 + "\n")
