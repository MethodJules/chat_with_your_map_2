"""
Master Agent Architecture for Dynamic Query Processing
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from haystack import component
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

# --- Data Models ---


@dataclass
class AgentCall:
    """Represents a single agent invocation"""

    agent_type: str  # "action", "layer", or "location"
    sub_query: str
    priority: int = 0  # For ordering execution if needed


@dataclass
class ExecutionPlan:
    """The master agent's execution plan"""

    agent_calls: List[AgentCall]
    reasoning: str  # Why this plan was chosen


# --- Master Agent Component ---


@component
class MasterAgentComponent:
    """
    Intelligent orchestrator that analyzes queries and coordinates specialized agents.
    """

    def __init__(
        self,
        api_key: str = "eyJ0eXAiOiJKV1QiLCJraWQiOiI0MGI3YjZmYy03ZjI2LTRmMjgtOWI4Ni00ZTNjMjY2NTQ4Y2IiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpb25vc2Nsb3VkIiwiaWF0IjoxNzU0OTIzNTkyLCJjbGllbnQiOiJVU0VSIiwiaWRlbnRpdHkiOnsiaXNQYXJlbnQiOmZhbHNlLCJwcml2aWxlZ2VzIjpbIkRBVEFfQ0VOVEVSX0NSRUFURSIsIlNOQVBTSE9UX0NSRUFURSIsIklQX0JMT0NLX1JFU0VSVkUiLCJNQU5BR0VfREFUQVBMQVRGT1JNIiwiQUNDRVNTX0FDVElWSVRZX0xPRyIsIlBDQ19DUkVBVEUiLCJBQ0NFU1NfUzNfT0JKRUNUX1NUT1JBR0UiLCJCQUNLVVBfVU5JVF9DUkVBVEUiLCJDUkVBVEVfSU5URVJORVRfQUNDRVNTIiwiSzhTX0NMVVNURVJfQ1JFQVRFIiwiRkxPV19MT0dfQ1JFQVRFIiwiQUNDRVNTX0FORF9NQU5BR0VfTU9OSVRPUklORyIsIkFDQ0VTU19BTkRfTUFOQUdFX0NFUlRJRklDQVRFUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0xPR0dJTkciLCJNQU5BR0VfREJBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ETlMiLCJNQU5BR0VfUkVHSVNUUlkiLCJBQ0NFU1NfQU5EX01BTkFHRV9DRE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9WUE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9BUElfR0FURVdBWSIsIkFDQ0VTU19BTkRfTUFOQUdFX05HUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0tBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ORVRXT1JLX0ZJTEVfU1RPUkFHRSIsIkFDQ0VTU19BTkRfTUFOQUdFX0FJX01PREVMX0hVQiIsIkNSRUFURV9ORVRXT1JLX1NFQ1VSSVRZX0dST1VQUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0lBTV9SRVNPVVJDRVMiXSwidXVpZCI6IjE1MDUyN2I2LTU5YjYtNDUyYi1hOTdlLWI5MGE4NDI1YWE2NiIsInJlc2VsbGVySWQiOjEsInJlZ0RvbWFpbiI6Imlvbm9zLmRlIiwicm9sZSI6Im93bmVyIiwiY29udHJhY3ROdW1iZXIiOjM1NTYxMDU0fSwiZXhwIjoxNzYyNjk5NTkyfQ.G-Su-oYRGZCCKfgrL0n-a2SLR4uD6Qobp0VBIbZxUZDm8yX6w7TjH5eCyLm_-8Rcwb-USdk8zhDYCYpVoZFY-gAWgGYdz_DIUFBKc3-71SOftFRdPN_IYoDVCBlhtOudyoKW5lh0nrCemRHwCrXMqVtp5U7gZZNMEEf1MZadMtTAh-8rVAW4GxEwEAcj99X3kuoJaos6wgU-6G8dPUFFBWX1vRlLsdDQafvCxy55FpqMi07oyhlpYNZTiFgBKc_-vLRMzI9cOWmQ-R5CaUNWk_Ry_g-wjxAntZadT0dcVQ-3OripfrZIVYk3mpnMExW5sfL6aS2FgrKBKryA0-pnUg",
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
    ):
        """
        Initialize the Master Agent with LLM capabilities.

        Args:
            api_key: OpenAI API key
            model: The LLM model to use for decision making
        """
        self.llm = OpenAIGenerator(
            api_key=Secret.from_token(api_key),
            api_base_url="https://openai.inference.de-txl.ionos.com/v1",
            model="meta-llama/Llama-3.3-70B-Instruct",
            generation_kwargs={
                "temperature": 0.1
            },  # Low temperature for consistent planning
        )

        self.planning_prompt_template = """You are a master agent orchestrator for a geospatial application.
Your job is to analyze user queries and create an execution plan that calls the appropriate specialized agents.

Available agents:
1. ACTION AGENT: Extracts action/operations from the query (e.g., "show", "display", "find", "filter")
2. LAYER AGENT: Identifies which map layers/data types are needed (e.g., "parking lots", "bike stations", "train stops")
3. LOCATION AGENT: Extracts geographic locations mentioned (e.g., "Altona", "Wandsbek", "Hamburg")

User Query: {query}

Your task:
1. Analyze the query complexity
2. Determine which agents are needed
3. If the query is complex, break it down into sub-queries for each agent
4. Create an execution plan

Output your plan as a JSON object with this EXACT structure:
{{
  "reasoning": "Brief explanation of your analysis",
  "agent_calls": [
    {{
      "agent_type": "action|layer|location",
      "sub_query": "The specific sub-query for this agent",
      "priority": 0
    }}
  ]
}}

Guidelines:
- Simple queries may only need 1-2 agents
- Complex queries with multiple locations or layers need multiple agent calls
- Break down "and" clauses into separate calls when they refer to different entities
- Always include reasoning for your decisions
- Priority: 0 = execute in parallel, higher numbers = execute later

Return ONLY the JSON object, no additional text."""

    @component.output_types(execution_plan=ExecutionPlan, plan_json=Dict[str, Any])
    def run(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query and create an execution plan.

        Args:
            query: The user's input query

        Returns:
            execution_plan: ExecutionPlan object
            plan_json: JSON representation of the plan
        """
        # Generate the execution plan using LLM
        prompt = self.planning_prompt_template.format(query=query)
        llm_response = self.llm.run(prompt=prompt)

        # Parse the LLM response
        response_text = llm_response["replies"][0]

        # Extract JSON from the response (in case LLM adds extra text)
        try:
            # Try to find JSON in the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            json_str = response_text[start_idx:end_idx]
            plan_dict = json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response was: {response_text}")
            # Fallback: call all agents with the original query
            plan_dict = {
                "reasoning": "Fallback plan due to parsing error",
                "agent_calls": [
                    {"agent_type": "action", "sub_query": query, "priority": 0},
                    {"agent_type": "layer", "sub_query": query, "priority": 0},
                    {"agent_type": "location", "sub_query": query, "priority": 0},
                ],
            }

        # Convert to ExecutionPlan object
        agent_calls = [
            AgentCall(
                agent_type=call["agent_type"],
                sub_query=call["sub_query"],
                priority=call.get("priority", 0),
            )
            for call in plan_dict["agent_calls"]
        ]

        execution_plan = ExecutionPlan(
            agent_calls=agent_calls, reasoning=plan_dict["reasoning"]
        )

        return {"execution_plan": execution_plan, "plan_json": plan_dict}


# --- Agent Executor Component ---


@component
class AgentExecutorComponent:
    """
    Executes the plan by calling the appropriate specialized agents.
    """

    def __init__(self, action_agent, layer_agent, location_agent):
        """
        Initialize with references to the specialized agents.

        Args:
            action_agent: ActionExtractorComponent instance
            layer_agent: LayerExtractorComponent instance
            location_agent: LocationExtractorComponent instance
        """
        self.agents = {
            "action": action_agent,
            "layer": layer_agent,
            "location": location_agent,
        }

    @component.output_types(results=Dict[str, List[Any]], execution_log=List[str])
    def run(self, execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Execute the plan by calling agents in the specified order.

        Args:
            execution_plan: The ExecutionPlan from the master agent

        Returns:
            results: Dictionary with results from each agent type
            execution_log: Log of what was executed
        """
        results = {"action": [], "layer": [], "location": []}
        execution_log = []

        # Sort agent calls by priority
        sorted_calls = sorted(execution_plan.agent_calls, key=lambda x: x.priority)

        # Execute each agent call
        for agent_call in sorted_calls:
            agent_type = agent_call.agent_type
            sub_query = agent_call.sub_query

            log_entry = f"Calling {agent_type} agent with query: '{sub_query}'"
            print(log_entry)
            execution_log.append(log_entry)

            # Get the appropriate agent
            agent = self.agents.get(agent_type)

            if agent:
                # Call the agent with the sub-query
                agent_result = agent.run(query=sub_query)

                # Store the result (adapt based on your component output names)
                if agent_type == "action" and "action_json" in agent_result:
                    results["action"].append(agent_result["action_json"])
                elif agent_type == "layer" and "layer_json" in agent_result:
                    results["layer"].append(agent_result["layer_json"])
                elif agent_type == "location" and "location_json" in agent_result:
                    results["location"].append(agent_result["location_json"])
            else:
                print(f"Warning: Unknown agent type '{agent_type}'")

        return {"results": results, "execution_log": execution_log}


# --- Result Aggregator Component ---


@component
class ResultAggregatorComponent:
    """
    Combines results from multiple agent calls into a final JSON output.
    """

    @component.output_types(final_json=Dict[str, Any])
    def run(
        self,
        results: Dict[str, List[Any]],
        execution_plan: ExecutionPlan,
        raw_query: str,
    ) -> Dict[str, Any]:
        """
        Aggregate all agent results into a final JSON structure.

        Args:
            results: Results from all agent calls
            execution_plan: The execution plan that was used
            raw_query: Original user query

        Returns:
            final_json: Consolidated output
        """
        # Merge multiple results of the same type
        merged_results = {
            "query": raw_query,
            "reasoning": execution_plan.reasoning,
            "command": self._merge_list(results.get("action", [])),
            "layers": self._merge_list(results.get("layer", [])),
            "locations": self._merge_list(results.get("location", [])),
        }

        return {"final_json": merged_results}

    def _merge_list(self, items: List[Any]) -> Any:
        """
        Merge multiple results into a single structure.
        Customize this based on your data format.
        """
        if not items:
            return None
        if len(items) == 1:
            return items[0]

        # For multiple items, try to merge them intelligently
        # This is a simple approach - you may need to customize
        if isinstance(items[0], dict):
            merged = {}
            for item in items:
                if isinstance(item, dict):
                    merged.update(item)
            return merged
        elif isinstance(items[0], list):
            merged = []
            for item in items:
                if isinstance(item, list):
                    merged.extend(item)
            return merged
        else:
            return items
