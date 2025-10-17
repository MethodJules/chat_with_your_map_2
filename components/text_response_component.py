"""
Text Response Component - Generates natural language responses
"""

from typing import Dict, Any
from haystack import component
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret


@component
class TextResponseComponent:
    """
    Generates natural language responses based on query results.
    Provides user-friendly feedback about what was found and processed.
    """

    def __init__(
        self,
        api_key: str = "eyJ0eXAiOiJKV1QiLCJraWQiOiI0MGI3YjZmYy03ZjI2LTRmMjgtOWI4Ni00ZTNjMjY2NTQ4Y2IiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpb25vc2Nsb3VkIiwiaWF0IjoxNzU0OTIzNTkyLCJjbGllbnQiOiJVU0VSIiwiaWRlbnRpdHkiOnsiaXNQYXJlbnQiOmZhbHNlLCJwcml2aWxlZ2VzIjpbIkRBVEFfQ0VOVEVSX0NSRUFURSIsIlNOQVBTSE9UX0NSRUFURSIsIklQX0JMT0NLX1JFU0VSVkUiLCJNQU5BR0VfREFUQVBMQVRGT1JNIiwiQUNDRVNTX0FDVElWSVRZX0xPRyIsIlBDQ19DUkVBVEUiLCJBQ0NFU1NfUzNfT0JKRUNUX1NUT1JBR0UiLCJCQUNLVVBfVU5JVF9DUkVBVEUiLCJDUkVBVEVfSU5URVJORVRfQUNDRVNTIiwiSzhTX0NMVVNURVJfQ1JFQVRFIiwiRkxPV19MT0dfQ1JFQVRFIiwiQUNDRVNTX0FORF9NQU5BR0VfTU9OSVRPUklORyIsIkFDQ0VTU19BTkRfTUFOQUdFX0NFUlRJRklDQVRFUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0xPR0dJTkciLCJNQU5BR0VfREJBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ETlMiLCJNQU5BR0VfUkVHSVNUUlkiLCJBQ0NFU1NfQU5EX01BTkFHRV9DRE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9WUE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9BUElfR0FURVdBWSIsIkFDQ0VTU19BTkRfTUFOQUdFX05HUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0tBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ORVRXT1JLX0ZJTEVfU1RPUkFHRSIsIkFDQ0VTU19BTkRfTUFOQUdFX0FJX01PREVMX0hVQiIsIkNSRUFURV9ORVRXT1JLX1NFQ1VSSVRZX0dST1VQUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0lBTV9SRVNPVVJDRVMiXSwidXVpZCI6IjE1MDUyN2I2LTU5YjYtNDUyYi1hOTdlLWI5MGE4NDI1YWE2NiIsInJlc2VsbGVySWQiOjEsInJlZ0RvbWFpbiI6Imlvbm9zLmRlIiwicm9sZSI6Im93bmVyIiwiY29udHJhY3ROdW1iZXIiOjM1NTYxMDU0fSwiZXhwIjoxNzYyNjk5NTkyfQ.G-Su-oYRGZCCKfgrL0n-a2SLR4uD6Qobp0VBIbZxUZDm8yX6w7TjH5eCyLm_-8Rcwb-USdk8zhDYCYpVoZFY-gAWgGYdz_DIUFBKc3-71SOftFRdPN_IYoDVCBlhtOudyoKW5lh0nrCemRHwCrXMqVtp5U7gZZNMEEf1MZadMtTAh-8rVAW4GxEwEAcj99X3kuoJaos6wgU-6G8dPUFFBWX1vRlLsdDQafvCxy55FpqMi07oyhlpYNZTiFgBKc_-vLRMzI9cOWmQ-R5CaUNWk_Ry_g-wjxAntZadT0dcVQ-3OripfrZIVYk3mpnMExW5sfL6aS2FgrKBKryA0-pnUg",
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
        language: str = "German",
    ):
        """
        Initialize the Text Response Generator.

        Args:
            api_key: OpenAI API key
            model: The LLM model to use for response generation
            language: Language for the response (German, English, etc.)
        """
        self.llm = OpenAIGenerator(
            api_key=Secret.from_token(api_key),
            api_base_url="https://openai.inference.de-txl.ionos.com/v1",
            model=model,
            generation_kwargs={"temperature": 0.7},  # Higher for more natural responses
        )
        self.language = language

        self.response_prompt_template = """You are a helpful assistant for a geospatial application.
Your task is to generate a natural, user-friendly response in {language} based on the query processing results.

Original User Query: {query}

System Processing:
- Reasoning: {reasoning}
- Actions extracted: {actions}
- Layers identified: {layers}
- Locations found: {locations}

Execution Summary:
{execution_log}

Guidelines for your response:
1. Be conversational and friendly
2. Confirm what the user requested
3. Summarize what was found/processed
4. If multiple locations or layers were identified, mention them clearly
5. If geocoding was successful, mention the coordinates were found
6. If there were any issues or empty results, communicate that clearly
7. Keep it concise but informative (2-4 sentences)
8. Use the language: {language}

Generate a natural language response for the user:"""

    @component.output_types(text_response=str, response_metadata=Dict[str, Any])
    def run(
        self, final_json: Dict[str, Any], execution_log: list, query: str
    ) -> Dict[str, Any]:
        """
        Generate a natural language response based on the results.

        Args:
            final_json: The consolidated results from ResultAggregatorComponent
            execution_log: Log of agent executions
            query: Original user query

        Returns:
            text_response: Natural language response string
            response_metadata: Additional metadata about the response
        """
        # Extract data from final_json
        reasoning = final_json.get("reasoning", "No reasoning available")
        actions = self._format_data(final_json.get("actions"))
        layers = self._format_data(final_json.get("layers"))
        locations = self._format_data(final_json.get("locations"))

        # Format execution log
        execution_summary = "\n".join([f"- {log}" for log in execution_log])

        # Generate the prompt
        prompt = self.response_prompt_template.format(
            language=self.language,
            query=query,
            reasoning=reasoning,
            actions=actions,
            layers=layers,
            locations=locations,
            execution_log=execution_summary,
        )

        # Generate response
        llm_response = self.llm.run(prompt=prompt)
        text_response = llm_response["replies"][0].strip()

        # Create metadata
        metadata = {
            "language": self.language,
            "has_actions": actions != "None",
            "has_layers": layers != "None",
            "has_locations": locations != "None",
            "num_agent_calls": len(execution_log),
        }

        return {"text_response": text_response, "response_metadata": metadata}

    def _format_data(self, data: Any) -> str:
        """
        Format data for the prompt in a readable way.

        Args:
            data: The data to format

        Returns:
            Formatted string representation
        """
        if data is None:
            return "None"

        if isinstance(data, dict):
            # Handle dictionary data
            if not data:
                return "None"

            # Try to extract meaningful information
            if "locations" in data and isinstance(data["locations"], list):
                return f"{len(data['locations'])} location(s): " + ", ".join(
                    [loc.get("name", "Unknown") for loc in data["locations"]]
                )

            # Generic dict formatting
            items = []
            for key, value in data.items():
                if isinstance(value, list):
                    items.append(f"{key}: {len(value)} items")
                else:
                    items.append(f"{key}: {value}")
            return ", ".join(items) if items else str(data)

        elif isinstance(data, list):
            if not data:
                return "None"
            return f"{len(data)} item(s)"

        else:
            return str(data)


@component
class TextResponseComponentEnhanced(TextResponseComponent):
    """
    Enhanced version with response templates for different scenarios.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Response templates for different scenarios
        self.templates = {
            "success_multiple": {
                "de": "Ich habe {num_layers} Layer-Typ(en) in {num_locations} Ort(en) gefunden: {summary}",
                "en": "I found {num_layers} layer type(s) in {num_locations} location(s): {summary}",
            },
            "success_single": {
                "de": "Ich zeige Ihnen {layers} in {locations}.",
                "en": "I'm showing you {layers} in {locations}.",
            },
            "no_results": {
                "de": "Leider konnte ich keine Ergebnisse für Ihre Anfrage finden.",
                "en": "Unfortunately, I couldn't find any results for your query.",
            },
            "partial_results": {
                "de": "Ich konnte {found} finden, aber {missing} wurde nicht gefunden.",
                "en": "I found {found}, but {missing} was not found.",
            },
        }

    def _generate_template_response(self, final_json: Dict[str, Any]) -> str:
        """
        Generate a response using templates for common scenarios.
        Falls back to LLM if scenario doesn't match templates.

        Args:
            final_json: The consolidated results

        Returns:
            Template-based response or None if no template matches
        """
        lang = "de" if self.language.lower() == "german" else "en"

        # Extract information
        layers = final_json.get("layers")
        locations = final_json.get("locations")

        # Check for no results
        if not layers and not locations:
            return self.templates["no_results"][lang]

        # Count items
        num_layers = self._count_items(layers)
        num_locations = self._count_items(locations)

        if num_layers > 0 and num_locations > 0:
            # Extract names for summary
            layer_names = self._extract_names(layers)
            location_names = self._extract_names(locations)

            summary = f"{layer_names} in {location_names}"

            if num_layers > 1 or num_locations > 1:
                return self.templates["success_multiple"][lang].format(
                    num_layers=num_layers, num_locations=num_locations, summary=summary
                )
            else:
                return self.templates["success_single"][lang].format(
                    layers=layer_names, locations=location_names
                )

        return None  # Fall back to LLM

    def _count_items(self, data: Any) -> int:
        """Count items in various data structures."""
        if data is None:
            return 0
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            # Try to find a list inside
            for value in data.values():
                if isinstance(value, list):
                    return len(value)
            return 1 if data else 0
        return 1

    def _extract_names(self, data: Any) -> str:
        """Extract readable names from data."""
        if data is None:
            return ""

        if isinstance(data, dict):
            if "locations" in data:
                names = [loc.get("name", "?") for loc in data.get("locations", [])]
                return ", ".join(names) if names else "unknown"
            # For other dict structures, try to get meaningful values
            return ", ".join(str(v) for v in data.values() if v)

        if isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                names = [item.get("name", str(item)) for item in data]
                return ", ".join(names)
            return ", ".join(str(item) for item in data)

        return str(data)
