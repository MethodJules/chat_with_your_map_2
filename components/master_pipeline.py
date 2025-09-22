import os
import json
from dotenv import load_dotenv

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage

from actiontype_agent import ActionTypeAgent
from location_agent import LocationAgent
from dataset_agent import DatasetAgent

load_dotenv()

master_prompt_template = """
Du bist ein Master-Agent (Orchestrator). Deine Aufgabe ist es, die JSON-Ausgaben von spezialisierten Agenten zu einer einzigen, sauberen JSON-Antwort zusammenzufassen.

Hier sind die Daten der einzelnen Agenten:
- Action Agent: {{ action_result }}
- Location Agent: {{ location_result }}
- Dataset Agent: {{ dataset_result }}

Kombiniere diese Informationen zu einem einzigen JSON-Objekt. Die Antwort darf nur das JSON-Objekt enthalten, keinen zusätzlichen Text.

Beispiel-Ausgabeformat:
{
  "action": {"type": "zeigen", "confidence": 0.95},
  "location": {
    "name": "Altona",
    "latitude": 53.55,
    "longitude": 9.93,
    "confidence": 0.9
  },
  "layer": {
    "name": "Fahrradstationen",
    "id": "18105",
    "confidence": 0.92
  },
  "options": {}
}
"""


def build_master_pipeline():

    action_agent = ActionTypeAgent()
    location_agent = LocationAgent()
    dataset_agent = DatasetAgent()
    prompt_builder = PromptBuilder(template=master_prompt_template)
    final_generator = OpenAIChatGenerator(
        api_key=Secret.from_env_var("IONOS_API_TOKEN"),
        model="meta-llama/Llama-3.3-70B-Instruct",
        api_base_url="https://openai.inference.de-txl.ionos.com/v1",
        generation_kwargs={"response_format": {"type": "json_object"}}
    )

    pipeline = Pipeline()
    pipeline.add_component("action_agent", action_agent)
    pipeline.add_component("location_agent", location_agent)
    pipeline.add_component("dataset_agent", dataset_agent)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("final_generator", final_generator)

    
    pipeline.connect("action_agent.action_data", "prompt_builder.action_result")
    pipeline.connect("location_agent.location_data", "prompt_builder.location_result")
    pipeline.connect("dataset_agent.dataset_data", "prompt_builder.dataset_result")
    pipeline.connect("prompt_builder.prompt", "final_generator.messages")

    return pipeline


if __name__ == '__main__':
    master_pipeline = build_master_pipeline()

    query = "Zeige mir alle Fahrradstationen in Altona."

    result = master_pipeline.run({
        "action_agent": {"query": query},
        "location_agent": {"query": query},
        "dataset_agent": {"query": query}
    })

    try:
        final_json = json.loads(result["final_generator"]["replies"][0].content)
        print(json.dumps(final_json, indent=4))
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Fehler bei der finalen JSON-Ausgabe: {e}")
        print("Rohe Ausgabe:", result["final_generator"]["replies"][0].content)