import os
import json
from typing import Dict, Any, List
import pandas as pd

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret


def get_layer_list(csv_path: str) -> str:
    """Lädt die Layer-Liste aus der CSV-Datei."""
    try:
        df = pd.read_csv(csv_path)
        layer_dict = {row["Name"]: row["ID"] for _, row in df.iterrows()}
        return "\n".join([f"- {name} (ID: {layer_id})" for name, layer_id in layer_dict.items()])
    except FileNotFoundError:
        return "Keine Layer-Daten gefunden."


# System-Prompt für die Layer-Erkennung
system_prompt_dataset = """
Du bist ein GIS-Datenexperte. Du erhältst einen deutschen Text und musst bestimmen, welche Layer aus einer GIS-Anwendung am besten passen.

Verfügbare Layer:
{layer_list}

Deine Antwort muss ein JSON-Objekt im folgenden Format sein:
{{
  "layer": {{
    "name": "LayerName",
    "id": "layer_id",
    "confidence": 0.92
  }}
}}

Beispiel:
Text: Zeige mir alle Fahrradstationen in Altona
Antwort (JSON):
{{
  "layer": {{
    "name": "Fahrradstationen",
    "id": "18105",
    "confidence": 0.92
  }}
}}
"""


@component
class DatasetAgent:
    """
    Eine Haystack-Komponente zur Identifizierung von GIS-Layern aus einem Text.
    """

    def __init__(self, csv_path: str = "./data/csv_datei.csv", model_name: str = "meta-llama/Llama-3.3-70B-Instruct"):
        self.csv_path = csv_path
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_env_var("IONOS_API_TOKEN"),
            model=model_name,
            api_base_url="https://openai.inference.de-txl.ionos.com/v1",
            generation_kwargs={"response_format": {"type": "json_object"}}
        )

    @component.output_types(dataset_data=Dict[str, Any])
    def run(self, query: str) -> Dict[str, Any]:
        """
        Identifiziert den passenden GIS-Layer für die gegebene Anfrage.
        """
        layer_list = get_layer_list(self.csv_path)
        prompt = system_prompt_dataset.format(layer_list=layer_list)

        messages = [ChatMessage.from_system(prompt), ChatMessage.from_user(query)]
        response = self.generator.run(messages=messages)

        try:
            # Der Inhalt der Antwort sollte bereits JSON sein
            json_response = json.loads(response["replies"][0].content)
            return {"dataset_data": json_response}
        except (json.JSONDecodeError, IndexError) as e:
            return {"dataset_data": {"error": f"Fehler bei der JSON-Verarbeitung: {e}"}}