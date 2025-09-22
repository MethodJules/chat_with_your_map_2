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


system_prompt_dataset = """
Du bist ein Experte für Geoinformationssysteme (GIS). Deine Aufgabe ist es, den folgenden deutschen Text zu analysieren und den am besten passenden GIS-Layer aus der bereitgestellten Liste zu identifizieren.

Verfügbare Layer:
{layer_list}

Anweisungen:

Lies den deutschen Text des Nutzers sorgfältig durch, um dessen Kernaussage zu verstehen.

Vergleiche die Absicht des Textes mit den Namen und Beschreibungen der verfügbaren Layer.

Wähle den Layer aus, der am relevantesten ist, und bestimme einen Konfidenzwert zwischen 0.0 und 1.0.

Gib deine Antwort ausschließlich als JSON-Objekt im unten definierten Format aus. Füge keine zusätzlichen Erklärungen, Notizen oder einleitenden Sätze hinzu.

Ausgabeformat (JSON):

JSON

{
  "layer": {
    "name": "NameDesLayers",
    "id": "ID_des_Layers",
    "confidence": 0.92
  }
}
Beispiel:

Text: Zeige mir alle Fahrradstationen in Altona.

Antwort (JSON):

JSON

{
  "layer": {
    "name": "Fahrradstationen",
    "id": "18105",
    "confidence": 0.92
  }
}
"""


@component
class DatasetAgent:

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

        layer_list = get_layer_list(self.csv_path)
        prompt = system_prompt_dataset.format(layer_list=layer_list)

        messages = [ChatMessage.from_system(prompt), ChatMessage.from_user(query)]
        response = self.generator.run(messages=messages)

        try:
            json_response = json.loads(response["replies"][0].content)
            return {"dataset_data": json_response}
        except (json.JSONDecodeError, IndexError) as e:
            return {"dataset_data": {"error": f"Fehler bei der JSON-Verarbeitung: {e}"}}