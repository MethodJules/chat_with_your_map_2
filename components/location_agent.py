import os
import json
from typing import Dict, Any, Optional

from haystack import component, Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

system_prompt_location = """
Du bist ein Stadtplaner und nutzt eine webbasierte GIS-Anwendung. Extrahiere aus dem folgenden Text alle geografischen Orte (z. B. Städte, Länder, Regionen).

Gib nur den geografischen Ort als einzelnes Wort oder eine kurze Phrase zurück, ohne zusätzliche Erklärungen.

Beispiele:
Text: "Wie lang fließt die Elbe durch Altona?"
Antwort: "Altona"

Text: "Zeige alle S-Bahn Stationen in Wandsbek"
Antwort: "Wandsbek"
"""

@component
class LocationAgent:
    """
    Eine Haystack-Komponente, die einen Standort aus einem Text extrahiert und geocodiert.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.3-70B-Instruct"):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_env_var("IONOS_API_TOKEN"),
            model=model_name,
            api_base_url="https://openai.inference.de-txl.ionos.com/v1"
        )
        self.geolocator = Nominatim(user_agent="location_agent_app/1.0")

    @component.output_types(location_data=Optional[Dict[str, Any]])
    def run(self, query: str) -> Dict[str, Any]:
        """
        Führt die Standorterkennung und Geocodierung durch.
        """
        messages = [ChatMessage.from_system(system_prompt_location), ChatMessage.from_user(query)]
        response = self.generator.run(messages=messages)
        location_name = response["replies"][0].content.strip()

        if not location_name:
            return {"location_data": None}

        
        try:
            location_data = self.geolocator.geocode(location_name)
            if location_data:
                coords = {
                    "name": location_name,
                    "latitude": location_data.latitude,
                    "longitude": location_data.longitude,
                    "confidence": 0.9  # Beispielhafter Konfidenzwert
                }
                return {"location_data": coords}
            else:
                return {"location_data": {"name": location_name, "error": "Geocoding fehlgeschlagen"}}
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            return {"location_data": {"name": location_name, "error": f"Geocoder-Fehler: {e}"}}
        except Exception as e:
            return {"location_data": {"name": location_name, "error": f"Unerwarteter Fehler: {e}"}}