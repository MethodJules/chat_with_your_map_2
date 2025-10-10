import os
import json
from typing import Dict
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
import openrouteservice
from .location_extractor_component import LocationExtractorComponent


@component
class RoutingAgentComponent:
    def __init__(self):
        self.location_extractor = LocationExtractorComponent()
        self.ors_client = openrouteservice.Client(key="eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjA2YmYzNTdkMmNmMzQ3YWFhZGMwZGU0ZWVmZGExNzZhIiwiaCI6Im11cm11cjY0In0=")
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(
                "eyJ0eXAiOiJKV1QiLCJraWQiOiI0MGI3YjZmYy03ZjI2LTRmMjgtOWI4Ni00ZTNjMjY2NTQ4Y2IiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpb25vc2Nsb3VkIiwiaWF0IjoxNzU0OTIzNTkyLCJjbGllbnQiOiJVU0VSIiwiaWRlbnRpdHkiOnsiaXNQYXJlbnQiOmZhbHNlLCJwcml2aWxlZ2VzIjpbIkRBVEFfQ0VOVEVSX0NSRUFURSIsIlNOQVBTSE9UX0NSRUFURSIsIklQX0JMT0NLX1JFU0VSVkUiLCJNQU5BR0VfREFUQVBMQVRGT1JNIiwiQUNDRVNTX0FDVElWSVRZX0xPRyIsIlBDQ19DUkVBVEUiLCJBQ0NFU1NfUzNfT0JKRUNUX1NUT1JBR0UiLCJCQUNLVVBfVU5JVF9DUkVBVEUiLCJDUkVBVEVfSU5URVJORVRfQUNDRVNTIiwiSzhTX0NMVVNURVJfQ1JFQVRFIiwiRkxPV19MT0dfQ1JFQVRFIiwiQUNDRVNTX0FORF9NQU5BR0VfTU9OSVRPUklORyIsIkFDQ0VTU19BTkRfTUFOQUdFX0NFUlRJRklDQVRFUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0xPR0dJTkciLCJNQU5BR0VfREJBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ETlMiLCJNQU5BR0VfUkVHSVNUUlkiLCJBQ0NFU1NfQU5EX01BTkFHRV9DRE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9WUE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9BUElfR0FURVdBWSIsIkFDQ0VTU19BTkRfTUFOQUdFX05HUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0tBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ORVRXT1JLX0ZJTEVfU1RPUkFHRSIsIkFDQ0VTU19BTkRfTUFOQUdFX0FJX01PREVMX0hVQiIsIkNSRUFURV9ORVRXT1JLX1NFQ1VSSVRZX0dST1VQUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0lBTV9SRVNPVVJDRVMiXSwidXVpZCI6IjE1MDUyN2I2LTU5YjYtNDUyYi1hOTdlLWI5MGE4NDI1YWE2NiIsInJlc2VsbGVySWQiOjEsInJlZ0RvbWFpbiI6Imlvbm9zLmRlIiwicm9sZSI6Im93bmVyIiwiY29udHJhY3ROdW1iZXIiOjM1NTYxMDU0fSwiZXhwIjoxNzYyNjk5NTkyfQ.G-Su-oYRGZCCKfgrL0n-a2SLR4uD6Qobp0VBIbZxUZDm8yX6w7TjH5eCyLm_-8Rcwb-USdk8zhDYCYpVoZFY-gAWgGYdz_DIUFBKc3-71SOftFRdPN_IYoDVCBlhtOudyoKW5lh0nrCemRHwCrXMqVtp5U7gZZNMEEf1MZadMtTAh-8rVAW4GxEwEAcj99X3kuoJaos6wgU-6G8dPUFFBWX1vRlLsdDQafvCxy55FpqMi07oyhlpYNZTiFgBKc_-vLRMzI9cOWmQ-R5CaUNWk_Ry_g-wjxAntZadT0dcVQ-3OripfrZIVYk3mpnMExW5sfL6aS2FgrKBKryA0-pnUg"),
            model="meta-llama/Llama-3.3-70B-Instruct",
            api_base_url="https://openai.inference.de-txl.ionos.com/v1",
            generation_kwargs={"response_format": {"type": "json_object"}}
        )
        self.system_prompt = """
        Deine Aufgabe ist es, aus einer Benutzeranfrage Start, Ziel und Verkehrsmittel für eine Routenplanung zu extrahieren.
        Mögliche Verkehrsmittel: 'driving-car', 'cycling-regular', 'foot-walking'. Wähle das passendste.

        Beispiel:
        User: "Wie komme ich mit dem Fahrrad vom Rathaus zum Stadtpark?"
        Antwort:
        {
          "start": "Rathaus",
          "end": "Stadtpark",
          "mode": "cycling-regular"
        }
        """

    @component.output_types(result_json=Dict)
    def run(self, query: str):
        # Schritt 1: LLM zur Extraktion von Start, Ziel, Modus
        messages = [ChatMessage.from_system(self.system_prompt), ChatMessage.from_user(query)]
        try:
            result = self.generator.run(messages=messages)
            route_params = json.loads(result["replies"][0].text)
            start_query = route_params.get("start")
            end_query = route_params.get("end")
            mode = route_params.get("mode", "driving-car")
        except (json.JSONDecodeError, IndexError, KeyError):
            return {"result_json": {"error": "Routenparameter konnten nicht extrahiert werden."}}

        # Schritt 2: Koordinaten für Start und Ziel
        start_loc = self.location_extractor.run(query=start_query).get("location_json", {}).get("locations")
        end_loc = self.location_extractor.run(query=end_query).get("location_json", {}).get("locations")

        if not start_loc or not end_loc:
            return {"result_json": {"error": "Start- oder Zielort konnte nicht gefunden werden."}}

        start_coords = [start_loc[0]['longitude'], start_loc[0]['latitude']]
        end_coords = [end_loc[0]['longitude'], end_loc[0]['latitude']]

        # Schritt 3: OpenRouteService API anfragen
        try:
            routes = self.ors_client.directions(coordinates=[start_coords, end_coords], profile=mode, format='geojson')

            # Schritt 4: Ergebnis formatieren
            summary = routes['features'][0]['properties']['summary']
            result = {
                "route": {
                    "from": start_loc[0]['name'],
                    "to": end_loc[0]['name'],
                    "mode": mode,
                    "duration_minutes": round(summary['duration'] / 60, 1),
                    "distance_km": round(summary['distance'] / 1000, 2),
                    "geometry": routes['features'][0]['geometry'],
                    "bbox": routes['bbox']
                }
            }
            return {"result_json": result}

        except openrouteservice.exceptions.ApiError as e:
            return {"result_json": {"error": f"Routing API Fehler: {e}"}}
        except Exception as e:
            return {"result_json": {"error": f"Ein unerwarteter Routing-Fehler ist aufgetreten: {e}"}}
