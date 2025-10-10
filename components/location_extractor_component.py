import json
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable


@component
class LocationExtractorComponent:
      # Extracts one or more location names from a query using an LLM, then geocodes each one.

    def __init__(self):
        self.geolocator = Nominatim(user_agent="master_agent_app/1.0")
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(
                "eyJ0eXAiOiJKV1QiLCJraWQiOiI0MGI3YjZmYy03ZjI2LTRmMjgtOWI4Ni00ZTNjMjY2NTQ4Y2IiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpb25vc2Nsb3VkIiwiaWF0IjoxNzU0OTIzNTkyLCJjbGllbnQiOiJVU0VSIiwiaWRlbnRpdHkiOnsiaXNQYXJlbnQiOmZhbHNlLCJwcml2aWxlZ2VzIjpbIkRBVEFfQ0VOVEVSX0NSRUFURSIsIlNOQVBTSE9UX0NSRUFURSIsIklQX0JMT0NLX1JFU0VSVkUiLCJNQU5BR0VfREFUQVBMQVRGT1JNIiwiQUNDRVNTX0FDVElWSVRZX0xPRyIsIlBDQ19DUkVBVEUiLCJBQ0NFU1NfUzNfT0JKRUNUX1NUT1JBR0UiLCJCQUNLVVBfVU5JVF9DUkVBVEUiLCJDUkVBVEVfSU5URVJORVRfQUNDRVNTIiwiSzhTX0NMVVNURVJfQ1JFQVRFIiwiRkxPV19MT0dfQ1JFQVRFIiwiQUNDRVNTX0FORF9NQU5BR0VfTU9OSVRPUklORyIsIkFDQ0VTU19BTkRfTUFOQUdFX0NFUlRJRklDQVRFUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0xPR0dJTkciLCJNQU5BR0VfREJBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ETlMiLCJNQU5BR0VfUkVHSVNUUlkiLCJBQ0NFU1NfQU5EX01BTkFHRV9DRE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9WUE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9BUElfR0FURVdBWSIsIkFDQ0VTU19BTkRfTUFOQUdFX05HUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0tBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ORVRXT1JLX0ZJTEVfU1RPUkFHRSIsIkFDQ0VTU19BTkRfTUFOQUdFX0FJX01PREVMX0hVQiIsIkNSRUFURV9ORVRXT1JLX1NFQ1VSSVRZX0dST1VQUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0lBTV9SRVNPVVJDRVMiXSwidXVpZCI6IjE1MDUyN2I2LTU5YjYtNDUyYi1hOTdlLWI5MGE4NDI1YWE2NiIsInJlc2VsbGVySWQiOjEsInJlZ0RvbWFpbiI6Imlvbm9zLmRlIiwicm9sZSI6Im93bmVyIiwiY29udHJhY3ROdW1iZXIiOjM1NTYxMDU0fSwiZXhwIjoxNzYyNjk5NTkyfQ.G-Su-oYRGZCCKfgrL0n-a2SLR4uD6Qobp0VBIbZxUZDm8yX6w7TjH5eCyLm_-8Rcwb-USdk8zhDYCYpVoZFY-gAWgGYdz_DIUFBKc3-71SOftFRdPN_IYoDVCBlhtOudyoKW5lh0nrCemRHwCrXMqVtp5U7gZZNMEEf1MZadMtTAh-8rVAW4GxEwEAcj99X3kuoJaos6wgU-6G8dPUFFBWX1vRlLsdDQafvCxy55FpqMi07oyhlpYNZTiFgBKc_-vLRMzI9cOWmQ-R5CaUNWk_Ry_g-wjxAntZadT0dcVQ-3OripfrZIVYk3mpnMExW5sfL6aS2FgrKBKryA0-pnUg"),
            model="meta-llama/Llama-3.3-70B-Instruct",
            api_base_url="https://openai.inference.de-txl.ionos.com/v1",
            generation_kwargs={"response_format": {"type": "json_object"}}
        )
        # NEUER PROMPT: Extrahiert eine Liste von Orten
        self.system_prompt = """
        Du bist ein Geolocation-Experte. Extrahiere aus dem folgenden Text ALLE geografischen Orte (z.B. Städte, Länder, Regionen).

        Deine Aufgabe ist es, die Orte als JSON-Objekt mit dem Schlüssel "locations" zurückzugeben, der ein Array von Strings enthält.

        Beispiele:
        Text: "Zeige mir die Route von Hamburg Hauptbahnhof nach Altona."
        Antwort: {"locations": ["Hamburg", "Altona"]} 

        Text: "Wo liegt Altona?"
        Antwort: {"locations": ["Altona"]}
        """

    @component.output_types(location_json=dict)
    def run(self, query: str):
        # Schritt 1: Extrahiere eine Liste von Ortsnamen mit dem LLM
        messages = [ChatMessage.from_system(self.system_prompt), ChatMessage.from_user(query)]
        result = self.generator.run(messages=messages)

        try:
            # Parse die JSON-Antwort, die eine Liste von Orten enthalten sollte
            response_data = json.loads(result["replies"][0].text)
            location_names = response_data.get("locations", [])
            if not isinstance(location_names, list):
                raise ValueError("LLM response for locations is not a list.")
        except (json.JSONDecodeError, IndexError, ValueError) as e:
            return {"location_json": {"locations": [
                {"name": "extraction_failed", "error": f"LLM did not return a valid list of locations: {e}"}]}}

        if not location_names:
            return {"location_json": {"locations": []}}

        # Schritt 2: Geokodiere jeden extrahierten Namen
        geocoded_locations = []
        for name in location_names:
            try:
                location_data = self.geolocator.geocode(name)
                if location_data:
                    geocoded_locations.append({
                        "name": name,
                        "coordinates": {
                            "latitude": location_data.latitude,
                            "longitude": location_data.longitude
                        },
                        "confidence": 0.95
                    })
                else:
                    geocoded_locations.append({"name": name, "error": "Geocoding failed"})
            except (GeocoderTimedOut, GeocoderUnavailable) as e:
                geocoded_locations.append({"name": name, "error": f"Geocoder error: {e}"})
            except Exception as e:
                geocoded_locations.append({"name": name, "error": f"Unexpected error: {e}"})

        return {"location_json": {"locations": geocoded_locations}}

