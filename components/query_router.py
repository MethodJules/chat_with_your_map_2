import json
from typing import List
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret


@component
class QueryRouterComponent:
    """
    Klassifiziert die Absicht der letzten Nutzernachricht im Kontext des Gesprächs.
    """

    def __init__(self):
        """
        Initialisiert die Komponente mit dem LLM-Generator und dem System-Prompt.
        """
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(
                "eyJ0eXAiOiJKV1QiLCJraWQiOiI0MGI3YjZmYy03ZjI2LTRmMjgtOWI4Ni00ZTNjMjY2NTQ4Y2IiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpb25vc2Nsb3VkIiwiaWF0IjoxNzU0OTIzNTkyLCJjbGllbnQiOiJVU0VSIiwiaWRlbnRpdHkiOnsiaXNQYXJlbnQiOmZhbHNlLCJwcml2aWxlZ2VzIjpbIkRBVEFfQ0VOVEVSX0NSRUFURSIsIlNOQVBTSE9UX0NSRUFURSIsIklQX0JMT0NLX1JFU0VSVkUiLCJNQU5BR0VfREFUQVBMQVRGT1JNIiwiQUNDRVNTX0FDVElWSVRZX0xPRyIsIlBDQ19DUkVBVEUiLCJBQ0NFU1NfUzNfT0JKRUNUX1NUT1JBR0UiLCJCQUNLVVBfVU5JVF9DUkVBVEUiLCJDUkVBVEVfSU5URVJORVRfQUNDRVNTIiwiSzhTX0NMVVNURVJfQ1JFQVRFIiwiRkxPV19MT0dfQ1JFQVRFIiwiQUNDRVNTX0FORF9NQU5BR0VfTU9OSVRPUklORyIsIkFDQ0VTU19BTkRfTUFOQUdFX0NFUlRJRklDQVRFUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0xPR0dJTkciLCJNQU5BR0VfREJBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ETlMiLCJNQU5BR0VfUkVHSVNUUlkiLCJBQ0NFU1NfQU5EX01BTkFHRV9DRE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9WUE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9BUElfR0FURVdBWSIsIkFDQ0VTU19BTkRfTUFOQUdFX05HUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0tBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ORVRXT1JLX0ZJTEVfU1RPUkFHRSIsIkFDQ0VTU19BTkRfTUFOQUdFX0FJX01PREVMX0hVQiIsIkNSRUFURV9ORVRXT1JLX1NFQ1VSSVRZX0dST1VQUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0lBTV9SRVNPVVJDRVMiXSwidXVpZCI6IjE1MDUyN2I2LTU5YjYtNDUyYi1hOTdlLWI5MGE4NDI1YWE2NiIsInJlc2VsbGVySWQiOjEsInJlZ0RvbWFpbiI6Imlvbm9zLmRlIiwicm9sZSI6Im93bmVyIiwiY29udHJhY3ROdW1iZXIiOjM1NTYxMDU0fSwiZXhwIjoxNzYyNjk5NTkyfQ.G-Su-oYRGZCCKfgrL0n-a2SLR4uD6Qobp0VBIbZxUZDm8yX6w7TjH5eCyLm_-8Rcwb-USdk8zhDYCYpVoZFY-gAWgGYdz_DIUFBKc3-71SOftFRdPN_IYoDVCBlhtOudyoKW5lh0nrCemRHwCrXMqVtp5U7gZZNMEEf1MZadMtTAh-8rVAW4GxEwEAcj99X3kuoJaos6wgU-6G8dPUFFBWX1vRlLsdDQafvCxy55FpqMi07oyhlpYNZTiFgBKc_-vLRMzI9cOWmQ-R5CaUNWk_Ry_g-wjxAntZadT0dcVQ-3OripfrZIVYk3mpnMExW5sfL6aS2FgrKBKryA0-pnUg"),
            model="meta-llama/Llama-3.3-70B-Instruct",
            api_base_url="https://openai.inference.de-txl.ionos.com/v1",
            generation_kwargs={"response_format": {"type": "json_object"}}
        )
        self.system_prompt = """
        Deine Aufgabe ist es, die Absicht der letzten Nutzernachricht zu klassifizieren.
        Die möglichen Absichten sind:
        - "gis_query": Der Nutzer möchte neue geografische Informationen suchen oder anzeigen (z.B. "Zeige mir...", "Wo ist...", "Finde alle...").
        - "distance_query": Der Nutzer möchte die Distanz zwischen zwei oder mehr Orten messen (z.B. "Messe die Distanz...", "Wie weit ist es von A nach B?").
        - "follow_up_question": Der Nutzer stellt eine Frage über das Ergebnis der vorherigen Anfrage (z.B. "Wie viele sind das?", "Zähle sie").
        - "visualization_query": Der Nutzer möchte das Ergebnis der vorherigen Anfrage auf einer Karte sehen (z.B. "Zeig mir das auf der Karte", "Visualisiere das").
        - "follow_up_question": Der Nutzer stellt eine andere Frage über das Ergebnis der vorherigen Anfrage (z.B. "Wie viele sind das?").
        - "proximity_query": Anfrage nach dem nächstgelegenen Objekt. (z.B. "Wo ist die nächste Haltestelle?")
        - "filter_query": Anfrage, die Daten nach Attributen filtert. (z.B. "Finde Schulen mit dem Namen 'Goethe'")
        - "buffer_query": Anfrage nach Objekten in einem bestimmten Umkreis. (z.B. "Was ist in 500m Umkreis vom Rathaus?")
  

        Analysiere den Chatverlauf und gib NUR ein JSON-Objekt mit dem Schlüssel "intent" zurück, der eine der oben genannten Absichten enthält.

        Beispiel 1:
        User: Messe die Distanz zwischen Hamburg und Berlin.
        Antwort: {"intent": "distance_query"}

        Beispiel 2:
        User: Zeige mir alle Parks in Hamburg.
        Antwort: {"intent": "gis_query"}

        Beispiel 3:
        User: Zeige mir alle Parks in Hamburg.
        Assistant: [JSON-Ergebnis]
        User: Wie viele sind das?
        Antwort: {"intent": "follow_up_question"}
        
        Beispiel 4:
        User: Messe die Distanz zwischen Hamburg und Berlin.
        Antwort: {"intent": "distance_query"}

        Beispiel 5:
        User: Messe die Distanz zwischen Hamburg Hauptbahnhof und Altona.
        Assistant: Die Distanz beträgt 4.5 km.
        User: Zeig mir das auf der Karte.
        Antwort: {"intent": "visualization_query"}
        """

    @component.output_types(intent=str)
    def run(self, chat_history: List[ChatMessage]):
        """
        Führt die Klassifizierung der Absicht aus.
        """
        messages = [ChatMessage.from_system(self.system_prompt)] + chat_history

        try:
            result = self.generator.run(messages=messages)
            response_data = json.loads(result["replies"][0].text)
            return {"intent": response_data.get("intent", "unknown")}
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"[Error in QueryRouter]: {e}")
            return {"intent": "unknown"}

