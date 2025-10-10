import json
from typing import List, Dict, Any
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret


@component
class ConversationalResponderComponent:
    """
    Beantwortet Folgefragen basierend auf dem Chatverlauf und einem vorherigen GIS-Ergebnis.
    """

    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(
                "eyJ0eXAiOiJKV1QiLCJraWQiOiI0MGI3YjZmYy03ZjI2LTRmMjgtOWI4Ni00ZTNjMjY2NTQ4Y2IiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpb25vc2Nsb3VkIiwiaWF0IjoxNzU0OTIzNTkyLCJjbGllbnQiOiJVU0VSIiwiaWRlbnRpdHkiOnsiaXNQYXJlbnQiOmZhbHNlLCJwcml2aWxlZ2VzIjpbIkRBVEFfQ0VOVEVSX0NSRUFURSIsIlNOQVBTSE9UX0NSRUFURSIsIklQX0JMT0NLX1JFU0VSVkUiLCJNQU5BR0VfREFUQVBMQVRGT1JNIiwiQUNDRVNTX0FDVElWSVRZX0xPRyIsIlBDQ19DUkVBVEUiLCJBQ0NFU1NfUzNfT0JKRUNUX1NUT1JBR0UiLCJCQUNLVVBfVU5JVF9DUkVBVEUiLCJDUkVBVEVfSU5URVJORVRfQUNDRVNTIiwiSzhTX0NMVVNURVJfQ1JFQVRFIiwiRkxPV19MT0dfQ1JFQVRFIiwiQUNDRVNTX0FORF9NQU5BR0VfTU9OSVRPUklORyIsIkFDQ0VTU19BTkRfTUFOQUdFX0NFUlRJRklDQVRFUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0xPR0dJTkciLCJNQU5BR0VfREJBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ETlMiLCJNQU5BR0VfUkVHSVNUUlkiLCJBQ0NFU1NfQU5EX01BTkFHRV9DRE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9WUE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9BUElfR0FURVdBWSIsIkFDQ0VTU19BTkRfTUFOQUdFX05HUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0tBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ORVRXT1JLX0ZJTEVfU1RPUkFHRSIsIkFDQ0VTU19BTkRfTUFOQUdFX0FJX01PREVMX0hVQiIsIkNSRUFURV9ORVRXT1JLX1NFQ1VSSVRZX0dST1VQUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0lBTV9SRVNPVVJDRVMiXSwidXVpZCI6IjE1MDUyN2I2LTU5YjYtNDUyYi1hOTdlLWI5MGE4NDI1YWE2NiIsInJlc2VsbGVySWQiOjEsInJlZ0RvbWFpbiI6Imlvbm9zLmRlIiwicm9sZSI6Im93bmVyIiwiY29udHJhY3ROdW1iZXIiOjM1NTYxMDU0fSwiZXhwIjoxNzYyNjk5NTkyfQ.G-Su-oYRGZCCKfgrL0n-a2SLR4uD6Qobp0VBIbZxUZDm8yX6w7TjH5eCyLm_-8Rcwb-USdk8zhDYCYpVoZFY-gAWgGYdz_DIUFBKc3-71SOftFRdPN_IYoDVCBlhtOudyoKW5lh0nrCemRHwCrXMqVtp5U7gZZNMEEf1MZadMtTAh-8rVAW4GxEwEAcj99X3kuoJaos6wgU-6G8dPUFFBWX1vRlLsdDQafvCxy55FpqMi07oyhlpYNZTiFgBKc_-vLRMzI9cOWmQ-R5CaUNWk_Ry_g-wjxAntZadT0dcVQ-3OripfrZIVYk3mpnMExW5sfL6aS2FgrKBKryA0-pnUg"),
            model="meta-llama/Llama-3.3-70B-Instruct",
            api_base_url="https://openai.inference.de-txl.ionos.com/v1"
        )

    @component.output_types(response=str)
    def run(self, chat_history: List[ChatMessage], last_gis_result: Dict[str, Any]):
        # Erstelle einen spezifischen Prompt für die Beantwortung der Frage
        prompt = f"""
        Du bist ein hilfsbereiter GIS-Assistent. Beantworte die letzte Frage des Nutzers kurz und präzise.
        Nutze dafür den bisherigen Gesprächsverlauf und das Ergebnis der letzten GIS-Abfrage.

        Letztes GIS-Ergebnis:
        {json.dumps(last_gis_result, indent=2, ensure_ascii=False)}

        Gesprächsverlauf:
        """

        # Formatiere den Chatverlauf für den Prompt
        for msg in chat_history:
            prompt += f"{msg.role.capitalize()}: {msg.text}\n"

        # Wir müssen die Nachrichten direkt übergeben, nicht nur einen langen String
        messages = [ChatMessage.from_system(prompt)]

        result = self.generator.run(messages=messages)

        try:
            return {"response": result["replies"][0].text}
        except IndexError:
            return {"response": "Ich konnte leider keine Antwort generieren."}
