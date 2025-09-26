import json
import pandas as pd
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

@component
class LayerExtractorComponent:

    # A component that extracts the GIS layer from a user query using an LLM.
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(
                "eyJ0eXAiOiJKV1QiLCJraWQiOiI0MGI3YjZmYy03ZjI2LTRmMjgtOWI4Ni00ZTNjMjY2NTQ4Y2IiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpb25vc2Nsb3VkIiwiaWF0IjoxNzU0OTIzNTkyLCJjbGllbnQiOiJVU0VSIiwiaWRlbnRpdHkiOnsiaXNQYXJlbnQiOmZhbHNlLCJwcml2aWxlZ2VzIjpbIkRBVEFfQ0VOVEVSX0NSRUFURSIsIlNOQVBTSE9UX0NSRUFURSIsIklQX0JMT0NLX1JFU0VSVkUiLCJNQU5BR0VfREFUQVBMQVRGT1JNIiwiQUNDRVNTX0FDVElWSVRZX0xPRyIsIlBDQ19DUkVBVEUiLCJBQ0NFU1NfUzNfT0JKRUNUX1NUT1JBR0UiLCJCQUNLVVBfVU5JVF9DUkVBVEUiLCJDUkVBVEVfSU5URVJORVRfQUNDRVNTIiwiSzhTX0NMVVNURVJfQ1JFQVRFIiwiRkxPV19MT0dfQ1JFQVRFIiwiQUNDRVNTX0FORF9NQU5BR0VfTU9OSVRPUklORyIsIkFDQ0VTU19BTkRfTUFOQUdFX0NFUlRJRklDQVRFUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0xPR0dJTkciLCJNQU5BR0VfREJBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ETlMiLCJNQU5BR0VfUkVHSVNUUlkiLCJBQ0NFU1NfQU5EX01BTkFHRV9DRE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9WUE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9BUElfR0FURVdBWSIsIkFDQ0VTU19BTkRfTUFOQUdFX05HUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0tBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ORVRXT1JLX0ZJTEVfU1RPUkFHRSIsIkFDQ0VTU19BTkRfTUFOQUdFX0FJX01PREVMX0hVQiIsIkNSRUFURV9ORVRXT1JLX1NFQ1VSSVRZX0dST1VQUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0lBTV9SRVNPVVJDRVMiXSwidXVpZCI6IjE1MDUyN2I2LTU5YjYtNDUyYi1hOTdlLWI5MGE4NDI1YWE2NiIsInJlc2VsbGVySWQiOjEsInJlZ0RvbWFpbiI6Imlvbm9zLmRlIiwicm9sZSI6Im93bmVyIiwiY29udHJhY3ROdW1iZXIiOjM1NTYxMDU0fSwiZXhwIjoxNzYyNjk5NTkyfQ.G-Su-oYRGZCCKfgrL0n-a2SLR4uD6Qobp0VBIbZxUZDm8yX6w7TjH5eCyLm_-8Rcwb-USdk8zhDYCYpVoZFY-gAWgGYdz_DIUFBKc3-71SOftFRdPN_IYoDVCBlhtOudyoKW5lh0nrCemRHwCrXMqVtp5U7gZZNMEEf1MZadMtTAh-8rVAW4GxEwEAcj99X3kuoJaos6wgU-6G8dPUFFBWX1vRlLsdDQafvCxy55FpqMi07oyhlpYNZTiFgBKc_-vLRMzI9cOWmQ-R5CaUNWk_Ry_g-wjxAntZadT0dcVQ-3OripfrZIVYk3mpnMExW5sfL6aS2FgrKBKryA0-pnUg"),
            model="meta-llama/Llama-3.3-70B-Instruct",
            api_base_url="https://openai.inference.de-txl.ionos.com/v1",
            generation_kwargs={"response_format": {"type": "json_object"}}
        )
        df = pd.read_csv(self.csv_path)
        layer_dict = {row["Name"]: row["ID"] for _, row in df.iterrows()}

        layer_list = "\n".join([f"- {name} (ID: {layer_id})" for name, layer_id in layer_dict.items()])
        self.system_prompt = f"""
        Du bist ein GIS-Datenexperte. Du erhältst einen deutschen Text und musst bestimmen, welche Layer aus einer GIS-Anwendung am besten passen.

        Verfügbare Layer:
        {layer_list}

        Gib deine Antwort ausschließlich als JSON-Objekt mit dem Schlüssel "layer" zurück.
        Beispiel(wenn 1 gefunden) : Text: Zeige mir alle Fahrradstationen. Antwort: {{"layers": {{"name": "[NAME]", "id": "[ID]", "confidence": 0.92}}}}
        Beispiel (wenn 2 gefunden) : Text: Zeige mir alle Fahrradstationen und Parkplätze. Antwort: {{"layers": {{"name": "[NAME]", "id": "[ID]", "confidence": 0.92}},{{"name": "[NAME]", "id": "[ID]", "confidence": 0.50}}}}
        
        Wenn Mehr gefunden Zeig ihnen alle

        """

    def _get_layer_list(self) -> str:
        try:

            df = pd.read_csv(self.csv_path, sep=';')
            for row in df.iterrows():
                print(row)
            return "\n".join([f"- {row['Name']} (ID: {row['ID']})" for _, row in df.iterrows()])
        except Exception:
            return "Layer data could not be loaded."

    @component.output_types(layer_json=dict)
    def run(self, query: str):

        messages = [ChatMessage.from_system(self.system_prompt), ChatMessage.from_user(query)]
        result = self.generator.run(messages=messages)
        try:
            # FIX: Changed .content to .text
            layer_data = json.loads(result["replies"][0].text)
            return {"layer_json": layer_data}
        except (json.JSONDecodeError, IndexError):
            return {"layer_json": {"layer": {"name": "error", "id": "error", "confidence": 0.0}}}

