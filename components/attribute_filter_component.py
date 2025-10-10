import json
import pandas as pd
from typing import Dict, List
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from .wfs_client import fetch_features_from_wfs
from .layer_extractor_component import LayerExtractorComponent


@component
class AttributeFilterAgentComponent:
    def __init__(self):
        self.layer_extractor = LayerExtractorComponent(csv_path="./data/csv datei.csv")
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(
                "eyJ0eXAiOiJKV1QiLCJraWQiOiI0MGI3YjZmYy03ZjI2LTRmMjgtOWI4Ni00ZTNjMjY2NTQ4Y2IiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpb25vc2Nsb3VkIiwiaWF0IjoxNzU0OTIzNTkyLCJjbGllbnQiOiJVU0VSIiwiaWRlbnRpdHkiOnsiaXNQYXJlbnQiOmZhbHNlLCJwcml2aWxlZ2VzIjpbIkRBVEFfQ0VOVEVSX0NSRUFURSIsIlNOQVBTSE9UX0NSRUFURSIsIklQX0JMT0NLX1JFU0VSVkUiLCJNQU5BR0VfREFUQVBMQVRGT1JNIiwiQUNDRVNTX0FDVElWSVRZX0xPRyIsIlBDQ19DUkVBVEUiLCJBQ0NFU1NfUzNfT0JKRUNUX1NUT1JBR0UiLCJCQUNLVVBfVU5JVF9DUkVBVEUiLCJDUkVBVEVfSU5URVJORVRfQUNDRVNTIiwiSzhTX0NMVVNURVJfQ1JFQVRFIiwiRkxPV19MT0dfQ1JFQVRFIiwiQUNDRVNTX0FORF9NQU5BR0VfTU9OSVRPUklORyIsIkFDQ0VTU19BTkRfTUFOQUdFX0NFUlRJRklDQVRFUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0xPR0dJTkciLCJNQU5BR0VfREJBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ETlMiLCJNQU5BR0VfUkVHSVNUUlkiLCJBQ0NFU1NfQU5EX01BTkFHRV9DRE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9WUE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9BUElfR0FURVdBWSIsIkFDQ0VTU19BTkRfTUFOQUdFX05HUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0tBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ORVRXT1JLX0ZJTEVfU1RPUkFHRSIsIkFDQ0VTU19BTkRfTUFOQUdFX0FJX01PREVMX0hVQiIsIkNSRUFURV9ORVRXT1JLX1NFQ1VSSVRZX0dST1VQUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0lBTV9SRVNPVVJDRVMiXSwidXVpZCI6IjE1MDUyN2I2LTU5YjYtNDUyYi1hOTdlLWI5MGE4NDI1YWE2NiIsInJlc2VsbGVySWQiOjEsInJlZ0RvbWFpbiI6Imlvbm9zLmRlIiwicm9sZSI6Im93bmVyIiwiY29udHJhY3ROdW1iZXIiOjM1NTYxMDU0fSwiZXhwIjoxNzYyNjk5NTkyfQ.G-Su-oYRGZCCKfgrL0n-a2SLR4uD6Qobp0VBIbZxUZDm8yX6w7TjH5eCyLm_-8Rcwb-USdk8zhDYCYpVoZFY-gAWgGYdz_DIUFBKc3-71SOftFRdPN_IYoDVCBlhtOudyoKW5lh0nrCemRHwCrXMqVtp5U7gZZNMEEf1MZadMtTAh-8rVAW4GxEwEAcj99X3kuoJaos6wgU-6G8dPUFFBWX1vRlLsdDQafvCxy55FpqMi07oyhlpYNZTiFgBKc_-vLRMzI9cOWmQ-R5CaUNWk_Ry_g-wjxAntZadT0dcVQ-3OripfrZIVYk3mpnMExW5sfL6aS2FgrKBKryA0-pnUg"),
            model="meta-llama/Llama-3.3-70B-Instruct",
            api_base_url="https://openai.inference.de-txl.ionos.com/v1",
            generation_kwargs={"response_format": {"type": "json_object"}}
        )
        self.system_prompt = """
        Deine Aufgabe ist es, aus einer Benutzeranfrage Filterkriterien zu extrahieren.
        Analysiere den Text und gib ein JSON-Objekt zurück, das den Layernamen und eine Liste von Filtern enthält.
        Jeder Filter muss 'property', 'operator' und 'value' enthalten.
        Mögliche Operatoren: 'greaterThan', 'lessThan', 'equals', 'contains'.

        Beispiel 1:
        User: "Finde Park+Ride Anlagen mit mehr als 200 Plätzen."
        Antwort:
        {
          "layer_query": "Park+Ride Anlagen",
          "filters": [
            { "property": "anzahl_stellplaetze", "operator": "greaterThan", "value": 200 }
          ]
        }

        Beispiel 2:
        User: "Gibt es Schulen mit dem Namen 'Goethe' in Hamburg?"
        Antwort:
        {
          "layer_query": "Schulen",
          "filters": [
            { "property": "schulname", "operator": "contains", "value": "Goethe" }
          ]
        }
        """

    @component.output_types(result_json=Dict)
    def run(self, query: str):
        # Schritt 1: LLM zur Extraktion von Filtern und Layer aufrufen
        messages = [ChatMessage.from_system(self.system_prompt), ChatMessage.from_user(query)]
        try:
            result = self.generator.run(messages=messages)
            filter_data = json.loads(result["replies"][0].text)
            layer_query = filter_data.get("layer_query")
            filters = filter_data.get("filters", [])
        except (json.JSONDecodeError, IndexError, KeyError):
            return {"result_json": {"error": "Filter konnten nicht extrahiert werden."}}

        # Schritt 2: Layer-Daten holen
        layer_result = self.layer_extractor.run(query=layer_query)
        layer_info = layer_result.get("layer_json", {}).get("layers", [{}])[0]
        if not layer_info.get("url"):
            return {"result_json": {"error": f"Layer für '{layer_query}' nicht gefunden."}}

        # Schritt 3: Alle Features des Layers abrufen
        gdf = fetch_features_from_wfs(layer_info["url"], layer_info["feature_type"])
        if gdf is None or gdf.empty:
            return {"result_json": {"error": f"Keine Daten für Layer '{layer_info['name']}' gefunden."}}

        # Schritt 4: Filter anwenden
        filtered_gdf = gdf
        for f in filters:
            prop = f.get("property")
            op = f.get("operator")
            val = f.get("value")

            if prop not in filtered_gdf.columns:
                continue  # Eigenschaft im Datensatz nicht gefunden

            # Konvertiere Spalte zu numerisch wenn möglich
            filtered_gdf[prop] = pd.to_numeric(filtered_gdf[prop], errors='coerce')

            if op == 'greaterThan':
                filtered_gdf = filtered_gdf[filtered_gdf[prop] > val]
            elif op == 'lessThan':
                filtered_gdf = filtered_gdf[filtered_gdf[prop] < val]
            elif op == 'equals':
                filtered_gdf = filtered_gdf[filtered_gdf[prop] == val]
            elif op == 'contains':
                # Konvertiere Spalte zu String für 'contains'
                filtered_gdf[prop] = filtered_gdf[prop].astype(str)
                filtered_gdf = filtered_gdf[filtered_gdf[prop].str.contains(str(val), case=False, na=False)]

        # Schritt 5: Ergebnis formatieren
        result = {
            "layer": layer_info,
            "applied_filters": filters,
            "feature_count": len(filtered_gdf),
            "features": json.loads(filtered_gdf.to_json())
        }
        return {"result_json": result}
