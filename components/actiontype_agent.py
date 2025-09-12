from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.validators import JsonSchemaValidator
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack import Pipeline

system_prompt = """
Du bist ein Stadtplaner und nutzt eine webbasierte GIS Anwendung. Diese Anwendung kann verschiedene Aktionen ausführen. Diese sind:

 Messen
 Zeigen
 Öffnen
 Zoomen
 Berechnen
 Aktivieren

 Deine Aufgabe ist es, aus einem deutschen Text die richtige Aktion zuzuordnen.

 Antworte im JSON-Format mit den Attributen
 action: Aktionsklasse
 confidence: wie viel ist der prozent von Information Sicherung

 Beispiel:
 
 Text: "Zoome um den Faktor 5 ?"
 Antwort (Json):{
 "action": {"type": "zoomen", "confidence": 0.95},
 "options": {"zoom": 5}

 Text: "Wie lang fließt die Elbe durch Altona?"
 Antwort (JSON): {
 "action": "Messen","confidence": 0.90
 }
"""
schema = {
    "type": "object",
    "properties": {
        "action": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["type", "confidence"]
            },
        "options": {
            "type": "object"
        }
    },
    "required": ["action"]
}

input_text = "Zeige mir alle Fahrradstationen in Altona."

generator = OpenAIChatGenerator(
    api_key=Secret.from_token("eyJ0eXAiOiJKV1QiLCJraWQiOiJkMTcxZGEwNi1hNzg2LTQwYzctYmQzNS1kODE3YmZkNDQ0MmQiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpb25vc2Nsb3VkIiwiaWF0IjoxNzU3MzM3MTI2LCJjbGllbnQiOiJVU0VSIiwiaWRlbnRpdHkiOnsiaXNQYXJlbnQiOmZhbHNlLCJjb250cmFjdE51bWJlciI6MzU1NjEwNTQsInJvbGUiOiJvd25lciIsInJlZ0RvbWFpbiI6Imlvbm9zLmRlIiwicmVzZWxsZXJJZCI6MSwidXVpZCI6IjE1MDUyN2I2LTU5YjYtNDUyYi1hOTdlLWI5MGE4NDI1YWE2NiIsInByaXZpbGVnZXMiOlsiREFUQV9DRU5URVJfQ1JFQVRFIiwiU05BUFNIT1RfQ1JFQVRFIiwiSVBfQkxPQ0tfUkVTRVJWRSIsIk1BTkFHRV9EQVRBUExBVEZPUk0iLCJBQ0NFU1NfQUNUSVZJVFlfTE9HIiwiUENDX0NSRUFURSIsIkFDQ0VTU19TM19PQkpFQ1RfU1RPUkFHRSIsIkJBQ0tVUF9VTklUX0NSRUFURSIsIkNSRUFURV9JTlRFUk5FVF9BQ0NFU1MiLCJLOFNfQ0xVU1RFUl9DUkVBVEUiLCJGTE9XX0xPR19DUkVBVEUiLCJBQ0NFU1NfQU5EX01BTkFHRV9NT05JVE9SSU5HIiwiQUNDRVNTX0FORF9NQU5BR0VfQ0VSVElGSUNBVEVTIiwiQUNDRVNTX0FORF9NQU5BR0VfTE9HR0lORyIsIk1BTkFHRV9EQkFBUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0ROUyIsIk1BTkFHRV9SRUdJU1RSWSIsIkFDQ0VTU19BTkRfTUFOQUdFX0NETiIsIkFDQ0VTU19BTkRfTUFOQUdFX1ZQTiIsIkFDQ0VTU19BTkRfTUFOQUdFX0FQSV9HQVRFV0FZIiwiQUNDRVNTX0FORF9NQU5BR0VfTkdTIiwiQUNDRVNTX0FORF9NQU5BR0VfS0FBUyIsIkFDQ0VTU19BTkRfTUFOQUdFX05FVFdPUktfRklMRV9TVE9SQUdFIiwiQUNDRVNTX0FORF9NQU5BR0VfQUlfTU9ERUxfSFVCIiwiQ1JFQVRFX05FVFdPUktfU0VDVVJJVFlfR1JPVVBTIiwiQUNDRVNTX0FORF9NQU5BR0VfSUFNX1JFU09VUkNFUyJdfSwiZXhwIjoxNzU3OTQxOTI2fQ.jXva7D1p3rQBwEE0qVSkJkGqDRFOTEWIm4GBxlgRm2ciVFgG5dlu_DYllvqItlOLLUnvBcqPssULF30FJlVhzLyjv0UvG_TBKxQZY9rCLqO8PGzcAZPI4Zc32vwCEHgkZt6oST-EWDkyLL7dS2AlbU9OvlQISwKbj2uGdcnxP1rTGCID10AYsCZfjWUBA6mGKHfngB1YxNGZPZAqDn2YUG-kYcr5U83wLrvxr5dE03428Tfb6-dMQ-CMqFp9WaOvlAmg2Lpwsi6Iz3CUljJEBVKmeLDXeaq8GxvDK3dVkrZL0G0E95t1pJN1pOxO_3ZfJKqU1luRjTBYwpSjBq8zeg"),
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_base_url="https://openai.inference.de-txl.ionos.com/v1",
    generation_kwargs = {"response_format": {"type": "json_object"}}
)

validator = JsonSchemaValidator(json_schema=schema)

pipeline = Pipeline()
pipeline.add_component("generator", generator)
pipeline.add_component("validator", validator)
pipeline.connect("generator.replies", "validator.messages")

result = pipeline.run({
    "generator": {
        "messages": [ChatMessage.from_system(system_prompt), ChatMessage.from_user(input_text)]
    }
})

print(result)

if result["validator"]["validated"]:
    print("Valid JSON:", result["validator"]["validated"][0].text)
else:
    print("Validation error:", result["validator"]["validation_error"][0].text)