from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.validators import JsonSchemaValidator
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack import Pipeline

system_prompt_dataset = """
Du bist ein Experte für Geoinformationssysteme (GIS). Deine Aufgabe ist es, den folgenden deutschen Text zu analysieren und den am besten passenden GIS-Layer aus der bereitgestellten Liste zu identifizieren.

Verfügbare Layer:
{layer_list}

Anweisungen:
Wähle den relevantesten Layer aus und bestimme einen Konfidenzwert. Gib deine Antwort ausschließlich als JSON-Objekt im definierten Format aus. Füge keine zusätzlichen Erklärungen hinzu.

Ausgabeformat (JSON):
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
{
  "layer": {
    "name": "Fahrradstationen",
    "id": "18105",
    "confidence": 0.92
  }
}
"""

schema = {
"type": "object",
"properties": {
    "layer":{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "id": {"type": "string"},
            "confidence": {"type": "number"}
        }
    }
}}

input_text = "Zeige mir alle Fahrradstationen in Altona."

# Configure the generator to use IONOS AI Model Hub
generator = OpenAIChatGenerator(
    api_key=Secret.from_token("eyJ0eXAiOiJKV1QiLCJraWQiOiI0MGI3YjZmYy03ZjI2LTRmMjgtOWI4Ni00ZTNjMjY2NTQ4Y2IiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpb25vc2Nsb3VkIiwiaWF0IjoxNzU0OTIzNTkyLCJjbGllbnQiOiJVU0VSIiwiaWRlbnRpdHkiOnsiaXNQYXJlbnQiOmZhbHNlLCJwcml2aWxlZ2VzIjpbIkRBVEFfQ0VOVEVSX0NSRUFURSIsIlNOQVBTSE9UX0NSRUFURSIsIklQX0JMT0NLX1JFU0VSVkUiLCJNQU5BR0VfREFUQVBMQVRGT1JNIiwiQUNDRVNTX0FDVElWSVRZX0xPRyIsIlBDQ19DUkVBVEUiLCJBQ0NFU1NfUzNfT0JKRUNUX1NUT1JBR0UiLCJCQUNLVVBfVU5JVF9DUkVBVEUiLCJDUkVBVEVfSU5URVJORVRfQUNDRVNTIiwiSzhTX0NMVVNURVJfQ1JFQVRFIiwiRkxPV19MT0dfQ1JFQVRFIiwiQUNDRVNTX0FORF9NQU5BR0VfTU9OSVRPUklORyIsIkFDQ0VTU19BTkRfTUFOQUdFX0NFUlRJRklDQVRFUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0xPR0dJTkciLCJNQU5BR0VfREJBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ETlMiLCJNQU5BR0VfUkVHSVNUUlkiLCJBQ0NFU1NfQU5EX01BTkFHRV9DRE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9WUE4iLCJBQ0NFU1NfQU5EX01BTkFHRV9BUElfR0FURVdBWSIsIkFDQ0VTU19BTkRfTUFOQUdFX05HUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0tBQVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9ORVRXT1JLX0ZJTEVfU1RPUkFHRSIsIkFDQ0VTU19BTkRfTUFOQUdFX0FJX01PREVMX0hVQiIsIkNSRUFURV9ORVRXT1JLX1NFQ1VSSVRZX0dST1VQUyIsIkFDQ0VTU19BTkRfTUFOQUdFX0lBTV9SRVNPVVJDRVMiXSwidXVpZCI6IjE1MDUyN2I2LTU5YjYtNDUyYi1hOTdlLWI5MGE4NDI1YWE2NiIsInJlc2VsbGVySWQiOjEsInJlZ0RvbWFpbiI6Imlvbm9zLmRlIiwicm9sZSI6Im93bmVyIiwiY29udHJhY3ROdW1iZXIiOjM1NTYxMDU0fSwiZXhwIjoxNzYyNjk5NTkyfQ.G-Su-oYRGZCCKfgrL0n-a2SLR4uD6Qobp0VBIbZxUZDm8yX6w7TjH5eCyLm_-8Rcwb-USdk8zhDYCYpVoZFY-gAWgGYdz_DIUFBKc3-71SOftFRdPN_IYoDVCBlhtOudyoKW5lh0nrCemRHwCrXMqVtp5U7gZZNMEEf1MZadMtTAh-8rVAW4GxEwEAcj99X3kuoJaos6wgU-6G8dPUFFBWX1vRlLsdDQafvCxy55FpqMi07oyhlpYNZTiFgBKc_-vLRMzI9cOWmQ-R5CaUNWk_Ry_g-wjxAntZadT0dcVQ-3OripfrZIVYk3mpnMExW5sfL6aS2FgrKBKryA0-pnUg"),
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_base_url="https://openai.inference.de-txl.ionos.com/v1",
    generation_kwargs = {"response_format": {"type": "json_object"}}
)

validator = JsonSchemaValidator(json_schema=schema)

# Create pipeline
pipeline = Pipeline()
pipeline.add_component("generator", generator)
pipeline.add_component("validator", validator)
pipeline.connect("generator.replies", "validator.messages")

# Run pipeline
result = pipeline.run({
    "generator": {
        "messages": [ChatMessage.from_system(system_prompt_dataset), ChatMessage.from_user(input_text)]
    }
})

print(result)

# Check results
if result["validator"]["validated"]:
    print("Valid JSON:", result["validator"]["validated"][0].text)
else:
    print("Validation error:", result["validator"]["validation_error"][0].text)


