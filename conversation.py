import os
import json
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.dataclasses import ChatMessage

from components.action_extractor_component import ActionExtractorComponent
from components.layer_extractor_component import LayerExtractorComponent
from components.location_extractor_component import LocationExtractorComponent
from components.combiner_component import JSONCombinerComponent
from components.query_router import QueryRouterComponent
from components.conversational_responder import ConversationalResponderComponent
from components.distance_calculator_component import DistanceCalculatorComponent
from components.proximity_finder import ProximityAgentComponent
from components.attribute_filter_component import AttributeFilterAgentComponent
from components.buffer_analysis_component import BufferAnalysisAgentComponent
from components.routing_component import RoutingAgentComponent




class ConversationState:
    """Eine einfache Klasse, um den Gesprächszustand zu speichern."""

    def __init__(self):
        self.chat_history = []
        self.last_gis_result = {}

    def add_message(self, message: ChatMessage):
        self.chat_history.append(message)


# --- Code wird jetzt hier direkt ausgeführt ---


# Initialisiere die Pipelines und den Zustand
state = ConversationState()

# Pipeline für die GIS-Datenextraktion
gis_pipeline = Pipeline()
gis_pipeline.add_component("action_extractor", ActionExtractorComponent())
gis_pipeline.add_component("layer_extractor", LayerExtractorComponent(csv_path="./data/csv datei.csv"))
gis_pipeline.add_component("location_extractor", LocationExtractorComponent())
gis_pipeline.add_component("json_combiner", JSONCombinerComponent())
gis_pipeline.connect("action_extractor.action_json", "json_combiner.action_json")
gis_pipeline.connect("layer_extractor.layer_json", "json_combiner.layer_json")
gis_pipeline.connect("location_extractor.location_json", "json_combiner.location_json")
# Einzelne Komponenten für die dynamische Steuerung
router = QueryRouterComponent()
responder = ConversationalResponderComponent()
distance_calculator = DistanceCalculatorComponent()
location_extractor = LocationExtractorComponent()
proximity_agent = ProximityAgentComponent()
attribute_filter_agent = AttributeFilterAgentComponent()
buffer_analysis_agent = BufferAnalysisAgentComponent()

print("Conversational GIS Agent gestartet. Geben Sie 'exit' ein, um zu beenden.")
print("-" * 30)

while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        break

    state.add_message(ChatMessage.from_user(user_input))
    intent = router.run(chat_history=state.chat_history).get("intent", "unknown")
    print(f"[Debug: Intent is '{intent}']")

    response_text = ""
    try:
        if intent == "gis_query":
            gis_result = gis_pipeline.run({
                "action_extractor": {"query": user_input},
                "layer_extractor": {"query": user_input},
                "location_extractor": {"query": user_input},
                "json_combiner": {"raw_query": user_input}
            })
            final_json = gis_result["json_combiner"]["final_json"]
            state.last_gis_result = final_json
            response_text = json.dumps(final_json, indent=2, ensure_ascii=False)
            state.add_message(ChatMessage.from_assistant("Ich habe die angeforderten Geodaten gefunden."))

        elif intent == "distance_query":
            locations = location_extractor.run(query=user_input).get("location_json", {}).get("locations", [])
            if len(locations) >= 2:
                result = distance_calculator.run(locations=locations)
                response_text = result["distance_json"]["readable_response"]
                state.last_gis_result = result["distance_json"]
            else:
                response_text = "Ich konnte nicht genügend Orte finden, um die Distanz zu messen."
            state.add_message(ChatMessage.from_assistant(response_text))

        elif intent == "proximity_query":
            result = proximity_agent.run(query=user_input)
            response_text = json.dumps(result["result_json"], indent=2, ensure_ascii=False)
            state.last_gis_result = result["result_json"]
            state.add_message(ChatMessage.from_assistant("Ich habe das nächstgelegene Objekt gefunden."))

        elif intent == "filter_query":
            result = attribute_filter_agent.run(query=user_input)
            response_text = json.dumps(result["result_json"], indent=2, ensure_ascii=False)
            state.last_gis_result = result["result_json"]
            state.add_message(ChatMessage.from_assistant("Hier sind die gefilterten Ergebnisse."))

        elif intent == "buffer_query":
            result = buffer_analysis_agent.run(query=user_input)
            response_text = json.dumps(result["result_json"], indent=2, ensure_ascii=False)
            state.last_gis_result = result["result_json"]
            state.add_message(ChatMessage.from_assistant("Hier ist die Umkreisanalyse."))


        elif intent == "follow_up_question":
            result = responder.run(chat_history=state.chat_history, last_gis_result=state.last_gis_result)
            response_text = result["response"]
            state.add_message(ChatMessage.from_assistant(response_text))

        else:
            response_text = "Entschuldigung, das habe ich nicht verstanden."
            state.add_message(ChatMessage.from_assistant(response_text))

    except Exception as e:
        response_text = f"Ein Fehler ist aufgetreten: {e}"
        print(f"[ERROR]: {e}")
        state.add_message(ChatMessage.from_assistant("Es ist ein interner Fehler aufgetreten."))

    print(f"\nAgent:\n{response_text}\n")

print("Agent wird beendet. Auf Wiedersehen!")


