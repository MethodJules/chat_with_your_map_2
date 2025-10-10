import json
from typing import Dict, Any, Optional
from haystack import component


@component
class JSONCombinerComponent:
    """
    Kombiniert die JSON-Ausgaben der Extraktor-Komponenten zu einem einzigen JSON-Objekt.
    """

    @component.output_types(final_json=dict)
    def run(self, raw_query: str, action_json: Optional[dict] = None, layer_json: Optional[dict] = None,
            location_json: Optional[dict] = None):

        final_command = {}

        # Füge die Ergebnisse der einzelnen Komponenten hinzu, wenn sie vorhanden sind
        if action_json and "action" in action_json:
            final_command["action"] = action_json["action"]

        if layer_json and "layers" in layer_json:
            # Wenn es nur einen Layer gibt, füge ihn direkt hinzu, ansonsten die ganze Liste
            if len(layer_json["layers"]) == 1:
                final_command["layer"] = layer_json["layers"][0]
            else:
                final_command["layers"] = layer_json["layers"]

        if location_json and "locations" in location_json:
            # Wenn es nur einen Ort gibt, füge ihn direkt hinzu, ansonsten die ganze Liste
            if len(location_json["locations"]) == 1:
                final_command["location"] = location_json["locations"][0]
            else:
                final_command["locations"] = location_json["locations"]

        # Füge die ursprüngliche Anfrage hinzu
        final_command["rawQuery"] = raw_query

        return {"final_json": {"command": final_command}}

