import json
import uuid
from datetime import datetime
from haystack import component


@component
class JsonCombinerComponent:
    #A component that merges JSON outputs from other components into a final structure.
    @component.output_types(final_json=dict)
    def run(self, action_json: dict, layer_json: dict, location_json: dict, raw_query: str):

        command = {
            **action_json,
            **layer_json,
            **location_json,
            "rawQuery": raw_query
        }

        # Remove empty keys if a component failed
        if not command.get("action"): command.pop("action", None)
        if not command.get("layer"): command.pop("layer", None)
        if not command.get("location"): command.pop("location", None)

        final_structure = {
            "command": command,
            "metadata": {
                "requestId": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        }
        return {"final_json": final_structure}
