import json

from haystack import Pipeline
from components.action_extractor_component import ActionExtractorComponent
from components.layer_extractor_component import LayerExtractorComponent
from components.location_extractor_component import LocationExtractorComponent
from components.combiner_component import JSONCombinerComponent

USER_QUERY = "Zeige mir alle Parkplätze und Bahnhalte sowie Fahrradparkplätze in Altona und auch in Wandsbek."
CSV_PATH = "./data/csv datei.csv"

master_pipeline = Pipeline()

master_pipeline.add_component("action_extractor", ActionExtractorComponent())
master_pipeline.add_component("layer_extractor", LayerExtractorComponent(csv_path=CSV_PATH))
master_pipeline.add_component("location_extractor", LocationExtractorComponent())
master_pipeline.add_component("json_combiner", JSONCombinerComponent())


master_pipeline.connect("action_extractor.action_json", "json_combiner.action_json")
master_pipeline.connect("layer_extractor.layer_json", "json_combiner.layer_json")
master_pipeline.connect("location_extractor.location_json", "json_combiner.location_json")

print(f"Processing query: '{USER_QUERY}'")

result = master_pipeline.run({
    "action_extractor": {"query": USER_QUERY},
    "layer_extractor": {"query": USER_QUERY},
    "location_extractor": {"query": USER_QUERY},
    "json_combiner": {"raw_query": USER_QUERY}
})

print("\n--- Final Consolidated JSON Output ---")
final_output = result["json_combiner"]["final_json"]
print(json.dumps(final_output, indent=2, ensure_ascii=False))
print("-" * 38)
