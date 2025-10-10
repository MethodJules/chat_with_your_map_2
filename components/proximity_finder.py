import json
import pandas as pd
from typing import Dict
from haystack import component
from geopy.distance import geodesic
from .wfs_client import fetch_features_from_wfs
from .location_extractor_component import LocationExtractorComponent
from .layer_extractor_component import LayerExtractorComponent


@component
class ProximityAgentComponent:
    def __init__(self):
        self.location_extractor = LocationExtractorComponent()
        self.layer_extractor = LayerExtractorComponent(csv_path="./data/csv datei.csv")

    @component.output_types(result_json=Dict)
    def run(self, query: str):
        # Schritt 1: Extrahiere Startort und Ziel-Layer mit einem LLM (vereinfacht hier)
        # In einer echten Anwendung würde hier eine LLM-Komponente stehen
        # Wir nehmen an, die Anfrage hat das Format "Finde [Layer] bei [Ort]"

        # Vereinfachte Extraktion - für eine robuste Lösung LLM verwenden
        parts = query.lower().split(" von ")
        if len(parts) != 2:
            parts = query.lower().split(" bei ")
            if len(parts) != 2:
                parts = query.lower().split(" in der nähe von ")
                if len(parts) != 2:
                    return {"result_json": {
                        "error": "Anfrageformat nicht erkannt. Bitte verwenden Sie 'Finde [Layer] von/bei/in der Nähe von [Ort]'.'"}}

        target_layer_query = parts[0].replace("nächste", "").replace("nächsten", "").strip()
        start_location_query = parts[1].strip()

        # Schritt 2: Koordinaten des Startpunkts
        location_result = self.location_extractor.run(query=start_location_query)
        start_locations = location_result.get("location_json", {}).get("locations")
        if not start_locations:
            return {"result_json": {"error": f"Startort '{start_location_query}' nicht gefunden."}}
        start_point = start_locations[0]
        start_coords = (start_point['latitude'], start_point['longitude'])

        # Schritt 3: Layer-Daten holen
        layer_result = self.layer_extractor.run(query=target_layer_query)
        layer_info = layer_result.get("layer_json", {}).get("layers", [{}])[0]
        if not layer_info.get("url"):
            return {"result_json": {"error": f"Layer für '{target_layer_query}' nicht gefunden."}}

        # Schritt 4: Alle Features des Ziel-Layers abrufen
        features_gdf = fetch_features_from_wfs(layer_info["url"], layer_info["feature_type"])
        if features_gdf is None or features_gdf.empty:
            return {"result_json": {"error": f"Keine Daten für Layer '{layer_info['name']}' gefunden."}}

        # Schritt 5: Nächstes Feature finden
        nearest_feature = None
        min_distance = float('inf')

        for index, feature in features_gdf.iterrows():
            # Annahme: Geometrie ist ein Punkt
            if feature.geometry.geom_type == 'Point':
                feature_coords = (feature.geometry.y, feature.geometry.x)
                distance = geodesic(start_coords, feature_coords).meters
                if distance < min_distance:
                    min_distance = distance
                    nearest_feature = feature.to_dict()

        if nearest_feature:
            feature_name = nearest_feature.get('name', nearest_feature.get('Name', 'N/A'))
            result = {
                "nearest_feature": {
                    "from_location": start_point['name'],
                    "target_layer": layer_info['name'],
                    "feature_name": feature_name,
                    "distance": {
                        "meters": round(min_distance, 2),
                        "kilometers": round(min_distance / 1000, 2),
                    },
                    "geometry": nearest_feature.get('geometry')
                }
            }
            return {"result_json": result}
        else:
            return {"result_json": {"error": "Kein nächstgelegenes Feature gefunden."}}
