import json
from typing import Dict

from geopy import Point
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from geopy.distance import geodesic
from .wfs_client import fetch_features_from_wfs
from .location_extractor_component import LocationExtractorComponent
from .layer_extractor_component import LayerExtractorComponent


@component
class BufferAnalysisAgentComponent:
    def __init__(self):
        self.location_extractor = LocationExtractorComponent()
        self.layer_extractor = LayerExtractorComponent(csv_path="./data/csv datei.csv")

    @component.output_types(result_json=Dict)
    def run(self, query: str, radius_km: float = 1.0):
        # Schritt 1: Extrahiere den zentralen Ort und den Ziel-Layer aus der Anfrage
        try:
            parts = query.lower().split(" um ")
            if len(parts) != 2:
                parts = query.lower().split(" in der nähe von ")

            target_layer_query = parts[0].replace("welche", "").strip()
            center_location_query = parts[1].strip().rstrip("?.,!")

        except Exception:
            return {"result_json": {
                "error": "Anfrageformat nicht erkannt. Bitte verwenden Sie 'Finde [Layer] im Umkreis von [Distanz] um [Ort]'."}}

        # Schritt 2: Finde die Koordinaten des zentralen Punktes
        location_result = self.location_extractor.run(query=center_location_query)

        locations = location_result.get("location_json", {}).get("locations")
        if not locations:
            return {
                "result_json": {"error": f"Der zentrale Ort '{center_location_query}' konnte nicht gefunden werden."}}

        center_point_data = locations[0]
        if 'latitude' not in center_point_data or 'longitude' not in center_point_data:
            return {"result_json": {
                "error": f"Für den Ort '{center_location_query}' konnten keine gültigen Koordinaten gefunden werden."}}

        center_coords = (center_point_data['latitude'], center_point_data['longitude'])

        # Schritt 3: Finde den relevanten Layer
        layer_result = self.layer_extractor.run(query=target_layer_query)
        layer_info = layer_result.get("layer_json", {}).get("layers", [{}])[0]
        if not layer_info.get("url"):
            return {"result_json": {"error": f"Layer für '{target_layer_query}' konnte nicht gefunden werden."}}

        # Schritt 4: Lade die Features des Layers
        features_gdf = fetch_features_from_wfs(layer_info["url"], layer_info["feature_type"])
        if features_gdf is None or features_gdf.empty:
            return {"result_json": {"error": f"Keine Daten für den Layer '{layer_info['name']}' gefunden."}}

        # Schritt 5: Finde alle Features innerhalb des Radius
        features_in_radius = []
        for index, feature in features_gdf.iterrows():
            if feature.geometry and feature.geometry.geom_type == 'Point':
                # === KORREKTUR HIER ===
                feature_coords = (feature.geometry.y, feature.geometry.x)
                # ======================

                distance_km = geodesic(center_coords, feature_coords).kilometers
                if distance_km <= radius_km:
                    feature_dict = feature.to_dict()
                    feature_dict['distance_from_center_km'] = round(distance_km, 2)
                    if 'geometry' in feature_dict:
                        feature_dict['geometry'] = str(feature_dict['geometry'])
                    features_in_radius.append(feature_dict)

        result = {
            "analysis_type": "buffer_query",
            "center_location": center_point_data['name'],
            "radius_km": radius_km,
            "target_layer": layer_info['name'],
            "found_features_count": len(features_in_radius),
            "features": features_in_radius
        }
        return {"result_json": result}