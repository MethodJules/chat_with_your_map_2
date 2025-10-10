from typing import List, Dict, Any
from haystack import component
from geopy.distance import geodesic


@component
class DistanceCalculatorComponent:
    """
    Berechnet die Distanz zwischen zwei geografischen Koordinaten.
    """

    @component.output_types(distance_json=dict)
    def run(self, locations: List[Dict[str, Any]]):
        """
        Nimmt eine Liste von Orten entgegen und berechnet die Distanz zwischen den ersten beiden.

        :param locations: Eine Liste von Location-Objekten, die vom LocationExtractorComponent generiert wurden.
        :return: Ein Dictionary mit den Distanzinformationen oder einer Fehlermeldung.
        """
        if not locations or len(locations) < 2:
            return {"distance_json": {"error": "Nicht genügend Orte für die Distanzmessung gefunden."}}

        try:
            # Nimm die ersten beiden gefundenen Orte
            point1_data = locations[0]
            point2_data = locations[1]

            point1_coords = (point1_data['coordinates']['latitude'], point1_data['coordinates']['longitude'])
            point2_coords = (point2_data['coordinates']['latitude'], point2_data['coordinates']['longitude'])

            # Berechne die Distanz
            distance_km = geodesic(point1_coords, point2_coords).kilometers
            distance_meters = distance_km * 1000

            result = {
                "distance": {
                    "from": point1_data['name'],
                    "to": point2_data['name'],
                    "meters": round(distance_meters, 2),
                    "kilometers": round(distance_km, 2),
                    "unit": "meters"
                },
                "readable_response": f"Die Distanz zwischen {point1_data['name']} und {point2_data['name']} beträgt {round(distance_km, 2)} km."
            }
            return {"distance_json": result}

        except (KeyError, IndexError) as e:
            return {"distance_json": {"error": f"Fehlende Koordinaten in den Eingabedaten: {e}"}}
        except Exception as e:
            return {"distance_json": {"error": f"Ein unerwarteter Fehler ist aufgetreten: {e}"}}
