import requests
import geopandas as gpd
from io import StringIO
from typing import Optional


def fetch_features_from_wfs(url: str, feature_type: str) -> Optional[gpd.GeoDataFrame]:
    """
    Holt Geodaten von einem WFS-Dienst und gibt sie als GeoDataFrame zurück.
    """
    params = {
        'service': 'WFS',
        'version': '2.0.0',
        'request': 'GetFeature',
        'typeName': feature_type,
        'outputFormat': 'application/json'  # GeoJSON ist einfacher zu parsen als GML
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()  # Wirft einen Fehler bei HTTP-Statuscodes 4xx/5xx

        # Geopandas kann GeoJSON direkt lesen
        gdf = gpd.read_file(StringIO(response.text))
        return gdf
    except requests.exceptions.RequestException as e:
        print(f"Fehler bei der WFS-Anfrage für {feature_type}: {e}")
        return None
    except Exception as e:
        print(f"Fehler beim Parsen der WFS-Antwort für {feature_type}: {e}")
        return None
