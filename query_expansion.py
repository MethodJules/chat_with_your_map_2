QUICK_MAPPINGS = {
    "Fahrradstationen": ["StadtRAD", "Fahrrad", "Bike"],
    "Bahnhaltestellen": ["S-Bahn", "U-Bahn", "Bahnhof", "Haltestelle", "Bahn"],
    "Bushaltestellen": ["HVV", "Bus", "Haltestelle"],
    "Parkplätze": ["Parkhaus", "Parkplatz", "Parken"],
    "Parkhäuser": ["Parkhaus", "Parkgarage", "Parken"],
    "Behindertenparkplätze": ["Behinderten", "Parkplatz", "Behindertenstellplatz"],
    "Reisebusparkplätze": ["Reisebus", "Bus", "Parkplatz"],
    "Fernbahn": ["Fernbahn", "Bahn", "Zug"],
    "Schmalspurbahn": ["Schmalspurbahn", "Bahn"],
    "Straßenbahn": ["Straßenbahn", "Stadtbahn", "Tram", "Bahn"],
}

def expand_query(entity: str):
  """Expand entity with domain synonyms"""
  # Check if we have a mapping
  if entity in QUICK_MAPPINGS:
    return QUICK_MAPPINGS[entity]

  # Otherwise return original
  return [entity]
