import json


def json_to_key_value_table(json_string: str):

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        print("Fehler: Der übergebene String ist kein valides JSON.")
        return

    def flatten_json(nested_dict, parent_key='', sep='.'):
        items = []
        for key, value in nested_dict.items():
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, dict):
                items.extend(flatten_json(value, new_key, sep=sep))
            else:
                items.append((new_key, value))
        return items

    flat_list = flatten_json(data)
    if not flat_list:
        print("Das JSON-Objekt ist leer.")
        return



    return flat_list

def format_json_string(input_string: str) -> str:

    try:

        parsed_dict = json.loads(input_string)

        formatted_json = json.dumps(parsed_dict, indent=2, ensure_ascii=False)

        return formatted_json
    except json.JSONDecodeError as e:
        return f"Fehler beim Parsen des JSON-Strings: {e}"


