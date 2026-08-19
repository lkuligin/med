import json
import os


def create_ehr_hint(ehr_path: str) -> str:
    """Reads EHR data from the given path and returns a formatted hint string."""
    if not os.path.exists(ehr_path):
        return f"EHR file not found at {ehr_path}"
    try:
        with open(ehr_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return json.dumps(data, ensure_ascii=False, indent=2)
            return str(data)
    except Exception as e:
        return f"Error loading EHR file {ehr_path}: {e}"
