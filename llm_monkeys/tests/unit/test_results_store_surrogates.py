"""A stored record survives text a model produced with a lone surrogate."""

import json

from results_store import _write_json


def test_a_lone_surrogate_is_stored_as_a_replacement_character(tmp_path):
    path = _write_json(tmp_path / "r.json", {"raw_response": "ok \udc5b ok \U0001F600"})
    assert json.loads(path.read_text(encoding="utf-8"))["raw_response"] == "ok � ok \U0001F600"
