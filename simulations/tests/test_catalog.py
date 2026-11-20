"""Unit tests for app.catalog."""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

_SIMULATIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIMULATIONS_DIR not in sys.path:
    sys.path.insert(0, _SIMULATIONS_DIR)

from app import catalog  # noqa: E402


def _write_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class _FakeBlob:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self._text = text

    def download_as_text(self, encoding: str = "utf-8") -> str:
        return self._text


class TestDisplayLabelAndPreview(unittest.TestCase):
    def test_extracts_name_from_fhir_bundle_given_and_family(self) -> None:
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Condition", "id": "c1"}},
                {
                    "resource": {
                        "resourceType": "Patient",
                        "name": [{"given": ["Sarah"], "family": "Levi"}],
                    }
                },
            ],
        }
        self.assertEqual(catalog.extract_display_label(bundle, "sarah-levi-brca1"), "Sarah Levi")

    def test_extracts_name_text_when_present(self) -> None:
        patient = {
            "resourceType": "Patient",
            "name": [{"text": "Dr. Max Mustermann"}],
        }
        self.assertEqual(catalog.extract_display_label(patient, "max-m"), "Dr. Max Mustermann")

    def test_falls_back_to_patient_text_narrative_and_strips_html(self) -> None:
        patient = {
            "resourceType": "Patient",
            "text": {
                "status": "generated",
                "div": '<div xmlns="http://www.w3.org/1999/xhtml"><p>Anna <b>Schmidt</b>, 54y</p></div>',
            },
        }
        self.assertEqual(catalog.extract_display_label(patient, "anna-s"), "Anna Schmidt , 54y")

    def test_falls_back_to_ehr_id_when_no_patient_name_or_text(self) -> None:
        ehr = {"resourceType": "Bundle", "entry": []}
        self.assertEqual(catalog.extract_display_label(ehr, "case-042"), "case-042")

    def test_make_profile_preview_collapses_whitespace_and_truncates(self) -> None:
        long_text = "Sehr besorgte Patientin.\n\n   " + ("Details " * 40)
        preview = catalog.make_profile_preview(long_text, max_chars=50)
        self.assertLessEqual(len(preview), 50)
        self.assertTrue(preview.endswith("..."))
        self.assertNotIn("\n", preview)


class TestLocalCatalogDiscovery(unittest.TestCase):
    def test_discovers_ehr_profiles_and_questions_and_hides_raw_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ehr_bundle = {
                "resourceType": "Bundle",
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "name": [{"given": ["Sarah"], "family": "Levi"}],
                        }
                    }
                ],
                "secretClinicalDetail": "BRCA1 positive, Olaparib 300mg",
            }
            full_profile_0 = "Sehr detailliertes Profil 0. " + ("X" * 300)
            full_profile_1 = "Kurzes Profil 1."

            _write_json(os.path.join(tmpdir, "sarah-levi-brca1.json"), ehr_bundle)
            _write_json(
                os.path.join(tmpdir, "profiles_sarah-levi-brca1.json"),
                {"profiles": [full_profile_0, full_profile_1]},
            )
            # Questions file with trailing comma to verify lenient parsing
            with open(
                os.path.join(tmpdir, "sarah-levi-brca1_questions.json"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write('{"questions": ["Frage 1?", "Frage 2?",]}')
            # Simulation output file should be ignored
            _write_json(
                os.path.join(tmpdir, "simulations_sarah-levi-brca1.json"),
                {"simulations": []},
            )

            cat = catalog.PatientCatalog(data_dir=tmpdir, gcs_data_prefix="")
            summary_list = cat.list_catalog()

            self.assertEqual(len(summary_list), 1)
            item = summary_list[0]
            self.assertEqual(item["ehr_id"], "sarah-levi-brca1")
            self.assertEqual(item["display_label"], "Sarah Levi")
            self.assertTrue(item["has_seed_questions"])
            self.assertEqual(item["seed_question_count"], 2)
            self.assertEqual([p["profile_id"] for p in item["profiles"]], [0, 1])

            # Ensure raw EHR and full long profile text are not leaked in UI summary
            serialized_summary = json.dumps(summary_list)
            self.assertNotIn("secretClinicalDetail", serialized_summary)
            self.assertNotIn(full_profile_0, serialized_summary)

            # Server-side helpers return full data
            self.assertIn("secretClinicalDetail", cat.get_ehr_text("sarah-levi-brca1"))
            self.assertEqual(cat.get_profile_text("sarah-levi-brca1", 0), full_profile_0)
            self.assertEqual(cat.get_profile_text("sarah-levi-brca1", 1), full_profile_1)
            self.assertIn(
                cat.sample_seed_question("sarah-levi-brca1", rng=random.Random(0)),
                {"Frage 1?", "Frage 2?"},
            )

            # Bounds & missing ID errors
            with self.assertRaises(KeyError):
                cat.get_patient("does-not-exist")
            with self.assertRaises(IndexError):
                cat.get_profile_text("sarah-levi-brca1", 2)
            with self.assertRaises(IndexError):
                cat.get_profile_text("sarah-levi-brca1", -1)

    def test_discovers_across_nested_raw_and_results_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_json(
                os.path.join(tmpdir, "raw", "patient-a.json"),
                {"resourceType": "Patient", "name": [{"text": "Patient Alpha"}]},
            )
            _write_json(
                os.path.join(tmpdir, "results", "profiles_patient-a.json"),
                ["Profil A0"],
            )

            cat = catalog.PatientCatalog(data_dir=tmpdir, gcs_data_prefix="")
            items = cat.list_catalog()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["ehr_id"], "patient-a")
            self.assertEqual(items[0]["display_label"], "Patient Alpha")
            self.assertFalse(items[0]["has_seed_questions"])
            self.assertIsNone(cat.sample_seed_question("patient-a"))
            self.assertEqual(cat.get_profile_text("patient-a", 0), "Profil A0")

    def test_missing_local_directory_returns_empty_catalog(self) -> None:
        cat = catalog.PatientCatalog(
            data_dir="/nonexistent/path/for/patient/catalog",
            gcs_data_prefix="",
        )
        self.assertEqual(cat.list_catalog(), [])

    def test_matches_prefix_questions_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_json(
                os.path.join(tmpdir, "Hannah_BRCA2.json"),
                {"resourceType": "Patient", "name": [{"given": ["Hannah"]}]},
            )
            _write_json(
                os.path.join(tmpdir, "profiles_Hannah_BRCA2.json"),
                {"profiles": ["Single Hannah profile"]},
            )
            _write_json(
                os.path.join(tmpdir, "Hannah_questions.json"),
                {"questions": ["Feed question 1?", "Feed question 2?"]},
            )

            cat = catalog.PatientCatalog(data_dir=tmpdir, gcs_data_prefix="")
            items = cat.list_catalog()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["ehr_id"], "Hannah_BRCA2")
            self.assertTrue(items[0]["has_seed_questions"])
            self.assertEqual(items[0]["seed_question_count"], 2)
            self.assertIn(
                cat.sample_seed_question("Hannah_BRCA2", rng=random.Random(0)),
                {"Feed question 1?", "Feed question 2?"},
            )

    def test_bundled_data_dir_has_two_patients_with_single_profile_and_feed_questions(
        self,
    ) -> None:
        bundled_dir = os.path.join(_SIMULATIONS_DIR, "data")
        cat = catalog.PatientCatalog(data_dir=bundled_dir, gcs_data_prefix="")
        items = cat.list_catalog()
        self.assertEqual([item["ehr_id"] for item in items], ["Hannah_BRCA2", "Sarah_BRCA1"])
        for item in items:
            self.assertEqual(len(item["profiles"]), 1)
            self.assertTrue(item["has_seed_questions"])
            self.assertGreater(item["seed_question_count"], 0)
            self.assertIsNotNone(cat.sample_seed_question(item["ehr_id"], rng=random.Random(0)))


class TestGcsCatalogDiscovery(unittest.TestCase):
    def test_loads_catalog_from_gs_uri(self) -> None:
        blobs = [
            _FakeBlob(
                "data/case-1.json",
                json.dumps({"resourceType": "Patient", "name": [{"given": ["Lena"], "family": "K"}]}),
            ),
            _FakeBlob(
                "data/profiles_case-1.json",
                json.dumps({"profiles": ["Profil Lena 0", "Profil Lena 1"]}),
            ),
            _FakeBlob(
                "data/case-1_questions.json",
                json.dumps(["Wie lange dauert die Therapie?"]),
            ),
            _FakeBlob("data/README.txt", "ignored"),
        ]
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_client.list_blobs.return_value = blobs

        cat = catalog.PatientCatalog(
            gcs_data_prefix="gs://my-med-bucket/data/",
            gcs_client=mock_client,
        )

        mock_client.bucket.assert_called_once_with("my-med-bucket")
        mock_client.list_blobs.assert_called_once_with(mock_bucket, prefix="data/")

        items = cat.list_catalog()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["ehr_id"], "case-1")
        self.assertEqual(items[0]["display_label"], "Lena K")
        self.assertEqual(len(items[0]["profiles"]), 2)
        self.assertEqual(cat.sample_seed_question("case-1"), "Wie lange dauert die Therapie?")

    def test_relative_gcs_prefix_requires_bucket(self) -> None:
        with self.assertRaises(ValueError):
            catalog.PatientCatalog(gcs_data_prefix="data/", gcs_bucket="")


if __name__ == "__main__":
    unittest.main()
