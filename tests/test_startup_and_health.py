"""Unit tests for Phase 9: Safe API Startup and Health Subsystem Reporting.

Verifies:
- Missing Mark 1 dataset does not prevent application startup.
- Mark 2 and Mark 3 startup functions are still executed.
- /health accurately reflects readiness for each subsystem.
- Mark 1 dataset-dependent endpoints return HTTP 503 when the dataset is unavailable.
- Existing Mark 2 and Mark 3 endpoints remain unaffected by the startup change.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import api.main as api


class SafeApiStartupTests(unittest.TestCase):
    def setUp(self):
        # Save original states to restore in tearDown
        self.orig_model = api.model
        self.orig_model_loaded = api.model_loaded
        self.orig_dataset = api.dataset
        self.orig_dataset_loaded = api.dataset_loaded
        self.orig_test_ids = api.test_ids
        self.orig_mark2_loaded = api.mark2_loaded
        self.orig_mark2_fusion = api.mark2_fusion_model
        self.orig_mark2_text = api.mark2_text_model
        self.orig_mark3_loaded = api.mark3_loaded
        self.orig_mark3_pipeline = api.mark3_raw_pipeline
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def tearDown(self):
        api.model = self.orig_model
        api.model_loaded = self.orig_model_loaded
        api.dataset = self.orig_dataset
        api.dataset_loaded = self.orig_dataset_loaded
        api.test_ids = self.orig_test_ids
        api.mark2_loaded = self.orig_mark2_loaded
        api.mark2_fusion_model = self.orig_mark2_fusion
        api.mark2_text_model = self.orig_mark2_text
        api.mark3_loaded = self.orig_mark3_loaded
        api.mark3_raw_pipeline = self.orig_mark3_pipeline

    def test_missing_dataset_does_not_prevent_startup(self):
        """Startup event must succeed gracefully even when Mark 1 dataset files are missing."""
        api.dataset = None
        api.dataset_loaded = False
        api.test_ids = None

        # Execute startup_event without throwing any exceptions
        try:
            api.startup_event()
        except Exception as exc:
            self.fail(f"startup_event() raised unexpected exception: {exc}")

        self.assertFalse(api.dataset_loaded)
        self.assertIsNone(api.dataset)
        self.assertIsNone(api.test_ids)

    def test_mark1_model_loading_state_is_separate_from_dataset(self):
        """Model loaded state must not be tied to dataset loaded state."""
        api.model_loaded = True
        api.dataset_loaded = False

        self.assertTrue(api.model_loaded)
        self.assertFalse(api.dataset_loaded)

    def test_mark2_and_mark3_startup_still_called_even_if_mark2_fails(self):
        """If load_mark2 fails, load_mark3 must still be attempted."""
        with (
            patch("api.main.load_mark2", side_effect=RuntimeError("Simulated Mark 2 failure")) as mock_m2,
            patch("api.main.load_mark3") as mock_m3,
        ):
            api.startup_mark2()
            mock_m2.assert_called_once()
            mock_m3.assert_called_once()

    def test_health_reports_dataset_readiness_accurately(self):
        """Health endpoint must accurately report readiness flags for each subsystem."""
        api.model = object()
        api.model_loaded = True
        api.dataset = None
        api.dataset_loaded = False
        api.mark2_loaded = True
        api.mark2_fusion_model = object()
        api.mark2_text_model = object()
        api.mark3_loaded = True
        api.mark3_raw_pipeline = object()

        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertFalse(data["dataset_loaded"])
        self.assertTrue(data["mark2_loaded"])
        self.assertTrue(data["mark3_loaded"])

    def test_health_reports_subsystem_unavailable_when_failed(self):
        """Health endpoint reports false when any subsystem is not loaded."""
        api.model = None
        api.model_loaded = False
        api.dataset = None
        api.dataset_loaded = False
        api.mark2_loaded = False
        api.mark2_fusion_model = None
        api.mark3_loaded = False
        api.mark3_raw_pipeline = None

        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertFalse(data["model_loaded"])
        self.assertFalse(data["dataset_loaded"])
        self.assertFalse(data["mark2_loaded"])
        self.assertFalse(data["mark3_loaded"])

    def test_mark1_samples_returns_503_when_dataset_unavailable(self):
        """GET /samples must return HTTP 503 if dataset is unavailable."""
        api.dataset_loaded = False
        api.dataset = None
        api.test_ids = None

        response = self.client.get("/samples")
        self.assertEqual(response.status_code, 503)
        self.assertIn("Dataset is not loaded", response.json()["detail"])

    def test_mark1_predict_returns_503_when_dataset_unavailable(self):
        """POST /predict must return HTTP 503 if dataset is unavailable even if model is loaded."""
        api.model = object()
        api.model_loaded = True
        api.dataset_loaded = False
        api.dataset = None
        api.test_ids = None

        response = self.client.post("/predict", json={"segment_id": "nonexistent_seg"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("Dataset is not loaded", response.json()["detail"])

    def test_mark2_and_mark3_remain_functional_when_dataset_missing(self):
        """Mark 2 and Mark 3 endpoints must remain unaffected by missing Mark 1 dataset."""
        api.dataset_loaded = False
        api.dataset = None

        # Verify Mark 2 returns 503 when its own pipeline is uninitialized
        api.mark2_fusion_model = None
        api.mark2_text_model = None
        res_m2 = self.client.post("/predict/raw", data={"text": "hello"})
        self.assertEqual(res_m2.status_code, 503)
        self.assertIn("Mark 2 pipeline is not initialized", res_m2.json()["detail"])

        # Verify Mark 3 returns 503 when its own pipeline is uninitialized
        api.mark3_raw_pipeline = None
        res_m3 = self.client.post("/predict/mark3/raw", data={"text": "hello"})
        self.assertEqual(res_m3.status_code, 503)
        self.assertIn("Mark 3 pipeline is not initialized", res_m3.json()["detail"])


if __name__ == "__main__":
    unittest.main()
