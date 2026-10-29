import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api.main as api


class MLOpsApiContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app, raise_server_exceptions=False)
        self.original_loaded = api.mark3_loaded
        self.original_pipeline = api.mark3_raw_pipeline

    def tearDown(self):
        api.mark3_loaded = self.original_loaded
        api.mark3_raw_pipeline = self.original_pipeline

    def test_liveness_is_independent_of_model_readiness(self):
        api.mark3_loaded = False
        api.mark3_raw_pipeline = None
        response = self.client.get("/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(self.client.get("/ready").status_code, 503)

    def test_readiness_exposes_mark3_contract(self):
        api.mark3_loaded = True
        api.mark3_raw_pipeline = object()
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["threshold_policy"], "fixed_0_5")

    def test_request_id_is_returned(self):
        response = self.client.get("/live", headers={"X-Request-ID": "audit-123"})
        self.assertEqual(response.headers["X-Request-ID"], "audit-123")

    def test_configured_cors_does_not_allow_arbitrary_origin(self):
        response = self.client.options(
            "/live",
            headers={
                "Origin": "https://not-approved.invalid",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertNotEqual(response.headers.get("access-control-allow-origin"), "*")


if __name__ == "__main__":
    unittest.main()
