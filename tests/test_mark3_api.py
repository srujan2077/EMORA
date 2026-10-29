"""Unit tests for Mark 3 API endpoints in api/main.py."""

from __future__ import annotations

import asyncio
import io
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from fastapi import HTTPException
from starlette.datastructures import UploadFile

import api.main as api
from src.inference.mark3 import Mark3Prediction


def upload(filename: str, content: bytes = b"mock media content") -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


class Mark3ApiTests(unittest.TestCase):
    def setUp(self):
        api.mark3_raw_pipeline = object()

    def tearDown(self):
        api.mark3_raw_pipeline = None

    def test_uninitialized_pipeline_raises_503(self):
        api.mark3_raw_pipeline = None
        with self.assertRaisesRegex(HTTPException, "Mark 3 pipeline is not initialized"):
            asyncio.run(api.predict_mark3_raw(text="hi", video=upload("vid.mp4")))

    def test_empty_request_rejected(self):
        with self.assertRaisesRegex(HTTPException, "Provide text and video/audio"):
            asyncio.run(api.predict_mark3_raw(text=None, audio=None, video=None))

    def test_missing_video_rejected(self):
        with self.assertRaisesRegex(HTTPException, "Video is required"):
            asyncio.run(api.predict_mark3_raw(text="Some text", audio=upload("a.wav"), video=None))

    def test_complete_raw_request_success_and_cleanup(self):
        removed_files = []

        mock_prediction = Mark3Prediction(
            probabilities={
                "happy": 0.85,
                "sad": 0.10,
                "anger": 0.05,
                "surprise": 0.15,
                "disgust": 0.02,
                "fear": 0.01,
            },
            thresholds={
                "happy": 0.32,
                "sad": 0.43,
                "anger": 0.43,
                "surprise": 0.41,
                "disgust": 0.52,
                "fear": 0.43,
            },
            predicted_emotions=("happy",),
            metadata={"raw_media_supported": True, "raw_sequence_length": 1},
        )

        mock_pipeline = SimpleNamespace(
            predict_single_segment=MagicMock(return_value=mock_prediction)
        )

        async def fake_save_upload(up, kind):
            return f"temp_{kind}.mp4"

        with (
            patch.object(api, "mark3_raw_pipeline", mock_pipeline),
            patch.object(api, "_save_upload", fake_save_upload),
            patch.object(api.os.path, "exists", return_value=True),
            patch.object(api.os, "remove", side_effect=removed_files.append),
        ):
            res = asyncio.run(
                api.predict_mark3_raw(
                    text="Super happy about this release!",
                    audio=upload("sound.wav"),
                    video=upload("clip.mp4"),
                )
            )

        self.assertTrue(res["success"])
        self.assertEqual(res["model"], "Mark 3")
        self.assertEqual(res["dominant_emotion"], "happy")
        self.assertTrue(res["emotions"]["happy"]["detected"])
        self.assertFalse(res["emotions"]["sad"]["detected"])
        self.assertEqual(res["predicted_emotions"], ["happy"])
        self.assertEqual(removed_files, ["temp_video.mp4", "temp_audio.mp4"])

    def test_manifest_endpoint_success(self):
        mock_prediction = Mark3Prediction(
            probabilities={
                "happy": 0.70,
                "sad": 0.20,
                "anger": 0.10,
                "surprise": 0.10,
                "disgust": 0.05,
                "fear": 0.05,
            },
            thresholds={
                "happy": 0.32,
                "sad": 0.43,
                "anger": 0.43,
                "surprise": 0.41,
                "disgust": 0.52,
                "fear": 0.43,
            },
            predicted_emotions=("happy",),
            metadata={"raw_media_supported": True, "split_sequence_length": 2},
        )
        mock_pipeline = SimpleNamespace(
            predict_manifest=MagicMock(return_value=mock_prediction)
        )

        manifest_data = {
            "video_id": "session_test",
            "segments": [
                {"segment_number": 0, "text": "Hi", "media_path": "a.mp4"},
                {"segment_number": 1, "text": "Hello", "media_path": "b.mp4"},
            ],
        }

        with patch.object(api, "mark3_raw_pipeline", mock_pipeline):
            res = asyncio.run(api.predict_mark3_manifest(manifest_data))

        self.assertTrue(res["success"])
        self.assertEqual(res["model"], "Mark 3")
        self.assertEqual(res["dominant_emotion"], "happy")
        mock_pipeline.predict_manifest.assert_called_once_with(
            ANY,
            target_segment_number=None,
        )

    def test_manifest_wrapper_passes_target_segment(self):
        mock_prediction = Mark3Prediction(
            probabilities={"happy": 0.70, "sad": 0.20, "anger": 0.10,
                           "surprise": 0.10, "disgust": 0.05, "fear": 0.05},
            thresholds={"happy": 0.32, "sad": 0.43, "anger": 0.43,
                        "surprise": 0.41, "disgust": 0.52, "fear": 0.43},
            predicted_emotions=("happy",),
            metadata={"target_segment_number": 0},
        )
        mock_pipeline = SimpleNamespace(
            predict_manifest=MagicMock(return_value=mock_prediction)
        )
        request = {
            "manifest": {
                "video_id": "session_test",
                "segments": [
                    {"segment_number": 0, "text": "Hi", "media_path": "a.mp4"},
                    {"segment_number": 1, "text": "Hello", "media_path": "b.mp4"},
                ],
            },
            "target_segment_number": 0,
        }

        with patch.object(api, "mark3_raw_pipeline", mock_pipeline):
            result = asyncio.run(api.predict_mark3_manifest(request))

        self.assertEqual(result["metadata"]["target_segment_number"], 0)
        mock_pipeline.predict_manifest.assert_called_once_with(
            ANY,
            target_segment_number=0,
        )

    def test_legacy_direct_manifest_with_target_remains_supported(self):
        mock_prediction = Mark3Prediction(
            probabilities={"happy": 0.70, "sad": 0.20, "anger": 0.10,
                           "surprise": 0.10, "disgust": 0.05, "fear": 0.05},
            thresholds={"happy": 0.32, "sad": 0.43, "anger": 0.43,
                        "surprise": 0.41, "disgust": 0.52, "fear": 0.43},
            predicted_emotions=("happy",),
            metadata={"target_segment_number": 1},
        )
        mock_pipeline = SimpleNamespace(
            predict_manifest=MagicMock(return_value=mock_prediction)
        )
        request = {
            "video_id": "session_test",
            "segments": [
                {"segment_number": 0, "text": "Hi", "media_path": "a.mp4"},
                {"segment_number": 1, "text": "Hello", "media_path": "b.mp4"},
            ],
            "target_segment_number": 1,
        }

        with patch.object(api, "mark3_raw_pipeline", mock_pipeline):
            result = asyncio.run(api.predict_mark3_manifest(request))

        self.assertEqual(result["metadata"]["target_segment_number"], 1)
        mock_pipeline.predict_manifest.assert_called_once_with(
            ANY,
            target_segment_number=1,
        )

    def test_manifest_malformed_requests_return_422(self):
        invalid_requests = (
            {"manifest": None},
            {"manifest": {"video_id": "session_test", "segments": [{}]}},
            {
                "manifest": {
                    "video_id": "session_test",
                    "segments": [
                        {"segment_number": 1, "media_path": "a.mp4"},
                        {"segment_number": 0, "media_path": "b.mp4"},
                    ],
                }
            },
            {
                "manifest": {
                    "video_id": "session_test",
                    "segments": [
                        {"segment_number": 0, "media_path": "a.mp4"},
                    ],
                },
                "target_segment_number": "0",
            },
        )

        with patch.object(
            api,
            "mark3_raw_pipeline",
            SimpleNamespace(predict_manifest=MagicMock()),
        ):
            for request in invalid_requests:
                with self.subTest(request=request):
                    with self.assertRaisesRegex(HTTPException, "Invalid|must be|Target"):
                        asyncio.run(api.predict_mark3_manifest(request))

    def test_predict_mark3_raw_value_error_returns_422(self):
        mock_pipeline = SimpleNamespace(
            predict_single_segment=MagicMock(side_effect=ValueError("Video contains undecodable frame stream."))
        )
        async def fake_save_upload(up, kind):
            return f"temp_{kind}.mp4"

        with (
            patch.object(api, "mark3_raw_pipeline", mock_pipeline),
            patch.object(api, "_save_upload", fake_save_upload),
            patch.object(api.os.path, "exists", return_value=True),
            patch.object(api.os, "remove", return_value=None),
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(
                    api.predict_mark3_raw(
                        text="Valid text",
                        video=upload("bad_video.mp4"),
                    )
                )
            self.assertEqual(ctx.exception.status_code, 422)
            self.assertIn("undecodable", ctx.exception.detail)

    def test_manifest_different_targets_passed_to_pipeline(self):
        mock_pipeline = SimpleNamespace(
            predict_manifest=MagicMock(
                return_value=Mark3Prediction(
                    probabilities={"happy": 0.5, "sad": 0.1, "anger": 0.1, "surprise": 0.1, "disgust": 0.1, "fear": 0.1},
                    thresholds={"happy": 0.32, "sad": 0.43, "anger": 0.43, "surprise": 0.41, "disgust": 0.52, "fear": 0.43},
                    predicted_emotions=("happy",),
                    metadata={"target_segment_number": 2},
                )
            )
        )
        payload = {
            "manifest": {
                "video_id": "test_video",
                "segments": [
                    {"segment_number": 0, "text": "Turn 0", "media_path": "0.mp4"},
                    {"segment_number": 1, "text": "Turn 1", "media_path": "1.mp4"},
                    {"segment_number": 2, "text": "Turn 2", "media_path": "2.mp4"},
                ]
            },
            "target_segment_number": 2
        }

        with patch.object(api, "mark3_raw_pipeline", mock_pipeline):
            res = asyncio.run(api.predict_mark3_manifest(payload))
            self.assertEqual(res["metadata"]["target_segment_number"], 2)
            mock_pipeline.predict_manifest.assert_called_once_with(ANY, target_segment_number=2)


if __name__ == "__main__":
    unittest.main()
