import asyncio
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi import HTTPException
from starlette.datastructures import UploadFile

import api.main as api


def upload(filename: str, content: bytes = b"media") -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


class FakeTokenizer:
    def __call__(self, *_args, **_kwargs):
        return {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }


class FakeProcessor:
    def __call__(self, *_args, **_kwargs):
        return {"input_values": torch.ones((1, 4))}


class FakeImageProcessor:
    def __call__(self, images=None, *_args, **_kwargs):
        num = len(images) if images is not None else 8
        return {"pixel_values": torch.ones((num, 3, 224, 224))}


class FakeModel:
    def __call__(self, pixel_values=None, **_kwargs):
        num = pixel_values.shape[0] if pixel_values is not None else 8
        return SimpleNamespace(last_hidden_state=torch.ones((num, 2, 768)))


class FakeAudioModel:
    def __call__(self, **_kwargs):
        return SimpleNamespace(last_hidden_state=torch.ones((1, 2, 768)))


class FakeTextModel:
    def __call__(self, **_kwargs):
        device = _kwargs["input_ids"].device
        return SimpleNamespace(
            last_hidden_state=torch.ones((1, 2, 768), device=device)
        )


class FakeFusion:
    def __call__(self, text, audio, video):
        assert text.shape == audio.shape == video.shape == (1, 768)
        return torch.zeros((1, 6)), torch.zeros((1, 256))


class FakeCapture:
    def __init__(self, _path):
        self.frames = 8

    def get(self, _property):
        return self.frames

    def set(self, *_args):
        return True

    def read(self):
        return True, torch.zeros((224, 224, 3), dtype=torch.uint8).numpy()

    def release(self):
        return None


class FakeCaptureWithCount:
    def __init__(self, valid_reads: int, total_frames: int = 8):
        self._valid_reads = valid_reads
        self._reads_so_far = 0
        self.total_frames = total_frames

    def get(self, _property):
        return self.total_frames

    def set(self, *_args):
        return True

    def read(self):
        if self._reads_so_far < self._valid_reads:
            self._reads_so_far += 1
            return True, torch.zeros((224, 224, 3), dtype=torch.uint8).numpy()
        return False, None

    def release(self):
        return None


class Mark2RawApiTests(unittest.TestCase):
    def test_upload_limit_is_50_mib_inclusive(self):
        path = asyncio.run(
            api._save_upload(
                upload("clip.mp4", b"x" * (api.MAX_UPLOAD_BYTES)),
                "video",
            )
        )
        try:
            self.assertTrue(path)
        finally:
            if api.os.path.exists(path):
                api.os.remove(path)

    def test_upload_over_limit_returns_413(self):
        with self.assertRaisesRegex(HTTPException, "50 MB limit"):
            asyncio.run(
                api._save_upload(
                    upload("clip.mp4", b"x" * (api.MAX_UPLOAD_BYTES + 1)),
                    "video",
                )
            )

    def test_duration_limit_is_300_seconds_inclusive(self):
        api._validate_media_duration(300.0, "audio")
        with self.assertRaisesRegex(HTTPException, "5-minute limit"):
            api._validate_media_duration(300.001, "audio")

    def setUp(self):
        api.mark2_text_model = object()
        api.mark2_fusion_model = object()

    def tearDown(self):
        api.mark2_text_model = None
        api.mark2_fusion_model = None

    def test_empty_request_is_rejected(self):
        with self.assertRaisesRegex(HTTPException, "Provide text"):
            asyncio.run(api.predict_raw(text=None, audio=None, video=None))

    def test_partial_request_is_rejected_before_inference(self):
        with self.assertRaisesRegex(HTTPException, "Text is required"):
            asyncio.run(api.predict_raw(text=None, video=upload("clip.mp4")))

    def test_unsupported_upload_is_rejected(self):
        with self.assertRaisesRegex(HTTPException, "Unsupported video format"):
            asyncio.run(api._save_upload(upload("clip.txt"), "video"))

    def test_empty_upload_is_rejected(self):
        with self.assertRaisesRegex(HTTPException, "upload is empty"):
            asyncio.run(api._save_upload(upload("clip.mp4", b""), "video"))

    def test_complete_request_uses_all_three_embeddings_and_cleans_files(self):
        removed = []

        async def save_upload(_upload, kind):
            return f"temporary-{kind}"

        with (
            patch.object(api, "mark2_fusion_model", FakeFusion()),
            patch.object(api, "mark2_text_model", FakeTextModel()),
            patch.object(api, "mark2_tokenizer", FakeTokenizer()),
            patch.object(api, "mark2_audio_processor", FakeProcessor()),
            patch.object(api, "mark2_audio_model", FakeAudioModel()),
            patch.object(api, "mark2_image_processor", FakeImageProcessor()),
            patch.object(api, "mark2_vision_model", FakeModel()),
            patch.object(api, "_save_upload", save_upload),
            patch.object(api, "convert_audio", return_value=torch.ones(16).numpy()),
            patch.object(
                api,
                "cv2",
                SimpleNamespace(
                    VideoCapture=FakeCapture,
                    CAP_PROP_FRAME_COUNT=1,
                    CAP_PROP_FPS=4,
                    CAP_PROP_POS_FRAMES=2,
                    COLOR_BGR2RGB=3,
                    cvtColor=lambda frame, _code: frame,
                ),
            ),
            patch.object(api.os.path, "exists", return_value=True),
            patch.object(api.os, "remove", side_effect=removed.append),
            patch.object(api, "mark2_thresholds", {}),
        ):
            result = asyncio.run(
                api.predict_raw(
                    text="A complete request",
                    audio=upload("sound.wav"),
                    video=upload("clip.mp4"),
                )
            )

        self.assertTrue(result["success"])
        self.assertEqual(
            result["modalities"],
            {"text": True, "audio": True, "video": True},
        )
        self.assertEqual(
            removed,
            ["temporary-video", "temporary-audio"],
        )

    def test_six_decoded_frames_succeeds_and_cleans_files(self):
        removed = []

        async def save_upload(_upload, kind):
            return f"temporary-{kind}"

        with (
            patch.object(api, "mark2_fusion_model", FakeFusion()),
            patch.object(api, "mark2_text_model", FakeTextModel()),
            patch.object(api, "mark2_tokenizer", FakeTokenizer()),
            patch.object(api, "mark2_audio_processor", FakeProcessor()),
            patch.object(api, "mark2_audio_model", FakeAudioModel()),
            patch.object(api, "mark2_image_processor", FakeImageProcessor()),
            patch.object(api, "mark2_vision_model", FakeModel()),
            patch.object(api, "_save_upload", save_upload),
            patch.object(api, "convert_audio", return_value=torch.ones(16).numpy()),
            patch.object(
                api,
                "cv2",
                SimpleNamespace(
                    VideoCapture=lambda _p: FakeCaptureWithCount(6),
                    CAP_PROP_FRAME_COUNT=1,
                    CAP_PROP_FPS=4,
                    CAP_PROP_POS_FRAMES=2,
                    COLOR_BGR2RGB=3,
                    cvtColor=lambda frame, _code: frame,
                ),
            ),
            patch.object(api.os.path, "exists", return_value=True),
            patch.object(api.os, "remove", side_effect=removed.append),
            patch.object(api, "mark2_thresholds", {}),
        ):
            result = asyncio.run(
                api.predict_raw(
                    text="Six frames request",
                    audio=upload("sound.wav"),
                    video=upload("clip.mp4"),
                )
            )

        self.assertTrue(result["success"])
        self.assertEqual(
            result["modalities"],
            {"text": True, "audio": True, "video": True},
        )
        self.assertEqual(
            removed,
            ["temporary-video", "temporary-audio"],
        )

    def test_zero_decoded_frames_returns_422_and_cleans_files(self):
        removed = []

        async def save_upload(_upload, kind):
            return f"temporary-{kind}"

        with (
            patch.object(api, "mark2_fusion_model", FakeFusion()),
            patch.object(api, "mark2_text_model", FakeTextModel()),
            patch.object(api, "mark2_tokenizer", FakeTokenizer()),
            patch.object(api, "mark2_audio_processor", FakeProcessor()),
            patch.object(api, "mark2_audio_model", FakeAudioModel()),
            patch.object(api, "mark2_image_processor", FakeImageProcessor()),
            patch.object(api, "mark2_vision_model", FakeModel()),
            patch.object(api, "_save_upload", save_upload),
            patch.object(api, "convert_audio", return_value=torch.ones(16).numpy()),
            patch.object(
                api,
                "cv2",
                SimpleNamespace(
                    VideoCapture=lambda _p: FakeCaptureWithCount(0),
                    CAP_PROP_FRAME_COUNT=1,
                    CAP_PROP_FPS=4,
                    CAP_PROP_POS_FRAMES=2,
                    COLOR_BGR2RGB=3,
                    cvtColor=lambda frame, _code: frame,
                ),
            ),
            patch.object(api.os.path, "exists", return_value=True),
            patch.object(api.os, "remove", side_effect=removed.append),
            patch.object(api, "mark2_thresholds", {}),
        ):
            with self.assertRaisesRegex(HTTPException, "no decodable frames"):
                asyncio.run(
                    api.predict_raw(
                        text="Zero frames request",
                        audio=upload("sound.wav"),
                        video=upload("clip.mp4"),
                    )
                )

        self.assertEqual(
            removed,
            ["temporary-video", "temporary-audio"],
        )


if __name__ == "__main__":
    unittest.main()
