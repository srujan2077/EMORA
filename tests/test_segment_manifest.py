"""Unit tests for SegmentManifest and SegmentInput."""

from __future__ import annotations

import unittest
import numpy as np

from src.inference.segment_manifest import SegmentInput, SegmentManifest


class SegmentManifestTests(unittest.TestCase):
    def test_valid_manifest_construction(self):
        seg1 = SegmentInput(
            segment_number=0,
            text="Hello world",
            media_path="clip0.mp4",
        )
        seg2 = SegmentInput(
            segment_number=1,
            text="How are you?",
            video_path="clip1.mp4",
            audio_path="clip1.wav",
        )
        manifest = SegmentManifest(video_id="video_01", segments=[seg1, seg2])
        self.assertEqual(manifest.video_id, "video_01")
        self.assertEqual(len(manifest.segments), 2)
        self.assertEqual(seg1.resolved_video_path, "clip0.mp4")
        self.assertEqual(seg1.resolved_audio_path, "clip0.mp4")

    def test_in_memory_modalities_accepted(self):
        seg = SegmentInput(
            segment_number=5,
            text="In-memory test",
            audio_waveform=np.zeros(16000, dtype=np.float32),
            video_frames=[np.zeros((224, 224, 3), dtype=np.uint8)],
        )
        manifest = SegmentManifest(video_id="vid_mem", segments=[seg])
        self.assertTrue(manifest.segments[0].has_audio_source())
        self.assertTrue(manifest.segments[0].has_video_source())

    def test_rejects_empty_manifest(self):
        with self.assertRaisesRegex(ValueError, "at least one segment"):
            SegmentManifest(video_id="vid", segments=[])

    def test_rejects_empty_video_id(self):
        seg = SegmentInput(segment_number=0, media_path="test.mp4")
        with self.assertRaisesRegex(ValueError, "video_id must be a non-empty string"):
            SegmentManifest(video_id="   ", segments=[seg])

    def test_rejects_non_increasing_segment_numbers(self):
        s1 = SegmentInput(segment_number=2, media_path="test.mp4")
        s2 = SegmentInput(segment_number=1, media_path="test.mp4")
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            SegmentManifest(video_id="vid", segments=[s1, s2])

        s_dup = SegmentInput(segment_number=2, media_path="test.mp4")
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            SegmentManifest(video_id="vid", segments=[s1, s_dup])

    def test_rejects_missing_modalities(self):
        # Missing audio
        s_no_audio = SegmentInput(
            segment_number=0,
            text="hello",
            video_path="v.mp4",
            audio_path=None,
            media_path=None,
        )
        with self.assertRaisesRegex(ValueError, "no audio source"):
            SegmentManifest(video_id="vid", segments=[s_no_audio])

        # Missing video
        s_no_video = SegmentInput(
            segment_number=0,
            text="hello",
            audio_path="a.wav",
            video_path=None,
            media_path=None,
        )
        with self.assertRaisesRegex(ValueError, "no video source"):
            SegmentManifest(video_id="vid", segments=[s_no_video])

    def test_rejects_invalid_timestamps(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            SegmentInput(segment_number=0, media_path="a.mp4", start_time=-1.0)

        with self.assertRaisesRegex(ValueError, "must be greater than start_time"):
            SegmentInput(
                segment_number=0,
                media_path="a.mp4",
                start_time=5.0,
                end_time=3.0,
            )

    def test_serialization_round_trip(self):
        s1 = SegmentInput(
            segment_number=0,
            text="First",
            media_path="data/clip0.mp4",
            start_time=0.0,
            end_time=2.5,
        )
        s2 = SegmentInput(
            segment_number=1,
            text="Second",
            media_path="data/clip1.mp4",
            start_time=2.5,
            end_time=5.0,
        )
        manifest = SegmentManifest(video_id="session_42", segments=[s1, s2])

        data = manifest.to_dict()
        restored = SegmentManifest.from_dict(data)

        self.assertEqual(restored.video_id, "session_42")
        self.assertEqual(len(restored.segments), 2)
        self.assertEqual(restored.segments[0].segment_number, 0)
        self.assertEqual(restored.segments[1].end_time, 5.0)


if __name__ == "__main__":
    unittest.main()
