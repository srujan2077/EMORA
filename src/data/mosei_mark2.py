"""
Mark 2 - CMU-MOSEI Raw Multimodal Dataset

Loads:
    Text  -> timestamp-aligned transcript
    Audio -> embedded audio from MP4
    Video -> sampled RGB frames from MP4
    Labels -> six emotion multi-label target

Mark 1 is untouched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mmsdk import mmdatasdk


class MOSEIMark2Dataset(Dataset):
    """
    CMU-MOSEI raw multimodal dataset for Mark 2.

    Each sample corresponds to a MOSEI segment such as:

        -ZgjBOA1Yhw[4]

    The sample contains:
        - segment-level emotion labels
        - timestamp-aligned transcript
        - corresponding raw MP4
        - audio embedded inside the MP4
        - sampled video frames
    """

    # Original MOSEI label indices used by Mark 1
    EMOTION_INDICES = [1, 2, 3, 4, 5, 6]

    EMOTION_NAMES = [
        "happy",
        "sad",
        "anger",
        "surprise",
        "disgust",
        "fear",
    ]

    def __init__(
        self,
        segment_ids,
        labels_path="data/processed/aligned_mark1/labels_aligned.csd",
        words_path="data/raw/cmu_mosei/CMU_MOSEI_TimestampedWords.csd",
        media_root="data/raw/cmu_mosei/Raw",
        test_media_root="data/raw/cmu_mosei/test",
        num_video_frames=8,
        audio_sample_rate=16000,
        image_size=224,
        drop_pause_tokens=True,
    ):
        self.segment_ids = list(segment_ids)

        self.labels_path = Path(labels_path)
        self.words_path = Path(words_path)
        self.media_root = Path(media_root)
        self.test_media_root = Path(test_media_root)

        self.num_video_frames = num_video_frames
        self.audio_sample_rate = audio_sample_rate
        self.image_size = image_size
        self.drop_pause_tokens = drop_pause_tokens

        # ---------------------------------------------------------
        # Load labels
        # ---------------------------------------------------------

        print("Loading Mark 2 labels...")

        self.label_dataset = mmdatasdk.mmdataset(
            {
                "labels": str(self.labels_path)
            },
            str(self.labels_path.parent),
        )

        self.label_sequence = (
            self.label_dataset
            .computational_sequences["labels"]
            .data
        )

        # ---------------------------------------------------------
        # Load timestamped words
        # ---------------------------------------------------------

        print("Loading timestamped transcripts...")

        self.words_dataset = mmdatasdk.mmdataset(
            {
                "words": str(self.words_path)
            },
            str(self.words_path.parent),
        )

        self.words_sequence = (
            self.words_dataset
            .computational_sequences["words"]
            .data
        )

        # ---------------------------------------------------------
        # Locate FFmpeg
        # ---------------------------------------------------------

        self.ffmpeg = self._find_executable("ffmpeg")

        if self.ffmpeg is None:
            raise RuntimeError(
                "FFmpeg was not found. "
                "Install FFmpeg or add it to PATH."
            )

        print(f"FFmpeg: {self.ffmpeg}")

        print(
            f"Mark 2 dataset initialized with "
            f"{len(self.segment_ids)} segments."
        )

    # =============================================================
    # Utility functions
    # =============================================================

    @staticmethod
    def _find_executable(name: str):
        """
        Find executable using PATH first,
        then search common WinGet locations.
        """

        executable = shutil.which(name)

        if executable is not None:
            return executable

        local_appdata = os.environ.get(
            "LOCALAPPDATA",
            ""
        )

        winget_root = (
            Path(local_appdata)
            / "Microsoft"
            / "WinGet"
            / "Packages"
        )

        if winget_root.exists():

            matches = list(
                winget_root.rglob(
                    f"{name}.exe"
                )
            )

            if matches:

                full_build = [
                    path
                    for path in matches
                    if "full_build" in str(path)
                    and "shared" not in str(path).lower()
                ]

                if full_build:
                    return str(full_build[0])

                return str(matches[0])

        return None

    @staticmethod
    def parse_segment_id(segment_id: str):
        """
        Convert:

            -ZgjBOA1Yhw[4]

        into:

            video_id = -ZgjBOA1Yhw
            segment_number = 4
        """

        if (
            "[" not in segment_id
            or not segment_id.endswith("]")
        ):
            raise ValueError(
                f"Invalid MOSEI segment ID: {segment_id}"
            )

        video_id, segment_number = (
            segment_id.rsplit("[", 1)
        )

        segment_number = int(
            segment_number[:-1]
        )

        return video_id, segment_number

    # =============================================================
    # Labels
    # =============================================================

    def load_label(self, segment_id: str):
        """
        Load:

            segment start
            segment end
            six emotion labels
        """

        if segment_id not in self.label_sequence:
            raise KeyError(
                f"Label not found for segment: {segment_id}"
            )

        item = self.label_sequence[segment_id]

        interval = np.asarray(
            item["intervals"][0],
            dtype=np.float32,
        )

        raw_labels = np.asarray(
            item["features"][0],
            dtype=np.float32,
        )

        # Select:
        # happy, sad, anger, surprise, disgust, fear
        labels = raw_labels[
            self.EMOTION_INDICES
        ]

        # Same convention as Mark 1:
        #
        # score > 0 -> 1
        # score <= 0 -> 0
        labels = (
            labels > 0
        ).astype(np.float32)

        return (
            float(interval[0]),
            float(interval[1]),
            torch.from_numpy(labels).float(),
        )

    # =============================================================
    # Transcript
    # =============================================================

    def load_transcript(
        self,
        video_id: str,
        segment_start: float,
        segment_end: float,
    ):
        """
        Extract words whose timestamps overlap
        the requested segment.
        """

        if video_id not in self.words_sequence:
            raise KeyError(
                f"Transcript not found for video: "
                f"{video_id}"
            )

        video_words = self.words_sequence[
            video_id
        ]

        features = video_words[
            "features"
        ]

        intervals = np.asarray(
            video_words["intervals"],
            dtype=np.float64,
        )

        words = []

        for i in range(len(features)):

            word = features[i][0]

            if isinstance(word, bytes):
                word = word.decode(
                    "utf-8",
                    errors="ignore",
                )
            else:
                word = str(word)

            word_start = float(
                intervals[i][0]
            )

            word_end = float(
                intervals[i][1]
            )

            # Word overlaps the segment if:
            #
            # word_end > segment_start
            # AND
            # word_start < segment_end

            overlaps = (
                word_end > segment_start
                and word_start < segment_end
            )

            if not overlaps:
                continue

            # "sp" represents a pause.
            if (
                self.drop_pause_tokens
                and word.lower() == "sp"
            ):
                continue

            words.append(word)

        return " ".join(words).strip()

    # =============================================================
    # Media path
    # =============================================================

    def get_media_path(
        self,
        segment_id: str,
    ):
        """
        Map:

            -ZgjBOA1Yhw[4]

        to:

            Raw/-ZgjBOA1Yhw/4.mp4
        """

        video_id, segment_number = (
            self.parse_segment_id(
                segment_id
            )
        )

        # ---------------------------------------------------------
        # Full extracted dataset
        # ---------------------------------------------------------

        full_path = (
            self.media_root
            / video_id
            / f"{segment_number}.mp4"
        )

        if full_path.exists():
            return full_path

        # ---------------------------------------------------------
        # TEST FALLBACK
        #
        # IMPORTANT:
        # Only the exact known test segment is allowed
        # to use test/4.mp4.
        # ---------------------------------------------------------

        if segment_id == "-ZgjBOA1Yhw[4]":

            test_path = (
                self.test_media_root
                / "4.mp4"
            )

            if test_path.exists():
                return test_path

        # ---------------------------------------------------------
        # Nothing found
        # ---------------------------------------------------------

        raise FileNotFoundError(
            f"Media file not found for {segment_id}\n\n"
            f"Expected full dataset path:\n"
            f"{full_path}\n\n"
            f"Test fallback is only available for:\n"
            f"-ZgjBOA1Yhw[4]"
        )

    # =============================================================
    # Video
    # =============================================================

    def load_video(
        self,
        video_path: Path,
    ):
        """
        Uniformly sample video frames.

        Returns:

            Tensor [T, 3, H, W]

        Example:

            [8, 3, 224, 224]
        """

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: "
                f"{video_path}"
            )

        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if frame_count <= 0:
            capture.release()

            raise RuntimeError(
                f"Video contains no frames: "
                f"{video_path}"
            )

        # Uniform frame sampling
        frame_indices = np.linspace(
            0,
            frame_count - 1,
            self.num_video_frames,
            dtype=np.int32,
        )

        frames = []

        for frame_index in frame_indices:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index),
            )

            success, frame = (
                capture.read()
            )

            if not success:
                continue

            # BGR -> RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            # Resize
            frame = cv2.resize(
                frame,
                (
                    self.image_size,
                    self.image_size,
                ),
                interpolation=cv2.INTER_AREA,
            )

            # uint8 -> float32 [0, 1]
            frame = (
                frame.astype(
                    np.float32
                )
                / 255.0
            )

            # HWC -> CHW
            frame = np.transpose(
                frame,
                (2, 0, 1),
            )

            frames.append(frame)

        capture.release()

        if not frames:
            raise RuntimeError(
                f"Could not decode frames: "
                f"{video_path}"
            )

        # If some frames failed,
        # repeat the final valid frame.
        while (
            len(frames)
            < self.num_video_frames
        ):
            frames.append(
                frames[-1].copy()
            )

        frames = np.stack(
            frames,
            axis=0,
        )

        return torch.from_numpy(
            frames
        ).float()

    # =============================================================
    # Audio
    # =============================================================

    def load_audio(
        self,
        video_path: Path,
    ):
        """
        Extract embedded MP4 audio using FFmpeg.

        Output:

            mono 16 kHz float32 waveform

        Shape:

            [num_samples]
        """

        command = [
            self.ffmpeg,

            "-hide_banner",
            "-loglevel",
            "error",

            "-i",
            str(video_path),

            # Disable video
            "-vn",

            # Mono
            "-ac",
            "1",

            # 16 kHz
            "-ar",
            str(self.audio_sample_rate),

            # Raw float32 PCM
            "-f",
            "f32le",

            # Send audio to stdout
            "pipe:1",
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if result.returncode != 0:

            error = result.stderr.decode(
                "utf-8",
                errors="ignore",
            )

            raise RuntimeError(
                "FFmpeg audio extraction failed:\n"
                f"{error}"
            )

        audio = np.frombuffer(
            result.stdout,
            dtype=np.float32,
        ).copy()

        if audio.size == 0:
            raise RuntimeError(
                f"No audio found in: "
                f"{video_path}"
            )

        return torch.from_numpy(
            audio
        ).float()

    # =============================================================
    # Dataset interface
    # =============================================================

    def __len__(self):
        return len(
            self.segment_ids
        )

    def __getitem__(self, index):

        segment_id = (
            self.segment_ids[index]
        )

        # ---------------------------------------------------------
        # Labels + segment timing
        # ---------------------------------------------------------

        (
            segment_start,
            segment_end,
            labels,
        ) = self.load_label(
            segment_id
        )

        # ---------------------------------------------------------
        # Video ID
        # ---------------------------------------------------------

        (
            video_id,
            segment_number,
        ) = self.parse_segment_id(
            segment_id
        )

        # ---------------------------------------------------------
        # Transcript
        # ---------------------------------------------------------

        text = self.load_transcript(
            video_id,
            segment_start,
            segment_end,
        )

        # ---------------------------------------------------------
        # Media
        # ---------------------------------------------------------

        media_path = (
            self.get_media_path(
                segment_id
            )
        )

        # ---------------------------------------------------------
        # Video
        # ---------------------------------------------------------

        video = self.load_video(
            media_path
        )

        # ---------------------------------------------------------
        # Audio
        # ---------------------------------------------------------

        audio = self.load_audio(
            media_path
        )

        # ---------------------------------------------------------
        # Final sample
        # ---------------------------------------------------------

        return {
            "text": text,

            "audio": audio,

            "video": video,

            "labels": labels,

            "segment_id": segment_id,

            "video_id": video_id,

            "segment_number": segment_number,

            "segment_start": segment_start,

            "segment_end": segment_end,

            "media_path": str(
                media_path
            ),
        }


# =============================================================
# Standalone test
# =============================================================

if __name__ == "__main__":

    # Known test segment
    segment_id = "-ZgjBOA1Yhw[4]"

    dataset = MOSEIMark2Dataset(
        segment_ids=[
            segment_id
        ]
    )

    sample = dataset[0]

    print()
    print("=" * 60)
    print("MARK 2 SAMPLE TEST")
    print("=" * 60)

    print()
    print("Segment ID:")
    print(sample["segment_id"])

    print()
    print("Video ID:")
    print(sample["video_id"])

    print()
    print("Segment Number:")
    print(sample["segment_number"])

    print()
    print("Segment:")
    print(
        sample["segment_start"],
        "->",
        sample["segment_end"],
    )

    print()
    print("Transcript:")
    print(sample["text"])

    print()
    print("Video tensor:")
    print(
        sample["video"].shape,
        sample["video"].dtype,
    )

    print()
    print("Audio tensor:")
    print(
        sample["audio"].shape,
        sample["audio"].dtype,
    )

    print()
    print("Labels:")
    print(sample["labels"])

    print()
    print("Media:")
    print(sample["media_path"])

    print()
    print("Emotion names:")

    for name, value in zip(
        MOSEIMark2Dataset.EMOTION_NAMES,
        sample["labels"].tolist(),
    ):
        print(
            f"  {name:10s}: {value}"
        )

    print()
    print("=" * 60)
    print("MARK 2 DATA LOADER TEST COMPLETE")
    print("=" * 60)