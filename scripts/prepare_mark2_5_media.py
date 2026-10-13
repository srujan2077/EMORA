import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MARK2_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mark2"
)

MEDIA_CACHE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mark2_5_media"
)

MEDIA_CACHE.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

NUM_FRAMES = 8

AUDIO_SAMPLE_RATE = 16000


# ============================================================
# FFMPEG
# ============================================================

FFMPEG = (
    r"C:\Users\sruja\AppData\Local\Microsoft\WinGet"
    r"\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_segments():

    all_rows = []

    for split in [
        "train",
        "val",
        "test",
    ]:

        csv_path = (
            MARK2_DIR
            / f"{split}.csv"
        )

        df = pd.read_csv(
            csv_path
        )

        df["split"] = split

        all_rows.append(
            df
        )

    return pd.concat(
        all_rows,
        ignore_index=True,
    )


# ============================================================
# AUDIO
# ============================================================

def extract_audio(
    video_path,
    output_path,
):

    command = [

        FFMPEG,

        "-hide_banner",

        "-loglevel",
        "error",

        "-i",
        str(video_path),

        "-vn",

        "-ac",
        "1",

        "-ar",
        str(AUDIO_SAMPLE_RATE),

        "-f",
        "f32le",

        "-",
    ]

    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr.decode(
                errors="ignore"
            )
        )

    audio = np.frombuffer(
        result.stdout,
        dtype=np.float32,
    )

    if len(audio) == 0:

        raise RuntimeError(
            "No audio extracted."
        )

    np.save(
        output_path,
        audio,
    )


# ============================================================
# VIDEO FRAMES
# ============================================================

def extract_frames(
    video_path,
    output_path,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open:\n"
            f"{video_path}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:

        cap.release()

        raise RuntimeError(
            "Video contains no frames."
        )

    indices = np.linspace(

        0,

        total_frames - 1,

        NUM_FRAMES,

    ).astype(int)

    frames = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index),
        )

        success, frame = cap.read()

        if not success:

            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(
            frame
        )

    cap.release()

    if len(frames) == 0:

        raise RuntimeError(
            "Could not extract frames."
        )

    frames = np.stack(
        frames,
        axis=0,
    )

    np.save(
        output_path,
        frames,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MARK 2.5 MEDIA PREPROCESSING")
    print("=" * 70)

    print(
        f"Cache directory:\n"
        f"{MEDIA_CACHE}"
    )

    df = load_segments()

    print()
    print(
        f"Total segments: {len(df)}"
    )

    print()

    completed = 0

    skipped = 0

    failed = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Preparing media",
    ):

        segment_id = str(
            row["segment_id"]
        )

        media_path = (
            PROJECT_ROOT
            / str(row["media_path"])
        )

        if not media_path.exists():

            print(
                f"\nMissing video:\n"
                f"{media_path}"
            )

            failed += 1

            continue

        # ----------------------------------------------------
        # Safe filename
        # ----------------------------------------------------

        safe_id = (
            segment_id
            .replace("[", "_")
            .replace("]", "")
            .replace("/", "_")
            .replace("\\", "_")
        )

        split = row["split"]

        split_dir = (
            MEDIA_CACHE
            / split
        )

        split_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        audio_path = (
            split_dir
            / f"{safe_id}_audio.npy"
        )

        frames_path = (
            split_dir
            / f"{safe_id}_frames.npy"
        )

        # ----------------------------------------------------
        # Already processed
        # ----------------------------------------------------

        if (
            audio_path.exists()
            and
            frames_path.exists()
        ):

            skipped += 1

            continue

        try:

            # ------------------------------------------------
            # Audio
            # ------------------------------------------------

            if not audio_path.exists():

                extract_audio(

                    media_path,

                    audio_path,
                )

            # ------------------------------------------------
            # Video
            # ------------------------------------------------

            if not frames_path.exists():

                extract_frames(

                    media_path,

                    frames_path,
                )

            completed += 1

        except Exception as exc:

            failed += 1

            print()

            print(
                f"FAILED: {segment_id}"
            )

            print(
                exc
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("MEDIA PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Processed: {completed}"
    )

    print(
        f"Already existed: {skipped}"
    )

    print(
        f"Failed: {failed}"
    )

    print()

    print(
        f"Cache:\n{MEDIA_CACHE}"
    )


if __name__ == "__main__":

    main()