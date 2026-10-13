import os
import sys
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    AutoImageProcessor,
)

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MARK2_DIR = DATA_DIR / "processed" / "mark2"
RAW_DIR = DATA_DIR / "raw" / "cmu_mosei"

OUTPUT_DIR = MARK2_DIR / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_MODEL = "microsoft/deberta-v3-base"
AUDIO_MODEL = "facebook/wav2vec2-base-960h"
VISION_MODEL = "google/vit-base-patch16-224-in21k"

NUM_FRAMES = 8
AUDIO_SAMPLE_RATE = 16000

EMOTION_NAMES = [
    "happy",
    "sad",
    "anger",
    "surprise",
    "disgust",
    "fear",
]

# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def find_ffmpeg():
    candidates = [
        "ffmpeg",
        r"C:\Users\sruja\AppData\Local\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe",
    ]

    for path in candidates:
        if path == "ffmpeg":
            return path

        if Path(path).exists():
            return path

    raise FileNotFoundError(
        "FFmpeg not found. Install FFmpeg or update the path in this script."
    )


FFMPEG = find_ffmpeg()


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print_header("LOADING PRETRAINED MODELS")

    print(f"Device: {DEVICE}")
    print()

    # ---------------- TEXT ----------------

    print("Loading DeBERTa...")
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    text_model = AutoModel.from_pretrained(TEXT_MODEL)

    text_model.to(DEVICE)
    text_model.eval()

    print("DeBERTa loaded.")

    # ---------------- AUDIO ----------------

    print("Loading Wav2Vec2...")

    audio_processor = AutoProcessor.from_pretrained(AUDIO_MODEL)
    audio_model = AutoModel.from_pretrained(AUDIO_MODEL)

    audio_model.to(DEVICE)
    audio_model.eval()

    print("Wav2Vec2 loaded.")

    # ---------------- VISION ----------------

    print("Loading ViT...")

    vision_processor = AutoImageProcessor.from_pretrained(VISION_MODEL)
    vision_model = AutoModel.from_pretrained(VISION_MODEL)

    vision_model.to(DEVICE)
    vision_model.eval()

    print("ViT loaded.")

    print()
    print("All pretrained models loaded.")

    return (
        text_tokenizer,
        text_model,
        audio_processor,
        audio_model,
        vision_processor,
        vision_model,
    )


# ============================================================
# TEXT EMBEDDING
# ============================================================

@torch.no_grad()
def extract_text_embedding(
    text,
    tokenizer,
    model,
):
    if not isinstance(text, str) or not text.strip():
        return torch.zeros(768, dtype=torch.float32)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    outputs = model(**encoded)

    hidden = outputs.last_hidden_state

    attention_mask = encoded["attention_mask"].unsqueeze(-1)

    masked_hidden = hidden * attention_mask

    summed = masked_hidden.sum(dim=1)

    count = attention_mask.sum(dim=1).clamp(min=1)

    embedding = summed / count

    return embedding.squeeze(0).float().cpu()


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio_ffmpeg(video_path):

    import subprocess

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
            f"FFmpeg audio extraction failed:\n"
            f"{result.stderr.decode(errors='ignore')}"
        )

    audio = np.frombuffer(
        result.stdout,
        dtype=np.float32,
    )

    return audio


@torch.no_grad()
def extract_audio_embedding(
    video_path,
    processor,
    model,
):
    audio = extract_audio_ffmpeg(video_path)

    if len(audio) == 0:
        return torch.zeros(768, dtype=torch.float32)

    inputs = processor(
        audio,
        sampling_rate=AUDIO_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    outputs = model(**inputs)

    hidden = outputs.last_hidden_state

    embedding = hidden.mean(dim=1)

    return embedding.squeeze(0).float().cpu()


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def load_video_frames(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()

        raise RuntimeError(
            f"No frames found: {video_path}"
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

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            f"Could not decode frames: {video_path}"
        )

    return frames


@torch.no_grad()
def extract_video_embedding(
    video_path,
    processor,
    model,
):
    frames = load_video_frames(video_path)

    inputs = processor(
        images=frames,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"]

    # [frames, 3, 224, 224]
    pixel_values = pixel_values.to(DEVICE)

    outputs = model(
        pixel_values=pixel_values
    )

    hidden = outputs.last_hidden_state

    # CLS token for each frame
    cls_embeddings = hidden[:, 0, :]

    # Temporal mean across frames
    embedding = cls_embeddings.mean(dim=0)

    return embedding.float().cpu()


# ============================================================
# PROCESS ONE SAMPLE
# ============================================================

def process_sample(
    row,
    tokenizer,
    text_model,
    audio_processor,
    audio_model,
    vision_processor,
    vision_model,
):

    segment_id = row["segment_id"]

    media_path = PROJECT_ROOT / row["media_path"]

    text = row["text"]

    print(f"\nProcessing: {segment_id}")
    print(f"Video: {media_path}")

    if not media_path.exists():
        raise FileNotFoundError(
            f"Missing video: {media_path}"
        )

    # ---------------- TEXT ----------------

    print("  Text...", end=" ", flush=True)

    text_embedding = extract_text_embedding(
        text,
        tokenizer,
        text_model,
    )

    print("OK", end=" ")

    # ---------------- AUDIO ----------------

    print("| Audio...", end=" ", flush=True)

    audio_embedding = extract_audio_embedding(
        media_path,
        audio_processor,
        audio_model,
    )

    print("OK", end=" ")

    # ---------------- VIDEO ----------------

    print("| Video...", end=" ", flush=True)

    video_embedding = extract_video_embedding(
        media_path,
        vision_processor,
        vision_model,
    )

    print("OK")

    return {
        "segment_id": segment_id,
        "text": text,

        "text_embedding": text_embedding,
        "audio_embedding": audio_embedding,
        "video_embedding": video_embedding,

        "labels": torch.tensor(
            [
                float(row["happy"]),
                float(row["sad"]),
                float(row["anger"]),
                float(row["surprise"]),
                float(row["disgust"]),
                float(row["fear"]),
            ],
            dtype=torch.float32,
        ),

        "video_id": row["video_id"],
        "segment_number": int(row["segment_number"]),
    }


# ============================================================
# SAVE SHARD
# ============================================================

def save_shard(samples, split, shard_number):

    output_file = (
        OUTPUT_DIR
        / f"{split}_shard_{shard_number:04d}.pt"
    )

    payload = {
        "segment_id": [
            item["segment_id"]
            for item in samples
        ],

        "video_id": [
            item["video_id"]
            for item in samples
        ],

        "segment_number": [
            item["segment_number"]
            for item in samples
        ],

        "text_embedding": torch.stack(
            [
                item["text_embedding"]
                for item in samples
            ]
        ),

        "audio_embedding": torch.stack(
            [
                item["audio_embedding"]
                for item in samples
            ]
        ),

        "video_embedding": torch.stack(
            [
                item["video_embedding"]
                for item in samples
            ]
        ),

        "labels": torch.stack(
            [
                item["labels"]
                for item in samples
            ]
        ),

        "text": [
            item["text"]
            for item in samples
        ],
    }

    torch.save(
        payload,
        output_file,
    )

    print(
        f"\nSaved shard: {output_file.name}"
    )

    return output_file


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split,
    tokenizer,
    text_model,
    audio_processor,
    audio_model,
    vision_processor,
    vision_model,
):

    csv_file = MARK2_DIR / f"{split}.csv"

    df = pd.read_csv(csv_file)

    print_header(
        f"PROCESSING {split.upper()} SPLIT"
    )

    print(f"Samples: {len(df)}")

    # Small shards keep RAM usage safe.
    SHARD_SIZE = 64

    samples = []

    shard_number = 0

    start_time = time.time()

    for index, row in df.iterrows():

        try:

            sample = process_sample(
                row,
                tokenizer,
                text_model,
                audio_processor,
                audio_model,
                vision_processor,
                vision_model,
            )

            samples.append(sample)

        except Exception as e:

            print()
            print(
                f"ERROR processing "
                f"{row['segment_id']}"
            )

            print(e)

            continue

        if len(samples) >= SHARD_SIZE:

            save_shard(
                samples,
                split,
                shard_number,
            )

            shard_number += 1

            samples = []

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        completed = index + 1

        elapsed = time.time() - start_time

        rate = completed / max(elapsed, 1)

        remaining = len(df) - completed

        eta = remaining / max(rate, 0.0001)

        print(
            f"Progress: "
            f"{completed}/{len(df)} "
            f"| {rate:.2f} samples/sec "
            f"| ETA: {eta / 60:.1f} min"
        )

    # Remaining samples

    if samples:

        save_shard(
            samples,
            split,
            shard_number,
        )

    elapsed = time.time() - start_time

    print()
    print(
        f"{split.upper()} COMPLETE"
    )

    print(
        f"Time: {elapsed / 60:.2f} minutes"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "MARK 2 - PRETRAINED FEATURE EXTRACTION"
    )

    print(f"Project: {PROJECT_ROOT}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Device:  {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU:     "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Check CSV files
    # --------------------------------------------------------

    for split in ["train", "val", "test"]:

        path = MARK2_DIR / f"{split}.csv"

        if not path.exists():

            raise FileNotFoundError(
                f"Missing dataset split: {path}"
            )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    (
        tokenizer,
        text_model,
        audio_processor,
        audio_model,
        vision_processor,
        vision_model,
    ) = load_models()

    # --------------------------------------------------------
    # Process splits
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test",
    ]:

        process_split(
            split,
            tokenizer,
            text_model,
            audio_processor,
            audio_model,
            vision_processor,
            vision_model,
        )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "text_model": TEXT_MODEL,
        "audio_model": AUDIO_MODEL,
        "vision_model": VISION_MODEL,

        "text_embedding_dim": 768,
        "audio_embedding_dim": 768,
        "video_embedding_dim": 768,

        "num_frames": NUM_FRAMES,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,

        "emotion_names": EMOTION_NAMES,

        "device": str(DEVICE),
    }

    metadata_file = (
        OUTPUT_DIR / "metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print_header(
        "MARK 2 FEATURE EXTRACTION COMPLETE"
    )

    print(
        f"Features saved to:\n{OUTPUT_DIR}"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Train the Mark 2 cross-modal fusion model."
    )


if __name__ == "__main__":
    main()