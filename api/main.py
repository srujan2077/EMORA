from pathlib import Path

import torch
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mmsdk import mmdatasdk

from src.models.multimodal_model import MultimodalEmotionModel
from src.data.mosei_dataset import MOSEIDataset
from src.data.split import create_video_splits


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "mark1_best.pt"
ALIGNED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "aligned_mark1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS = [
    "Happy",
    "Sad",
    "Anger",
    "Surprise",
    "Disgust",
    "Fear",
]


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="EMORA API",
    description="Multimodal Emotion Recognition API - Mark 1",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request schema
# ============================================================

class PredictionRequest(BaseModel):
    segment_id: str


# ============================================================
# Global objects
# ============================================================

model = None
dataset = None
test_ids = None


# ============================================================
# Load model
# ============================================================

def load_model():
    global model

    model = MultimodalEmotionModel(
        embedding_dim=128,
        fusion_dim=256,
        num_emotions=6,
        dropout=0.2,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # Support either a raw state_dict or checkpoint dictionary
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print(f"Model loaded on: {DEVICE}")


# ============================================================
# Load aligned CMU-MOSEI data
# ============================================================

def load_dataset():
    global dataset, test_ids

    dataset_files = {
        "glove_vectors": str(
            ALIGNED_DATA_PATH / "glove_vectors_aligned.csd"
        ),
        "COVAREP": str(
            ALIGNED_DATA_PATH / "COVAREP_aligned.csd"
        ),
        "FACET 4.2": str(
            ALIGNED_DATA_PATH / "FACET42_aligned.csd"
        ),
        "All Labels": str(
            ALIGNED_DATA_PATH / "labels_aligned.csd"
        ),
    }

    dataset = mmdatasdk.mmdataset(dataset_files)

    # Get IDs common to all modalities
    text_ids = set(
        dataset.computational_sequences["glove_vectors"].data.keys()
    )

    audio_ids = set(
        dataset.computational_sequences["COVAREP"].data.keys()
    )

    vision_ids = set(
        dataset.computational_sequences["FACET 4.2"].data.keys()
    )

    label_ids = set(
        dataset.computational_sequences["All Labels"].data.keys()
    )

    common_ids = sorted(
        text_ids
        & audio_ids
        & vision_ids
        & label_ids
    )

    # Same video-level split used by Mark 1
    _, _, test_ids = create_video_splits(
        common_ids,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
    )

    print(f"Test samples: {len(test_ids)}")


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup_event():
    load_model()
    load_dataset()


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "project": "EMORA",
        "version": "Mark 1",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "dataset_loaded": dataset is not None,
        "mark2_loaded": mark2_fusion_model is not None if 'mark2_fusion_model' in globals() else False,
    }


# ============================================================
# Samples endpoint
# ============================================================

@app.get("/samples")
def get_samples():
    if test_ids is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset is not loaded.",
        )

    return {
        "count": len(test_ids),
        "samples": test_ids,
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    if model is None or dataset is None:
        raise HTTPException(
            status_code=503,
            detail="Model or dataset is not loaded.",
        )

    if request.segment_id not in test_ids:
        raise HTTPException(
            status_code=404,
            detail="Segment ID not found in the test set.",
        )

    # Create dataset wrapper for this sample
    sample_dataset = MOSEIDataset(
        dataset,
        [request.segment_id],
    )

    sample = sample_dataset[0]

    text = sample["text"].unsqueeze(0).to(DEVICE)
    audio = sample["audio"].unsqueeze(0).to(DEVICE)
    vision = sample["vision"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(text, audio, vision)

        probabilities = torch.sigmoid(logits)[0]

    scores = {
        emotion: round(
            float(probabilities[i].cpu()),
            4,
        )
        for i, emotion in enumerate(EMOTIONS)
    }

    predicted_emotions = [
        emotion
        for emotion, score in scores.items()
        if score >= 0.5
    ]

    # Ground truth labels if available in sample
    ground_truth = []
    if "labels" in sample:
        gt_arr = sample["labels"].cpu().numpy().astype(int)
        ground_truth = [
            EMOTIONS[i] for i, val in enumerate(gt_arr) if val == 1
        ]

    modalities = {
        "text": {
            "processed": True,
            "frames": int(sample["text"].shape[0]),
            "feature_dim": int(sample["text"].shape[1]),
        },
        "audio": {
            "processed": True,
            "frames": int(sample["audio"].shape[0]),
            "feature_dim": int(sample["audio"].shape[1]),
        },
        "vision": {
            "processed": True,
            "frames": int(sample["vision"].shape[0]),
            "feature_dim": int(sample["vision"].shape[1]),
        },
    }

    return {
        "segment_id": request.segment_id,
        "predicted_emotions": predicted_emotions,
        "scores": scores,
        "threshold": 0.5,
        "modalities": modalities,
        "ground_truth": ground_truth,
    }

# ============================================================
# MARK 2 Endpoints & Logic
# ============================================================

import json
import tempfile
import subprocess
import numpy as np
import cv2
import os
import shutil
from typing import Optional
from fastapi import File, Form, UploadFile
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor, AutoModel
from src.models.fusion_v2 import Mark2FusionModel

mark2_fusion_model = None
mark2_tokenizer = None
mark2_text_model = None
mark2_audio_processor = None
mark2_audio_model = None
mark2_image_processor = None
mark2_vision_model = None
mark2_thresholds = None

def _find_ffmpeg():
    try:
        which_path = shutil.which("ffmpeg")
        if which_path:
            return which_path
        if os.path.exists("C:\\ffmpeg\\bin\\ffmpeg.exe"):
            return "C:\\ffmpeg\\bin\\ffmpeg.exe"
    except Exception:
        pass
    return "ffmpeg"

ffmpeg_path = _find_ffmpeg()

def load_mark2():
    global mark2_fusion_model, mark2_tokenizer, mark2_text_model
    global mark2_audio_processor, mark2_audio_model, mark2_image_processor, mark2_vision_model
    global mark2_thresholds

    print("\n" + "=" * 50)
    print("Loading Mark 2 Models & Encoders...")
    print(f"Device: {DEVICE}")

    ckpt_path = PROJECT_ROOT / "checkpoints" / "mark2_best.pt"
    thresh_path = PROJECT_ROOT / "checkpoints" / "mark2_thresholds.json"

    # 1. Load thresholds
    with open(thresh_path, 'r', encoding='utf-8') as f:
        tdata = json.load(f)
    mark2_thresholds = tdata.get("thresholds", tdata)
    print(f"Loaded Mark 2 thresholds: {mark2_thresholds}")

    # 2. Load Fusion Model (mark2_best.pt)
    mark2_fusion_model = Mark2FusionModel(
        input_dim=768,
        projection_dim=256,
        num_heads=8,
        num_layers=2,
        ff_dim=1024,
        dropout=0.1,
        num_emotions=6,
    ).to(DEVICE)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in checkpoint:
        mark2_fusion_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        mark2_fusion_model.load_state_dict(checkpoint)
    mark2_fusion_model.eval()
    print(f"Mark 2 fusion model loaded from {ckpt_path.name} (Epoch {checkpoint.get('epoch', '?')})")

    # 3. Load Pretrained Encoders
    print("Loading DeBERTa-v3-base...")
    mark2_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    mark2_text_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(DEVICE).eval()

    print("Loading Wav2Vec2-base-960h...")
    mark2_audio_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    mark2_audio_model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE).eval()

    print("Loading ViT-base-patch16-224-in21k...")
    mark2_image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    mark2_vision_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(DEVICE).eval()

    print("All Mark 2 models and encoders successfully loaded on GPU/Device.")
    print("=" * 50 + "\n")

@app.on_event("startup")
def startup_mark2():
    load_mark2()

def extract_audio_from_video(video_path: str):
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "f32le",
        "pipe:1"
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return None
    return np.frombuffer(result.stdout, dtype=np.float32)

def convert_audio(audio_path: str):
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(audio_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "f32le",
        "pipe:1"
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return None
    return np.frombuffer(result.stdout, dtype=np.float32)

@app.post("/predict/raw")
async def predict_raw(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None)
):
    if mark2_fusion_model is None or mark2_text_model is None:
        raise HTTPException(status_code=503, detail="Mark 2 pipeline is not initialized.")

    print("\n--- INFERENCE REQUEST ---")
    print("MODEL = Mark 2")

    mod_text_used = False
    mod_audio_used = False
    mod_video_used = False

    temp_video_path = None
    temp_audio_path = None

    try:
        # 1. TEXT ENCODER: DeBERTa-v3-base -> [1, 768]
        if text and text.strip():
            inputs = mark2_tokenizer(
                text.strip(),
                max_length=256,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                out = mark2_text_model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                text_emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            mod_text_used = True
            print("TEXT = processed")
        else:
            text_emb = torch.zeros((1, 768), dtype=torch.float32, device=DEVICE)
            print("TEXT = skipped")

        # Handle uploaded media files
        if video is not None:
            content = await video.read()
            if len(content) > 0:
                fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
                with os.fdopen(fd, 'wb') as f:
                    f.write(content)

        if audio is not None:
            content = await audio.read()
            if len(content) > 0:
                fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
                with os.fdopen(fd, 'wb') as f:
                    f.write(content)

        # 2. AUDIO ENCODER: Wav2Vec2-base-960h -> [1, 768]
        audio_array = None
        if temp_audio_path:
            audio_array = convert_audio(temp_audio_path)
        elif temp_video_path:
            # If no separate audio supplied, extract audio from video via FFmpeg
            audio_array = extract_audio_from_video(temp_video_path)

        if audio_array is not None and len(audio_array) > 0:
            inputs = mark2_audio_processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                out = mark2_audio_model(**inputs)
                audio_emb = out.last_hidden_state.mean(dim=1)
            mod_audio_used = True
            print("AUDIO = processed")
        else:
            audio_emb = torch.zeros((1, 768), dtype=torch.float32, device=DEVICE)
            print("AUDIO = skipped")

        # 3. VIDEO ENCODER: ViT-base-patch16-224-in21k (8 frames) -> [1, 768]
        if temp_video_path:
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            if total_frames > 0:
                indices = np.linspace(0, total_frames - 1, 8, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
            cap.release()

            if len(frames) > 0:
                while len(frames) < 8:
                    frames.append(frames[-1])

                inputs = mark2_image_processor(images=frames, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(DEVICE)
                with torch.no_grad():
                    out = mark2_vision_model(pixel_values=pixel_values)
                    cls_tokens = out.last_hidden_state[:, 0, :]
                    video_emb = cls_tokens.mean(dim=0, keepdim=True)
                mod_video_used = True
                print("VIDEO = processed")
            else:
                video_emb = torch.zeros((1, 768), dtype=torch.float32, device=DEVICE)
                print("VIDEO = skipped")
        else:
            video_emb = torch.zeros((1, 768), dtype=torch.float32, device=DEVICE)
            print("VIDEO = skipped")

        # 4. CROSS-MODAL FUSION: Mark2FusionModel (mark2_best.pt)
        print("FUSION = processed")
        print("CHECKPOINT = mark2_best.pt")

        with torch.no_grad():
            logits, _ = mark2_fusion_model(text_emb, audio_emb, video_emb)
            probs = torch.sigmoid(logits)[0]

        # 5. MULTI-LABEL EMOTION EVALUATION USING OPTIMIZED THRESHOLDS
        emotion_names = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
        emotions_result = {}
        dominant_emotion = None
        highest_prob = -1.0

        for i, emp in enumerate(emotion_names):
            p = float(probs[i].cpu())
            t = float(mark2_thresholds.get(emp, 0.5)) if mark2_thresholds else 0.5
            emotions_result[emp] = {
                "probability": round(p, 4),
                "threshold": round(t, 4),
                "detected": bool(p >= t)
            }
            if p > highest_prob:
                highest_prob = p
                dominant_emotion = emp

        response_payload = {
            "success": True,
            "model": "Mark 2",
            "modalities": {
                "text": mod_text_used,
                "audio": mod_audio_used,
                "video": mod_video_used
            },
            "emotions": emotions_result,
            "dominant_emotion": dominant_emotion
        }
        return response_payload

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass
