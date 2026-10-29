from pathlib import Path
import logging
import time
import uuid

import torch
import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel

from mmsdk import mmdatasdk

from src.models.multimodal_model import MultimodalEmotionModel
from src.data.mosei_dataset import MOSEIDataset
from src.data.split import create_video_splits
from api.config import (
    CORS_ORIGINS,
    MARK3_CHECKPOINT_ROOT,
    MARK3_REPORT_PATH,
    MAX_REQUEST_BYTES,
    MAX_UPLOAD_BYTES,
)
from api.logging_config import request_finished


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
    description="Multimodal Emotion Recognition API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestTelemetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        content_length = request.headers.get("content-length")
        try:
            request_size = int(content_length) if content_length else 0
        except ValueError:
            request_size = MAX_REQUEST_BYTES + 1
        if request_size > MAX_REQUEST_BYTES:
            response = JSONResponse(
                status_code=413,
                content={"detail": "Request body exceeds the configured size limit."},
            )
            response.headers["X-Request-ID"] = request_id
            return response
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logging.getLogger("emora.api").exception(
                "request_failed request_id=%s method=%s path=%s",
                request_id,
                request.method,
                request.url.path,
            )
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error.", "request_id": request_id},
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        request_finished(
            request_id, request.method, request.url.path, response.status_code, elapsed_ms
        )
        return response


app.add_middleware(RequestTelemetryMiddleware)


# ============================================================
# Request schema
# ============================================================

class PredictionRequest(BaseModel):
    segment_id: str


# ============================================================
# Global objects
# ============================================================

model = None
model_loaded = False
dataset = None
dataset_loaded = False
test_ids = None

mark2_loaded = False
mark3_loaded = False


# ============================================================
# Load model
# ============================================================

def load_model():
    global model, model_loaded

    try:
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
        model_loaded = True
        print(f"Model loaded on: {DEVICE}")
    except Exception as exc:
        model = None
        model_loaded = False
        print(f"WARNING: Mark 1 model initialization failed ({exc.__class__.__name__}): {exc}")


# ============================================================
# Load aligned CMU-MOSEI data
# ============================================================

def load_dataset():
    global dataset, test_ids, dataset_loaded

    dataset = None
    test_ids = None
    dataset_loaded = False

    dataset_files = {
        "glove_vectors": ALIGNED_DATA_PATH / "glove_vectors_aligned.csd",
        "COVAREP": ALIGNED_DATA_PATH / "COVAREP_aligned.csd",
        "FACET 4.2": ALIGNED_DATA_PATH / "FACET42_aligned.csd",
        "All Labels": ALIGNED_DATA_PATH / "labels_aligned.csd",
    }

    missing_files = [k for k, p in dataset_files.items() if not p.exists()]
    if missing_files:
        print(
            f"WARNING: Mark 1 dataset files missing: {', '.join(missing_files)}. "
            "Mark 1 dataset-dependent endpoints (/samples, /predict) will return 503."
        )
        return

    try:
        csd_str_files = {k: str(p) for k, p in dataset_files.items()}
        dataset = mmdatasdk.mmdataset(csd_str_files)

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

        dataset_loaded = True
        print(f"Test samples: {len(test_ids)}")
    except Exception as exc:
        dataset = None
        test_ids = None
        dataset_loaded = False
        print(
            f"WARNING: Mark 1 dataset initialization failed ({exc.__class__.__name__}): {exc}. "
            "Mark 1 dataset-dependent endpoints will return 503."
        )


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as exc:
        print(f"WARNING: Mark 1 model startup failed: {exc}")
    try:
        load_dataset()
    except Exception as exc:
        print(f"WARNING: Mark 1 dataset startup failed: {exc}")


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():
    active_model = (
        "Mark 3"
        if mark3_loaded and mark3_raw_pipeline is not None
        else "Mark 2"
        if mark2_loaded and mark2_fusion_model is not None
        else "Mark 1"
        if model_loaded and model is not None
        else None
    )
    return {
        "status": "ok",
        "project": "EMORA",
        "version": active_model,
        "api_version": app.version,
        "active_model": active_model,
        "device": str(DEVICE),
        "model_loaded": bool(model_loaded and model is not None),
        "dataset_loaded": bool(dataset_loaded and dataset is not None),
        "mark2_loaded": bool(mark2_loaded and mark2_fusion_model is not None and mark2_text_model is not None),
        "mark3_loaded": bool(mark3_loaded and mark3_raw_pipeline is not None),
    }


@app.get("/live")
def live():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if not mark3_loaded or mark3_raw_pipeline is None:
        raise HTTPException(status_code=503, detail="Mark 3 model is not ready.")
    return {
        "status": "ready",
        "model": "Mark 3",
        "candidate": "split_at_gap",
        "ensemble_size": 5,
        "threshold_policy": "fixed_0_5",
    }


# ============================================================
# Samples endpoint
# ============================================================

@app.get("/samples")
def get_samples():
    if not dataset_loaded or dataset is None or test_ids is None:
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

    if model is None or not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    if dataset is None or test_ids is None or not dataset_loaded:
        raise HTTPException(
            status_code=503,
            detail="Dataset is not loaded.",
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
from pathlib import Path
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
mark3_raw_pipeline = None

import asyncio
from contextlib import asynccontextmanager

DEFAULT_MAX_CONCURRENT_INFERENCE = 1
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 30.0

MAX_CONCURRENT_INFERENCE = int(
    os.getenv("EMORA_MAX_CONCURRENT_INFERENCE", str(DEFAULT_MAX_CONCURRENT_INFERENCE))
)
INFERENCE_TIMEOUT_SECONDS = float(
    os.getenv("EMORA_INFERENCE_TIMEOUT_SECONDS", str(DEFAULT_INFERENCE_TIMEOUT_SECONDS))
)

_inference_semaphore = None
_inference_semaphore_loop = None


def get_inference_semaphore() -> asyncio.Semaphore:
    """Return the singleton inference semaphore for the current event loop."""
    global _inference_semaphore, _inference_semaphore_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _inference_semaphore is None or (_inference_semaphore_loop is not None and _inference_semaphore_loop is not current_loop):
        _inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)
        _inference_semaphore_loop = current_loop
    return _inference_semaphore


def reset_inference_semaphore(limit: Optional[int] = None) -> asyncio.Semaphore:
    """Reset or reconfigure the inference semaphore (useful for testing)."""
    global _inference_semaphore, _inference_semaphore_loop
    val = limit if limit is not None else MAX_CONCURRENT_INFERENCE
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    _inference_semaphore = asyncio.Semaphore(val)
    _inference_semaphore_loop = current_loop
    return _inference_semaphore


@asynccontextmanager
async def inference_guard(timeout: Optional[float] = None):
    """Async context manager bounding concurrent GPU-heavy model and encoder inference.

    Ensures that at most MAX_CONCURRENT_INFERENCE requests execute GPU inference
    simultaneously. Waits up to timeout seconds (default INFERENCE_TIMEOUT_SECONDS);
    if capacity is unavailable, raises HTTP 503. Guarantees permit release on success,
    exception, HTTP error, or task cancellation.
    """
    sem = get_inference_semaphore()
    effective_timeout = timeout if timeout is not None else INFERENCE_TIMEOUT_SECONDS
    acquired = False
    try:
        try:
            await asyncio.wait_for(sem.acquire(), timeout=effective_timeout)
            acquired = True
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="GPU inference queue is busy; request timed out waiting for capacity.",
            )
        yield
    finally:
        if acquired:
            sem.release()


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

MAX_MEDIA_DURATION_SECONDS = 300
FFMPEG_TIMEOUT_SECONDS = 30
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _validate_media_duration(duration_seconds: float, media_kind: str) -> None:
    if duration_seconds > MAX_MEDIA_DURATION_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{media_kind.title()} duration exceeds the "
                f"{MAX_MEDIA_DURATION_SECONDS // 60}-minute limit."
            ),
        )


async def _save_upload(upload: UploadFile, kind: str) -> str:
    filename = Path(upload.filename or "").name
    extension = Path(filename).suffix.lower()
    supported_extensions = (
        SUPPORTED_AUDIO_EXTENSIONS
        if kind == "audio"
        else SUPPORTED_VIDEO_EXTENSIONS
    )
    if extension not in supported_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported {kind} format. "
            f"Supported extensions: {', '.join(sorted(supported_extensions))}.",
        )

    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=422, detail=f"{kind.title()} upload is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{kind.title()} upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    suffix = extension
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as file:
        file.write(content)
    return path

def load_mark2():
    global mark2_fusion_model, mark2_tokenizer, mark2_text_model
    global mark2_audio_processor, mark2_audio_model, mark2_image_processor, mark2_vision_model
    global mark2_thresholds, mark2_loaded

    mark2_loaded = False
    print("\n" + "=" * 50)
    print("Loading Mark 2 Models & Encoders...")
    print(f"Device: {DEVICE}")

    try:
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

        mark2_loaded = True
        print("All Mark 2 models and encoders successfully loaded on GPU/Device.")
    except Exception as exc:
        mark2_loaded = False
        mark2_fusion_model = None
        mark2_tokenizer = None
        mark2_text_model = None
        mark2_audio_processor = None
        mark2_audio_model = None
        mark2_image_processor = None
        mark2_vision_model = None
        mark2_thresholds = None
        print(f"WARNING: Mark 2 model initialization failed ({exc.__class__.__name__}): {exc}")
    print("=" * 50 + "\n")


def load_mark3():
    global mark3_raw_pipeline, mark3_loaded
    mark3_loaded = False
    print("\n" + "=" * 50)
    print("Loading Mark 3 Pipeline...")
    print(f"Device: {DEVICE}")

    try:
        from src.encoders.audio_encoder import AudioEncoder
        from src.encoders.text_encoder import TextEncoder
        from src.encoders.video_encoder import VideoEncoder
        from src.inference.mark3 import Mark3Inference
        from src.inference.mark3_raw import Mark3RawInference

        text_enc = TextEncoder(
            device=DEVICE,
            tokenizer=mark2_tokenizer,
            model=mark2_text_model,
        )
        audio_enc = AudioEncoder(
            device=DEVICE,
            processor=mark2_audio_processor,
            model=mark2_audio_model,
            ffmpeg_path=ffmpeg_path,
        )
        video_enc = VideoEncoder(
            device=DEVICE,
            processor=mark2_image_processor,
            model=mark2_vision_model,
        )
        mark3_inf = Mark3Inference(
            checkpoint_root=MARK3_CHECKPOINT_ROOT,
            report_path=MARK3_REPORT_PATH,
            device=DEVICE,
        )
        mark3_raw_pipeline = Mark3RawInference(
            mark3_inference=mark3_inf,
            text_encoder=text_enc,
            audio_encoder=audio_enc,
            video_encoder=video_enc,
            device=DEVICE,
        )
        mark3_loaded = True
        print("Mark 3 raw-media pipeline loaded successfully.")
    except Exception as exc:
        mark3_loaded = False
        mark3_raw_pipeline = None
        print(f"WARNING: Mark 3 initialization failed ({exc.__class__.__name__}): {exc}")
    print("=" * 50 + "\n")


@app.on_event("startup")
def startup_mark2():
    try:
        load_mark2()
    except Exception as exc:
        print(f"WARNING: Mark 2 startup failed: {exc}")
    try:
        load_mark3()
    except Exception as exc:
        print(f"WARNING: Mark 3 startup failed: {exc}")

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
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while extracting audio from video.") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"Video audio extraction failed: "
            f"{result.stderr.decode(errors='ignore').strip() or 'unknown FFmpeg error'}"
        )
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
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while converting audio.") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"Audio conversion failed: "
            f"{result.stderr.decode(errors='ignore').strip() or 'unknown FFmpeg error'}"
        )
    return np.frombuffer(result.stdout, dtype=np.float32)

def _run_mark2_inference(
    text: Optional[str],
    temp_video_path: Optional[str],
    temp_audio_path: Optional[str],
) -> dict:
    """Synchronous CPU/GPU inference for Mark 2, executed in a worker thread."""
    mod_text_used = False
    mod_audio_used = False
    mod_video_used = False

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

    # 2. AUDIO ENCODER: Wav2Vec2-base-960h -> [1, 768]
    audio_array = None
    try:
        if temp_audio_path:
            audio_array = convert_audio(temp_audio_path)
        elif temp_video_path:
            # If no separate audio supplied, extract audio from video via FFmpeg
            audio_array = extract_audio_from_video(temp_video_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if audio_array is not None and len(audio_array) > 0:
        _validate_media_duration(len(audio_array) / 16000, "audio")
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
        raise HTTPException(
            status_code=422,
            detail="No usable audio was found in the uploaded audio or video.",
        )

    # 3. VIDEO ENCODER: ViT-base-patch16-224-in21k (8 frames) -> [1, 768]
    if temp_video_path:
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps > 0 and total_frames / fps > MAX_MEDIA_DURATION_SECONDS:
            cap.release()
            _validate_media_duration(total_frames / fps, "video")
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
            if len(frames) < 8:
                print(f"WARNING: Decoded {len(frames)} frames (expected 8) from video.")
            inputs = mark2_image_processor(images=frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(DEVICE)
            with torch.no_grad():
                out = mark2_vision_model(pixel_values=pixel_values)
                cls_tokens = out.last_hidden_state[:, 0, :]
                video_emb = cls_tokens.mean(dim=0, keepdim=True)
            mod_video_used = True
            print("VIDEO = processed")
        else:
            raise HTTPException(
                status_code=422,
                detail="Video contains no decodable frames.",
            )

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

    return {
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

    temp_video_path = None
    temp_audio_path = None

    try:
        if not (text and text.strip()) and audio is None and video is None:
            raise HTTPException(
                status_code=400,
                detail="Provide text, audio, and video for a complete Mark 2 prediction.",
            )
        if not text or not text.strip():
            raise HTTPException(
                status_code=422,
                detail="Text is required because Mark 2 was trained with all three modalities.",
            )
        if video is None:
            raise HTTPException(
                status_code=422,
                detail="Video is required because Mark 2 was trained with all three modalities.",
            )

        # Handle uploaded media files
        if video is not None:
            temp_video_path = await _save_upload(video, "video")

        if audio is not None:
            temp_audio_path = await _save_upload(audio, "audio")

        async with inference_guard():
            return await asyncio.to_thread(
                _run_mark2_inference,
                text=text,
                temp_video_path=temp_video_path,
                temp_audio_path=temp_audio_path,
            )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError as exc:
                print(f"WARNING: failed to remove temporary video file: {exc}")
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError as exc:
                print(f"WARNING: failed to remove temporary audio file: {exc}")


# ============================================================
# MARK 3 Endpoints & Logic
# ============================================================

@app.post("/predict/mark3/raw")
async def predict_mark3_raw(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
):
    """Raw-input inference for Mark 3 temporal fusion model."""
    if mark3_raw_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Mark 3 pipeline is not initialized.",
        )

    print("\n--- INFERENCE REQUEST ---")
    print("MODEL = Mark 3")

    temp_video_path = None
    temp_audio_path = None

    try:
        if not (text and text.strip()) and audio is None and video is None:
            raise HTTPException(
                status_code=400,
                detail="Provide text and video/audio for a complete Mark 3 prediction.",
            )
        if video is None:
            raise HTTPException(
                status_code=422,
                detail="Video is required for Mark 3 raw multimodal inference.",
            )

        temp_video_path = await _save_upload(video, "video")
        if audio is not None:
            temp_audio_path = await _save_upload(audio, "audio")

        async with inference_guard():
            # Predict single segment via Mark3RawInference in worker thread
            prediction = await asyncio.to_thread(
                mark3_raw_pipeline.predict_single_segment,
                text=text,
                video_path=temp_video_path,
                audio_path=temp_audio_path or temp_video_path,
            )

            dominant_emotion = max(
                prediction.probabilities,
                key=prediction.probabilities.get,
            )

            emotions_result = {
                emp: {
                    "probability": round(prob, 4),
                    "threshold": round(prediction.thresholds[emp], 4),
                    "detected": emp in prediction.predicted_emotions,
                }
                for emp, prob in prediction.probabilities.items()
            }

            response_payload = {
                "success": True,
                "model": "Mark 3",
                "modalities": {
                    "text": bool(text and text.strip()),
                    "audio": True,
                    "video": True,
                },
                "emotions": emotions_result,
                "dominant_emotion": dominant_emotion,
                "predicted_emotions": list(prediction.predicted_emotions),
                "metadata": prediction.metadata,
            }
            return response_payload

    except HTTPException:
        raise
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError as exc:
                print(f"WARNING: failed to remove temporary video file: {exc}")
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError as exc:
                print(f"WARNING: failed to remove temporary audio file: {exc}")


@app.post("/predict/mark3/manifest")
async def predict_mark3_manifest(manifest_data: dict):
    """Conversational temporal sequence inference for Mark 3 from a segment manifest."""
    if mark3_raw_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Mark 3 pipeline is not initialized.",
        )

    try:
        from src.inference.segment_manifest import SegmentManifest

        if not isinstance(manifest_data, dict):
            raise ValueError("Request body must be a JSON object.")

        if "manifest" in manifest_data:
            raw_manifest = manifest_data["manifest"]
            target_segment_number = manifest_data.get("target_segment_number")
            if not isinstance(raw_manifest, dict):
                raise ValueError("'manifest' must be a JSON object.")
        else:
            raw_manifest = manifest_data
            target_segment_number = manifest_data.get("target_segment_number")

        if target_segment_number is not None and (
            isinstance(target_segment_number, bool)
            or not isinstance(target_segment_number, int)
        ):
            raise ValueError("'target_segment_number' must be an integer.")

        try:
            manifest = SegmentManifest.from_dict(raw_manifest)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Mark 3 manifest: {exc}") from exc
        async with inference_guard():
            prediction = await asyncio.to_thread(
                mark3_raw_pipeline.predict_manifest,
                manifest,
                target_segment_number=target_segment_number,
            )

            dominant_emotion = max(
                prediction.probabilities,
                key=prediction.probabilities.get,
            )

            emotions_result = {
                emp: {
                    "probability": round(prob, 4),
                    "threshold": round(prediction.thresholds[emp], 4),
                    "detected": emp in prediction.predicted_emotions,
                }
                for emp, prob in prediction.probabilities.items()
            }

            return {
                "success": True,
                "model": "Mark 3",
                "emotions": emotions_result,
                "dominant_emotion": dominant_emotion,
                "predicted_emotions": list(prediction.predicted_emotions),
                "metadata": prediction.metadata,
            }
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")
