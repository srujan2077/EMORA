import os

# Must be set before importing torch.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
)


# ============================================================
# MARK 2.5 FAST / RESUME TRAINER
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

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "mark2_5_fast_best.pt"
)

RESUME_CHECKPOINT = (
    CHECKPOINT_DIR
    / "mark2_5_fast_resume.pt"
)

THRESHOLD_FILE = (
    CHECKPOINT_DIR
    / "mark2_5_fast_thresholds.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PRETRAINED MODELS
# ============================================================

TEXT_MODEL_NAME = (
    "microsoft/deberta-v3-base"
)

AUDIO_MODEL_NAME = (
    "facebook/wav2vec2-base-960h"
)

VISION_MODEL_NAME = (
    "google/vit-base-patch16-224-in21k"
)


# ============================================================
# MODEL CONFIG
# ============================================================

NUM_FRAMES = 8

AUDIO_SAMPLE_RATE = 16000

TEXT_MAX_LENGTH = 256

PROJECTION_DIM = 256

NUM_EMOTIONS = 6

EMOTION_NAMES = [
    "happy",
    "sad",
    "anger",
    "surprise",
    "disgust",
    "fear",
]

NUM_HEADS = 8

NUM_FUSION_LAYERS = 2

FUSION_FF_DIM = 1024

DROPOUT = 0.1


# ============================================================
# TRAINING CONFIG
# ============================================================

# IMPORTANT:
# Batch 2 caused Wav2Vec2 OOM on the 6 GB RTX 4050.
#
# Therefore we use batch 1 and retain an effective batch
# size of 8 through gradient accumulation.

BATCH_SIZE = 1

GRADIENT_ACCUMULATION = 8

# We only want to continue to Epoch 9.
TARGET_EPOCH = 9

BACKBONE_LR = 1e-5

NEW_LAYER_LR = 5e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 3

MAX_GRAD_NORM = 1.0

USE_AMP = True


# ============================================================
# DATALOADER
# ============================================================

NUM_WORKERS = 0

PIN_MEMORY = True


# ============================================================
# SEED
# ============================================================

SEED = 42

torch.manual_seed(SEED)

np.random.seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)

    torch.set_float32_matmul_precision(
        "high"
    )


# ============================================================
# HEADER
# ============================================================

def header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# SAFE SEGMENT ID
# ============================================================

def safe_segment_id(segment_id):

    return (
        str(segment_id)
        .replace("[", "_")
        .replace("]", "")
        .replace("/", "_")
        .replace("\\", "_")
    )


# ============================================================
# DATASET
# ============================================================

class FastMark25Dataset(Dataset):

    def __init__(self, split):

        self.split = split

        csv_path = (
            MARK2_DIR
            / f"{split}.csv"
        )

        if not csv_path.exists():

            raise FileNotFoundError(
                f"Missing CSV:\n{csv_path}"
            )

        self.df = pd.read_csv(
            csv_path
        )

        print(
            f"{split}: "
            f"{len(self.df)} samples"
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        segment_id = str(
            row["segment_id"]
        )

        safe_id = safe_segment_id(
            segment_id
        )

        split_dir = (
            MEDIA_CACHE
            / self.split
        )

        audio_path = (
            split_dir
            / f"{safe_id}_audio.npy"
        )

        frames_path = (
            split_dir
            / f"{safe_id}_frames.npy"
        )

        if not audio_path.exists():

            raise FileNotFoundError(
                f"Missing cached audio:\n"
                f"{audio_path}"
            )

        if not frames_path.exists():

            raise FileNotFoundError(
                f"Missing cached frames:\n"
                f"{frames_path}"
            )

        audio = np.load(
            audio_path,
            mmap_mode="r"
        )

        frames = np.load(
            frames_path,
            mmap_mode="r"
        )

        labels = np.array(
            [
                float(row["happy"]),
                float(row["sad"]),
                float(row["anger"]),
                float(row["surprise"]),
                float(row["disgust"]),
                float(row["fear"]),
            ],
            dtype=np.float32
        )

        return {
            "segment_id": segment_id,
            "text": str(row["text"]),
            "audio": np.asarray(
                audio,
                dtype=np.float32
            ),
            "frames": np.asarray(
                frames,
                dtype=np.uint8
            ),
            "labels": labels,
        }


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    return {
        "segment_ids": [
            item["segment_id"]
            for item in batch
        ],

        "texts": [
            item["text"]
            for item in batch
        ],

        "audios": [
            item["audio"]
            for item in batch
        ],

        "frames": [
            item["frames"]
            for item in batch
        ],

        "labels": torch.tensor(
            np.stack(
                [
                    item["labels"]
                    for item in batch
                ]
            ),
            dtype=torch.float32
        ),
    }


# ============================================================
# FIND TRANSFORMER LAYERS
# ============================================================

def find_layer_list(
    module,
    model_name
):

    paths = [
        "encoder.layer",
        "encoder.layers",
        "transformer.layer",
        "transformer.layers",
        "layers",
        "layer",
    ]

    candidates = []

    for path in paths:

        current = module

        valid = True

        for part in path.split("."):

            if not hasattr(
                current,
                part
            ):

                valid = False

                break

            current = getattr(
                current,
                part
            )

        if valid and isinstance(
            current,
            nn.ModuleList
        ):

            candidates.append(
                (
                    path,
                    current
                )
            )

    if not candidates:

        for name, child in (
            module.named_modules()
        ):

            if isinstance(
                child,
                nn.ModuleList
            ):

                if len(child) >= 4:

                    candidates.append(
                        (
                            name,
                            child
                        )
                    )

    if not candidates:

        raise RuntimeError(
            f"Could not find "
            f"transformer layers for "
            f"{model_name}"
        )

    candidates.sort(
        key=lambda x: len(x[1]),
        reverse=True
    )

    path, layers = candidates[0]

    print(
        f"{model_name} layer container: "
        f"{path}"
    )

    print(
        f"{model_name} number of layers: "
        f"{len(layers)}"
    )

    return layers


# ============================================================
# MODEL
# ============================================================

class Mark25FastModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        self.text_tokenizer = (
            AutoTokenizer.from_pretrained(
                TEXT_MODEL_NAME
            )
        )

        self.text_encoder = (
            AutoModel.from_pretrained(
                TEXT_MODEL_NAME
            )
        )

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        self.audio_processor = (
            AutoProcessor.from_pretrained(
                AUDIO_MODEL_NAME
            )
        )

        self.audio_encoder = (
            AutoModel.from_pretrained(
                AUDIO_MODEL_NAME
            )
        )

        # ----------------------------------------------------
        # VISION
        # ----------------------------------------------------

        self.vision_processor = (
            AutoImageProcessor.from_pretrained(
                VISION_MODEL_NAME
            )
        )

        self.vision_encoder = (
            AutoModel.from_pretrained(
                VISION_MODEL_NAME
            )
        )

        # Keep parameters FP32.
        self.text_encoder.float()
        self.audio_encoder.float()
        self.vision_encoder.float()

        # ----------------------------------------------------
        # LAYERS
        # ----------------------------------------------------

        self.text_layers = (
            find_layer_list(
                self.text_encoder,
                "DeBERTa"
            )
        )

        self.audio_layers = (
            find_layer_list(
                self.audio_encoder,
                "Wav2Vec2"
            )
        )

        self.vision_layers = (
            find_layer_list(
                self.vision_encoder,
                "ViT"
            )
        )

        # ----------------------------------------------------
        # SELECTIVE FINE-TUNING
        # ----------------------------------------------------

        self.configure_backbones()

        # ----------------------------------------------------
        # PROJECTIONS
        # ----------------------------------------------------

        self.text_projection = nn.Sequential(
            nn.Linear(
                768,
                PROJECTION_DIM
            ),
            nn.LayerNorm(
                PROJECTION_DIM
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            )
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(
                768,
                PROJECTION_DIM
            ),
            nn.LayerNorm(
                PROJECTION_DIM
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            )
        )

        self.video_projection = nn.Sequential(
            nn.Linear(
                768,
                PROJECTION_DIM
            ),
            nn.LayerNorm(
                PROJECTION_DIM
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            )
        )

        # ----------------------------------------------------
        # CLS
        # ----------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                PROJECTION_DIM
            )
        )

        # ----------------------------------------------------
        # MODALITY EMBEDDINGS
        # ----------------------------------------------------

        self.modality_embeddings = nn.Parameter(
            torch.randn(
                1,
                4,
                PROJECTION_DIM
            )
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=PROJECTION_DIM,
                nhead=NUM_HEADS,
                dim_feedforward=FUSION_FF_DIM,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.fusion = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=NUM_FUSION_LAYERS
            )
        )

        self.fusion_norm = (
            nn.LayerNorm(
                PROJECTION_DIM
            )
        )

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                PROJECTION_DIM,
                PROJECTION_DIM
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                PROJECTION_DIM,
                NUM_EMOTIONS
            )
        )

    # ========================================================
    # BACKBONE CONFIG
    # ========================================================

    def configure_backbones(self):

        for parameter in (
            self.text_encoder.parameters()
        ):

            parameter.requires_grad = False

        for parameter in (
            self.audio_encoder.parameters()
        ):

            parameter.requires_grad = False

        for parameter in (
            self.vision_encoder.parameters()
        ):

            parameter.requires_grad = False

        print()
        print(
            "Selective fine-tuning:"
        )

        for layer in (
            self.text_layers[-2:]
        ):

            for parameter in (
                layer.parameters()
            ):

                parameter.requires_grad = True

        print(
            "  DeBERTa: "
            "unfrozen last 2 layers"
        )

        for layer in (
            self.audio_layers[-2:]
        ):

            for parameter in (
                layer.parameters()
            ):

                parameter.requires_grad = True

        print(
            "  Wav2Vec2: "
            "unfrozen last 2 layers"
        )

        for layer in (
            self.vision_layers[-2:]
        ):

            for parameter in (
                layer.parameters()
            ):

                parameter.requires_grad = True

        print(
            "  ViT: "
            "unfrozen last 2 layers"
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self, mode=True):

        super().train(mode)

        self.text_encoder.eval()
        self.audio_encoder.eval()
        self.vision_encoder.eval()

        if mode:

            for layer in (
                self.text_layers[-2:]
            ):

                layer.train()

            for layer in (
                self.audio_layers[-2:]
            ):

                layer.train()

            for layer in (
                self.vision_layers[-2:]
            ):

                layer.train()

        return self

    # ========================================================
    # TEXT
    # ========================================================

    def encode_text(
        self,
        texts
    ):

        encoded = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TEXT_MAX_LENGTH
        )

        encoded = {
            key: value.to(
                DEVICE,
                non_blocking=True
            )
            for key, value
            in encoded.items()
        }

        outputs = self.text_encoder(
            **encoded
        )

        hidden = (
            outputs.last_hidden_state
        )

        mask = (
            encoded["attention_mask"]
            .unsqueeze(-1)
            .to(hidden.dtype)
        )

        hidden = (
            hidden * mask
        )

        embedding = (
            hidden.sum(dim=1)
            /
            mask.sum(dim=1).clamp(
                min=1
            )
        )

        return embedding

    # ========================================================
    # AUDIO
    # ========================================================

    def encode_audio(
        self,
        audios
    ):

        processed = (
            self.audio_processor(
                audios,
                sampling_rate=AUDIO_SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
        )

        processed = {
            key: value.to(
                DEVICE,
                non_blocking=True
            )
            for key, value
            in processed.items()
        }

        outputs = self.audio_encoder(
            **processed
        )

        hidden = (
            outputs.last_hidden_state
        )

        if "attention_mask" in processed:

            mask = (
                processed["attention_mask"]
                .float()
            )

            mask = (
                torch.nn.functional.interpolate(
                    mask.unsqueeze(1),
                    size=hidden.size(1),
                    mode="nearest"
                )
                .squeeze(1)
            )

            mask = mask.to(
                hidden.dtype
            )

            hidden = (
                hidden
                *
                mask.unsqueeze(-1)
            )

            embedding = (
                hidden.sum(dim=1)
                /
                mask.sum(dim=1)
                .clamp(min=1)
                .unsqueeze(-1)
            )

        else:

            embedding = (
                hidden.mean(dim=1)
            )

        return embedding

    # ========================================================
    # VIDEO
    # ========================================================

    def encode_video(
        self,
        frames_batch
    ):

        flattened_frames = []

        frame_counts = []

        for frames in frames_batch:

            frames = np.asarray(
                frames
            )

            frame_counts.append(
                len(frames)
            )

            for frame in frames:

                flattened_frames.append(
                    frame
                )

        processed = (
            self.vision_processor(
                images=flattened_frames,
                return_tensors="pt"
            )
        )

        pixels = (
            processed["pixel_values"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )

        outputs = self.vision_encoder(
            pixel_values=pixels
        )

        cls = (
            outputs
            .last_hidden_state[
                :,
                0,
                :
            ]
        )

        embeddings = []

        start = 0

        for count in frame_counts:

            end = start + count

            embeddings.append(
                cls[
                    start:end
                ].mean(dim=0)
            )

            start = end

        return torch.stack(
            embeddings
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        texts,
        audios,
        frames
    ):

        text_features = (
            self.encode_text(
                texts
            )
        )

        audio_features = (
            self.encode_audio(
                audios
            )
        )

        video_features = (
            self.encode_video(
                frames
            )
        )

        text = (
            self.text_projection(
                text_features
            )
        )

        audio = (
            self.audio_projection(
                audio_features
            )
        )

        video = (
            self.video_projection(
                video_features
            )
        )

        batch_size = (
            text.size(0)
        )

        cls = (
            self.cls_token.expand(
                batch_size,
                -1,
                -1
            )
        )

        sequence = torch.cat(
            [
                cls,
                text.unsqueeze(1),
                audio.unsqueeze(1),
                video.unsqueeze(1)
            ],
            dim=1
        )

        sequence = (
            sequence
            +
            self.modality_embeddings
        )

        fused_sequence = (
            self.fusion(
                sequence
            )
        )

        fusion = (
            fused_sequence[
                :,
                0,
                :
            ]
        )

        fusion = (
            self.fusion_norm(
                fusion
            )
        )

        logits = (
            self.classifier(
                fusion
            )
        )

        return logits, fusion


# ============================================================
# PARAMETER REPORT
# ============================================================

def parameter_report(model):

    groups = {
        "DeBERTa":
            model.text_encoder,

        "Wav2Vec2":
            model.audio_encoder,

        "ViT":
            model.vision_encoder,

        "Text projection":
            model.text_projection,

        "Audio projection":
            model.audio_projection,

        "Video projection":
            model.video_projection,

        "Fusion":
            model.fusion,

        "Classifier":
            model.classifier
    }

    total = 0
    trainable = 0

    print()
    print("=" * 70)
    print(
        "PARAMETER REPORT"
    )
    print("=" * 70)

    for name, module in groups.items():

        module_total = sum(
            p.numel()
            for p in module.parameters()
        )

        module_trainable = sum(
            p.numel()
            for p in module.parameters()
            if p.requires_grad
        )

        total += module_total
        trainable += module_trainable

        print(
            f"{name:20s}"
            f" total={module_total:,}"
            f" trainable={module_trainable:,}"
        )

    extra = (
        model.cls_token.numel()
        +
        model.modality_embeddings.numel()
    )

    total += extra
    trainable += extra

    print()
    print(
        f"Total parameters: "
        f"{total:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Frozen parameters: "
        f"{total - trainable:,}"
    )


# ============================================================
# POS WEIGHT
# ============================================================

def calculate_pos_weight(
    dataset
):

    labels = []

    for index in range(
        len(dataset)
    ):

        labels.append(
            dataset[index]["labels"]
        )

    labels = torch.tensor(
        np.stack(labels),
        dtype=torch.float32
    )

    positives = (
        labels.sum(dim=0)
    )

    negatives = (
        labels.size(0)
        -
        positives
    )

    return (
        negatives
        /
        positives.clamp(min=1)
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    labels,
    thresholds=None
):

    probabilities = (
        torch.sigmoid(
            logits
        )
        .cpu()
        .numpy()
    )

    labels_np = (
        labels
        .cpu()
        .numpy()
    )

    if thresholds is None:

        thresholds = (
            np.ones(
                NUM_EMOTIONS
            )
            * 0.5
        )

    predictions = (
        probabilities
        >=
        np.asarray(
            thresholds
        )
    ).astype(int)

    return {
        "micro_f1": f1_score(
            labels_np,
            predictions,
            average="micro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            labels_np,
            predictions,
            average="macro",
            zero_division=0
        ),

        "subset_accuracy": accuracy_score(
            labels_np,
            predictions
        ),

        "emotion_f1": f1_score(
            labels_np,
            predictions,
            average=None,
            zero_division=0
        )
    }


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    total_loss = 0.0

    all_logits = []

    all_labels = []

    optimizer.zero_grad(
        set_to_none=True
    )

    use_amp = (
        USE_AMP
        and
        torch.cuda.is_available()
    )

    for step, batch in enumerate(
        loader
    ):

        texts = batch["texts"]

        audios = batch["audios"]

        frames = batch["frames"]

        labels = (
            batch["labels"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp
        ):

            logits, _ = model(
                texts,
                audios,
                frames
            )

            logits = logits.float()

            loss = criterion(
                logits,
                labels
            )

            loss_for_backward = (
                loss
                /
                GRADIENT_ACCUMULATION
            )

        if use_amp:

            scaler.scale(
                loss_for_backward
            ).backward()

        else:

            loss_for_backward.backward()

        should_step = (
            (
                (step + 1)
                %
                GRADIENT_ACCUMULATION
            )
            == 0
            or
            (
                step + 1
                ==
                len(loader)
            )
        )

        if should_step:

            if use_amp:

                scaler.unscale_(
                    optimizer
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM
            )

            if use_amp:

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        total_loss += loss.item()

        all_logits.append(
            logits.detach().cpu()
        )

        all_labels.append(
            labels.detach().cpu()
        )

        if (
            (step + 1) % 50 == 0
            or
            (step + 1)
            == len(loader)
        ):

            print(
                f"\r  Training: "
                f"{step + 1}/"
                f"{len(loader)}",
                end="",
                flush=True
            )

    print()

    all_logits = torch.cat(
        all_logits
    )

    all_labels = torch.cat(
        all_labels
    )

    metrics = calculate_metrics(
        all_logits,
        all_labels
    )

    return (
        total_loss / len(loader),
        metrics
    )


# ============================================================
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_logits = []

    all_labels = []

    use_amp = (
        USE_AMP
        and
        torch.cuda.is_available()
    )

    for step, batch in enumerate(
        loader
    ):

        texts = batch["texts"]

        audios = batch["audios"]

        frames = batch["frames"]

        labels = (
            batch["labels"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp
        ):

            logits, _ = model(
                texts,
                audios,
                frames
            )

            logits = logits.float()

            loss = criterion(
                logits,
                labels
            )

        total_loss += loss.item()

        all_logits.append(
            logits.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

        if (
            (step + 1) % 50 == 0
            or
            (step + 1)
            == len(loader)
        ):

            print(
                f"\r  Evaluating: "
                f"{step + 1}/"
                f"{len(loader)}",
                end="",
                flush=True
            )

    print()

    all_logits = torch.cat(
        all_logits
    )

    all_labels = torch.cat(
        all_labels
    )

    metrics = calculate_metrics(
        all_logits,
        all_labels
    )

    return (
        total_loss / len(loader),
        metrics,
        all_logits,
        all_labels
    )


# ============================================================
# SAVE RESUME CHECKPOINT
# ============================================================

def save_resume_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_f1,
    patience_counter
):

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_f1":
                float(best_f1),

            "patience_counter":
                int(patience_counter),

            "config": {
                "batch_size":
                    BATCH_SIZE,

                "gradient_accumulation":
                    GRADIENT_ACCUMULATION,

                "use_amp":
                    USE_AMP
            }
        },
        RESUME_CHECKPOINT
    )


# ============================================================
# LOAD RESUME
# ============================================================

def load_resume_checkpoint(
    model,
    optimizer,
    scheduler
):

    if not RESUME_CHECKPOINT.exists():

        return (
            0,
            -1.0,
            0
        )

    print()
    print(
        "Resume checkpoint found:"
    )

    print(
        RESUME_CHECKPOINT
    )

    checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )

    start_epoch = int(
        checkpoint["epoch"]
    )

    best_f1 = float(
        checkpoint["best_f1"]
    )

    patience_counter = int(
        checkpoint[
            "patience_counter"
        ]
    )

    print(
        f"Resuming after Epoch "
        f"{start_epoch}"
    )

    print(
        f"Best validation Micro-F1: "
        f"{best_f1:.4f}"
    )

    return (
        start_epoch,
        best_f1,
        patience_counter
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_thresholds(
    logits,
    labels
):

    probabilities = (
        torch.sigmoid(
            logits
        )
        .cpu()
        .numpy()
    )

    labels = (
        labels
        .cpu()
        .numpy()
    )

    thresholds = []

    for emotion_index in range(
        NUM_EMOTIONS
    ):

        best_threshold = 0.50

        best_score = -1.0

        for threshold in np.arange(
            0.05,
            0.96,
            0.01
        ):

            predictions = (
                probabilities[
                    :,
                    emotion_index
                ]
                >= threshold
            ).astype(int)

            score = f1_score(
                labels[
                    :,
                    emotion_index
                ],
                predictions,
                zero_division=0
            )

            if score > best_score:

                best_score = score

                best_threshold = float(
                    threshold
                )

        thresholds.append(
            best_threshold
        )

    return thresholds


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "MARK 2.5 FAST - RESUME TRAINING"
    )

    print(
        f"Project: {PROJECT_ROOT}"
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            /
            1024**3
        )

        print(
            f"CUDA memory: "
            f"{memory:.2f} GB"
        )

    print()
    print(
        "TRAINING CONFIG"
    )

    print(
        f"  Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"  Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION}"
    )

    print(
        f"  Effective batch size: "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION}"
    )

    print(
        f"  Target epoch: "
        f"{TARGET_EPOCH}"
    )

    print(
        f"  AMP: "
        f"{USE_AMP}"
    )

    # ========================================================
    # DATA
    # ========================================================

    header(
        "LOADING CACHED DATASETS"
    )

    train_dataset = (
        FastMark25Dataset(
            "train"
        )
    )

    val_dataset = (
        FastMark25Dataset(
            "val"
        )
    )

    test_dataset = (
        FastMark25Dataset(
            "test"
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn
    )

    # ========================================================
    # MODEL
    # ========================================================

    header(
        "LOADING MARK 2.5 MODEL"
    )

    model = (
        Mark25FastModel()
        .to(DEVICE)
        .float()
    )

    parameter_report(
        model
    )

    # ========================================================
    # LOSS
    # ========================================================

    pos_weight = (
        calculate_pos_weight(
            train_dataset
        )
        .to(DEVICE)
        .float()
    )

    print()
    print(
        "Positive class weights:"
    )

    for emotion, weight in zip(
        EMOTION_NAMES,
        pos_weight
    ):

        print(
            f"  {emotion:10s}: "
            f"{weight.item():.4f}"
        )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    backbone_parameters = []

    new_parameters = []

    for module in [
        model.text_encoder,
        model.audio_encoder,
        model.vision_encoder
    ]:

        for parameter in (
            module.parameters()
        ):

            if parameter.requires_grad:

                backbone_parameters.append(
                    parameter
                )

    for module in [
        model.text_projection,
        model.audio_projection,
        model.video_projection,
        model.fusion,
        model.fusion_norm,
        model.classifier
    ]:

        new_parameters.extend(
            list(
                module.parameters()
            )
        )

    new_parameters.extend(
        [
            model.cls_token,
            model.modality_embeddings
        ]
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    backbone_parameters,
                "lr":
                    BACKBONE_LR
            },
            {
                "params":
                    new_parameters,
                "lr":
                    NEW_LAYER_LR
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # AMP
    # ========================================================

    use_amp = (
        USE_AMP
        and
        torch.cuda.is_available()
    )

    if use_amp:

        scaler = torch.amp.GradScaler(
            "cuda"
        )

        print()
        print(
            "AMP: ENABLED"
        )

    else:

        scaler = None

        print()
        print(
            "AMP: DISABLED"
        )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1
        )
    )

    # ========================================================
    # RESUME
    # ========================================================

    (
        start_epoch,
        best_f1,
        patience_counter
    ) = load_resume_checkpoint(
        model,
        optimizer,
        scheduler
    )

    # --------------------------------------------------------
    # If the previous run crashed during Epoch 6 but the
    # resume checkpoint was saved at Epoch 5, we start at 6.
    # --------------------------------------------------------

    if start_epoch >= TARGET_EPOCH:

        print()
        print(
            "Target epoch already reached."
        )

    # ========================================================
    # TRAIN
    # ========================================================

    overall_start = time.time()

    for epoch in range(
        start_epoch + 1,
        TARGET_EPOCH + 1
    ):

        epoch_start = time.time()

        print()
        print(
            f"Epoch "
            f"{epoch:02d}/{TARGET_EPOCH}"
        )

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler
            )
        )

        (
            val_loss,
            val_metrics,
            _,
            _
        ) = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_metrics[
                "micro_f1"
            ]
        )

        epoch_time = (
            time.time()
            -
            epoch_start
        )

        print()

        print(
            f"  Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"  Train Micro-F1: "
            f"{train_metrics['micro_f1']:.4f}"
        )

        print(
            f"  Train Macro-F1: "
            f"{train_metrics['macro_f1']:.4f}"
        )

        print(
            f"  Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"  Val Micro-F1: "
            f"{val_metrics['micro_f1']:.4f}"
        )

        print(
            f"  Val Macro-F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"  Val Subset Accuracy: "
            f"{val_metrics['subset_accuracy']:.4f}"
        )

        print(
            f"  Epoch time: "
            f"{epoch_time / 60:.2f} min"
        )

        print()

        print(
            "  Emotion F1:"
        )

        for emotion, score in zip(
            EMOTION_NAMES,
            val_metrics["emotion_f1"]
        ):

            print(
                f"    {emotion:10s}: "
                f"{score:.4f}"
            )

        # ----------------------------------------------------
        # Best
        # ----------------------------------------------------

        if (
            val_metrics[
                "micro_f1"
            ]
            >
            best_f1
        ):

            best_f1 = (
                val_metrics[
                    "micro_f1"
                ]
            )

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_micro_f1":
                        float(best_f1)
                },
                BEST_CHECKPOINT
            )

            print()
            print(
                "  ★ NEW BEST MODEL"
            )

            print(
                f"  Saved: "
                f"{BEST_CHECKPOINT}"
            )

        else:

            patience_counter += 1

            print(
                f"  No improvement "
                f"({patience_counter}/"
                f"{PATIENCE})"
            )

        # ----------------------------------------------------
        # ALWAYS save resume checkpoint
        # ----------------------------------------------------

        save_resume_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_f1,
            patience_counter
        )

        print()
        print(
            f"  Resume checkpoint saved:"
        )

        print(
            f"  {RESUME_CHECKPOINT}"
        )

        # ----------------------------------------------------
        # GPU information
        # ----------------------------------------------------

        if torch.cuda.is_available():

            allocated = (
                torch.cuda
                .memory_allocated()
                /
                1024**3
            )

            reserved = (
                torch.cuda
                .memory_reserved()
                /
                1024**3
            )

            peak = (
                torch.cuda
                .max_memory_allocated()
                /
                1024**3
            )

            print()
            print(
                "  GPU memory:"
            )

            print(
                f"    Allocated: "
                f"{allocated:.2f} GB"
            )

            print(
                f"    Reserved:  "
                f"{reserved:.2f} GB"
            )

            print(
                f"    Peak:      "
                f"{peak:.2f} GB"
            )

            torch.cuda.reset_peak_memory_stats()

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            patience_counter
            >=
            PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # BEST MODEL
    # ========================================================

    header(
        "LOADING BEST MARK 2.5 MODEL"
    )

    if BEST_CHECKPOINT.exists():

        checkpoint = torch.load(
            BEST_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        print(
            f"Best epoch: "
            f"{checkpoint['epoch']}"
        )

        print(
            f"Best validation Micro-F1: "
            f"{checkpoint['val_micro_f1']:.4f}"
        )

    else:

        print(
            "Best checkpoint not found."
        )

    # ========================================================
    # VALIDATION THRESHOLDS
    # ========================================================

    print()
    print(
        "Optimizing validation thresholds..."
    )

    (
        val_loss,
        val_metrics,
        val_logits,
        val_labels
    ) = evaluate(
        model,
        val_loader,
        criterion
    )

    thresholds = optimize_thresholds(
        val_logits,
        val_labels
    )

    print()
    print(
        "Optimized thresholds:"
    )

    for emotion, threshold in zip(
        EMOTION_NAMES,
        thresholds
    ):

        print(
            f"  {emotion:10s}: "
            f"{threshold:.2f}"
        )

    # ========================================================
    # TEST
    # ========================================================

    print()
    print(
        "Evaluating TEST set..."
    )

    (
        test_loss,
        test_metrics,
        test_logits,
        test_labels
    ) = evaluate(
        model,
        test_loader,
        criterion
    )

    test_default = calculate_metrics(
        test_logits,
        test_labels
    )

    test_optimized = calculate_metrics(
        test_logits,
        test_labels,
        thresholds
    )

    # ========================================================
    # RESULTS
    # ========================================================

    header(
        "MARK 2.5 FINAL RESULTS"
    )

    print(
        "TEST - Default thresholds"
    )

    print(
        f"  Micro-F1: "
        f"{test_default['micro_f1']:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{test_default['macro_f1']:.4f}"
    )

    print(
        f"  Subset Accuracy: "
        f"{test_default['subset_accuracy']:.4f}"
    )

    print()

    print(
        "TEST - Optimized thresholds"
    )

    print(
        f"  Micro-F1: "
        f"{test_optimized['micro_f1']:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{test_optimized['macro_f1']:.4f}"
    )

    print(
        f"  Subset Accuracy: "
        f"{test_optimized['subset_accuracy']:.4f}"
    )

    print()

    print(
        "Test Emotion F1:"
    )

    for emotion, score in zip(
        EMOTION_NAMES,
        test_optimized["emotion_f1"]
    ):

        print(
            f"  {emotion:10s}: "
            f"{score:.4f}"
        )

    # ========================================================
    # SAVE THRESHOLDS
    # ========================================================

    threshold_data = {

        "model":
            "Mark2.5-Fast",

        "thresholds": {
            emotion: float(
                threshold
            )
            for emotion, threshold
            in zip(
                EMOTION_NAMES,
                thresholds
            )
        },

        "test_micro_f1":
            float(
                test_optimized[
                    "micro_f1"
                ]
            ),

        "test_macro_f1":
            float(
                test_optimized[
                    "macro_f1"
                ]
            ),

        "test_subset_accuracy":
            float(
                test_optimized[
                    "subset_accuracy"
                ]
            )
    }

    with open(
        THRESHOLD_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            threshold_data,
            file,
            indent=2
        )

    print()
    print(
        f"Thresholds saved:"
    )

    print(
        THRESHOLD_FILE
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    total_time = (
        time.time()
        -
        overall_start
    )

    header(
        "MARK 2.5 RESUME RUN COMPLETE"
    )

    print(
        f"Best validation Micro-F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Final test Micro-F1: "
        f"{test_optimized['micro_f1']:.4f}"
    )

    print(
        f"Final test Macro-F1: "
        f"{test_optimized['macro_f1']:.4f}"
    )

    print()

    print(
        "Checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print()

    print(
        "Resume checkpoint:"
    )

    print(
        RESUME_CHECKPOINT
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()