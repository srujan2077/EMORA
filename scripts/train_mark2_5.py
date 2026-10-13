import gc
import json
import subprocess
import time
from pathlib import Path

import cv2
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
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MARK2_DIR = PROJECT_ROOT / "data" / "processed" / "mark2"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CHECKPOINT = (
    CHECKPOINT_DIR / "mark2_5_best.pt"
)

OUTPUT_THRESHOLDS = (
    CHECKPOINT_DIR / "mark2_5_thresholds.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PRETRAINED MODELS
# ============================================================

TEXT_MODEL_NAME = "microsoft/deberta-v3-base"

AUDIO_MODEL_NAME = "facebook/wav2vec2-base-960h"

VISION_MODEL_NAME = "google/vit-base-patch16-224-in21k"


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_FRAMES = 8

AUDIO_SAMPLE_RATE = 16000

TEXT_MAX_LENGTH = 256

PROJECTION_DIM = 256

NUM_HEADS = 8

NUM_TRANSFORMER_LAYERS = 2

TRANSFORMER_FF_DIM = 1024

DROPOUT = 0.1

NUM_EMOTIONS = 6

EMOTION_NAMES = [
    "happy",
    "sad",
    "anger",
    "surprise",
    "disgust",
    "fear",
]


# ============================================================
# TRAINING CONFIG
# ============================================================

BATCH_SIZE = 1

GRADIENT_ACCUMULATION = 8

EPOCHS = 12

BACKBONE_LR = 1e-5

NEW_LAYER_LR = 5e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 3

MAX_GRAD_NORM = 1.0


# ============================================================
# SEED
# ============================================================

torch.manual_seed(SEED)

np.random.seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PERFORMANCE
# ============================================================

if torch.cuda.is_available():

    torch.set_float32_matmul_precision("high")


# ============================================================
# HEADER
# ============================================================

def header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# FFMPEG
# ============================================================

def find_ffmpeg():

    candidates = [

        "ffmpeg",

        r"C:\Users\sruja\AppData\Local\Microsoft\WinGet"
        r"\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe",

    ]

    for candidate in candidates:

        if candidate == "ffmpeg":

            return candidate

        if Path(candidate).exists():

            return candidate

    raise FileNotFoundError(
        "FFmpeg was not found."
    )


FFMPEG = find_ffmpeg()


# ============================================================
# DATASET
# ============================================================

class Mark25Dataset(Dataset):

    def __init__(self, split):

        self.split = split

        csv_path = (
            MARK2_DIR / f"{split}.csv"
        )

        if not csv_path.exists():

            raise FileNotFoundError(
                f"Missing dataset file:\n{csv_path}"
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

        labels = torch.tensor(

            [
                float(row["happy"]),
                float(row["sad"]),
                float(row["anger"]),
                float(row["surprise"]),
                float(row["disgust"]),
                float(row["fear"]),
            ],

            dtype=torch.float32,
        )

        media_path = (
            PROJECT_ROOT / str(row["media_path"])
        )

        return {

            "segment_id":
                str(row["segment_id"]),

            "text":
                str(row["text"]),

            "media_path":
                str(media_path),

            "labels":
                labels,
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

        "media_paths": [
            item["media_path"]
            for item in batch
        ],

        "labels": torch.stack(

            [
                item["labels"]
                for item in batch
            ]

        ),
    }


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio(video_path):

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

            f"No audio extracted from:\n"
            f"{video_path}"
        )

    return audio


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def extract_frames(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(

            f"Cannot open video:\n"
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

        frames.append(frame)

    cap.release()

    if not frames:

        raise RuntimeError(

            "Could not extract video frames."
        )

    return frames


# ============================================================
# FIND TRANSFORMER LAYERS
# ============================================================

def find_layer_list(
    module,
    model_name,
):

    candidates = []

    paths = [

        "encoder.layer",

        "encoder.layers",

        "transformer.layer",

        "transformer.layers",

        "deberta.encoder.layer",

        "wav2vec2.encoder.layers",

        "vit.encoder.layer",

        "vit.encoder.layers",

        "vit.layer",

        "layers",

        "layer",

    ]

    for path in paths:

        current = module

        valid = True

        for part in path.split("."):

            if not hasattr(
                current,
                part,
            ):

                valid = False

                break

            current = getattr(
                current,
                part,
            )

        if not valid:

            continue

        if isinstance(
            current,
            nn.ModuleList,
        ):

            candidates.append(
                (
                    path,
                    current,
                )
            )

    # --------------------------------------------------------
    # Recursive fallback
    # --------------------------------------------------------

    if not candidates:

        for name, child in (
            module.named_modules()
        ):

            if isinstance(
                child,
                nn.ModuleList,
            ):

                if len(child) >= 4:

                    candidates.append(
                        (
                            name,
                            child,
                        )
                    )

    if not candidates:

        print()

        print(
            f"Could not find transformer "
            f"layers for {model_name}."
        )

        print(
            "Available ModuleLists:"
        )

        for name, child in (
            module.named_modules()
        ):

            if isinstance(
                child,
                nn.ModuleList,
            ):

                print(

                    f"  {name}: "
                    f"{len(child)} modules"
                )

        raise RuntimeError(

            f"Could not find layers "
            f"for {model_name}"
        )

    candidates.sort(

        key=lambda item:
        (
            len(
                item[0].split(".")
            ),

            len(item[1]),
        ),

        reverse=True,
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
# MARK 2.5 MODEL
# ============================================================

class Mark25Model(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # TEXT
        # ====================================================

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

        # ====================================================
        # AUDIO
        # ====================================================

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

        # ====================================================
        # VISION
        # ====================================================

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

        # ====================================================
        # IMPORTANT:
        # FORCE ALL BACKBONES TO FP32
        # ====================================================

        self.text_encoder.float()

        self.audio_encoder.float()

        self.vision_encoder.float()

        # ====================================================
        # FIND TRANSFORMER BLOCKS
        # ====================================================

        self.text_layers = (
            find_layer_list(
                self.text_encoder,
                "DeBERTa",
            )
        )

        self.audio_layers = (
            find_layer_list(
                self.audio_encoder,
                "Wav2Vec2",
            )
        )

        self.vision_layers = (
            find_layer_list(
                self.vision_encoder,
                "ViT",
            )
        )

        # ====================================================
        # SELECTIVE FINE-TUNING
        # ====================================================

        self.configure_backbones()

        # ====================================================
        # PROJECTIONS
        # ====================================================

        self.text_projection = nn.Sequential(

            nn.Linear(
                768,
                PROJECTION_DIM,
            ),

            nn.LayerNorm(
                PROJECTION_DIM
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),
        )

        self.audio_projection = nn.Sequential(

            nn.Linear(
                768,
                PROJECTION_DIM,
            ),

            nn.LayerNorm(
                PROJECTION_DIM
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),
        )

        self.video_projection = nn.Sequential(

            nn.Linear(
                768,
                PROJECTION_DIM,
            ),

            nn.LayerNorm(
                PROJECTION_DIM
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),
        )

        # ====================================================
        # CLS TOKEN
        # ====================================================

        self.cls_token = nn.Parameter(

            torch.randn(

                1,

                1,

                PROJECTION_DIM,
            )
        )

        # ====================================================
        # MODALITY EMBEDDINGS
        # ====================================================

        self.modality_embeddings = nn.Parameter(

            torch.randn(

                1,

                4,

                PROJECTION_DIM,
            )
        )

        # ====================================================
        # FUSION TRANSFORMER
        # ====================================================

        encoder_layer = (
            nn.TransformerEncoderLayer(

                d_model=PROJECTION_DIM,

                nhead=NUM_HEADS,

                dim_feedforward=
                    TRANSFORMER_FF_DIM,

                dropout=DROPOUT,

                activation="gelu",

                batch_first=True,

                norm_first=True,
            )
        )

        self.fusion = (
            nn.TransformerEncoder(

                encoder_layer,

                num_layers=
                    NUM_TRANSFORMER_LAYERS,
            )
        )

        self.fusion_norm = (
            nn.LayerNorm(
                PROJECTION_DIM
            )
        )

        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                PROJECTION_DIM,
                PROJECTION_DIM,
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                PROJECTION_DIM,
                NUM_EMOTIONS,
            ),
        )

    # ========================================================
    # FREEZE / UNFREEZE
    # ========================================================

    def configure_backbones(self):

        # ----------------------------------------------------
        # Freeze everything
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # DeBERTa
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Wav2Vec2
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ViT
        # ----------------------------------------------------

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
    # TRAIN MODE
    # ========================================================

    def train(self, mode=True):

        super().train(mode)

        # ----------------------------------------------------
        # Freeze pretrained backbones in eval mode
        # ----------------------------------------------------

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
    # TEXT ENCODING
    # ========================================================

    def encode_text(
        self,
        texts,
    ):

        encoded = self.text_tokenizer(

            texts,

            return_tensors="pt",

            padding=True,

            truncation=True,

            max_length=TEXT_MAX_LENGTH,
        )

        encoded = {

            key: value.to(DEVICE)

            for key, value
            in encoded.items()
        }

        outputs = self.text_encoder(
            **encoded
        )

        hidden = (
            outputs.last_hidden_state
        )

        # IMPORTANT:
        # Force output to FP32

        hidden = hidden.float()

        mask = (
            encoded[
                "attention_mask"
            ]
            .unsqueeze(-1)
            .to(hidden.dtype)
        )

        hidden = (
            hidden * mask
        )

        embedding = (

            hidden.sum(dim=1)

            /

            mask.sum(
                dim=1
            ).clamp(min=1)
        )

        return embedding.float()

    # ========================================================
    # AUDIO ENCODING
    # ========================================================

    def encode_audio(
        self,
        media_paths,
    ):

        audio_list = []

        for path in media_paths:

            audio = extract_audio(
                path
            )

            audio_list.append(
                audio
            )

        processed = (
            self.audio_processor(

                audio_list,

                sampling_rate=
                    AUDIO_SAMPLE_RATE,

                return_tensors="pt",

                padding=True,
            )
        )

        processed = {

            key: value.to(DEVICE)

            for key, value
            in processed.items()
        }

        outputs = (
            self.audio_encoder(
                **processed
            )
        )

        hidden = (
            outputs.last_hidden_state
        )

        # IMPORTANT:
        # Force output to FP32

        hidden = hidden.float()

        embedding = (
            hidden.mean(
                dim=1
            )
        )

        return embedding.float()

    # ========================================================
    # VIDEO ENCODING
    # ========================================================

    def encode_video(
        self,
        media_paths,
    ):

        embeddings = []

        for path in media_paths:

            frames = extract_frames(
                path
            )

            processed = (
                self.vision_processor(

                    images=frames,

                    return_tensors="pt",
                )
            )

            pixels = (
                processed[
                    "pixel_values"
                ]
                .to(DEVICE)
                .float()
            )

            outputs = (
                self.vision_encoder(

                    pixel_values=pixels
                )
            )

            cls = (
                outputs
                .last_hidden_state[
                    :,
                    0,
                    :
                ]
            )

            # IMPORTANT:
            # Force output to FP32

            cls = cls.float()

            embedding = (
                cls.mean(
                    dim=0
                )
            )

            embeddings.append(
                embedding
            )

        return torch.stack(
            embeddings
        ).float()

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        texts,
        media_paths,
    ):

        # ----------------------------------------------------
        # Encode modalities
        # ----------------------------------------------------

        text_features = (
            self.encode_text(
                texts
            )
        )

        audio_features = (
            self.encode_audio(
                media_paths
            )
        )

        video_features = (
            self.encode_video(
                media_paths
            )
        )

        # ----------------------------------------------------
        # FINAL SAFETY CAST
        # ----------------------------------------------------

        text_features = (
            text_features.float()
        )

        audio_features = (
            audio_features.float()
        )

        video_features = (
            video_features.float()
        )

        # ----------------------------------------------------
        # Projections
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CLS
        # ----------------------------------------------------

        cls = (
            self.cls_token
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        # ----------------------------------------------------
        # [CLS, TEXT, AUDIO, VIDEO]
        # ----------------------------------------------------

        sequence = torch.cat(

            [

                cls,

                text.unsqueeze(1),

                audio.unsqueeze(1),

                video.unsqueeze(1),

            ],

            dim=1,
        )

        # ----------------------------------------------------
        # Modality embeddings
        # ----------------------------------------------------

        sequence = (

            sequence

            +

            self.modality_embeddings
        )

        # ----------------------------------------------------
        # Cross-modal transformer
        # ----------------------------------------------------

        fused_sequence = (
            self.fusion(
                sequence
            )
        )

        # ----------------------------------------------------
        # CLS representation
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

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

    total = 0

    trainable = 0

    print()
    print("=" * 70)
    print("PARAMETER REPORT")
    print("=" * 70)

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
            model.classifier,
    }

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
# CLASS WEIGHTS
# ============================================================

def calculate_pos_weight(dataset):

    labels = torch.stack(

        [
            dataset[i]["labels"]

            for i in range(
                len(dataset)
            )
        ]
    )

    positives = (
        labels.sum(
            dim=0
        )
    )

    negatives = (
        labels.size(0)
        -
        positives
    )

    return (

        negatives

        /

        positives.clamp(
            min=1
        )
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    labels,
    thresholds=None,
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

        np.array(
            thresholds
        )

    ).astype(int)

    micro_f1 = f1_score(

        labels,

        predictions,

        average="micro",

        zero_division=0,
    )

    macro_f1 = f1_score(

        labels,

        predictions,

        average="macro",

        zero_division=0,
    )

    subset_accuracy = (
        accuracy_score(
            labels,
            predictions,
        )
    )

    emotion_f1 = f1_score(

        labels,

        predictions,

        average=None,

        zero_division=0,
    )

    return {

        "micro_f1":
            micro_f1,

        "macro_f1":
            macro_f1,

        "subset_accuracy":
            subset_accuracy,

        "emotion_f1":
            emotion_f1,
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    total_loss = 0.0

    all_logits = []

    all_labels = []

    optimizer.zero_grad(
        set_to_none=True
    )

    for step, batch in enumerate(
        loader
    ):

        texts = (
            batch["texts"]
        )

        media_paths = (
            batch["media_paths"]
        )

        labels = (
            batch["labels"]
            .to(DEVICE)
            .float()
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits, _ = model(

            texts,

            media_paths,
        )

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        logits = logits.float()

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(

            logits,

            labels,
        )

        scaled_loss = (

            loss

            /

            GRADIENT_ACCUMULATION
        )

        scaled_loss.backward()

        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        should_step = (

            (

                (step + 1)

                %

                GRADIENT_ACCUMULATION
            )

            == 0

            or

            (

                (step + 1)

                ==

                len(loader)
            )
        )

        if should_step:

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                MAX_GRAD_NORM,
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += (
            loss.item()
        )

        all_logits.append(
            logits.detach().cpu()
        )

        all_labels.append(
            labels.detach().cpu()
        )

        del logits

        del loss

        del scaled_loss

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (

            (step + 1) % 100 == 0

            or

            (step + 1) == len(loader)

        ):

            print(

                f"\r  Training: "
                f"{step + 1}/"
                f"{len(loader)}",

                end="",

                flush=True,
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

        all_labels,
    )

    average_loss = (

        total_loss

        /

        len(loader)
    )

    return (

        average_loss,

        metrics,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0

    all_logits = []

    all_labels = []

    for step, batch in enumerate(
        loader
    ):

        texts = (
            batch["texts"]
        )

        media_paths = (
            batch["media_paths"]
        )

        labels = (
            batch["labels"]
            .to(DEVICE)
            .float()
        )

        logits, _ = model(

            texts,

            media_paths,
        )

        logits = logits.float()

        loss = criterion(

            logits,

            labels,
        )

        total_loss += (
            loss.item()
        )

        all_logits.append(
            logits.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

        if (

            (step + 1) % 100 == 0

            or

            (step + 1) == len(loader)

        ):

            print(

                f"\r  Evaluating: "
                f"{step + 1}/"
                f"{len(loader)}",

                end="",

                flush=True,
            )

        del logits

        del loss

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    print()

    all_logits = torch.cat(
        all_logits
    )

    all_labels = torch.cat(
        all_labels
    )

    metrics = calculate_metrics(

        all_logits,

        all_labels,
    )

    average_loss = (

        total_loss

        /

        len(loader)
    )

    return (

        average_loss,

        metrics,

        all_logits,

        all_labels,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_f1,
    path,
):

    torch.save(

        {

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "val_micro_f1":
                val_f1,

            "emotion_names":
                EMOTION_NAMES,

            "config": {

                "projection_dim":
                    PROJECTION_DIM,

                "num_heads":
                    NUM_HEADS,

                "num_transformer_layers":
                    NUM_TRANSFORMER_LAYERS,

                "transformer_ff_dim":
                    TRANSFORMER_FF_DIM,

                "dropout":
                    DROPOUT,

                "backbone_lr":
                    BACKBONE_LR,

                "new_layer_lr":
                    NEW_LAYER_LR,

                "gradient_accumulation":
                    GRADIENT_ACCUMULATION,
            },
        },

        path,
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_thresholds(
    logits,
    labels,
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

        best_f1 = -1.0

        for threshold in np.arange(

            0.05,

            0.96,

            0.01,
        ):

            predictions = (

                probabilities[
                    :,
                    emotion_index
                ]

                >=

                threshold

            ).astype(int)

            score = f1_score(

                labels[
                    :,
                    emotion_index
                ],

                predictions,

                zero_division=0,
            )

            if score > best_f1:

                best_f1 = score

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
        "MARK 2.5 - SELECTIVE FINE-TUNING"
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

        print(

            f"CUDA memory: "

            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # ========================================================
    # DATASETS
    # ========================================================

    header(
        "LOADING DATASETS"
    )

    train_dataset = (
        Mark25Dataset("train")
    )

    val_dataset = (
        Mark25Dataset("val")
    )

    test_dataset = (
        Mark25Dataset("test")
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=False,

        collate_fn=collate_fn,
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=False,

        collate_fn=collate_fn,
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=False,

        collate_fn=collate_fn,
    )

    # ========================================================
    # MODEL
    # ========================================================

    header(
        "LOADING MARK 2.5 MODEL"
    )

    model = (
        Mark25Model()
        .to(DEVICE)
    )

    parameter_report(
        model
    )

    # ========================================================
    # CLASS WEIGHTS
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

        pos_weight,
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

    backbone_modules = [

        model.text_encoder,

        model.audio_encoder,

        model.vision_encoder,

    ]

    new_modules = [

        model.text_projection,

        model.audio_projection,

        model.video_projection,

        model.fusion,

        model.fusion_norm,

        model.classifier,

    ]

    for module in backbone_modules:

        for parameter in (
            module.parameters()
        ):

            if parameter.requires_grad:

                backbone_parameters.append(
                    parameter
                )

    for module in new_modules:

        new_parameters.extend(

            list(
                module.parameters()
            )
        )

    new_parameters.extend(

        [

            model.cls_token,

            model.modality_embeddings,

        ]

    )

    optimizer = torch.optim.AdamW(

        [

            {

                "params":
                    backbone_parameters,

                "lr":
                    BACKBONE_LR,
            },

            {

                "params":
                    new_parameters,

                "lr":
                    NEW_LAYER_LR,
            },

        ],

        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=1,
        )
    )

    # ========================================================
    # TRAINING STATE
    # ========================================================

    best_f1 = -1.0

    best_epoch = -1

    patience_counter = 0

    # ========================================================
    # TRAINING
    # ========================================================

    header(
        "STARTING MARK 2.5 TRAINING"
    )

    for epoch in range(

        1,

        EPOCHS + 1,
    ):

        start_time = time.time()

        print()

        print(
            f"Epoch "
            f"{epoch:02d}/{EPOCHS}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_metrics = (

            train_one_epoch(

                model,

                train_loader,

                criterion,

                optimizer,
            )
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        (

            val_loss,

            val_metrics,

            _,

            _,

        ) = evaluate(

            model,

            val_loader,

            criterion,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(

            val_metrics[
                "micro_f1"
            ]
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

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

            f"  Time: "
            f"{elapsed / 60:.2f} min"
        )

        print()

        print(
            "  Emotion F1:"
        )

        for emotion, score in zip(

            EMOTION_NAMES,

            val_metrics[
                "emotion_f1"
            ],
        ):

            print(

                f"    {emotion:10s}: "
                f"{score:.4f}"
            )

        # ----------------------------------------------------
        # Best checkpoint
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

            best_epoch = epoch

            patience_counter = 0

            save_checkpoint(

                model,

                optimizer,

                epoch,

                best_f1,

                OUTPUT_CHECKPOINT,
            )

            print()

            print(
                "  ★ NEW BEST MARK 2.5 MODEL"
            )

            print(

                f"  Saved: "
                f"{OUTPUT_CHECKPOINT}"
            )

        else:

            patience_counter += 1

            print(

                f"  No improvement "
                f"({patience_counter}/"
                f"{PATIENCE})"
            )

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
    # LOAD BEST MODEL
    # ========================================================

    header(
        "LOADING BEST MARK 2.5 MODEL"
    )

    checkpoint = torch.load(

        OUTPUT_CHECKPOINT,

        map_location=DEVICE,

        weights_only=False,
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

    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    print()

    print(
        "Collecting validation predictions..."
    )

    (

        val_loss,

        val_metrics,

        val_logits,

        val_labels,

    ) = evaluate(

        model,

        val_loader,

        criterion,
    )

    thresholds = optimize_thresholds(

        val_logits,

        val_labels,
    )

    print()

    print(
        "Optimized thresholds:"
    )

    for emotion, threshold in zip(

        EMOTION_NAMES,

        thresholds,
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

        test_labels,

    ) = evaluate(

        model,

        test_loader,

        criterion,
    )

    test_default = calculate_metrics(

        test_logits,

        test_labels,
    )

    test_optimized = calculate_metrics(

        test_logits,

        test_labels,

        thresholds,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    header(
        "FINAL MARK 2.5 RESULTS"
    )

    print(

        f"Best epoch: "
        f"{best_epoch}"
    )

    print()

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

        test_optimized[
            "emotion_f1"
        ],
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
            "Mark2.5",

        "checkpoint":
            str(
                OUTPUT_CHECKPOINT
            ),

        "thresholds": {

            emotion: threshold

            for emotion, threshold
            in zip(

                EMOTION_NAMES,

                thresholds,
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
            ),
    }

    with open(

        OUTPUT_THRESHOLDS,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            threshold_data,

            file,

            indent=2,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    header(
        "MARK 2.5 TRAINING COMPLETE"
    )

    print()

    print(
        "Checkpoint:"
    )

    print(
        OUTPUT_CHECKPOINT
    )

    print()

    print(
        "Thresholds:"
    )

    print(
        OUTPUT_THRESHOLDS
    )

    print()

    print(
        "MARK 2 BENCHMARK"
    )

    print(
        "Optimized Test Micro-F1: 0.5316"
    )

    print()

    print(
        "MARK 2.5 RESULT"
    )

    print(

        "Optimized Test Micro-F1: "
        f"{test_optimized['micro_f1']:.4f}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()