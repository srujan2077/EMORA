import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mark2"
    / "features"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EPOCHS = 30
PATIENCE = 6

INPUT_DIM = 768
PROJECTION_DIM = 256

NUM_HEADS = 8
NUM_LAYERS = 2
FF_DIM = 1024

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
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PRINT
# ============================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# DATASET
# ============================================================

class CachedMark2Dataset(Dataset):

    def __init__(self, split):

        self.split = split

        pattern = f"{split}_shard_*.pt"

        shard_files = sorted(
            FEATURE_DIR.glob(pattern)
        )

        if not shard_files:
            raise FileNotFoundError(
                f"No feature shards found for {split}:\n"
                f"{FEATURE_DIR}"
            )

        print(
            f"\nLoading {split}: "
            f"{len(shard_files)} shards"
        )

        text_embeddings = []
        audio_embeddings = []
        video_embeddings = []
        labels = []

        segment_ids = []

        for shard_file in shard_files:

            payload = torch.load(
                shard_file,
                map_location="cpu",
                weights_only=False,
            )

            text_embeddings.append(
                payload["text_embedding"]
            )

            audio_embeddings.append(
                payload["audio_embedding"]
            )

            video_embeddings.append(
                payload["video_embedding"]
            )

            labels.append(
                payload["labels"]
            )

            segment_ids.extend(
                payload["segment_id"]
            )

        self.text = torch.cat(
            text_embeddings,
            dim=0,
        )

        self.audio = torch.cat(
            audio_embeddings,
            dim=0,
        )

        self.video = torch.cat(
            video_embeddings,
            dim=0,
        )

        self.labels = torch.cat(
            labels,
            dim=0,
        )

        self.segment_ids = segment_ids

        print(
            f"{split}: {len(self.labels)} samples"
        )

        print(
            f"Text:  {tuple(self.text.shape)}"
        )

        print(
            f"Audio: {tuple(self.audio.shape)}"
        )

        print(
            f"Video: {tuple(self.video.shape)}"
        )

        print(
            f"Labels:{tuple(self.labels.shape)}"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return {
            "text": self.text[index],
            "audio": self.audio[index],
            "video": self.video[index],
            "labels": self.labels[index],
        }


# ============================================================
# MARK 2 FUSION MODEL
# ============================================================

class Mark2FusionModel(nn.Module):

    def __init__(
        self,
        input_dim=768,
        projection_dim=256,
        num_heads=8,
        num_layers=2,
        ff_dim=1024,
        dropout=0.1,
        num_emotions=6,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Modality projections
        # ----------------------------------------------------

        self.text_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                projection_dim,
            ),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                projection_dim,
            ),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.video_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                projection_dim,
            ),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ----------------------------------------------------
        # Learnable CLS token
        # ----------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                projection_dim,
            )
        )

        # ----------------------------------------------------
        # Modality embeddings
        #
        # 0 = CLS
        # 1 = Text
        # 2 = Audio
        # 3 = Vision
        # ----------------------------------------------------

        self.modality_embeddings = nn.Parameter(
            torch.randn(
                1,
                4,
                projection_dim,
            )
        )

        # ----------------------------------------------------
        # Cross-modal Transformer
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.fusion_norm = nn.LayerNorm(
            projection_dim
        )

        # ----------------------------------------------------
        # Emotion classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                projection_dim,
                projection_dim,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                projection_dim,
                num_emotions,
            ),
        )

    def forward(
        self,
        text,
        audio,
        video,
    ):

        # ----------------------------------------------------
        # Project modalities
        # ----------------------------------------------------

        text = self.text_projection(text)

        audio = self.audio_projection(audio)

        video = self.video_projection(video)

        batch_size = text.size(0)

        # ----------------------------------------------------
        # CLS token
        # ----------------------------------------------------

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        # ----------------------------------------------------
        # Build multimodal sequence
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
        # Add modality embeddings
        # ----------------------------------------------------

        sequence = (
            sequence
            + self.modality_embeddings
        )

        # ----------------------------------------------------
        # Cross-modal attention
        # ----------------------------------------------------

        encoded = self.transformer(
            sequence
        )

        # ----------------------------------------------------
        # CLS = fused multimodal representation
        # ----------------------------------------------------

        fusion = encoded[:, 0, :]

        fusion = self.fusion_norm(
            fusion
        )

        # ----------------------------------------------------
        # Emotion logits
        # ----------------------------------------------------

        logits = self.classifier(
            fusion
        )

        return logits, fusion


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_pos_weight(dataset):

    labels = dataset.labels

    positives = labels.sum(dim=0)

    negatives = (
        labels.size(0)
        - positives
    )

    pos_weight = (
        negatives
        / positives.clamp(min=1)
    )

    return pos_weight


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    labels,
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= 0.5
    ).float()

    y_true = labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    subset_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    emotion_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "subset_accuracy": subset_accuracy,
        "emotion_f1": emotion_f1,
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

    for batch in loader:

        text = batch["text"].to(
            DEVICE,
            non_blocking=True,
        )

        audio = batch["audio"].to(
            DEVICE,
            non_blocking=True,
        )

        video = batch["video"].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits, _ = model(
            text,
            audio,
            video,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        all_logits.append(
            logits.detach().cpu()
        )

        all_labels.append(
            labels.detach().cpu()
        )

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
        / len(loader.dataset)
    )

    return average_loss, metrics


# ============================================================
# VALIDATION
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

    for batch in loader:

        text = batch["text"].to(
            DEVICE,
            non_blocking=True,
        )

        audio = batch["audio"].to(
            DEVICE,
            non_blocking=True,
        )

        video = batch["video"].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )

        logits, _ = model(
            text,
            audio,
            video,
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        all_logits.append(
            logits.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

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
        / len(loader.dataset)
    )

    return average_loss, metrics


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    val_f1,
    path,
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "val_loss": val_loss,

        "val_micro_f1": val_f1,

        "config": {

            "input_dim":
                INPUT_DIM,

            "projection_dim":
                PROJECTION_DIM,

            "num_heads":
                NUM_HEADS,

            "num_layers":
                NUM_LAYERS,

            "ff_dim":
                FF_DIM,

            "dropout":
                DROPOUT,

            "num_emotions":
                NUM_EMOTIONS,

            "emotion_names":
                EMOTION_NAMES,
        },
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "MARK 2 - FUSION TRAINING"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Feature directory: {FEATURE_DIR}"
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_dataset = CachedMark2Dataset(
        "train"
    )

    val_dataset = CachedMark2Dataset(
        "val"
    )

    test_dataset = CachedMark2Dataset(
        "test"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Mark2FusionModel(
        input_dim=INPUT_DIM,
        projection_dim=PROJECTION_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        num_emotions=NUM_EMOTIONS,
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Class imbalance handling
    # --------------------------------------------------------

    pos_weight = calculate_pos_weight(
        train_dataset
    ).to(DEVICE)

    print()
    print("Positive class weights:")

    for emotion, weight in zip(
        EMOTION_NAMES,
        pos_weight,
    ):

        print(
            f"  {emotion:10s}: "
            f"{weight.item():.4f}"
        )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

    mlflow.set_experiment(
        "Mark2-Multimodal-Emotion"
    )

    best_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / "mark2_best.pt"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    with mlflow.start_run():

        mlflow.log_params({

            "model": "Mark2FusionModel",

            "text_backbone":
                "microsoft/deberta-v3-base",

            "audio_backbone":
                "facebook/wav2vec2-base-960h",

            "vision_backbone":
                "google/vit-base-patch16-224-in21k",

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "epochs":
                EPOCHS,

            "projection_dim":
                PROJECTION_DIM,

            "transformer_heads":
                NUM_HEADS,

            "transformer_layers":
                NUM_LAYERS,

            "transformer_ff_dim":
                FF_DIM,

            "dropout":
                DROPOUT,

            "seed":
                SEED,
        })

        for epoch in range(
            1,
            EPOCHS + 1,
        ):

            epoch_start = time.time()

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            train_loss, train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )

            # ------------------------------------------------
            # VALIDATE
            # ------------------------------------------------

            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                criterion,
            )

            # ------------------------------------------------
            # Scheduler
            # ------------------------------------------------

            scheduler.step(
                val_metrics["micro_f1"]
            )

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            epoch_time = (
                time.time()
                - epoch_start
            )

            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            print()
            print(
                f"Epoch "
                f"{epoch:02d}/{EPOCHS}"
            )

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
                f"  Learning Rate: "
                f"{current_lr:.6f}"
            )

            print(
                f"  Time: "
                f"{epoch_time:.2f}s"
            )

            # ------------------------------------------------
            # Emotion F1
            # ------------------------------------------------

            print(
                "  Emotion F1:"
            )

            for emotion, score in zip(
                EMOTION_NAMES,
                val_metrics["emotion_f1"],
            ):

                print(
                    f"    {emotion:10s}: "
                    f"{score:.4f}"
                )

            # ------------------------------------------------
            # MLflow
            # ------------------------------------------------

            mlflow.log_metrics({

                "train_loss":
                    train_loss,

                "train_micro_f1":
                    train_metrics["micro_f1"],

                "train_macro_f1":
                    train_metrics["macro_f1"],

                "val_loss":
                    val_loss,

                "val_micro_f1":
                    val_metrics["micro_f1"],

                "val_macro_f1":
                    val_metrics["macro_f1"],

                "val_subset_accuracy":
                    val_metrics["subset_accuracy"],

                "learning_rate":
                    current_lr,
            }, step=epoch)

            for emotion, score in zip(
                EMOTION_NAMES,
                val_metrics["emotion_f1"],
            ):

                mlflow.log_metric(
                    f"val_f1_{emotion}",
                    float(score),
                    step=epoch,
                )

            # ------------------------------------------------
            # Best checkpoint
            # ------------------------------------------------

            if (
                val_metrics["micro_f1"]
                > best_f1
            ):

                best_f1 = (
                    val_metrics["micro_f1"]
                )

                best_epoch = epoch

                patience_counter = 0

                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    best_f1,
                    checkpoint_path,
                )

                print()
                print(
                    "  ★ NEW BEST MODEL"
                )

                print(
                    f"  Saved: "
                    f"{checkpoint_path}"
                )

            else:

                patience_counter += 1

                print(
                    f"  No improvement "
                    f"({patience_counter}/{PATIENCE})"
                )

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            if (
                patience_counter
                >= PATIENCE
            ):

                print()
                print(
                    "Early stopping triggered."
                )

                break

        # ----------------------------------------------------
        # LOAD BEST MODEL
        # ----------------------------------------------------

        print()
        print(
            "Loading best Mark 2 checkpoint..."
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # ----------------------------------------------------
        # FINAL TEST
        # ----------------------------------------------------

        test_loss, test_metrics = evaluate(
            model,
            test_loader,
            criterion,
        )

        print()
        print("=" * 70)
        print("FINAL MARK 2 TEST RESULTS")
        print("=" * 70)

        print(
            f"Best epoch: {best_epoch}"
        )

        print(
            f"Best validation Micro-F1: "
            f"{best_f1:.4f}"
        )

        print(
            f"Test Loss: "
            f"{test_loss:.4f}"
        )

        print(
            f"Test Micro-F1: "
            f"{test_metrics['micro_f1']:.4f}"
        )

        print(
            f"Test Macro-F1: "
            f"{test_metrics['macro_f1']:.4f}"
        )

        print(
            f"Test Subset Accuracy: "
            f"{test_metrics['subset_accuracy']:.4f}"
        )

        print()
        print("Test Emotion F1:")

        for emotion, score in zip(
            EMOTION_NAMES,
            test_metrics["emotion_f1"],
        ):

            print(
                f"  {emotion:10s}: "
                f"{score:.4f}"
            )

        # ----------------------------------------------------
        # MLflow final metrics
        # ----------------------------------------------------

        mlflow.log_metrics({

            "test_loss":
                test_loss,

            "test_micro_f1":
                test_metrics["micro_f1"],

            "test_macro_f1":
                test_metrics["macro_f1"],

            "test_subset_accuracy":
                test_metrics["subset_accuracy"],

        })

        for emotion, score in zip(
            EMOTION_NAMES,
            test_metrics["emotion_f1"],
        ):

            mlflow.log_metric(
                f"test_f1_{emotion}",
                float(score),
            )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MARK 2 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best checkpoint:\n"
        f"{checkpoint_path}"
    )

    print()
    print(
        "Mark 2 has now been trained and evaluated."
    )


if __name__ == "__main__":
    main()