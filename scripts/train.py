import os

import torch
import torch.nn as nn
import mlflow

from torch.optim import Adam
from mmsdk import mmdatasdk

from src.models.multimodal_model import MultimodalEmotionModel
from src.data.dataloaders import create_dataloaders
from src.data.split import create_video_splits
from src.training.training import train_one_epoch, evaluate


# ============================================================
# Configuration
# ============================================================

BASE_PATH = (
    "C:/SRUJAN/multimodal_ai/"
    "data/processed/aligned_mark1"
)

DATA_FILES = {
    "glove_vectors":
        f"{BASE_PATH}/glove_vectors_aligned.csd",

    "COVAREP":
        f"{BASE_PATH}/COVAREP_aligned.csd",

    "FACET 4.2":
        f"{BASE_PATH}/FACET42_aligned.csd",

    "All Labels":
        f"{BASE_PATH}/labels_aligned.csd",
}

# Training configuration
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 10
SEED = 42

# Model configuration
EMBEDDING_DIM = 128
FUSION_DIM = 256
NUM_EMOTIONS = 6
DROPOUT = 0.2

# Checkpoint
CHECKPOINT_DIR = (
    "C:/SRUJAN/multimodal_ai/"
    "checkpoints"
)

CHECKPOINT_PATH = (
    f"{CHECKPOINT_DIR}/"
    "mark1_best.pt"
)

# MLflow SQLite database
MLFLOW_DB = (
    "sqlite:///C:/SRUJAN/multimodal_ai/"
    "mlflow.db"
)


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# MLflow setup
# ============================================================

print("\nSetting up MLflow...")

mlflow.set_tracking_uri(
    MLFLOW_DB
)

mlflow.set_experiment(
    "Mark1-Multimodal-Emotion"
)

print("MLflow tracking enabled.")


# ============================================================
# Load aligned dataset
# ============================================================

print("\nLoading aligned dataset...")

dataset = mmdatasdk.mmdataset(
    DATA_FILES,
    BASE_PATH
)

print("Dataset loaded.")


# ============================================================
# Find common multimodal segments
# ============================================================

text_ids = set(
    dataset.computational_sequences[
        "glove_vectors"
    ].data.keys()
)

audio_ids = set(
    dataset.computational_sequences[
        "COVAREP"
    ].data.keys()
)

vision_ids = set(
    dataset.computational_sequences[
        "FACET 4.2"
    ].data.keys()
)

label_ids = set(
    dataset.computational_sequences[
        "All Labels"
    ].data.keys()
)

common_ids = sorted(
    text_ids
    & audio_ids
    & vision_ids
    & label_ids
)

print(
    "Common multimodal segments:",
    len(common_ids)
)


# ============================================================
# Create video-level train/validation/test split
# ============================================================

train_ids, val_ids, test_ids = create_video_splits(
    common_ids,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=SEED
)

print("\nDataset split:")

print(
    "Train:",
    len(train_ids)
)

print(
    "Validation:",
    len(val_ids)
)

print(
    "Test:",
    len(test_ids)
)


# ============================================================
# Verify no segment overlap
# ============================================================

train_set = set(train_ids)
val_set = set(val_ids)
test_set = set(test_ids)

assert len(
    train_set & val_set
) == 0

assert len(
    train_set & test_set
) == 0

assert len(
    val_set & test_set
) == 0

print(
    "✓ No segment overlap between splits."
)


# ============================================================
# Create DataLoaders
# ============================================================

train_loader, val_loader, test_loader = (
    create_dataloaders(
        dataset,
        train_ids,
        val_ids,
        test_ids,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
)

print(
    "✓ DataLoaders created."
)


# ============================================================
# Create model
# ============================================================

model = MultimodalEmotionModel(
    embedding_dim=EMBEDDING_DIM,
    fusion_dim=FUSION_DIM,
    num_emotions=NUM_EMOTIONS,
    dropout=DROPOUT
)

model = model.to(device)

print(
    "\n✓ Mark 1 model created."
)


# ============================================================
# Loss function
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Create checkpoint directory
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# Start MLflow run
# ============================================================

with mlflow.start_run(
    run_name="Mark1_Baseline"
) as run:

    print(
        "\nMLflow Run ID:",
        run.info.run_id
    )

    # --------------------------------------------------------
    # Log experiment parameters
    # --------------------------------------------------------

    mlflow.log_params({

        # Training
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "seed": SEED,

        # Text
        "text_input_dim": 300,

        # Audio
        "audio_input_dim": 74,

        # Vision
        "vision_input_dim": 35,

        # Encoder
        "encoder_hidden_dim": 128,

        # Embeddings
        "embedding_dim": EMBEDDING_DIM,

        # Fusion
        "fusion_dim": FUSION_DIM,

        # Classification
        "num_emotions": NUM_EMOTIONS,

        # Regularization
        "dropout": DROPOUT,

        # Optimizer
        "optimizer": "Adam",

        # Loss
        "loss_function":
            "BCEWithLogitsLoss",

        # Temporal processing
        "audio_stride": 10,
        "vision_stride": 5,

        # Dataset
        "total_segments":
            len(common_ids),

        "train_segments":
            len(train_ids),

        "validation_segments":
            len(val_ids),

        "test_segments":
            len(test_ids),

        # Hardware
        "device": str(device)
    })


    # ========================================================
    # Training loop
    # ========================================================

    best_val_loss = float("inf")
    best_epoch = 0

    print("\n")
    print("=" * 60)
    print("STARTING MARK 1 TRAINING")
    print("=" * 60)


    for epoch in range(EPOCHS):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device
        )


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )


        # ----------------------------------------------------
        # Log metrics to MLflow
        # ----------------------------------------------------

        mlflow.log_metric(
            "train_loss",
            train_loss,
            step=epoch + 1
        )

        mlflow.log_metric(
            "validation_loss",
            val_loss,
            step=epoch + 1
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch":
                        epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_loss":
                        val_loss
                },
                CHECKPOINT_PATH
            )

            print(
                "✓ Best Mark 1 model saved."
            )


    # ========================================================
    # Log best training information
    # ========================================================

    mlflow.log_metric(
        "best_validation_loss",
        best_val_loss
    )

    mlflow.log_metric(
        "best_epoch",
        best_epoch
    )


    # ========================================================
    # Log model checkpoint as artifact
    # ========================================================

    mlflow.log_artifact(
        CHECKPOINT_PATH,
        artifact_path="checkpoint"
    )


    # ========================================================
    # Training summary
    # ========================================================

    print("\n")
    print("=" * 60)
    print("MARK 1 TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}"
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "MLflow Run ID:",
        run.info.run_id
    )

    print("=" * 60)