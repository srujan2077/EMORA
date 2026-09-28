import numpy as np
import torch
import mlflow

from mmsdk import mmdatasdk
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.models.multimodal_model import MultimodalEmotionModel
from src.data.mosei_dataset import MOSEIDataset
from src.data.split import create_video_splits


# ============================================================
# Configuration
# ============================================================

BASE_PATH = "C:/SRUJAN/multimodal_ai/data/processed/aligned_mark1"

DATA_FILES = {
    "glove_vectors": f"{BASE_PATH}/glove_vectors_aligned.csd",
    "COVAREP": f"{BASE_PATH}/COVAREP_aligned.csd",
    "FACET 4.2": f"{BASE_PATH}/FACET42_aligned.csd",
    "All Labels": f"{BASE_PATH}/labels_aligned.csd",
}

CHECKPOINT_PATH = (
    "C:/SRUJAN/multimodal_ai/"
    "checkpoints/mark1_best.pt"
)

MLFLOW_DB = (
    "sqlite:///C:/SRUJAN/multimodal_ai/"
    "mlflow.db"
)

EMOTIONS = [
    "Happy",
    "Sad",
    "Anger",
    "Surprise",
    "Disgust",
    "Fear"
]

THRESHOLD = 0.5
SEED = 42


# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri(MLFLOW_DB)

mlflow.set_experiment(
    "Mark1-Multimodal-Emotion"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Load dataset
# ============================================================

print("\nLoading dataset...")

dataset = mmdatasdk.mmdataset(
    DATA_FILES,
    BASE_PATH
)


# ============================================================
# Common IDs
# ============================================================

sequences = dataset.computational_sequences

common_ids = sorted(
    set(sequences["glove_vectors"].data.keys())
    & set(sequences["COVAREP"].data.keys())
    & set(sequences["FACET 4.2"].data.keys())
    & set(sequences["All Labels"].data.keys())
)

print(
    "Common segments:",
    len(common_ids)
)


# ============================================================
# Recreate exact split
# ============================================================

_, _, test_ids = create_video_splits(
    common_ids,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=SEED
)

print(
    "Test segments:",
    len(test_ids)
)


# ============================================================
# Dataset
# ============================================================

test_dataset = MOSEIDataset(
    dataset,
    test_ids
)


# ============================================================
# Model
# ============================================================

model = MultimodalEmotionModel()

model = model.to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Loaded checkpoint from epoch:",
    checkpoint["epoch"]
)


# ============================================================
# Predictions
# ============================================================

all_true = []
all_pred = []

print("\nRunning test evaluation...")

with torch.no_grad():

    for index in range(
        len(test_dataset)
    ):

        sample = test_dataset[index]

        text = (
            sample["text"]
            .unsqueeze(0)
            .to(device)
        )

        audio = (
            sample["audio"]
            .unsqueeze(0)
            .to(device)
        )

        vision = (
            sample["vision"]
            .unsqueeze(0)
            .to(device)
        )

        labels = (
            sample["labels"]
            .numpy()
            .astype(int)
        )

        logits = model(
            text,
            audio,
            vision
        )

        probabilities = torch.sigmoid(
            logits
        )[0].cpu().numpy()

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        all_true.append(labels)
        all_pred.append(predictions)


# ============================================================
# Convert arrays
# ============================================================

y_true = np.asarray(all_true)

y_pred = np.asarray(all_pred)


# ============================================================
# Metrics
# ============================================================

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

micro_precision = precision_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)

micro_recall = recall_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)

micro_f1 = f1_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)


# ============================================================
# Display
# ============================================================

print("\n")
print("=" * 60)
print("MARK 1 TEST RESULTS")
print("=" * 60)

print(
    f"Macro Precision : {macro_precision:.4f}"
)

print(
    f"Macro Recall    : {macro_recall:.4f}"
)

print(
    f"Macro F1        : {macro_f1:.4f}"
)

print(
    f"Micro Precision : {micro_precision:.4f}"
)

print(
    f"Micro Recall    : {micro_recall:.4f}"
)

print(
    f"Micro F1        : {micro_f1:.4f}"
)


# ============================================================
# Per-emotion metrics
# ============================================================

print("\nPer-emotion results:")

report = classification_report(
    y_true,
    y_pred,
    target_names=EMOTIONS,
    zero_division=0
)

print(report)


# ============================================================
# Find existing MLflow run
# ============================================================

client = mlflow.MlflowClient()

experiment = client.get_experiment_by_name(
    "Mark1-Multimodal-Emotion"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=[
        "attributes.start_time DESC"
    ],
    max_results=1
)

if runs:

    run_id = runs[0].info.run_id

    print(
        "\nLogging evaluation to run:",
        run_id
    )

    with mlflow.start_run(
        run_id=run_id
    ):

        mlflow.log_metrics({

            "test_macro_precision":
                macro_precision,

            "test_macro_recall":
                macro_recall,

            "test_macro_f1":
                macro_f1,

            "test_micro_precision":
                micro_precision,

            "test_micro_recall":
                micro_recall,

            "test_micro_f1":
                micro_f1,

            "test_threshold":
                THRESHOLD
        })

        # Save classification report
        report_path = (
            "C:/SRUJAN/multimodal_ai/"
            "classification_report.txt"
        )

        with open(
            report_path,
            "w"
        ) as f:

            f.write(
                "MARK 1 CLASSIFICATION REPORT\n\n"
            )

            f.write(report)

        mlflow.log_artifact(
            report_path,
            artifact_path="evaluation"
        )

        print(
            "✓ Evaluation metrics logged."
        )

else:

    print(
        "WARNING: No MLflow training run found."
    )


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)