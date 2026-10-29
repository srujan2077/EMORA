
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "mark3"

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
MODELS = ["mark3a_concat", "mark3b_gated"]

for model_name in MODELS:
    results_path = RESULTS_DIR / f"{model_name}_results.json"

    if not results_path.exists():
        print(f"Missing results file: {results_path}")
        continue

    with results_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    default_f1 = results["test_default_thresholds"]["emotion_f1"]
    calibrated_f1 = results["test_calibrated_thresholds"]["emotion_f1"]
    thresholds = results["thresholds_from_validation"]

    print(f"\n{'=' * 65}")
    print(model_name.upper())
    print(f"{'=' * 65}")
    print(f"{'Emotion':<13}{'Default F1':>13}{'Threshold':>13}{'Calibrated F1':>18}")

    for index, emotion in enumerate(EMOTIONS):
        print(
            f"{emotion:<13}"
            f"{default_f1[index]:>13.4f}"
            f"{thresholds[emotion]:>13.2f}"
            f"{calibrated_f1[index]:>18.4f}"
        )
