
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "mark3"

FILES = [
    "mark3a_concat_best.pt",
    "mark3a_concat_results.json",
    "mark3b_gated_best.pt",
    "mark3b_gated_results.json",
    "mark3_pooled_summary.json",
]

print("=" * 70)
print("EMORA MARK 3 — ARTIFACT AUDIT")
print("=" * 70)

all_present = True

for filename in FILES:
    path = CHECKPOINT_DIR / filename

    if not path.is_file():
        print(f"[MISSING] {filename}")
        all_present = False
    else:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[OK]      {filename} ({size_mb:.2f} MB)")

print("\n" + "-" * 70)
print("RESULTS JSON VALIDATION")
print("-" * 70)

for model_name in ["mark3a_concat", "mark3b_gated"]:
    path = CHECKPOINT_DIR / f"{model_name}_results.json"

    if not path.is_file():
        continue

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        required = [
            "test_default_thresholds",
            "test_calibrated_thresholds",
            "thresholds_from_validation",
        ]

        missing_keys = [key for key in required if key not in data]

        if missing_keys:
            print(f"[CHECK] {model_name}: missing keys {missing_keys}")
        else:
            macro_f1 = data["test_calibrated_thresholds"].get("macro_f1")
            micro_f1 = data["test_calibrated_thresholds"].get("micro_f1")

            print(f"[OK] {model_name}: valid JSON")
            print(f"     Calibrated test Macro-F1: {macro_f1}")
            print(f"     Calibrated test Micro-F1: {micro_f1}")

    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] {model_name}: {exc}")

print("\n" + "-" * 70)
summary_path = CHECKPOINT_DIR / "mark3_pooled_summary.json"

if summary_path.is_file():
    try:
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        print("[OK] Summary JSON parses successfully.")
        print("Summary top-level structure:", type(summary).__name__)
        if isinstance(summary, dict):
            print("Summary keys:", list(summary.keys()))
    except (json.JSONDecodeError, OSError) as exc:
        print("[ERROR] Summary JSON:", exc)

print("\nAudit file presence:", "PASS" if all_present else "INCOMPLETE")
print("No model training or checkpoint modifications were performed.")
