"""Compare completed Step 3 candidate reports without loading test data."""

from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-report", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [
        json.loads(args.metadata_report.read_text(encoding="utf-8")),
        json.loads(args.split_report.read_text(encoding="utf-8")),
    ]
    if any(report.get("test_data_loaded") for report in reports):
        raise ValueError("Comparison refuses reports that loaded test data.")
    if {report["seed"] for report in reports} != {42}:
        raise ValueError("Candidates do not use the fixed seed 42.")
    configs = {json.dumps(report["config"], sort_keys=True) for report in reports}
    if len(configs) != 1:
        raise ValueError("Candidates do not have identical configurations.")
    if any(report["split_integrity"]["overlap_count"] != 0 for report in reports):
        raise ValueError("A candidate has train/validation video overlap.")
    result = {
        "test_data_loaded": False,
        "protocol_checks": {
            "same_seed": True,
            "same_config": True,
            "no_train_validation_video_overlap": True,
            "same_maximum_epoch_and_patience": len({
                (report["config"]["max_epochs"], report["config"]["patience"])
                for report in reports
            }) == 1,
        },
        "candidates": {
            report["candidate"]: {
                "best_epoch": report["best_epoch"],
                "primary_validation_metrics": report["validation_primary_at_0_5"],
                "calibrated_validation_metrics": report[
                    "validation_calibrated_metrics"
                ],
                "context_length_distributions": report[
                    "context_length_distributions"
                ],
            }
            for report in reports
        },
        "interpretation": (
            "The split-at-gap policy is not considered better unless its "
            "controlled validation results support that conclusion. Test "
            "evaluation remains reserved for a later locked evaluation."
        ),
    }
    output = args.output
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
