"""Aggregate paired, validation-only Mark 3 Step 3 multi-seed experiments."""

import argparse
import json
import statistics
from pathlib import Path

EXPECTED_SEEDS = [42, 123, 456, 789, 2026]
CANDIDATES = ["metadata_aware", "split_at_gap"]
METRICS = [
    ("primary", "macro_f1"),
    ("primary", "micro_f1"),
    ("primary", "subset_accuracy"),
    ("calibrated", "macro_f1"),
    ("calibrated", "micro_f1"),
    ("calibrated", "subset_accuracy"),
]


def metric(report, family, name):
    section = (
        "validation_primary_at_0_5"
        if family == "primary"
        else "validation_calibrated_metrics"
    )
    return float(report[section][name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    reports = {}

    for seed in EXPECTED_SEEDS:
        reports[seed] = {}
        for candidate in CANDIDATES:
            path = root / f"seed_{seed}" / candidate / "results.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing report: {path}")

            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("seed") != seed:
                raise ValueError(f"Wrong seed recorded in {path}")
            if report.get("candidate") != candidate:
                raise ValueError(f"Wrong candidate recorded in {path}")
            if report.get("test_data_loaded") is not False:
                raise ValueError(f"Test-data isolation check failed: {path}")
            if report.get("deterministic_algorithms") is not True:
                raise ValueError(f"Deterministic mode not recorded: {path}")
            if report["split_integrity"]["overlap_count"] != 0:
                raise ValueError(f"Train/validation overlap found: {path}")

            reports[seed][candidate] = report

    configs = [
        json.dumps(reports[seed][candidate]["config"], sort_keys=True)
        for seed in EXPECTED_SEEDS
        for candidate in CANDIDATES
    ]
    if len(set(configs)) != 1:
        raise ValueError("Configurations differ across runs.")

    for seed in EXPECTED_SEEDS:
        for candidate in CANDIDATES:
            report = reports[seed][candidate]
            if report["split_integrity"]["train_videos"] != 800:
                raise ValueError(f"Unexpected training split: seed {seed}")
            if report["split_integrity"]["validation_videos"] != 171:
                raise ValueError(f"Unexpected validation split: seed {seed}")

    summary = {}
    paired_differences = {}

    for family, name in METRICS:
        key = f"{family}_{name}"
        summary[key] = {}
        values = {}

        for candidate in CANDIDATES:
            series = [
                metric(reports[seed][candidate], family, name)
                for seed in EXPECTED_SEEDS
            ]
            values[candidate] = series
            summary[key][candidate] = {
                "mean": statistics.mean(series),
                "sample_std": statistics.stdev(series),
                "per_seed": dict(zip(map(str, EXPECTED_SEEDS), series)),
            }

        deltas = [
            b - a
            for a, b in zip(
                values["metadata_aware"], values["split_at_gap"]
            )
        ]
        paired_differences[key] = {
            "definition": "split_at_gap minus metadata_aware",
            "mean": statistics.mean(deltas),
            "sample_std": statistics.stdev(deltas),
            "per_seed": dict(zip(map(str, EXPECTED_SEEDS), deltas)),
        }

    result = {
        "experiment": "Mark 3 Step 3 controlled multi-seed policy comparison",
        "seeds": EXPECTED_SEEDS,
        "run_count": len(EXPECTED_SEEDS) * len(CANDIDATES),
        "test_data_loaded": False,
        "deterministic_algorithms": True,
        "protocol_checks": {
            "all_expected_reports_present": True,
            "matching_configurations": True,
            "zero_train_validation_video_overlap": True,
            "expected_split_sizes": True,
            "all_seeds_matched_between_candidates": True,
        },
        "summary": summary,
        "paired_differences": paired_differences,
        "interpretation": (
            "Compare mean paired differences and variability across seeds. "
            "These validation results guide policy selection; they do not "
            "replace a final locked test evaluation."
        ),
    }

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nReport written to: {output}")


if __name__ == "__main__":
    main()
