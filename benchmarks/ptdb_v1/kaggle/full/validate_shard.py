"""Validate a completed PTDB-1 base-development dataset shard."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(root_value: str, dataset: str, model: str | None = None) -> None:
    root = Path(root_value)
    required = (
        "manifest.json",
        "run_summary.csv",
        "epoch_metrics.csv",
        "example_scores.csv.gz",
        "paired_noninterference.csv",
        "errors.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"missing shard artifacts: {missing}")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors = json.loads((root / "errors.json").read_text(encoding="utf-8"))
    runs = read_csv(root / "run_summary.csv")
    epochs = read_csv(root / "epoch_metrics.csv")
    pairs = read_csv(root / "paired_noninterference.csv")
    declared_hashes = manifest.get("outputs", {})
    expected_instrumented = 27 if model is None else 9
    expected_plain = 6 if model is None else (3 if model in {"resnet18", "wrn28_2"} else 0)
    expected_pairs = expected_plain
    expected_executions = expected_instrumented + expected_plain
    checks = {
        "no_error_rows": not errors,
        "expected_completed_executions": len(runs) == expected_executions,
        "expected_instrumented_runs": sum(int(row["instrumented"]) for row in runs) == expected_instrumented,
        "expected_plain_twins": sum(1 - int(row["instrumented"]) for row in runs) == expected_plain,
        "expected_completed_pairs": len(pairs) == expected_pairs,
        "all_pairs_exact": all(int(row["exact_final_hash"]) for row in pairs),
        "all_runs_are_declared_dataset": {row["dataset"] for row in runs} == {dataset},
        "all_runs_are_declared_model": model is None or {row["model"] for row in runs} == {model},
        "all_runs_have_25_epochs": all(int(row["epochs"]) == 25 for row in runs),
        "all_epoch_rows_present": len(epochs) == expected_executions * 25,
        "public_package_is_0_6_2": manifest["environment"].get("traintools") == "0.6.2",
        "t4_recorded": "T4" in str(manifest["environment"].get("gpu")),
        "output_hashes_match": bool(declared_hashes) and all(
            (root / name).is_file() and sha256_file(root / name) == expected
            for name, expected in declared_hashes.items()
        ),
    }
    report = {
        "valid_base_shard": all(checks.values()),
        "dataset": dataset,
        "model": model,
        "checks": checks,
        "completed_executions": len(runs),
        "exact_hash_pairs": sum(int(row["exact_final_hash"]) for row in pairs),
        "max_abs_accuracy_difference": max(
            (abs(float(row["accuracy_difference"])) for row in pairs), default=None
        ),
        "interpretation_boundary": "Base development evidence only; protected holdout remains sealed.",
    }
    (root / "shard_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["valid_base_shard"]:
        raise SystemExit("PTDB-1 base shard failed validation")


if __name__ == "__main__":
    if len(sys.argv) not in {3, 4}:
        raise SystemExit("usage: validate_shard.py OUTPUT_DIRECTORY DATASET [MODEL]")
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
