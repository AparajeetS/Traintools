"""Validate PTDB-1 pilot completeness without interpreting pilot metric quality."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(root: str) -> None:
    output = Path(root)
    required = [
        "manifest.json",
        "run_summary.csv",
        "epoch_metrics.csv",
        "example_scores.csv.gz",
        "paired_noninterference.csv",
        "timing_projection.json",
        "errors.json",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise SystemExit(f"missing pilot artifacts: {missing}")

    errors = json.loads((output / "errors.json").read_text(encoding="utf-8"))
    runs = read_csv(output / "run_summary.csv")
    pairs = read_csv(output / "paired_noninterference.csv")
    projection = json.loads((output / "timing_projection.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    declared_hashes = manifest.get("outputs", {})
    hashes_match = bool(declared_hashes) and all(
        (output / name).is_file() and sha256_file(output / name) == expected
        for name, expected in declared_hashes.items()
    )

    checks = {
        "no_error_rows": len(errors) == 0,
        "twelve_completed_runs": len(runs) == 12,
        "six_completed_pairs": len(pairs) == 6,
        "six_instrumented_runs": sum(int(row["instrumented"]) for row in runs) == 6,
        "six_plain_runs": sum(1 - int(row["instrumented"]) for row in runs) == 6,
        "all_cells_have_three_epochs": all(int(row["epochs"]) == 3 for row in runs),
        "gpu_recorded": bool(manifest["environment"].get("gpu")),
        "package_is_0_6_2": manifest["environment"].get("traintools") == "0.6.2",
        "projection_present": projection.get("projected_81_run_core_gpu_hours") is not None,
        "output_hashes_match": hashes_match,
    }
    report = {
        "valid_timing_pilot": all(checks.values()),
        "checks": checks,
        "exact_hash_pairs": sum(int(row["exact_final_hash"]) for row in pairs),
        "max_abs_accuracy_difference": max((abs(float(row["accuracy_difference"])) for row in pairs), default=None),
        "median_runtime_ratio": sorted(float(row["runtime_ratio"]) for row in pairs)[len(pairs) // 2] if pairs else None,
        "projected_81_run_core_gpu_hours": projection.get("projected_81_run_core_gpu_hours"),
        "interpretation_boundary": "Timing and implementation only; diagnostic thresholds remain frozen.",
    }
    (output / "pilot_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["valid_timing_pilot"]:
        raise SystemExit("PTDB-1 timing pilot failed completeness validation")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_pilot.py OUTPUT_DIRECTORY")
    main(sys.argv[1])
