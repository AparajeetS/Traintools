"""Pre-frozen PTDB-1 base-matrix analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np


BOOTSTRAP_SEED = 20260718
BOOTSTRAP_REPS = 10_000
DECISION_EPOCHS = (6, 10, 13)
SCORES = ("aum", "el2n", "forgetting", "mean_confidence", "mean_loss", "random")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: str | float | int | None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def quantile_interval(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def clustered_mean_interval(
    rows: list[dict[str, str]],
    value: Callable[[dict[str, str]], float | None],
    cluster: Callable[[dict[str, str]], tuple],
) -> dict:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        measured = value(row)
        if measured is not None and math.isfinite(measured):
            grouped[cluster(row)].append(measured)
    cell_means = np.asarray([np.mean(values) for values in grouped.values()], dtype=float)
    if not len(cell_means):
        return {"mean": None, "ci95": [None, None], "clusters": 0}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(cell_means, size=(BOOTSTRAP_REPS, len(cell_means)), replace=True).mean(axis=1)
    return {
        "mean": float(cell_means.mean()),
        "ci95": quantile_interval(draws.tolist()),
        "clusters": int(len(cell_means)),
    }


def load_inputs(roots: list[Path]) -> tuple[list[dict], list[dict], list[dict], list[Path]]:
    runs: list[dict] = []
    epochs: list[dict] = []
    pairs: list[dict] = []
    files: list[Path] = []
    for root in roots:
        errors_path = root / "errors.json"
        errors = json.loads(errors_path.read_text(encoding="utf-8"))
        if errors:
            raise RuntimeError(f"error rows present in {root}: {len(errors)}")
        for name, destination in (
            ("run_summary.csv", runs),
            ("epoch_metrics.csv", epochs),
            ("paired_noninterference.csv", pairs),
        ):
            path = root / name
            destination.extend(read_csv(path))
            files.append(path)
        files.append(errors_path)
        files.append(root / "manifest.json")
    return runs, epochs, pairs, files


def noise_analysis(runs: list[dict]) -> tuple[list[dict], dict]:
    noisy = [row for row in runs if row["regime"] != "clean" and int(row["instrumented"])]
    retained = []
    for row in noisy:
        item = {key: row[key] for key in ("run_id", "dataset", "model", "seed", "regime")}
        item["prevalence"] = row.get("prevalence")
        for score in SCORES:
            for outcome in ("auroc", "average_precision", "precision_at_noise"):
                item[f"{score}_{outcome}"] = row.get(f"{score}_{outcome}")
        retained.append(item)

    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in noisy}):
        selected = [row for row in noisy if row["dataset"] == dataset]
        cluster = lambda row: (row["model"], row["regime"])
        metrics = {}
        for score in SCORES:
            for outcome in ("auroc", "average_precision", "precision_at_noise"):
                key = f"{score}_{outcome}"
                metrics[key] = clustered_mean_interval(selected, lambda row, k=key: finite(row.get(k)), cluster)
        metrics["aum_minus_mean_loss_auroc"] = clustered_mean_interval(
            selected,
            lambda row: (
                finite(row.get("aum_auroc")) - finite(row.get("mean_loss_auroc"))
                if finite(row.get("aum_auroc")) is not None and finite(row.get("mean_loss_auroc")) is not None
                else None
            ),
            cluster,
        )
        by_dataset[dataset] = metrics
    return retained, by_dataset


def noninterference_analysis(pairs: list[dict]) -> dict:
    ratios = [float(row["runtime_ratio"]) for row in pairs]
    accuracy = [abs(float(row["accuracy_difference"])) for row in pairs]
    loss = [abs(float(row["loss_difference"])) for row in pairs]
    exact = [int(row["exact_final_hash"]) for row in pairs]
    return {
        "pairs": len(pairs),
        "exact_hash_rate": sum(exact) / len(exact) if exact else None,
        "max_abs_accuracy_difference": max(accuracy, default=None),
        "max_abs_loss_difference": max(loss, default=None),
        "runtime_ratio_median": statistics.median(ratios) if ratios else None,
        "runtime_ratio_p90": float(np.quantile(ratios, 0.9)) if ratios else None,
    }


def guard_targets(runs: list[dict], epochs: list[dict]) -> list[dict]:
    metadata = {row["run_id"]: row for row in runs if int(row["instrumented"])}
    curves: dict[str, list[dict]] = defaultdict(list)
    for row in epochs:
        if row["run_id"] in metadata:
            curves[row["run_id"]].append(row)
    output = []
    for run_id, curve in sorted(curves.items()):
        ordered = sorted(curve, key=lambda row: int(row["epoch"]))
        by_epoch = {int(row["epoch"]): row for row in ordered}
        for decision_epoch in DECISION_EPOCHS:
            current = by_epoch.get(decision_epoch)
            future = [finite(row["val_loss"]) for row in ordered if int(row["epoch"]) > decision_epoch]
            future = [value for value in future if value is not None]
            if current is None or not future:
                continue
            recent = [finite(by_epoch[e]["val_loss"]) for e in range(decision_epoch - 2, decision_epoch + 1)]
            if any(value is None for value in recent):
                continue
            meta = metadata[run_id]
            output.append({
                "run_id": run_id,
                "dataset": meta["dataset"],
                "model": meta["model"],
                "seed": meta["seed"],
                "regime": meta["regime"],
                "decision_epoch": decision_epoch,
                "current_val_loss": finite(current["val_loss"]),
                "realized_future_improvement": finite(current["val_loss"]) - min(future),
                "three_point_loss_slope": (recent[-1] - recent[0]) / 2.0,
                "guard_should_stop": current.get("guard_should_stop"),
                "guard_predicted_improvement": current.get("guard_predicted_improvement"),
                "guard_ci_low": current.get("guard_ci_low"),
                "guard_ci_high": current.get("guard_ci_high"),
            })
    return output


def main(output_value: str, root_values: list[str]) -> None:
    output = Path(output_value)
    output.mkdir(parents=True, exist_ok=True)
    roots = [Path(value) for value in root_values]
    runs, epochs, pairs, input_files = load_inputs(roots)
    noise_rows, noise_summary = noise_analysis(runs)
    guard_rows = guard_targets(runs, epochs)
    summary = {
        "analysis": "PTDB-1 base matrix",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "completed_executions": len(runs),
        "label_noise": noise_summary,
        "noninterference": noninterference_analysis(pairs),
        "protected_holdout_used": False,
    }
    write_csv(output / "label_noise_by_run.csv", noise_rows)
    write_csv(output / "trainguard_decision_targets.csv", guard_rows)
    (output / "base_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "script_sha256": sha256_file(Path(__file__)),
        "inputs": {str(path): sha256_file(path) for path in input_files},
        "outputs": {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "analysis_manifest.json"
        },
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: analyze_base.py OUTPUT_DIRECTORY SHARD_OUTPUT [SHARD_OUTPUT ...]")
    main(sys.argv[1], sys.argv[2:])
