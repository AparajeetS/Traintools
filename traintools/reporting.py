"""Stable JSON serialization for TrainTools reports and agent integrations."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return to_jsonable(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def report_envelope(
    report: Any,
    *,
    diagnostic: str,
    package_version: Optional[str] = None,
) -> Dict[str, Any]:
    if package_version is None:
        from traintools import __version__

        package_version = __version__
    return {
        "schema_version": 1,
        "package": "traintools",
        "package_version": package_version,
        "diagnostic": diagnostic,
        "report": to_jsonable(report),
    }


def write_json_report(
    report: Any,
    path: Union[str, Path],
    *,
    diagnostic: str,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report_envelope(report, diagnostic=diagnostic), indent=2) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = ["report_envelope", "to_jsonable", "write_json_report"]
