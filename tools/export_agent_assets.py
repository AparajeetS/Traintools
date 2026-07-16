"""Export machine-readable assets from the canonical Python registry."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traintools.capabilities import list_capabilities  # noqa: E402


def main() -> int:
    payload = {
        "schema_version": 1,
        "package": "traintools",
        "diagnostics": list_capabilities(),
    }
    text = json.dumps(payload, indent=2) + "\n"
    targets = [
        ROOT / "capabilities.json",
        ROOT / "traintools" / "data" / "capabilities.json",
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
