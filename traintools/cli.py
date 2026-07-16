"""Command-line discovery and integration helpers for TrainTools."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from traintools import __version__
from traintools.capabilities import (
    get_capability,
    integration_snippet,
    list_capabilities,
    recommend_diagnostics,
)


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2))


def _recommend_payload(
    problem: str, framework: Optional[str], limit: int
) -> Dict[str, object]:
    recommendations = recommend_diagnostics(
        problem, framework=framework, limit=limit
    )
    return {
        "schema_version": 1,
        "package": "traintools",
        "package_version": __version__,
        "problem": problem,
        "framework": framework,
        "recommendations": recommendations,
        "abstained": not recommendations,
        "abstention_reason": (
            "No diagnostic matched the supplied problem. Add concrete symptoms "
            "such as batch size, gradients, labels, validation plateau, or plasticity."
            if not recommendations
            else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traintools",
        description="Discover and integrate paper-backed PyTorch training diagnostics.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available diagnostics.")
    list_parser.add_argument("--framework", choices=("pytorch", "huggingface"))
    list_parser.add_argument("--json", action="store_true")

    recommend_parser = subparsers.add_parser(
        "recommend", help="Recommend diagnostics for a concrete training problem."
    )
    recommend_parser.add_argument("problem")
    recommend_parser.add_argument("--framework", choices=("pytorch", "huggingface"))
    recommend_parser.add_argument("--limit", type=int, default=3)
    recommend_parser.add_argument("--json", action="store_true")

    explain_parser = subparsers.add_parser(
        "explain", help="Explain one diagnostic and its limitations."
    )
    explain_parser.add_argument("diagnostic")
    explain_parser.add_argument("--json", action="store_true")

    integration_parser = subparsers.add_parser(
        "integration", help="Generate a minimal integration snippet."
    )
    integration_parser.add_argument("diagnostic")
    integration_parser.add_argument(
        "--framework", choices=("pytorch", "huggingface"), default="pytorch"
    )
    integration_parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            items = list_capabilities(args.framework)
            if args.json:
                _print_json({"schema_version": 1, "diagnostics": items})
            else:
                for item in items:
                    print(f"{item['id']}: {item['summary']}")
            return 0
        if args.command == "recommend":
            payload = _recommend_payload(args.problem, args.framework, args.limit)
            if args.json:
                _print_json(payload)
            elif payload["abstained"]:
                print(payload["abstention_reason"])
            else:
                for index, item in enumerate(payload["recommendations"], start=1):
                    print(f"{index}. {item['name']} ({item['id']})")
                    print(f"   {item['summary']}")
                    print(f"   matched: {', '.join(item['matched_terms'])}")
            return 0
        if args.command == "explain":
            item = get_capability(args.diagnostic).to_dict()
            if args.json:
                _print_json({"schema_version": 1, "diagnostic": item})
            else:
                print(f"{item['name']}: {item['summary']}")
                print(f"Call timing: {item['call_timing']}")
                print("Limitations:")
                for limitation in item["limitations"]:
                    print(f"- {limitation}")
            return 0
        if args.command == "integration":
            snippet = integration_snippet(args.diagnostic, args.framework)
            if args.json:
                _print_json(
                    {
                        "schema_version": 1,
                        "diagnostic": args.diagnostic,
                        "framework": args.framework,
                        "snippet": snippet,
                    }
                )
            else:
                print(snippet)
            return 0
    except ValueError as exc:
        print(f"traintools: {exc}")
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
