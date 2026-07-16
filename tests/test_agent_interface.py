from __future__ import annotations

from dataclasses import dataclass
import asyncio
import json
from pathlib import Path
import re

import pytest

import traintools
from traintools.capabilities import (
    get_capability,
    integration_snippet,
    list_capabilities,
    recommend_diagnostics,
)
from traintools.cli import main
from traintools.reporting import report_envelope, to_jsonable, write_json_report


ROOT = Path(__file__).parents[1]


def test_version_matches_project_metadata() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
    assert match
    assert traintools.__version__ == match.group(1)


def test_mcp_registry_metadata_matches_release() -> None:
    manifest = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    package = manifest["packages"][0]
    assert manifest["name"] == "io.github.aparajeets/traintools"
    assert manifest["version"] == traintools.__version__
    assert package["identifier"] == "traintools"
    assert package["version"] == traintools.__version__
    assert package["transport"] == {"type": "stdio"}
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"mcp-name: {manifest['name']}" in readme


def test_registry_ids_are_unique_and_docs_exist() -> None:
    capabilities = list_capabilities()
    ids = [item["id"] for item in capabilities]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 10
    for item in capabilities:
        assert (ROOT / item["docs"]).is_file()
        assert item["limitations"]


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        ("loss is NaN and gradients are exploding", "gradient-health"),
        ("how should I choose my batch size", "gradient-noise-scale"),
        ("validation loss plateau should I stop training", "train-guard"),
        ("find mislabeled examples and bad labels", "aum"),
        ("my input batch has invalid labels", "batch-inspector"),
    ],
)
def test_recommender_routes_concrete_problems(problem: str, expected: str) -> None:
    results = recommend_diagnostics(problem)
    assert results
    assert results[0]["id"] == expected


def test_recommender_can_abstain() -> None:
    assert recommend_diagnostics("make everything generally better") == []


def test_framework_filter_and_integration() -> None:
    huggingface = list_capabilities("huggingface")
    assert {item["id"] for item in huggingface} >= {
        "gradient-noise-scale",
        "plasticity",
        "train-guard",
    }
    snippet = integration_snippet("train-guard", "huggingface")
    assert "TraintoolsCallback" in snippet
    with pytest.raises(ValueError, match="does not currently expose"):
        integration_snippet("batch-inspector", "huggingface")


def test_unknown_capability_is_actionable() -> None:
    with pytest.raises(ValueError, match="choose from"):
        get_capability("does-not-exist")


@dataclass
class ExampleReport:
    score: float
    values: tuple


def test_report_serialization_and_write(tmp_path: Path) -> None:
    report = ExampleReport(score=float("inf"), values=(1, 2))
    assert to_jsonable(report) == {"score": "inf", "values": [1, 2]}
    envelope = report_envelope(report, diagnostic="example")
    assert envelope["schema_version"] == 1
    output = write_json_report(report, tmp_path / "report.json", diagnostic="example")
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["report"]["score"] == "inf"


def test_cli_json_and_abstention(capsys) -> None:
    assert main(["recommend", "gradients explode", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommendations"][0]["id"] == "gradient-health"
    assert payload["abstained"] is False

    assert main(["recommend", "make it good", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["abstained"] is True


def test_cli_mcp_subcommand_uses_server_entrypoint(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("traintools.mcp_server.main", lambda: calls.append("run"))
    assert main(["mcp"]) == 0
    assert calls == ["run"]


def test_exported_capability_manifests_match_registry() -> None:
    expected = list_capabilities()
    for path in (
        ROOT / "capabilities.json",
        ROOT / "traintools" / "data" / "capabilities.json",
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["diagnostics"] == expected


def test_packaged_and_repository_schemas_match() -> None:
    for name in ("report-envelope.schema.json", "recommendation.schema.json"):
        assert json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8")) == json.loads(
            (ROOT / "traintools" / "data" / name).read_text(encoding="utf-8")
        )


def test_mcp_server_exposes_discovery_tools_when_installed() -> None:
    pytest.importorskip("mcp")
    from traintools.mcp_server import create_server

    async def tool_names():
        return {tool.name for tool in await create_server().list_tools()}

    assert asyncio.run(tool_names()) == {
        "recommend_training_diagnostics",
        "list_training_diagnostics",
        "explain_training_diagnostic",
        "generate_training_integration",
    }
