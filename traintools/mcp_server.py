"""Optional Model Context Protocol server for TrainTools discovery."""

from __future__ import annotations

from typing import Dict, List, Optional

from traintools.capabilities import (
    get_capability,
    integration_snippet,
    list_capabilities,
    recommend_diagnostics,
)


def create_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The MCP server requires the optional dependency: "
            "pip install 'traintools[mcp]'"
        ) from exc

    server = FastMCP(
        "TrainTools",
        instructions=(
            "Recommend paper-backed PyTorch training diagnostics. State call "
            "timing and limitations; do not present heuristic warnings as truth."
        ),
    )

    @server.tool()
    def recommend_training_diagnostics(
        problem: str,
        framework: Optional[str] = None,
        limit: int = 3,
    ) -> List[Dict[str, object]]:
        """Recommend diagnostics for a concrete ML training symptom."""
        return recommend_diagnostics(problem, framework=framework, limit=limit)

    @server.tool()
    def list_training_diagnostics(
        framework: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """List TrainTools capabilities, requirements, and limitations."""
        return list_capabilities(framework)

    @server.tool()
    def explain_training_diagnostic(diagnostic: str) -> Dict[str, object]:
        """Explain one diagnostic, including when not to trust it."""
        return get_capability(diagnostic).to_dict()

    @server.tool()
    def generate_training_integration(
        diagnostic: str,
        framework: str = "pytorch",
    ) -> Dict[str, str]:
        """Generate a minimal PyTorch or Hugging Face integration snippet."""
        return {
            "diagnostic": diagnostic,
            "framework": framework,
            "snippet": integration_snippet(diagnostic, framework),
        }

    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
