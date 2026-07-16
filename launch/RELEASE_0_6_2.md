# TrainTools 0.6.2: MCP Registry Namespace Correction

This metadata-only patch preserves the case-sensitive GitHub namespace required
by MCP Registry OIDC authentication. It includes the registry-compatible
`traintools mcp` command introduced in 0.6.1:

```bash
uvx --with mcp traintools mcp
```
