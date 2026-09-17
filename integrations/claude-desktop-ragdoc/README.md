# Ragdoc for Claude Desktop

This directory builds a Claude Desktop MCP Bundle that connects to the hosted
Ragdoc MCP server and supplies Ragdoc branding to Claude's connector UI.

## Build

```bash
npm ci --omit=dev
npx @anthropic-ai/mcpb validate manifest.json
npx @anthropic-ai/mcpb pack . dist/ragdoc-1.0.0.mcpb
```

Install the generated `.mcpb` file through Claude Desktop's Extensions settings.
Once the extension is connected and tested, disable the older `ragdoc` entry in
`claude_desktop_config.json` to avoid duplicate tools.
