## llm-provider-sample

**Author:** jl1nie  
**Version:** 0.0.4  
**Type:** model provider plugin

### Overview

Azure OpenAI-compatible LLM and text-embedding provider for Dify 1.9.1. Supports
predefined deployments for streaming and non-streaming models, including chunked
responses for models without SSE support.

### Development

Use [uv](https://docs.astral.sh/uv/) to manage dependencies during local
development:

```bash
uv sync
```

This will install the dependencies declared in `pyproject.toml`. When packaging
the plugin, the plugin daemon will also rely on these definitions—no vendored
wheels are required.
