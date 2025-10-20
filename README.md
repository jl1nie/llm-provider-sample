## llm-provider-sample

**Author:** jl1nie  
**Version:** 0.1.0  
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

To produce a distributable `.difypkg`, run:

```bash
uv run dify plugin package --output dist/llm-provider-sample.difypkg
```

This packages the current workspace so it can be uploaded to a Dify instance.

### Adding Models

To introduce an additional deployment:

1. **Register the base model** in `models/constants.py`.  
   - Append an entry to `LLM_BASE_MODELS` (or `EMBEDDING_BASE_MODELS`) with the expected Azure deployment defaults, pricing metadata, parameter rules, and the `supports_streaming` flag.  
   - Set `ModelPropertyKey.CONTEXT_SIZE` (and related keys) so runtime heuristics pick up the correct limits.
2. **Expose the deployment as a preset** by creating a YAML file under `models/llm/` (or `models/text_embedding/`) that mirrors the structure of the existing definitions.
3. **Update the provider schema** in `provider/llm-provider-sample.yaml` to add the new `base_model_name` option so it appears in Dify’s credential form.
4. **Adjust runtime logic when needed** in `models/llm/llm.py` or `models/text_embedding/text_embedding.py` if the model demands custom behaviour (e.g., forced non-streaming).

After making changes, run `uv sync` if dependencies changed and package the plugin with `uv run dify plugin package`.
