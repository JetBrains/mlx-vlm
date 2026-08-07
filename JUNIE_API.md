# Junie local gateway API

The public server is the lightweight gateway at `http://localhost:19239` by
default. Its `host` and `port` come from `server-config.json`.
The gateway spawns the inference worker on the same `host` at `worker_port`
(`19240` by default) and may restart or stop it without going down itself.
The worker also watches its parent process and exits if the gateway crashes,
so an orphan cannot keep model memory or port `19240` occupied.

Start both through the single supported entrypoint — the frozen
`junie-mlx-vlm` on a shipped machine, or from a checkout:

```bash
./start_dev.sh
```

Use `./serverctl.sh` for lifecycle and settings commands. Except for
`/health`, each control endpoint also accepts the `/v1` prefix.

## Inference

`POST /v1/chat/completions` accepts an OpenAI chat-completions JSON body.
The gateway always forwards it as a non-streaming request. If the worker was
stopped by the idle timeout, the gateway starts it, waits until it is ready,
and then forwards the same request.

Corrupted-generation bandage: if the model emits a long run of token id 0
("!", the argmax of zeroed/NaN logits — the signature of corrupted serving
state), the worker fails the request with HTTP 508 (a private
worker-to-gateway signal, tunable via `MLX_VLM_MAX_ZERO_TOKEN_RUN`,
default 8, 0 disables). The gateway maps it to a 503 for the client and
restarts the worker process, so a retry lands on a freshly loaded model.

## Status

`GET /status` reports gateway-owned lifecycle and request state:

```json
{
  "phase": "ready",
  "phase_detail": null,
  "phase_since_unix": 1785925149.289,
  "uptime_s": 512.3,
  "model": {
    "loaded": true,
    "id": "mlx-community/Qwen3.6-27B-4bit",
    "draft_model": "mlx-community/Qwen3.6-27B-MTP-4bit",
    "context_limit": null
  },
  "memory": {
    "total_gb": 19.06,
    "peak_gb": 21.49,
    "kv_cache_gb": 1.9
  },
  "inference": {
    "in_progress": false,
    "in_flight": 0,
    "queue_depth": 0,
    "requests": []
  }
}
```

Phases are `loading_model`, `ready`, `restarting`, `stopping`, and `error`.
After three consecutive startup failures, the gateway enters `error` instead
of restarting forever; settings can still be changed to retry startup. An
auto-unloaded worker is represented as `phase: "ready"` with
`model.loaded: false`, because the gateway is ready to start it on demand.

- `memory.total_gb` — the worker process's physical footprint (same number
  Activity Monitor shows), including the model weights and all caches.
- `memory.peak_gb` — lifetime maximum of that footprint (worst case this
  worker run has needed).
- `memory.kv_cache_gb` — in-RAM KV held by the prefix cache (warm
  conversations, plus a pinned seed when one is configured). This is the
  one part that can be freed without unloading the model:
  `POST /v1/cache/reset` releases it at the cost of the next requests
  re-prefilling their context.
- `memory` is `{}` when the worker is stopped (auto-unloaded or not yet
  started) — it comes from the worker's own `/ready` probe, which the
  gateway polls every few seconds while the worker is up.

## Settings

`GET /current_settings` returns:

```json
{
  "model_name": "mlx-community/Qwen3.6-27B-4bit",
  "max_context_length": null,
  "kv_quantization": true,
  "auto_unload_time": 600
}
```

`POST /apply_settings` accepts any subset of those fields plus `force`:

```json
{
  "max_context_length": 150000,
  "kv_quantization": true,
  "force": false
}
```

- `model_name`: informational; only the installed
  `mlx-community/Qwen3.6-27B-4bit` value is accepted. Other values return
  `400` without changing config or stopping the worker.
- `max_context_length`: positive integer or `null`.
- `kv_quantization`: boolean.
- `auto_unload_time`: positive integer seconds or `null` to disable.
- `force`: boolean. When inference is active, restart settings require
  `force: true`; otherwise the gateway returns `409`.

`auto_unload_time` alone is applied live:

```json
{
  "status": "applied",
  "changes": ["auto_unload_time"],
  "settings": {
    "model_name": "mlx-community/Qwen3.6-27B-4bit",
    "max_context_length": null,
    "kv_quantization": true,
    "auto_unload_time": 600
  }
}
```

The other settings restart the worker and return immediately after the new
worker process was launched:

```json
{
  "status": "applying",
  "model": "mlx-community/Qwen3.6-27B-4bit",
  "changes": ["kv_quantization", "max_context_length"],
  "message": "Model serving is restarting; poll GET /status until phase is 'ready'."
}
```

Settings are stored in
`~/.local/share/junie-local/server-config.json`. The file is created from
defaults on first start. Writes preserve worker-only tuning fields and use an
atomic file replacement, so an interrupted write does not leave partial JSON.
There is only one persistent config file. Restart settings are validated and
saved before the running worker is stopped; if saving fails, that worker keeps
running with the previous settings.

The gateway reads and validates this file once at startup, then keeps only a
small in-memory copy. Inference and idle checks do not read the file. Applying
settings atomically updates the same file and the in-memory copy; a restarted
worker reads that file once. Manual file edits therefore require a gateway
restart.

`host` and `port` are launch settings rather than `/apply_settings` fields,
because changing the gateway's own listening socket requires restarting the
gateway. Stop it, edit the same `server-config.json`, and start it again.
Startup fails with a clear error if either the requested public port or
private worker port is already occupied.

## Shutdown and monitoring

- `POST /shutdown`: stop the worker, release model memory, then terminate the
  gateway. Response: `{"status":"shutting_down"}`.
- `POST /unload`: stop the worker and release model memory while keeping the
  gateway available. The next inference request starts a fresh worker.
- `GET /v1/models`: return the worker's OpenAI-compatible local model list;
  it remains available from gateway memory after `/unload`.
- `GET /health`: cheap gateway liveness. It remains `200` while the worker is
  stopped or loading.
- `GET /metrics`: worker metrics plus a `gateway` block.
- `GET /cache/stats`: prompt-cache statistics.

Worker-only monitoring endpoints return `503` while the worker is stopped.

## Command examples

```bash
./serverctl.sh status
./serverctl.sh settings
./serverctl.sh apply auto_unload_time=600
./serverctl.sh apply max_context_length=150000
./serverctl.sh wait
./serverctl.sh stop
```

When inference is active, interactive `serverctl.sh apply` asks whether it
should stop the request and retry with `force=true`. Non-interactive callers
must pass `force=true` explicitly.
