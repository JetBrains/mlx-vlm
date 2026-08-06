# Junie local gateway API

The public server is the lightweight gateway at `http://localhost:8085`.
The inference worker is private on `127.0.0.1:8086` and may be restarted or
stopped without taking the gateway down.
The worker also watches its parent process and exits if the gateway crashes,
so an orphan cannot keep model memory or port `8086` occupied.

Start both through the single supported entrypoint:

```bash
./start.sh
```

Use `./serverctl.sh` for lifecycle and settings commands. Except for
`/health`, each control endpoint also accepts the `/v1` prefix.

## Inference

`POST /v1/chat/completions` accepts an OpenAI chat-completions JSON body.
The gateway always forwards it as a non-streaming request. If the worker was
stopped by the idle timeout, the gateway starts it, waits until it is ready,
and then forwards the same request.

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
  "inference": {
    "in_progress": false,
    "in_flight": 0,
    "queue_depth": 0,
    "requests": []
  }
}
```

Phases are `loading_model`, `ready`, `restarting`, and `stopping`. An
auto-unloaded worker is represented as `phase: "ready"` with
`model.loaded: false`, because the gateway is ready to start it on demand.

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

## Shutdown and monitoring

- `POST /shutdown`: stop the worker, release model memory, then terminate the
  gateway. Response: `{"status":"shutting_down"}`.
- `GET /health`: cheap gateway liveness. It remains `200` while the worker is
  stopped or loading.
- `GET /metrics`: worker metrics plus a `gateway` block.
- `GET /cache/stats`: prompt-cache statistics.
- `POST /cache/reset`: clear the worker prompt cache.

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
