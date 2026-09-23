# Experimental Splash worker

The Junie gateway can supervise a Splash worker instead of MLX. MLX remains the
default (`worker_backend: "mlx"`). The installer and published engine are
unchanged. This is an offline source-checkout prototype, not a shipped runtime.

Use the maintained `JetBrains/mlx-vlm` repository's `junie-mlx-vlm` branch. The
old `JetBrains/junie-mlx-vlm` repository has moved here.

## Configuration

Start from a separate `server-config.json`, using separate public/worker ports.
Set `JUNIE_SERVER_CONFIG` to its absolute path. Example (replace all paths and
supply a fresh private API key):

```json
{
  "worker_backend": "splash",
  "splash_python": "/path/to/splash/.venv/bin/python",
  "splash_source": "/path/to/splash",
  "splash_package": "/path/to/packed-blend-v03",
  "model_name": "Qwen3.8-3.6-27B-blend-MLX-4bit",
  "draft_model": "Qwen3.8-27B-test-DFlash2-v0.3",
  "draft_kind": "dflash",
  "host": "127.0.0.1",
  "port": 19539,
  "worker_port": 19540,
  "api_key": "replace-with-a-private-key",
  "kv_quantization": true,
  "auto_unload_time": 60
}
```

Launch the gateway with `python -m mlx_vlm_gateway.cli daemon` from this
checkout, using an environment with the gateway dependencies installed.
Its worker uses `splash_python`, not that gateway interpreter. No model or
runtime is downloaded by the adapter. The package must contain `manifest.json`,
`target/`, `draft/`, `vision/` and `tokenizer/`; ordinary MLX weights do not work.
Use a verified package and pinned source/binary. The initial target is the
benchmark runtime `incoai/splash@f58d36ddb046726adc8937ab67d07bdde015c0d6`.

## Behavior

- Junie's public model ID stays stable. The gateway substitutes the package
  manifest's model ID for worker requests; completion responses retain Splash's
  actual package identity. Tools, reasoning history and sampling are passed
  through. The gateway retains its existing non-streaming behavior.
- Gateway health, status, settings, authentication, unload, shutdown, idle
  unloading, on-demand startup and error recovery remain owned by the gateway.
- Splash listens on loopback at `worker_port`; its credential is passed through
  the environment, never command-line arguments.
- The wrapper owns the Splash HTTP/native process group and tears it down on
  normal exit, child failure, or loss of its gateway parent.
- `/status.backend.status` exposes native Splash telemetry. The existing MLX
  footprint/cache fields remain empty rather than mislabeling Metal allocation
  bytes as process memory.
- Context limits and the soft request timeout are passed to Splash. Existing
  gateway deadlines still apply. No extra memory ceiling is imposed.

## Prototype limitations

This pinned runtime supports INT8 KV only. `kv_quantization=false` is rejected
before startup or a settings update; it is not silently ignored. Support for
newer Splash BF16-KV options requires a separately validated runtime update.

Only the configured package is exposed. Switching to another installed MLX
model is rejected; multi-package selection is a follow-up. MLX-specific APC,
pinned-prefix and cache-management APIs are not implemented by Splash. Native
cache reuse works independently. Packaging, installer changes and release
publication are outside this prototype. Real Junie task quality, long-context
latency and image round trips still need evaluation.
