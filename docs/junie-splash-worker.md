# Splash in the Junie Local Nightly preview

The preview keeps the existing MLX path and adds Splash behind the same gateway.
Junie selects an installed model using its existing local-model interface. The
model descriptor selects the worker; neither Junie nor junie-agent needs a
Splash-specific launcher or settings implementation.

## Packaging and release

`packaging/build_splash.py --output dist/junie-local-splash-preview-macos-arm64.tar.gz`
builds the single-root `junie-mlx-vlm/` archive used by the existing installer.
It contains:

- The gateway and its hash-locked dependencies in `gateway-deps/`.
- The published MLX 0.3.2 worker, unchanged, in `runtime/mlx/`.
- Splash 1.1.0, its standalone Python, native executable, Metal library and
  upstream license in `splash/`.
- `serverctl.sh`, a relative-path launcher, runtime pins and `build-info.json`.

The two runtime archives have independent SHA-256 pins in `packaging/`.
The build records the gateway commit, dirty state and dependency-lock hash.
Build on Apple silicon using Python 3.13+ and uv. Users need neither uv, a
system Python, source checkouts nor a separately installed inference engine.
Splash dependencies are isolated from the gateway and frozen MLX dependencies.

The EAP engine catalog supplies this combined archive; the EAP model catalog
adds a separate Splash model choice and keeps existing MLX entries. Nightly
already uses the EAP catalog. Stable metadata is unchanged. Publish immutable
engine/model artifacts and verify their checksums before merging catalog URLs.
This is an interim macOS distribution; migration into junie-local's common
platform/runtime packaging can follow independently.

## Model selection

The installed model descriptor contains:

```json
{
  "worker_backend": "splash",
  "splash_package": "Qwen3.8-3.6-27B-blend-Splash-v0.3",
  "draft_model": "Qwen3.8-27B-test-DFlash2-v0.3",
  "draft_kind": "dflash"
}
```

The package must be a single directory under the managed models directory,
containing `manifest.json`, `target/`, `draft/`, `vision/` and `tokenizer/`.
It uses Splash's packed format, not MLX safetensors. The adapter validates the
package and requires Apple silicon with macOS 26.4+ before persisting a Splash
selection. The existing installer continues to require M5 or newer; MLX keeps
its existing OS requirements. Switching back to MLX restores that model's own
drafter pair. A busy worker rejects a model switch with HTTP 409.

`--junie-config` writes the usual custom model profile. The Splash profile uses
Chat Completions, reasoning low, temperature 1.0, top-p 0.95, top-k 20 and
`extraBody.stop=[]`. This overrides the legacy XML command stop string, which
is incompatible with native tool calls. No API switch is required.

## Serving and lifecycle

- Public requests use the catalog model ID. The gateway substitutes the packed
  manifest's model ID for Splash while preserving tools, sampling and history.
- `kv_quantization=true` selects INT8 KV; false selects BF16 KV. The existing UI
  toggle works for both workers. Changing it restarts an idle worker.
- Splash has a 3,600-second gateway deadline and 3,540-second worker deadline;
  MLX retains its existing deadlines. Configuration keys are
  `splash_request_timeout_s` and `splash_soft_request_timeout_s`.
- Health, authentication, settings, startup, unload, idle unload, shutdown and
  cancellation remain gateway responsibilities. Splash listens on loopback.
- Splash owns its native prefix cache. MLX-specific APC and cache-management
  endpoints do not describe or control that cache.
- A non-streaming request can recover from `runtime_unavailable` or a broken
  worker connection, with at most three attempts under one gateway deadline.
  Invalid model output does not restart the worker or trigger a blind replay.
- The adapter owns the Splash process group and cleans up native descendants
  when it stops or loses its parent. The API key is passed through the
  environment, never command-line arguments.
- `splash_max_memory_bytes` optionally sets Splash's allocation budget. Without
  it the runtime uses its own memory policy. `/status.backend.status` exposes
  Splash telemetry; it does not relabel Metal allocation as process RSS.

The gateway also supports streamed Responses for integrations that need it:
data-bearing progress events, monotonic sequence numbers, cancellation, and
recovery only before any model output is exposed. The Nightly local profile
uses Chat Completions. Neither tunnel behavior nor Responses recovery is a
prerequisite for local inference.

## Development and validation

For source development, configure `splash_python`, `splash_source` and
`splash_package`, set `worker_backend` to `splash`, and use separate ports and
an isolated `JUNIE_SERVER_CONFIG`. Packaged installations resolve the runtime
relative to the launcher instead. Never point a development test at a user's
installed configuration.

Run the gateway, CLI, settings, Responses and Splash-worker tests. Before
release, extract the actual archive and test MLX/Splash/MLX switching, tool
calls and follow-ups, idle reload, both KV formats, cancellation, engine
recovery and an unmodified Junie task. Unit tests do not establish Metal
execution or long-context quality. Installer rollback and broad hardware,
concurrency and long-session qualification remain separate release concerns.
