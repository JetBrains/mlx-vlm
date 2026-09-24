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

## Distributable engine

`packaging/build_splash.py --output dist/junie-splash-macos-arm64.tar.gz`
produces the same single-root archive layout consumed by Junie's installer.
The archive includes the gateway, `serverctl.sh`, checksum-pinned Splash 1.0.2,
its standalone Python, native executable, Metal library and upstream license.
The gateway dependencies are locked in `packaging/gateway-requirements.lock`.
Build with Python 3.13+ and uv on Apple silicon. The installed archive does not
need uv, system Python, an MLX environment, or a Splash source checkout.

The launcher resolves its bundled runtime relative to itself; moving a complete
installation does not preserve build-machine paths. The installer-owned model
descriptor selects `worker_backend: splash` and a single-component
`splash_package` below the managed models directory. `--junie-config` verifies
the runtime/package before writing the engine configuration and profile. A
fresh Junie home receives a default model selection as well as the profile.

The gateway reports `capabilities.kv_quantization_configurable=false` from
`/current_settings`. Splash 1.0.2 uses INT8 KV; the coordinated Junie branch
shows this setting as fixed. Reject unsupported settings rather than silently
changing their meaning. The model profile sets `extraBody.stop=[]` because
Junie's legacy XML command stop string cannot be combined with Splash's native
tool grammar. Reasoning and sampling settings remain explicit in that profile.

The preview workflow builds artifacts only. It does not replace stable release
metadata. The companion `JetBrains/junie` branch has a release staging tool
that creates checksummed model/engine metadata and the normal installer. The
companion Junie application branch reads backend capabilities and accepts
`JUNIE_LOCAL_INSTALL_SCRIPT_URL` for an isolated release channel.

## Streamed Responses and recovery

The gateway supports `POST /v1/responses` for Splash with `stream: true`.
Authentication and configured model validation are applied before opening the
stream. The public model name maps to the packed worker model as for Chat
Completions. Chat Completions behavior is unchanged.

The adapter assigns one public response ID and monotonic event sequence numbers.
It sends data-bearing `response.in_progress` events every 15 seconds while
waiting, including worker startup/recovery; it does not rely on SSE comments.
Request accounting stays active until the stream closes, preventing idle unload.
Client disconnection cancels the worker HTTP stream.

A `runtime_unavailable` failure, connection failure, or premature stream end
can restart the supervised worker and retry up to three total attempts within
one configured request deadline. Retries happen only before any output event
(including reasoning or tool-item creation) has been exposed. After output,
the adapter emits `response.failed` instead of replaying visible generation.
The caller must handle this failure; this does not implement Junie continuation.
Validation errors and `response.incomplete` are forwarded without retries.

This path needs a deployed gateway and tunnel smoke test before an evaluation;
unit/integration tests with simulated workers do not establish Metal recovery.

### Quota-6 TeamCity evaluation

The existing parent job
`Matterhorn_BenchmarksStaging_ExecutionsJbResearchPrivateSweBenchAllValidIdsWithoutSeedsErokhins2quota`
uses `erokhins2_quota` (verified enabled with quota 6 on 2026-09-24).
The suite's `max_parallel: 1` limits concurrent *cases*, not child task agents.
Keep it at one for this single-model evaluation; do not change the shared quota.

Use the previous `qwen-blend-splash-responses-20260924-full.json` contract in
cai-llm-stack as the template, with a new run identity and the tested gateway
endpoint. Keep `PRIMARY_MODEL_API=responses`, `PRIMARY_MODEL_STREAM=true`,
reasoning low, temperature 1.0, the same 100 tasks and task limits. Pin the
adapter commit in the run's serving provenance. Retain the source revisions
unless deliberately testing a separate Junie change.

For this evaluation explicitly configure gateway `request_timeout_s: 3600`
and Splash `soft_request_timeout: 3600`; the gateway defaults are 275 and 270
seconds and are inappropriate for this workload. Route ngrok to the gateway
public port, not directly to Splash's worker port. Retain authentication via
the existing protected local configuration; never put a key in the manifest.
The existing unauthenticated evaluation contract requires a separately reviewed
network/authentication setup if used with an authenticated gateway.

Before launching, test the public Responses stream, delayed progress events,
cancellation and recovery; verify model identity and that six callers can queue
without error (the native runtime still has four active slots). In cai-llm-stack,
commit the new manifest, run `bin/teamcity-evals preflight <manifest>`, then
`bin/teamcity-evals launch-next <manifest> --confirm-serving-validated`.
Do not reuse the old suite identity: its completed state belongs to the old run.
