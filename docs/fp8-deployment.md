# GLM-5.3-Flash FP8 on four Spark nodes

The TP4 launcher and switchless ConnectX transport can also serve compatible FP8
GLM checkpoints. vLLM reads the checkpoint's quantization configuration; this
recipe does not convert FP8 weights to NVFP4. `--dtype bfloat16` controls the
non-quantized computation dtype, rather than replacing the weight quantization.

## Launcher options

- `SPEC_DISABLE=1` omits both the draft mount and speculative configuration, so
  `DRAFT_DIR` is unnecessary. Set `SCHEDULER_CLS=none` for ordinary scheduling.
- `KV_DTYPE` selects the target KV cache dtype. Its default remains `fp8_e4m3`;
  the verified FP8 recipe uses `auto`.
- Keep `GLM5NEXT_PATCH=0` for the tested checkpoint: its attention projections
  include BF16 weights, and enabling the NVFP4 attention patch produced garbled
  output despite a passing health check. This finding is checkpoint-specific.
- Use Triton MoE and linear backends and `GLM_MAMBA_ALIGN_FIX=1` as recorded in
  the example. Default speculative/NVFP4 behavior is preserved when the new
  options are omitted.

## Configuration and launch

```sh
cp .env.fp8.switchless.example .env
# Edit host/IP mappings, model/image/NCCL paths and the pinned library hash.
# Model weights and the image must exist on all four nodes.
tmux new-session -s glm53-fp8
./start.sh serve 2>&1 | tee -a logs/serve-fp8.log
./start.sh status
./start.sh logs 0 80
```

The example records rank order spark1 → spark2 → spark4 → sparkX → spark1.
Verify topology if cables changed. Bulk weights and image synchronization on
this fleet use the ConnectX interconnect; verify fabric routes before transfer.
The launcher's small overlay synchronization uses SSH/rsync.

For a parameter change, stop with the old profile first (`./start.sh stop`), then
choose fresh `CTN` and `OVERLAY_REMOTE` values and launch again. Stopped containers
and their overlays are preserved for inspection; do not overwrite them to bypass
preflight. Startup may take several minutes for loading and kernel compilation.

## Verified deployment and limits

On 2026-10-01, `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` served as
`glm-5.3-flash` on port 8888 with image `glm53-roce:v11-b58f34ea`.
Applied settings were:

| Setting | Value |
| --- | --- |
| Single-sequence context limit | 1,048,576 tokens |
| KV allocation per rank | 17 GiB / 18,253,611,008 bytes |
| Actual shared KV token capacity | 1,535,656 tokens |
| Maximum active sequences | 3 |
| Chunked prefill batch token budget | 4,096 |
| Images in the entire prompt, including history | 32 |
| Multimodal processor cache | 2 GiB |

KV capacity is shared across requests, not multiplied by rank count or maximum
sequences. Three simultaneous 1M sequences do not fit this pool. KV allocation
is per rank, and changing concurrency or other model settings can change its
effective token capacity; read the new boot log rather than assuming a ratio.

Validation included all four containers running, `/health` HTTP 200, the expected
name/context in `/v1/models`, boot-log KV capacity and applied arguments, and a
32-small-image chat request returning HTTP 200. Full 1M-context, high-resolution
32-image and maximum-concurrency memory stress tests were not performed.
DGX Spark uses unified memory: inspect host `MemAvailable`, container OOM state
and logs rather than relying solely on Docker stats or `nvidia-smi` memory fields.

An upstream HTTP 400 stating “At most 4 image(s)” was resolved by changing
`MM_IMAGES` from 4 to 32 and restarting. This limit includes images retained in
conversation history, even when a proxy reports the upstream error.
