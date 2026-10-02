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

## Enable DFlash2 for the FP8 target

The verified DFlash configuration is in `.env.fp8.dflash.switchless.example`.
It uses the original BF16 `incoai/GLM-5.3-Flash-DFlash2` drafter, with no draft
re-quantization. The target remains the dealignai FP8 checkpoint.

Download the draft on the head, then synchronize it to every node over verified
ConnectX routes. For example, with Hugging Face CLI:

```sh
hf download incoai/GLM-5.3-Flash-DFlash2 \
  --revision bf582e4eacc1810f76656d1811693ff6c6737d2a \
  --local-dir /home/kemi/models/incoai/GLM-5.3-Flash-DFlash2
```

If direct connectivity fails, configure your HTTP(S) proxy for the download.
The local fleet also has `/home/kemi/models/hfd.sh`; inspect its options before
using it. The verified `model.safetensors` SHA256 on all four nodes was
`b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac`.
See the [draft model card](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
for its base target and license. This drafter was trained for the official target;
acceptance and speed on a modified checkpoint must be measured.

Stop the previous instance using its old profile, then copy/edit the DFlash
example with fresh runtime names and start it in tmux as above. Key settings:

```sh
SPEC_DISABLE=0
DRAFT_DIR=/home/kemi/models/incoai/GLM-5.3-Flash-DFlash2
K_HI=3
K_LO=3
SPEC_TABLE='[[1,3,3]]'
SCHEDULER_CLS=none
CAPTURE_SIZES='[1,2,4,8,12,16,32]'
```

This uses a fixed three-token draft for one to three active sequences. Capture
sizes include 4, 8 and 12 verification tokens. Keep `GLM5NEXT_PATCH=0` for this
target; importing the full NVFP4 profile would change incompatible precision
settings. To disable DFlash, stop the instance and redeploy with fresh names and
`SPEC_DISABLE=1` using the non-speculative example.

### Observed results (2026-10-02)

Four containers ran without OOM and `/health` returned 200. Three concurrent
short arithmetic requests returned correct answers. With the same 17 GiB KV
allocation per rank, DFlash reported **1,478,427 shared KV tokens**, compared with
1,535,656 without speculation. Context remained 1,048,576, concurrency 3 and the
image limit 32. Draft weights, recurrent state and runtime workspace add memory
overhead; the non-speculative token capacity cannot be assumed to carry over.

One temperature-zero hash-table explanation prompt with a 192-token output cap
gave these measurements:

| Mode | Total elapsed | First token | Decode rate |
| --- | --- | --- | --- |
| No draft | 8.406 s | 0.454 s | ~24 tokens/s |
| DFlash, first test request | 6.427 s | 1.702 s | ~40 tokens/s |
| DFlash, warmed repeat | 4.999 s | 0.361 s | ~41 tokens/s |

Over the first two DFlash requests, metrics counted 247 accepted tokens out of
423 drafted tokens (~58%). The warmed total time was ~1.68x faster on this short
workload. These are limited measurements, not a general speedup guarantee:
the repeated prompt may benefit from prefix caching, and there were no full 1M,
long-prompt, high-resolution multi-image or sustained concurrency stress tests.
Generated text differed between the greedy modes, so bitwise output equivalence
was not established. Check `vllm:spec_decode_*` metrics and representative output
quality when evaluating other workloads.


### Single-request tuning update (2026-10-02)

The current FP8+DFlash example keeps K=3 and MAX_SEQS=3, raises
BATCHED_TOKENS to 8192, and enables GLM_DRAFT_CONV_FUSED=1 (with
GLM_DRAFT_CONV_FUSED_QUAL=1) and GLM_ROUTER_DEDUP=1. With 17 GiB/rank
the final boot reported 1,469,300 shared KV tokens. The earlier capacity and
timings above describe the initial 4096-budget deployment.

K=2 and K=4 were slower overall than K=3 on the three tested prompt types.
The two kernel optimizations together reduced paired decode step time by
about 1.1–1.3%; aggregate token/s and first-token improvement were not
established beyond the identical-configuration control's variation.
MAX_SEQS=4 was not tested because the primary workload is one request.
See [the measurements, limitations and reproduction steps](results/2026-10-02-fp8-tuning.md).
