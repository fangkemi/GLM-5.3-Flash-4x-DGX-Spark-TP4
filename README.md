# GLM-5.3-Flash on 4x NVIDIA DGX Spark (vLLM TP4, NVFP4 experts, lossless 8-bit dense, DFlash2)

Serve [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (321B, 18B active) on four
DGX Spark boxes (GB10, SM121, 128 GB unified memory each) behind a RoCE switch. One OpenAI-compatible
endpoint with tool calling, reasoning and images, 262k context, up to 32 concurrent sequences.

- **Weights:** NVIDIA's NVFP4 routed experts, untouched. The dense layers that checkpoint leaves in BF16
  (attention, KDA, shared experts, dense MLP) are stored on 8-bit grids they already fit
  (MXFP8 / block FP8), so they read at half the bytes with 0.25 % output error.
- **Decode:** DFlash2 speculative decoding with a batch-uniform adaptive draft length, Marlin W4A16 MoE,
  FP8 KV cache, RoCEnante one-shot RDMA collectives on both ConnectX-7 rails, and a set of exact
  kernel and scheduling fixes listed below.

## Results (2026-09-28, this commit, fresh clone)

sparkDash 1.8.8, 256 new tokens, temperature 0, thinking off, idle endpoint, same boot as the gates
below. Two warm-up prose c1 runs discarded before each series.

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **71.8** | 107.6 (54.1) | **156.9** (40.6) | 233.2 (30.2) | 329.2 (21.7) |
| code | 114.8 | 159.7 (83.1) | 208.9 (56.5) | 297.3 (40.5) | 342.7 (24.5) |
| structured | 158.4 | 259.3 (129.7) | 356.6 (92.2) | 645.9 (82.8) | 821.3 (53.5) |
| JSON | 104.0 | 173.7 (86.8) | 260.2 (68.7) | 358.7 (48.7) | 504.0 (35.0) |

Prose c1 is the median of three runs (73.1 / 69.3 / 71.8) and prose c4 the median of three
(143.8 / 156.9 / 176.4); every other cell is one run. The gate series on the same boot half an hour
earlier read prose c1 71.4 (74.3 / 71.4 / 62.9) and c4 151.0. Acceptance moves prose by 3-5 % run to
run, and sparkDash uses different prompts at each concurrency, so per-stream values are not
comparable across columns. The stable number is the decode step: `bench/accept_probe.py` at c1 gives
prose 39.0 ms per step at 2.2-2.3 tokens per step, code 48.9 ms at 4.5-4.7, JSON 50.1 ms at 6.3 (two
runs each, identical within 0.2 ms).

**Prefill, cold, tok/s by prompt length** (sparkDash prefill bench, one pass, same boot; 256k is
258,073 tokens, the longest prompt that fits 262,144 with the reply)

| 4k | 16k | 32k | 64k | 128k | 256k |
|---:|---:|---:|---:|---:|---:|
| 2098 | 2243 | 2255 | 2264 | 2229 | 2150 |

Time to first token at 32k is 14.5 s, at 128k 58.8 s, at 256k 120 s. sparkDash's prefill filler is
one repeated token (unique prefix per size, so the prefix cache does not apply); on varied random-word
text `bench/prefill_checked.py` measured 2,202 / 2,209 / 2,197 tok/s at 16k / 32k / 64k on the
2026-09-27 stack, within 2 % of these. Prefill is not optimised yet: in a prefill step MoE takes 35 %, attention
16 %, all-reduce 13 % and mHC 11 %.

- **Quality:** `bench/qeval.py` 75/75 (75 auto-scored checks: code run against hidden asserts, JSON
  schema, numeric answers, format constraints, degeneration). KL divergence 0.0293 over 6618
  teacher-forced positions against a BF16-attention reference (`bench/kld_probe.py`,
  `bench/compare_kld_strict.py`).
- **Boot:** ~2 min to `/health` 200 with warm JIT caches; 7.4 min on the first boot of a fresh clone (cold FlashInfer / Triton / TileLang caches).

Earlier measurements: [measurement history](docs/history.md) (2026-09-26/27 L2 and GDN/router panels) and the
[2026-09-18 first release](docs/history-2026-09-18.md) (prose c1 37 tok/s). These numbers include the 2026-09-27
KDA speculative-block-boundary fix ([notes](docs/results/2026-09-27-kda-boundary-fix.md)).

## What is in the stack

Each row was measured alone against the stack without it, in the same boot where possible
(`overlay/glm_ab.py` switches kernels between two CUDA-graph sets in one boot; a result is
promoted only when its confidence interval clears zero and an identical-arm control does not).

| Piece | Where | Effect | Credit |
|---|---|---|---|
| NVFP4 experts on Marlin (W4A16) | image, `MOE_BACKEND=marlin` | code +40 %, JSON +50 % vs the official FP8 checkpoint, same quality gate | NVIDIA, LibertAI, Red Hat AI checkpoints; alexellis's launch line |
| Dense layers on 8-bit grids | `scripts/build_lossless8.sh`, `glm_quant_mix.py`, `overlay/qmix_patch.py` | step −7.5 ms, output error 0.25 % | tonyd2wild found the 18 GiB left in BF16 |
| DFlash2 drafter, block-FP8 linears | `scripts/drafter_fp8.py` | draft graph 4.24 → 3.10 ms, acceptance unchanged | incoai (drafter) |
| Batch-uniform draft length | `overlay/glm_levers_sched.py`, `profiles/levers_policy.json` | prose c4 136.7 → 148.2 (no eager mixed-k steps) | builds on jnardiello's adaptive-k scheduler and Reederey87's verify-only idea |
| RoCEnante one-shot all-reduce / all-gather | `Dockerfile.roce`, `roce/` | decode collectives over RDMA, both rails | Luke Alonso, Jason Cook (local-inference-lab/b12x#295, vllm#597); tonyd2wild's v11 port; rhys101 |
| Replicated-linear TP split, FP8 draft head, one-gather top-k | `overlay/glm_ds_*.py` | dense −0.48 ms, bit-exact where marked | ported from our DeepSeek-V4.1 stack |
| KDA verify stash, no-copy reads, fused flags | `overlay/glm_kda_stash*.py`, `kda_stash.py` | −0.90 ms (no-copy) | ours |
| L2 prefetch of the next weights | `overlay/glm_l2_prefetch.py` | −0.5 ms | ours (from the DeepSeek-V4.1 stack) |
| Router GEMM dedup | `overlay/glm_router_dedup.py` | −0.5 ms | vllm#55736 (JaredforReal), MiaAI-Lab issue #271 |
| GDN metadata fast path | `overlay/glm_gdn_metadata_fast.py` | exact, host side | ours, on vLLM's builder |
| Vocab-parallel target argmax, greedy path also for `min_tokens` | `overlay/glm_target_argmax.py` | no full-vocab gather at verify; −0.59 ms on `min_tokens` requests | vLLM's draft-side argmax from vllm#34049 (zixi-qi) |
| Padded-vocab clamp in both samplers | `overlay/gumbel.py`, `rejection_sampler_utils.py` | correctness | vllm#50843 (alexbi29) |
| DSA indexer kpool tail fixes | `GLM_KPOOL_FIX=1`, `overlay/glm5next_*.py`, `mla_indexer.py`, `mamba_hybrid.py` | correctness past the first KV block | vllm#57477 (JaredforReal), #58454 (mmastrac, on ivanium's #55219), #53906 (ZJY0516), root cause vcruz305 |
| Prefill cadence + end drain | `overlay/glm_prefill_sched.py`, `glm_prefill_hooks.py` | decoders under a 32k prefill 1.3 → 7.3 tok/s; short newcomer at c4 −22 %; TTFT −8 %; step unchanged | jnardiello (E27, E27b/c, E29), FujitsuPolycom (SparkRing non-DP throttle) |
| Prefix cache for the DFlash2 draft group | `overlay/kv_cache_coordinator.py` | repeated 20k prompt 8.1 s → 0.63 s TTFT | tonyd2wild |
| Fast weight loader, persistent FlashInfer JIT cache | `overlay/glm_fast_load.py`, `start.sh` | boot 271 → 128 s | vllm#58726 (Willian-Zhang) |

Tried and rejected, with numbers: confidence-based verify cut (prose −5 to −9 %), Marlin tile M=32
(+1.2 ms), fused mHC kernels (not bit-exact), a CUDA graph for the DFlash context KV (+0.05 ms),
W4A4 / MXFP4 experts for prefill (1.1-1.3x on MoE at 16-21 % MoE output error).

## Build

1. **Image**, on every node (no CUDA compile, a minute or two):
   ```bash
   docker build --platform linux/arm64 -f Dockerfile.roce -t glm53-roce:v11-b58f34ea .
   ```
   It adds the b12x RoCEnante subset (Apache-2.0, `roce/b12x/LICENSE`, pinned in
   `roce/b12x/PROVENANCE.json`) to tonyd2wild's `ghcr.io/tonyd2wild/vllm-glm53-flash` (vLLM `487ecf187`).
2. **Weights**, on every node at the same path (CPU only):
   ```bash
   hf download nvidia/GLM-5.3-Flash-NVFP4 --local-dir ~/models/nvidia/GLM-5.3-Flash-NVFP4
   scripts/build_lossless8.sh ~/models/nvidia/GLM-5.3-Flash-NVFP4 ~/models/glm-quant-mix
   hf download incoai/GLM-5.3-Flash-DFlash2 --local-dir ~/models/incoai/GLM-5.3-Flash-DFlash2
   python3 scripts/drafter_fp8.py ~/models/incoai/GLM-5.3-Flash-DFlash2 ~/models/incoai/GLM-5.3-Flash-DFlash2-fp8blk
   ```
   The drafter is CC BY-NC-ND 4.0: keep the re-encoded copy local.
3. **NCCL** 2.30.7 built for the host (optional, `NCCL_HOST_DIR`; the image's NCCL also works).

## Quick start

```bash
cp .env.example .env            # hosts, fabric, image and weight paths; sources profiles/current.env
./start.sh serve                # workers first, then the head
./start.sh status               # until health 200
./start.sh logs 0 80
./start.sh stop                 # stops, never removes; a new deployment needs a fresh CTN
```

The endpoint binds to loopback on the head; put your own tunnel or proxy in front of it.

```bash
curl http://127.0.0.1:8093/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-FP8", "messages": [{"role": "user", "content": "What is 19 + 23?"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

`reasoning_effort` is `low`, `high` or `max`. Tool calls use the `glm47` parser, reasoning the `glm45` parser.

Every switch is a line in `profiles/current.env`; removing a line turns that piece off. The
`serve` step first checks that no existing container already uses the target name or overlay path.

## Benchmarks and gates

```
bench/qeval.py              75-check quality gate (run / compare)
bench/kld_probe.py          teacher-forced top-20 logprobs from the live endpoint; compare_kld_strict.py
bench/accept_probe.py       step ms and accepted tokens per step at c1 (the stable speed number)
bench/conc_bench.py         concurrency sweep with 32 distinct prompts per type
bench/prefill_bench.py      cold prefill tok/s by prompt length
overlay/glm_ab.py, scripts/ab_inboot_glm.py   in-boot A/B of kernel switches with an A/A control
```

`tests/` hold the unit tests; the CPU ones run inside the image
(`docker exec ... python3 tests/<file>` with `PYTHONPATH=<overlay dir>` and `CUDA_VISIBLE_DEVICES=`),
the `*_gpu.py` ones need a free GPU, so run them with the model stopped.

## Known limits

- Prefill is ~1.7x behind recipes that use W4A4 experts and a sparse-MLA prefill plugin; that is the next
  piece of work, without giving up the weights above.
- GLM greedy output is not bit-reproducible across runs on this stack (batch-dependent kernels), so
  exactness of a change is checked per kernel, not by comparing text.
- The DFlash2 drafter is CC BY-NC-ND 4.0.

## More documentation

- [docs/install.md](docs/install.md): host prerequisites, NCCL, image build, launch and preflight rules
- [docs/weights.md](docs/weights.md): preparing the lossless8 target and the FP8 drafter
- [docs/runtime.md](docs/runtime.md), [docs/validation.md](docs/validation.md): runtime switches and the validation scope
- [docs/results/](docs/results/): raw result files per release

See [CREDITS.md](CREDITS.md) for authors and pull requests, and [NOTICE](NOTICE) with [LICENSES/](LICENSES/) for licence boundaries.
