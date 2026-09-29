# GLM-5.3-Flash on 4x NVIDIA DGX Spark (vLLM TP4, NVFP4 experts, lossless 8-bit dense, DFlash2)

Serve [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (321B, 18B active) on four
DGX Spark boxes (GB10, SM121, 128 GB unified memory each) behind a RoCE switch. One OpenAI-compatible
endpoint with tool calling, reasoning and images, 262k context, up to 32 concurrent sequences.

- **Weights:** NVIDIA's NVFP4 routed experts, untouched. The dense layers that checkpoint leaves in BF16
  (attention, KDA, shared experts, dense MLP) are stored on 8-bit grids they already fit
  (MXFP8 / block FP8), so they read at half the bytes with 0.25 % output error.
- **Decode:** DFlash2 speculative decoding. The target verifies a draft whose length is chosen per step from the
  drafter's own confidence (on the GPU at one request), a certified argmax skips most of the LM head, and
  sampling at T > 0 couples the draft to the target's noise so more of it is accepted. Marlin W4A16 MoE,
  FP8 KV cache, RoCEnante one-shot RDMA collectives on both ConnectX-7 rails.
- **Prefill:** sharded mHC, FlashKDA, Triton sparse MLA, dense 8-bit GEMMs on cuBLAS and fused routed-MoE
  kernels: +42 % cold prefill at 32k-128k over the 2026-09-28 release before the routed-MoE kernels and the
  gather route (+3-4 % more in their own A/B).

## Results (2026-09-29)

All numbers at the fleet's GPU clock cap of 2200 MHz. sparkDash 1.8.8: 256 new tokens, temperature 0, thinking
off, idle endpoint, two warm-up prose c1 runs discarded; c1 is the median of 3 runs, c4 the median of 2, c16 one
run. Prefill: sparkDash prefill bench, cold (salted), after a 4k warm-up, median of 2. RigMark 1.0.0 decode
screen: code / prose / structured at effort low, median of 4 passes (not a conformant receipt).

Both columns of the tables come from one gate window (gate 0003, 2026-09-29 00:43-01:50), same boot order and
benchmark code. The right column is this stack without the five additions marked "not in gate 0003" further down.

**This release, fresh clone, everything on** (the gates and every number are in
[docs/results/2026-09-29-release.md](docs/results/2026-09-29-release.md)): cold boot from a new tree 561 s with every
piece armed on 4/4 ranks; sparkDash prose c1 80.3, code c1 126.3 (quick bench, median of 2 series); cold
prefill 3015 / 3122 / 3165 tok/s at 8k / 32k / 128k (runs 2-3 after a restart); short-prompt TTFT 326 ms.

**Decode, per-stream tok/s (aggregate in brackets)**

| prompt type | 2026-09-28 release | 2026-09-29 stack, gate 0003 | change |
|---|---:|---:|---:|
| prose c1 | 72.3 | 83.8 | +16.0 % |
| code c1 | 109.5 | 125.2 | +14.4 % |
| JSON c1 | 99.7 | 114.8 | +15.1 % |
| prose c4 | 42.0 (161.4) | 39.5 (154.2) | -6.0 % |
| code c4 | 55.8 (211.5) | 56.9 (210.8) | +1.9 % |
| JSON c4 | 65.6 (252.0) | 72.8 (281.0) | +11.0 % |
| prose c16 (1 run) | 22.1 (335.6) | 21.7 (326.1) | -1.9 % |

**Prefill, cold**

| prompt | 2026-09-28 release | 2026-09-29 stack, gate 0003 | change |
|---|---:|---:|---:|
| 32k tokens | 2150 tok/s, TTFT 15.3 s | 3053 tok/s, TTFT 10.7 s | +42.0 % |
| 128k tokens | 2127 tok/s, TTFT 61.6 s | 3036 tok/s, TTFT 43.2 s | +42.7 % |

**RigMark 1.0.0 decode screen, tok/s**

| workload | 2026-09-28 release | 2026-09-29 stack, gate 0003 | change |
|---|---:|---:|---:|
| code | 95.5 | 110.4 | +15.6 % |
| prose | 51.7 | 57.6 | +11.4 % |
| structured | 140.5 | 146.4 | +4.2 % |

The full RigMark protocols on the same boot (effort low, 1.0.0 / 1.1.0): code 108.3 / 109.0, prose 58.7 / 57.7,
structured 146.6 / 147.0 tok/s, prefill 8k / 32k / 64k cold 2802 / 3052 / 3058 tok/s, C4 aggregate 174.5 / 172.6 tok/s.

- **Quality:** KL divergence 0.0285 over 6618 teacher-forced positions against a BF16-attention reference
  (`bench/kld_probe.py`, `bench/compare_kld_strict.py`), and 0.0098 over 98,500 positions of four 16.5k-40k-token
  prompts against the 2026-09-28 release (`bench/final_bench.py kldlong`), so the prefill path is covered too; these
  two are the primary gates. `bench/qeval.py` (75 auto-scored checks) with the fixed number extractor: 75 and 74 on
  this release; qeval moves by 2-3 points run to run on every stack here (the 2026-09-28-based stack read 73 / 75 / 72
  in one boot), so it is run three times against a same-window reference, not as a single-run floor. T > 0 scan
  (`bench/final_bench.py tscan`, 35 outputs at T 1.0 / 0.6 and mixed batches): 0 garbled outputs.
- **Boot:** 3.6 min to `/health` 200 with warm caches, 9.4 min on the first boot of a fresh tree (every JIT cold).

Earlier measurements, including the 2026-09-28 release table measured without the clock cap:
[measurement history](docs/history.md) and the [2026-09-18 first release](docs/history-2026-09-18.md).

## What is in the stack

Each row was measured against the stack without it, in the same boot where possible
(`overlay/glm_ab.py` switches kernels between two CUDA-graph sets in one boot; a result is
promoted only when its confidence interval clears zero and an identical-arm control does not).
"Exact" means the output is bit-identical to the path it replaces; the others are judged by the KL gates above.

**Added on 2026-09-29** (each is one switch in `profiles/current.env`)

| Piece | Where | Effect | Credit |
|---|---|---|---|
| Draft-shape truncation | `GLM_DRAFT_TRUNC`, `overlay/glm_draft_trunc.py` | verifies only the draft prefix that pays for its rows; prose c1 +5.8 %, c4 +4.5 %; exact | ours |
| Device-side verify-shape selection (not in gate 0003) | `DEVSELECT=1`, `overlay/glm_devselect.py` | at one request the GPU picks the verify length itself (one parent graph over the captured 2..8-row graphs), so the host no longer waits for the draft: step −0.45 ms, prose +1.9 %, RigMark T > 0 prose +3.9 %; buffer-identical to host truncation, and device and host share one λ, so both pick the same length | ours |
| Truncation cost refit for batch > 1 (not in gate 0003) | `TRUNC_COST=c4fit` | c4 prose +3.0 % (fleet A/B); c1 unchanged by construction | ours |
| Gumbel-coupled drafting (T > 0) | `GLM_GUMBEL_COUPLED`, `overlay/glm_gumbel_coupled.py` | the drafter reuses the target's Gumbel noise: +3.2 % at T = 1; committed tokens equal plain sampling with the same seed | method: Jim Routh, llama.cpp-lab PR #26 |
| Certified target head | `GLM_CERT_HEAD`, `overlay/glm_cert_head.py`, `cert_math.py` | an 8-bit screen of the LM head with a proven error bound picks the argmax; the full BF16 row is read only when the bound cannot decide; `min_tokens` requests too; exact | ours |
| Decode kernel set | `GLM_GATE_GEMV`, `GLM_EARLY_PLAN`, `GLM_MARLIN_TUNE_ON`, `GLM_DENSE_FAST`, `GLM_L2_PREFETCH_MLA/_DRAFT`, `GLM_MHC_BF16W`, `GLM_DRAFT_CONV_FUSED`, `GLM_ROCE_PROXY_CPUS` | with the certified head: −1.25 ms per c1 step (gate GEMV, early plan, cert head), then −1.90 ms more for the rest together (in-boot A/B, 10 rounds) | Marlin (IST-DASLab, Neural Magic, vLLM); rest ours |
| Prefill package | `GLM_PREFILL_SHARD(_PAD)`, `GLM_FLASHKDA_PREFILL`, `GLM_KDA_CONV_SPLIT`, `GLM_TRITON_MLA_PREFILL`, `GLM_DENSE_FAST_PREFILL` | cold prefill +28 % (4k) to +43 % (128k) | FlashKDA (MoonshotAI, MIT; Matt Mastracci's fp32-state branch); ideas: mmastrac (conv split, sparse MLA), Jacopo Nardiello and FujitsuPolycom (mHC sharding) |
| Routed-MoE prefill kernels (not in gate 0003) | `PF3_ARM=samemath`, `overlay/glm_pf3_*.py` | MoE sum + shared add fused (exact, self-checked before it arms), smaller down tile, gate_up + SiLU in one kernel: prefill +3-4 % | vLLM MoE Marlin template; epilogue idea: mmastrac |
| Large row gathers over NCCL (not in gate 0003) | `GATHER_ROUTE=1`, `overlay/glm_roce_gather_route.py` | the mHC prefill-shard row gathers (> 4 MiB per rank) take NCCL, −20 % per gather; exact | ours, on the RoCEnante shim |
| Corrected L2 prefetch tables (not in gate 0003) | `L2PF_V2=1`, `overlay/glm_l2pf_v2.py` | the all-reduce L2 windows warm the BF16 mHC weights the kernel actually reads instead of the unused FP32 copy; c1 −0.29 ms (n.s.), tok/s c1 +0.45 %, c4 +2.6 % (in-boot A/B); exact | ours |

**From the 2026-09-28 release**

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
W4A4 / MXFP4 experts for prefill (1.1-1.3x on MoE at 16-21 % MoE output error), a lower truncation
row cost of 1.6 ms (not measured on the fleet; 2.0 is), a 9216-row prefill chunk budget
(`BATCHED_TOKENS=9223`: prefill −9 to −34 %).

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

Every switch is a line in `profiles/current.env`; removing a line turns that piece off. The five newest pieces
are knobs you can set on the command line, e.g. `DEVSELECT=0 PF3_ARM=off ./start.sh serve` (`DEVSELECT`,
`TRUNC_COST`, `PF3_ARM`, `GATHER_ROUTE`, `L2PF_V2`; see the comments in the profile). The overlay's CUDA
extensions (dense 8-bit kernels, FlashKDA, routed-MoE prefill kernels) are JIT-built on each node on first use and
cached under the overlay directory; a first boot of a fresh tree takes longer. The `serve` step first checks that
no existing container already uses the target name or overlay path.

## Benchmarks and gates

```
bench/qeval.py              75-check quality gate (run / compare)
bench/kld_probe.py          teacher-forced top-20 logprobs from the live endpoint; compare_kld_strict.py
bench/accept_probe.py       step ms and accepted tokens per step at c1 (the stable speed number)
bench/conc_bench.py         concurrency sweep with 32 distinct prompts per type
bench/prefill_bench.py      cold prefill tok/s by prompt length
bench/final_bench.py        long-prompt KL panel, T > 0 garble scan, sparkDash prefill, RigMark decode screen
overlay/glm_ab.py, scripts/ab_inboot_glm.py   in-boot A/B of kernel switches with an A/A control
```

`tests/run_cpu_tests.sh` runs every CPU test (28 suites; `GLM_IMAGE_SRC` points the source-drift checks at an
extracted copy of the image's vLLM, or run it inside the image). The `*_gpu.py` tests need a free GPU, so run them
with the model stopped.

## Known limits

- Prefill is still behind recipes that use W4A4 experts; this stack keeps the NVFP4 / 8-bit weights.
- Device-side verify-shape selection works at one request only; batches use host truncation.
- GLM greedy output is not bit-reproducible across runs on this stack (batch-dependent kernels), so
  exactness of a change is checked per kernel, not by comparing text.
- The DFlash2 drafter is CC BY-NC-ND 4.0.

## More documentation

- [docs/install.md](docs/install.md): host prerequisites, NCCL, image build, launch and preflight rules
- [docs/weights.md](docs/weights.md): preparing the lossless8 target and the FP8 drafter
- [docs/runtime.md](docs/runtime.md), [docs/validation.md](docs/validation.md): runtime switches and the validation scope
- [docs/results/](docs/results/): raw result files per release

See [CREDITS.md](CREDITS.md) for authors and pull requests, and [NOTICE](NOTICE) with [LICENSES/](LICENSES/) for licence boundaries.
