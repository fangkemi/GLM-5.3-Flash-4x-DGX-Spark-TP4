# Runtime provenance and publication boundary

## Current release (2026-09-28)

The serving profile is `profiles/current.env`: the LVKP-S-L2 base (lossless8 target, block-128 FP8
incoai DFlash2 drafter, batch-uniform adaptive 3/7, kpool fixes, KDA stash with the 2026-09-27
[boundary repair](results/2026-09-27-kda-boundary-fix.md), L2 prefetch), the GDN metadata fast path
and router dedup, plus three additions:

- `GLM_ARGMAX_CLAMP=1`: vLLM #50843's padded-vocab clamp in the Gumbel and rejection samplers
  (`overlay/gumbel.py`, `overlay/rejection_sampler_utils.py`, mounted over the image files) and in
  `overlay/glm_target_argmax.py`. Bit-exact for every valid token id.
- The prefill scheduler: `GLM_PREFILL_CADENCE=8 GLM_PREFILL_SHORT_TOKENS=2048
  GLM_PREFILL_CADENCE_WHEN_QUEUED=1 GLM_END_DRAIN=1 GLM_IDLE_COALESCE_MS=4`
  (`overlay/glm_prefill_sched.py`, `glm_prefill_hooks.py`). Only rank 0 runs the engine core and
  scheduler in this TP4-over-4-nodes layout, so its ready lines appear in the rank-0 log only.
- The draft-length policy ships as `profiles/levers_policy.json` and is read from
  `/overlay/profiles/levers_policy.json`. Without the file the scheduler silently falls back to
  the launch table.

`overlay/sitecustomize.py` is the file the fleet serves. It also carries default-off registrations
for experiments that were measured and not adopted (in-boot A/B harness, verify cut, draft-context
graph, fused mHC, Marlin M=32, KV-lens exactness check, prefill sharding). Each is inert unless its
environment switch is set; `profiles/current.env` sets none of them.

## How this release was checked

- **Launch identity:** a `DRY=1` launch of this checkout was compared with `docker inspect` of the
  containers serving before the release. Arguments and bind mounts were identical; the environment
  differed only by the scheduler switches and the policy path above.
- **Fresh-clone boot:** the image was built from `Dockerfile.roce` in a fresh clone on all four
  nodes, and `start.sh serve` from that clone booted in 441 s with cold FlashInfer / Triton /
  TileLang caches. The rank-0 log showed every enabled piece installing (router dedup, L2 windows,
  RoCEnante, prefill scheduler, policy file).
- **Measurements on that boot:** sparkDash, `bench/accept_probe.py` step time, `bench/qeval.py`
  (75/75), `bench/kld_probe.py` + `bench/compare_kld_strict.py` (0.0293 over 6618 positions, same
  panel and reference as the earlier releases). Numbers are in the README.
- **Tests:** the CPU tests pass inside the image (`tests/test_glm_ab.py`, `test_glm_fast_load.py`,
  `test_glm_prefill_hooks.py` drift tables against the image's vLLM, `test_glm_prefill_shard.py`,
  `bench/test_*.py`). `tests/test_glm_prefill_sched.py` needs a vLLM source checkout
  (`GLM_VLLM_SRC`); the `*_gpu.py` tests need a free GPU and were run when their pieces were admitted.

## Source boundaries

The RoCEnante Docker layer pins its base image digest and the vendored b12x commit; source and
licence provenance is in `roce/`. The shim is based on Local Inference Lab's vLLM #597 via
tonyd2wild's port, with b12x/RoCEnante by Luke Alonso and Jason Cook. The base image contains the
vLLM / torch / CUDA / compiler stack; this repository provides the source-pinned derivative build,
not a from-source rebuild of every package in it.

`runtime-source-manifest.json` records the hashes of the 2026-09-27 publication; the files changed
since are listed in this release's commit. The launcher's `stop` preserves containers and signals
only verified auxiliary PIDs.

Cold caches are disposable compilation artifacts, not hidden model dependencies. A fresh checkout,
image and weight conversion must still pass transport, quality and end-to-end measurement checks
on your hardware before claiming the published performance.
