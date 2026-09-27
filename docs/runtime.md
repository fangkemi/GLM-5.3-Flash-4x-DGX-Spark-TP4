# Runtime provenance and publication boundary

The previously accepted deployment is LVKP-S-L2: the lossless8 target, block128-FP8 incoai
DFlash2 drafter, batch-uniform adaptive3/7, kpool fixes, exact/ULP-qualified KDA
stash changes and L2 prefetch. This release adds the qualified
GDN integer-metadata and router-dedup code to that 262k base, with bounded
checks. Completed four-rank quality and benchmark results are in
[the current results](results/2026-09-27-gdn-router-admitted.md). No
mHC/CPU-placement/gather/dense candidate from later experiments is enabled.

`profiles/current.env` flattens the previously nested accepted profile. Site-specific
paths and network settings live in `.env`; custom HCA selection is explicitly
forwarded to both NCCL and RoCEnante. Automatic page-cache prewarm, boot traffic
and memory compaction are disabled in the public launcher for operator control.
These operational defaults are distinct from the measured model configuration.
The portable profile appends `GLM_GDN_METADATA_FAST=1`,
`GLM_ROUTER_DEDUP=1`, `GLM_GDN_METADATA_CHECK_CALLS=8` and
`GLM_ROUTER_DEDUP_CHECK=1` after the original baseline flags. The checked
deployment uses the same effective four values, with site paths kept outside
the portable profile. A pre-existing local `.env` must still source
`profiles/current.env` to select the current stack.

`runtime-source-manifest.json` records historical and published source hashes.
The GDN/router files are pinned to the qualified running deployment. The
portable launcher itself has not undergone a fresh-clone four-node boot.
Active kernel, model, scheduler and loader sources are preserved except process-origin
documentation comments. The publication's sitecustomize removes
inactive diagnostic/experimental registrations while preserving baseline active statement
order and logic. It adds default-off GDN/router registrations that the current
profile enables. Removed modules include AB/dev counters, experimental mHC,
verify-cut, draft-context graph and prefill scheduling hooks. Dynamic optional
imports in retained helpers are unreachable under current.env; unsupported
experimental flags are not part of this package. This is not a claim that every
published file is byte-identical to the deployment. The public launcher stop path
was also changed to preserve containers and signal only verified auxiliary PIDs.

The RoCEnante Docker layer was recovered from the tested local source tree. Its
base image digest and vendored b12x commit are pinned, and source/license provenance
is retained in `roce/`. The shim is based on Local Inference Lab's vLLM#597 port via
tonyd2wild, with b12x/RoCEnante by Luke Alonso and Jason Cook. Retain all third-party
notices. The base itself contains the vLLM/torch/CUDA/compiler stack; this repository
provides the source-pinned derivative build, not a complete from-source rebuild
of every package inside that externally supplied base image.

Cold caches are disposable compilation artifacts, not hidden model dependencies.
L2 dynamically compiles its helper with the image's compiler; FlashInfer/Triton/
TileLang similarly populate per-node caches. The measured fleet used warmed caches.
A fresh checkout/image/weight conversion must pass transport, tensor/quality and
end-to-end measurement checks before claiming the published performance.

Publication checks performed: source/hash inventory against the accepted running
containers, flattened profile audit, Python AST/shell syntax, mocked actual launcher
name-refusal and PID-safe stopping tests. No fresh-clone four-node build/boot or
complete checkpoint conversion was executed during publication preparation.
