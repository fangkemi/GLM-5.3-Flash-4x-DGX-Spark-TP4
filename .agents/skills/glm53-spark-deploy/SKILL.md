---
name: glm53-spark-deploy
description: Deploy, restart, stop, inspect, or tune the FP8 GLM-5.3-Flash TP4 service on four DGX Spark nodes using this repository and a switchless ConnectX ring. Use for the FP8 target with optional BF16 DFlash2; the NVFP4 switched profile is a separate recipe.
---

# Deploy GLM-5.3-Flash FP8 on the Spark ring

Operate the checkout containing this skill. Its root is three directories above
this folder; if the skill was copied elsewhere, locate the user's checkout of
`fangkemi/GLM-5.3-Flash-4x-DGX-Spark-TP4` rather than assuming a workstation path.
Run launcher commands from that checkout.

Read [references/operations.md](references/operations.md) for deployment,
restart, stop and verification. Use the live profile and container arguments as
the source of truth. On the original fleet the profile is `.env.dealign-ring`;
on another installation identify the profile the operator actually uses. Do
not replace an existing profile with an example during a restart.

## Deployment constraints

- Original physical/TP order: spark1 → spark2 → spark4 → sparkX → spark1.
  Hosts: spark1.local, spark2.local, spark4.local, sparkx.local. spark3 is not in
  this deployment. Adapt host mappings for another fleet; verify physical
  topology if cables or hosts changed.
- Use IPv4 SSH with `-4 -o BatchMode=yes` on the original fleet. Investigate host
  identity mismatches without disabling host-key verification. Keep credential
  prompts in an interactive tmux pane; use the operator's configured credential
  store, never put secrets into profiles, scripts, arguments or logs.
- Start and restart through named tmux, normally
  `glm53-dealign-fp8-service`. Preserve accessible startup and container logs.
- Stop with the previous profile before changing `CTN` or `OVERLAY_REMOTE`.
  Every new deployment gets a fresh prefix and overlay directory. Preserve old
  containers and mounts; never remove them to bypass runtime preflight.
- Verify ConnectX routes before transferring bulk weights or Docker images.
  The launcher uses management SSH/rsync for small overlay files. Matching the
  pinned switchless NCCL hash verifies artifact identity, not network function.
- For the tested dealignai FP8 checkpoint retain `GLM5NEXT_PATCH=0`,
  `KV_DTYPE=auto`, `SCHEDULER_CLS=none`, `GLM_MAMBA_ALIGN_FIX=1`, Triton MoE
  and `--linear-backend triton`. Enabling the NVFP4 attention patch previously
  produced garbled output despite healthy endpoints. Do not import
  `profiles/current.env` precision settings into this FP8 deployment.

## Recipe and measurements

For a new FP8+DFlash deployment, start from the repository's
[FP8+DFlash example](../../../.env.fp8.dflash.switchless.example).
The 2026-10-02 single-request recipe uses prefill 8192, fixed K=3,
MAX_SEQS=3, the original BF16 incoai drafter, qualified convolution fusion and
router dedup. Preserve subsequent user settings. For a deployment without a
drafter use [the non-speculative example](../../../.env.fp8.switchless.example),
`SPEC_DISABLE=1` and `SCHEDULER_CLS=none`.

`KV_BYTES` allocates per rank; the resulting token pool is shared across
sequences. The tuned 17 GiB/rank recipe reported 1,469,300 KV tokens with a
1,048,576 per-sequence context limit. Read actual boot capacity after every
change. Check host MemAvailable and OOM state: Docker and nvidia-smi alone do
not account reliably for DGX Spark unified memory.

When tuning, read [the measured comparison](../../../docs/results/2026-10-02-fp8-tuning.md).
K=2/4 were slower overall on the tested prompts. Fusion plus router dedup
lowered step time about 1.1–1.3%; overall throughput and first-token improvement
were not established. Test MAX_SEQS=4 when the workload needs concurrency.
Keep fusion qualification enabled. Test-only AB/dev-mode flags must be removed
before restoring the production endpoint.

Deploy/restart requests authorize the necessary coordinated restart. Inspection
and documentation requests do not authorize a service interruption. Report the
applied configuration, four-rank state, endpoint, actual KV capacity, bounded
smoke/quality results and any unresolved issue; do not equate health 200 with
correct model output or a full-context memory validation.
