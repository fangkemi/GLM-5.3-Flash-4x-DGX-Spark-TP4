# Deployment operations

Paths below are relative to the repository root. Use the selected live profile
explicitly through `ENV_FILE`; the launcher otherwise defaults to `.env`.
Existing user authorization and requested settings take precedence over example
values. This workflow does not authorize publishing code or changing clients.

## Inspect before changing anything

1. Identify the checkout, current profile, host/rank mapping, image, model/draft
   paths, runtime prefix, overlay, bind address and port. Capture a timestamped
   profile backup under ignored `logs/` before editing. Do not expose secrets
   if the installation has added credentials to its profile.
2. Use `ENV_FILE=PATH ./start.sh status` and `./start.sh logs RANK LINES` with
   the same ENV_FILE. For each configured rank inspect Docker State, OOMKilled,
   RestartCount, image ID, mounts, actual command and selected deployment flags;
   do not dump unrelated environment secrets. Check host MemAvailable with
   `free -b` or `/proc/meminfo`. The status summary alone is insufficient.
3. Derive the API base from the head address and port. Use direct LAN requests
   (`curl --noproxy '*'` or an HTTP client with proxies disabled); local SSH
   tunnels must also bypass workstation proxies. Inspect `/health` and
   `/v1/models`, and boot logs for applied arguments and GPU KV cache size.
   On the original fleet the normal base is port 8888 on spark1, with served
   name `glm-5.3-flash`; discover current values rather than assuming them.

## First deployment or missing artifacts

Read `docs/fp8-deployment.md` and the image build section of `README.md`.
Select the FP8 example with or without a drafter. Adapt host/IP mapping,
interface/HCA names, paths and the operator-provided NCCL hash. Leave the
NVFP4 precision/conversion recipe out of this FP8 setup.

- Confirm SSH access, Docker/GPU access and image identity on all four nodes.
- Confirm the chosen target's config/tokenizer/shards and optional drafter
  exist at the configured path on every node. Compare manifests/checksums,
  not only directory names. Check target/drafter hidden and vocabulary sizes.
  The tuned drafter is original BF16, not the separate FP8-converted draft.
- Verify direct ConnectX routes/topology before bulk synchronization. Follow
  the installation's network configuration; this skill does not guess cable
  neighbors or overwrite persistent networking. Verify the pinned NCCL library
  is present on every node. `start.sh serve` performs the artifact hash check.
- Select fresh CTN and OVERLAY_REMOTE values, then use the launch procedure
  below. Do not stop unrelated models merely because one node is occupied;
  resolve that conflict within the user's requested scope.

## Stop, restart or change parameters

1. Save the old profile and remember its old CTN and OVERLAY_REMOTE.
   Run `ENV_FILE=OLD_PROFILE ./start.sh stop` in the named tmux runner.
   The launcher calls `scripts/stop_preserving.py` on every rank; wait until
   all four old containers have stopped. For stop-only requests, verify this
   state and finish without starting a replacement.
2. If profile names were already edited, use the saved old profile. If no
   backup exists, recover the old mounts/name from Docker inspect, then invoke
   the old overlay's `scripts/stop_preserving.py --root OLD_OVERLAY
   --container OLD_PREFIX-rRANK` on that rank's host. Do not stop a different
   deployment inferred only from a similar name.
3. Apply the requested changes and fresh runtime names, preserving unrelated
   settings. Check profile shell syntax (`bash -n PROFILE`). Inspect what the
   new speculative table, capture sizes and limits mean before launching.
   A fixed K should update K_HI/K_LO and SPEC_TABLE together; capture sizes
   must cover n*(K+1) verification rows for active batch sizes n.
4. Start through an existing named tmux session's new window, or create a
   named session if absent. The original fleet uses
   `glm53-dealign-fp8-service`. Inside the pane, from the repository root:

   ```sh
   mkdir -p logs
   set -o pipefail
   ENV_FILE=PROFILE ./start.sh serve 2>&1 | tee -a logs/serve-RUN.log
   ```

   Replace PROFILE and RUN with the chosen profile and unique run label; do
   not leave literal placeholders in the command. The launcher's detached
   Docker containers remain manageable through status/logs/stop. Keep the
   tmux pane and its run log accessible. A zero launcher exit means the ranks
   were launched, not that the API or inference is ready.
5. Poll roughly every 30 seconds and inspect rank logs for progress. Loading,
   kernel compilation and graph capture can take several minutes (observed
   about 4–8 minutes depending on cache). A 60-second shared-memory broadcast
   warning alone is not a failure. On a crashed/OOM rank, preserve its logs,
   inspect the other ranks and stop partial candidate processes before a
   corrected restart. Identify a concrete cause instead of repeatedly
   launching the unchanged failing configuration. If a blocker cannot be
   resolved, report it and the remaining service state. Roll back only within
   the authorized restart/recovery scope, using preserved settings and fresh
   runtime names; do not substitute incompatible defaults to get health 200.

## Completion gates

Require all of the following for a successful deployment:

- Four expected ranks running, no OOM, expected image and model/draft mounts.
  Confirm there are no unexplained restarts or unexpected competing runtimes.
- Health 200 and expected served name/context in `/v1/models`.
- Actual Docker command agrees with requested prefill, concurrency, context,
  KV dtype/allocation and speculative config. Confirm checkpoint-specific
  precision flags, applied backends and actual KV token capacity from logs.
- A bounded inference smoke test returns readable, correct content. Use the
  configured model name, temperature zero and a small max_tokens value;
  checking a short arithmetic question is adequate for a routine restart.
  For image-limit changes use a small-image request above the old limit and
  distinguish API-limit success from vision quality.
- For tuning, run representative requests after warmup and record TTFT,
  decode rate and acceptance separately. `bench/fp8_tune.py` provides the
  current synthetic panel and foreign-request detection for served name
  `glm-5.3-flash`; adapt it if using another name. Do not run it unsolicited
  for an inspection-only request. See `docs/results/2026-10-02-fp8-tuning.md`
  for the comparison method and test-only AB configuration.

Before finishing an AB experiment, stop the test deployment and launch normal
production settings with fresh runtime names: remove GLM_AB_* and
VLLM_SERVER_DEV_MODE and test-only diagnostics, restore the intended binding,
and rerun the completion gates. Keep GLM_DRAFT_CONV_FUSED_QUAL=1 when fusion
is enabled; failed shape qualification must keep its original fallback.

State exactly what was validated. Bounded arithmetic, 13k-token retrieval or
small images do not establish full 1M-context, high-resolution image or
sustained concurrency safety. Record the final profile, container prefix,
tmux/log location, capacity and performance evidence so a later restart can
use the live state instead of a stale example.
