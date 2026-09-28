"""LVKP-S-L2 startup hooks. Diagnostic and unqualified registrations omitted.
Active registration statements preserve the accepted deployment's ordering.
"""
import importlib.abc
import importlib.util
import os
import sys

_orig = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_orig):
    try:
        spec = importlib.util.spec_from_file_location("_distro_sitecustomize", _orig)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        pass



_POST = {}  # module name -> list of callables(module)




if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_LV_CGSTAT", "GLM_LV_SPLIT_FIX", "GLM_LV_ARGMAX_MINTOK", "GLM_LV_DRAFT_FP8_KV")):
    try:
        import glm_levers as _glv
        _glv.register_early()
        for _name, _fn in _glv.HOOKS.items():
            _POST.setdefault(_name, []).append(_fn)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-levers: import failed: %r\n" % (exc,))

if _POST:

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _POST:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            exec_module = loader.exec_module
            fns = _POST.pop(name)

            def patched_exec(module):
                exec_module(module)
                for fn in fns:
                    try:
                        fn(module)
                    except Exception as exc:  # noqa: BLE001
                        sys.stderr.write("overlay hook for %s failed: %r\n" % (name, exc))
                        raise

            loader.exec_module = patched_exec
            return spec

    sys.meta_path.insert(0, _Hook())


if any(os.environ.get(k) == "1" for k in ("QMIX_FP8_BLOCK", "QMIX_DRAFT_HEAD_FP8", "QMIX_DEBUG_LAYERS")):
    import qmix_patch
    qmix_patch.register()

if os.environ.get("GLM_FAST_LOAD", "0").strip().lower() in ("1", "on", "true"):
    import glm_fast_load
    glm_fast_load.register()
if any(k.startswith("GLM_DS_") and os.environ[k].strip() not in ("", "0", "off") for k in os.environ):
    import glm_ds_hooks
    glm_ds_hooks.register()

if "1" in (os.environ.get("GLM_KDA_STASH"), os.environ.get("GLM_ROUTER_FP32OUT")):
    import glm_kda_stash
    glm_kda_stash.register()
if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_TARGET_VOCAB_ARGMAX", "GLM_KDA_NOCOPY")):
    import glm_exact_hooks
    glm_exact_hooks.register()
if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_KDA_STASH_NOCOPY", "GLM_KDA_FLAG_FUSED")):
    import glm_kda_stash_fast
    glm_kda_stash_fast.register()
if os.environ.get("GLM_L2_PREFETCH", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_l2_prefetch
    glm_l2_prefetch.register()

# Default-off candidate registration; source identical to the scored diagnostic.
# Router dedup: JaredforReal, vLLM #55736. Integer GDN producer: vLLM metadata builder.
if os.environ.get("GLM_ROUTER_DEDUP", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_router_dedup
    glm_router_dedup.register()
if os.environ.get("GLM_GDN_METADATA_FAST", "0") == "1":
    import glm_gdn_hook
    glm_gdn_hook.register()

# Default-off candidate: exact sync-free kv_lens. Cheap flag pre-check so
# the disabled default does not import torch at interpreter startup.
if (
    os.environ.get("GLM_KVLENS_EXACT", "").strip().lower()
        not in ("", "0", "off", "false", "no")
    or os.path.exists("/overlay/overlay/glm_kvlens_exact.enabled")
):
    try:
        import glm_kv_lens_exact as _gkv
        _gkv.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-kvlens-exact: import failed: %r\n" % (exc,))

# Default-off candidate: forced fused_marlin_moe M-tile (decode MoE sweep,
# glm-window-20260927 night loop). Env-only arm; no marker files.
if os.environ.get("GLM_MARLIN_MOE_BLOCK_M", "0").strip() not in ("", "0", "off", "false", "no"):
    try:
        import glm_marlin_m32
        glm_marlin_m32.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-marlin-m32: import failed: %r\n" % (exc,))

# Default-off candidate: DFlash context-KV precompute replayed from
# exact-shape CUDA graphs (glm-window-20260927 night loop, t<->d glue).
if os.environ.get("GLM_DFLASH_CTX_GRAPH", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    try:
        import glm_dflash_ctx_graph
        glm_dflash_ctx_graph.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-dflash-ctx-graph: import failed: %r\n" % (exc,))
#   GLM_MHC_FUSED=1          bf16-load mHC projection kernels, bit-exact
#                            (overlay/mhc_fused.py + glm_mhc_hook.py)
if os.environ.get("GLM_MHC_FUSED", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_mhc_hook
    glm_mhc_hook.register()

# --- GLM prefill adapters (overlay/glm_prefill_hooks.py; diagnostics/glm-prefill) --------------------
# Inert unless GLM_PREFILL_SHARD / GLM_PREFILL_CADENCE / GLM_PREFILL_SHORT_TOKENS /
# GLM_PREFILL_CADENCE_WHEN_QUEUED / GLM_END_DRAIN / GLM_IDLE_COALESCE_MS is set to a non-zero value.
if any(k.startswith(("GLM_PREFILL_", "GLM_END_DRAIN", "GLM_IDLE_COALESCE")) for k in os.environ):
    import glm_prefill_hooks
    glm_prefill_hooks.register()
