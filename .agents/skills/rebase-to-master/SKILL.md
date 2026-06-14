---
name: rebase-to-master
description: Rebase workflow onto origin/master and resolve merge conflicts
---

# there are several remotes:

`origin` is real llama.cpp
`turboquant` is a fork which added TurboQuant support for KV cache
`my` is a fork of `turboquant`, with the only `turboquant` release squashed and rebased to `origin/master` with necessary fixes.
It has:
* main branch `tqp-v0.1.1-rebased-to-master`, which is periodically rebased to `origin/master`
* each succesful rebase is tagged with `tqp-v0.1.1-rebased-to-<origin_tag>` or `tqp-v0.1.1-rebased-to-<nearest_origin_tag>+<origin_commit_hash>` if the master HEAD is not tagged at the time of the rebase

## Procedure
1. Run `git fetch origin`.
2. Run `git rebase origin/master`.
3. If there are conflicts, DO NOT ignore them. Run `git diff --name-only --diff-filter=U` to list the conflicted files.
4. For each conflicted file:
    * Read to understand the context and the conflicting changes.
    * Check the upstream diff between the old base and the new base to understand what changed in the master branch.
    * Locate conflict markers using regex `^<<<<<<< |^=======|^>>>>>>>`.
    * Synthesize the changes carefully, preserving independent features from both branches. Keep an eye on specific index values or overlapping properties.
5. Resolve the conflicts:
    * Always use file editing tools
    * Verify all conflict markers are removed
6. Run `git add <resolved_files>` then `git rebase --continue`. Repeat until complete.
7. Build the project (e.g., `..\build.cmd`) to verify the integrated code compiles correctly.

## Compatibility checks for rebased ggml changes

When carrying a fork-specific ggml change across a rebase:

1. Compare the old commit's exact diff with the rebased tree. If upstream changes the operation count, this rebase will have to update it too.
2. Recount `GGML_OP_*` entries in `ggml/include/ggml.h` and update all matching compile-time assertions in `ggml/src/ggml.c` and `ggml/include/ggml-rpc.h` when the enum count changed.
3. Inspect upstream additions separately. An upstream operation inserted into the enum can increase `GGML_OP_COUNT`, making it equal to the pre-rebase fork count; after the rebase, the fork needs to increase it to to include the operations introduced in it.
4. Treat `RPC_PROTO_PATCH_VERSION` as fork protocol history, implicitly incompatible with upstream. Keep the same value as upstream to avoid rebase/merge conflicts on the following rebase. Even though the fork adds operations, DO NOT change the RPC version from upstream: even if you do, the upstream will eventually catch up, still making the numerically identical versions incompatible - and they will not be adding the same operations: the upstream maintainers denied merging this fork or adding the fork's functionality.

## CUDA mul_mat compatibility

The TurboQuant CUDA changes were originally written against an older, larger `ggml_cuda_mul_mat()` dispatcher. Upstream later simplified this dispatcher. During a rebase, preserve the current upstream implementation and translate the TurboQuant intent into the smallest possible additions. Do not restore the removed selector variables, split-buffer loop, batched-cuBLAS conditions, debug helpers, or `ggml_cuda_op_mul_mat()` dispatch chain from the old fork diff.

Check these integration points in `ggml/src/ggml-cuda/ggml-cuda.cu`:

1. In `ggml_cuda_should_fuse_mul_mat_vec_q()`, exclude `GGML_TYPE_TQ4_1S` and `GGML_TYPE_TQ3_1S` from the generic MMVQ fusion path.
2. In `ggml_cuda_mul_mat()`, keep all upstream guards and dispatch decisions unchanged. Add one TQ-only branch after the upstream MMVF fast path and before the generic MMF/MMVQ/MMQ checks:
    * For either TQ weight type with `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`, call `ggml_cuda_mul_mat_tq()`.
    * For larger `GGML_TYPE_TQ4_1S` batches, call `ggml_cuda_mul_mat_tq4_1s_cublas()`.
    * For larger `GGML_TYPE_TQ3_1S` batches, call `ggml_cuda_mul_mat_cublas()`.
    * Return after handling the TQ type so it cannot reach the generic MMVQ/MMQ helpers.
3. In `ggml_cuda_mul_mat_id()`, keep TQ weight types out of the generic MMVQ path. They must continue to the existing synchronized cuBLAS fallback rather than use kernels that do not support TQ blocks.
4. In `ggml_cuda_graph_check_compability()`, keep `GGML_OP_MUL_MAT_ID` with TQ weights ineligible for CUDA graphs because its fallback synchronizes the stream.

After resolving this area, compare the complete file against `origin/master`. The `ggml_cuda_mul_mat()` hunk should contain only the compact TQ branch described above; a large deletion or replacement of upstream dispatch code indicates an over-resolved conflict. Build with CUDA enabled to verify that all referenced TQ kernels compile.
