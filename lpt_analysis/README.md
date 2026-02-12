# OpenVINO CPU LPT Analysis Bundle

This bundle documents how Low Precision Transformations (LPT) are implemented and consumed by the Intel CPU plugin, with code-traceable references.

## Scope used

Mandatory code scope (read and cross-checked):

- `src/plugins/intel_cpu/src/graph_optimizer.cpp`
- `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp`
- `src/plugins/intel_cpu/src/transformations/**`
- `src/plugins/intel_cpu/docs/fake_quantize.md`
- `src/plugins/intel_cpu/docs/cpu_emulation.md`
- `src/plugins/intel_cpu/docs/convolution_post_ops.md`
- `src/plugins/intel_cpu/docs/compilation_options.md`
- `src/plugins/intel_cpu/docs/internal_cpu_plugin_optimization.md`

## Document map

1. [01_overview.md](./01_overview.md)
   - End-to-end LPT mechanism and where it sits in CPU plugin compilation/runtime flow.
2. [02_pipeline_order.md](./02_pipeline_order.md)
   - Ordered pass registration and execution mapping (pre-LPT, LPT, post-LPT, post-snippets, CPU opset stage).
3. [03_transformations_catalog.md](./03_transformations_catalog.md)
   - Per-transformation technical catalog for CPU-plugin-local passes in the LPT-related pipeline, with before/after graphs.
4. [04_graph_optimizer_integration.md](./04_graph_optimizer_integration.md)
   - Runtime graph-level fusions that finalize low-precision behavior after nGraph pass pipelines.
5. [05_x64_vs_arm.md](./05_x64_vs_arm.md)
   - x64 vs ARM/AArch64 differences: pass presence, order, guards, and expected graph impact.
6. [appendix_transformation_matrix.md](./appendix_transformation_matrix.md)
   - Coverage matrix and validation checklist.
7. [COVERAGE_REPORT.md](./COVERAGE_REPORT.md)
   - Short completion report of analyzed classes/files and skip check.

## Key boundary used in this analysis

- “LPT-related pipeline” is treated as:
  - `Transformations::UpToLpt()` pre-processing + `runLptPasses()` (`transformation_pipeline.cpp:435-1090`)
  - `Transformations::PostLpt()` (`transformation_pipeline.cpp:1092-1208`)
  - `Transformations::PostSnippets()` FQ decomposition controls (`transformation_pipeline.cpp:1663-1692`)
  - Runtime CPU graph fusion stage relevant to quantization/FQ (`graph_optimizer.cpp:86-240`, and specific fusion functions).
- Passes from OpenVINO common transformation libraries are included in order tables for completeness, but deep internals are documented only when implementation is inside the scoped CPU plugin sources.
