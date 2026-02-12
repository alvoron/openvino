# Coverage Report

## Scope confirmation

Coverage basis:

- `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp`
- `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp`
- `src/plugins/intel_cpu/src/graph_optimizer.cpp`
- CPU-local transformation sources under `src/plugins/intel_cpu/src/transformations/**` that are referenced by the pass registration points above.

## Deep-analyzed CPU-local transformations (LPT-critical path)

1. `SDPASubgraphFusion`
2. `InsertConvertAfterExtension`
3. `SwapConvertTranspose`
4. `ConvertReduceNoKeepDims`
5. `ConvertReduceMultiAxis`
6. `MishDecomposition`
7. `ConvertConv1D`
8. `ConvertGroupConv1D`
9. `ConvertGroupConvolution`
10. `GridSampleDecomposition`
11. `Deconv1DDecomposition`
12. `DecomposeIntegerDivide`
13. `PermuteSliceAndInterpolation`
14. `ConvertToInteraction`
15. `ConvertInteractionInt8`
16. `ConvertConvolutionBias`
17. `FallbackUnsupportedLPConvToFP16`
18. `FuseFQtoInteraction`
19. `ConvertFqRnnToQuantizedRnn`
20. `CausalMaskPreprocessFusion`
21. `MLPFusion` / `MLPFusionPass`
22. `QKVProjFusion` / `QKVProjFusionPass1/2`
23. `DecomposeRMSNorm`
24. `NgramFusion`
25. ARM LPT helper matchers in `src/plugins/intel_cpu/src/transformations/utils.cpp`

## Enumerated (full pipeline matrix, not deep-expanded)

CPU-specific-opset and snippets-local passes were fully enumerated in `lpt_analysis/appendix_transformation_matrix.md` section B/D and included in ordering analysis (`lpt_analysis/02_pipeline_order.md`).

## Skip check

- No CPU-local transformation class referenced by pass registration in the scoped pipeline files was skipped.
- External OpenVINO pass classes (outside `src/plugins/intel_cpu/src/transformations/**`) are explicitly listed in the matrix and marked external; their internals are outside this bundle’s deep-analysis boundary.
