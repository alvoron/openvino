# 02. Pipeline Order

## 1. Architecture gating macros used by pipeline

Architecture-specific registration is compile-time expanded via macros in `src/plugins/intel_cpu/src/transformations/defs.hpp:7-78`:

- `CPU_REGISTER_PASS_X64`: active only on `OPENVINO_ARCH_X86_64`
- `CPU_REGISTER_PASS_X86`: active only on `OPENVINO_ARCH_X86`
- `CPU_REGISTER_PASS_ARM`: active on `OPENVINO_ARCH_ARM` and `OPENVINO_ARCH_ARM64`
- `CPU_REGISTER_PASS_ARM64`: active only on `OPENVINO_ARCH_ARM64`
- `CPU_REGISTER_PASS_ARM32`: active only on `OPENVINO_ARCH_ARM`

The same applies for `CPU_DISABLE_PASS_*`, `CPU_ENABLE_PASS_*`, `CPU_SET_CALLBACK_*`.

## 2. Transformations stage order

Stage entry points (`src/plugins/intel_cpu/src/plugin.cpp:365-377`):

1. `UpToLpt()`
2. `PostLpt()`
3. `Snippets()`
4. `CpuSpecificOpSet()`

`UpToLpt()` internals (`src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:435-453`):

1. `PreLpt(defaultPrecisions)`
2. If enabled, `Lpt(defaultPrecisions)`

## 3. PreLpt order

### 3.1 Decompression handling manager (runs first)

Source: `transformation_pipeline.cpp:464-510`.

Order:

1. `InitNodeInfo` (`469`)
2. `CompressedGatherTransformation` (`471`)
3. `MarkShapeOfSubgraphs` (`472`)
4. `TransposeMatMul` x64 (`474`) / ARM (`475`)
5. `MarkDequantization` (`478-482`)

Callbacks:

- `MarkDequantization` skips non-decompression multiply (`483-488`).
- `ConvertGatherToGatherCompressed` callback preserves const precision and keeps quantized gather paths for LPT handling (`490-508`).

### 3.2 Main pre-LPT manager

Source: `transformation_pipeline.cpp:511-917`.

High-level order blocks:

1. Optional `MarkDequantization` if `useLpt` (`513-515`)
2. Optional fp32->fp16 `ConvertPrecision` when inference precision is fp16 (`543-557`)
3. `KeepConstAndDecompression` + callback (`558-572`)
4. General graph canonicalization and decompositions (`573-644`)
5. If LPT enabled: `low_precision::ConvertSubtractConstant` (`646-649`)
6. `InsertConvertAfterExtension` then `ConvertPrecision` (must stay adjacent) (`655-665`)
7. Cleanup and plugin-local canonicalization passes (`667-687`)
8. Callback setup for conversion/decomposition passes (`689-843`)
9. Explicit pass disable/enable configuration (`845-883`)
10. If LPT enabled: callback tuning for fake-quantize-related common passes (`885-905`)
11. Final decompression re-mark + CF + LoRA fusion (`907-914`)

Plugin-local passes in this stage (registration lines):

- `InsertConvertAfterExtension` (`656`)
- `SwapConvertTranspose` (`669`)
- `ConvertToInteraction` x64 (`670`)
- `ConvertInteractionInt8` x64 (`671`)
- `ConvertReduceNoKeepDims` ARM (`672`)
- `ConvertReduceMultiAxis` ARM (`673`)
- `MishDecomposition` ARM32 (`674`)
- `ConvertConv1D` ARM (`675`)
- `ConvertGroupConv1D` ARM (`676`)
- `ConvertGroupConvolution` ARM (`677`)
- `GridSampleDecomposition` ARM (`679`)
- `Deconv1DDecomposition` ARM (`680`)
- `DecomposeIntegerDivide` ARM (`683`) and x86 (`684`)
- `PermuteSliceAndInterpolation` (`686`)
- `SDPASubgraphFusion` (`574`)

## 4. LPT manager order (`runLptPasses`)

Source: `transformation_pipeline.cpp:919-1081`.

1. Build precision and quantization restrictions:
   - ARM path (`923-930`)
   - non-ARM (x86/x64) path (`932-967`)
2. Register `LowPrecision` (`968-973`)
3. ARM-only extra passes:
   - `ConvertConvolutionBias` (`974`)
   - `FallbackUnsupportedLPConvToFP16` (`975`)
4. Callback tuning and selective disable:
   - ARM callbacks for `ConvolutionTransformation`, `FuseSubtractToFakeQuantizeTransformation`, `FuseMultiplyToFakeQuantizeTransformation`, `MatMulTransformation`, `FakeQuantizeTransformation` (`976-1041`)
   - x64 callbacks for `ConvolutionBackpropDataTransformation`, `FoldConvertTransformation`, `FuseConvertTransformation` (`1043-1078`)
   - disable `MultiplyToGroupConvolutionTransformation` globally (`1000`)
   - disable multiple ARM LPT transformations (`1002-1012`)

## 5. PostLpt order

Source: `transformation_pipeline.cpp:1092-1208`.

Order (selected):

1. `ConvertBroadcast3`, `UnrollTensorIterator`, `ReshapePRelu` (`1097-1099`)
2. `MoveEltwiseUpThroughDataMov` (`1107`)
3. `Validate`, `ConstantFolding` (`1124`, `1126`)
4. x64 low-precision/fusion extras:
   - `FuseFQtoInteraction` (`1128`)
   - `ConvertFqRnnToQuantizedRnn` (`1131`)
5. `RoPEFusion` x64/ARM64 (`1133-1134`)
6. x64 `CausalMaskPreprocessFusion` (`1136`)
7. x64 conditional `MLPFusion`/`QKVProjFusion` under AMX and precision checks (`1138-1183`)
8. x64 `RMSFusion`, `DecomposeRMSNorm` (`1186-1194`)
9. Optional mixed-precision rope markup (`1196-1200`)
10. Symbolic optimizations + `NgramFusion` (`1202-1205`)

## 6. Snippets and post-snippets parts relevant to LPT

Source: `transformation_pipeline.cpp:1210-1708`.

- On ARM/ARM64, snippets are skipped for quantized models (`1212-1222`).
- Post-snippets registers `FakeQuantizeDecomposition` (`1666`) with:
  - x64 callback: keep FQ if CPU FakeQuantize supports it (`1667-1673`).
  - ARM callback: keep only conv-compatible low-precision FQ patterns (`1676-1688`).

## 7. CPU specific opset stage order

Source: `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:39-94`.

Order (registration):

1. `MoEMatMulsFusion` x64 (`45`)
2. `ConvertBatchGatherMatmulToBatchGatherMatmulCompressed` x64 (`47-57`)
3. `ConvertMatMulToFC` (`59`)
4. `FullyConnectedBiasFusion` (`60`)
5. `ConvertFullyConnectedToFullyConnectedCompressed` (`62-69`)
6. `ConvertFCToFCQuantizedLegacy` x64 (`71`)
7. `MoveFCReshapeToWeights` (`72`)
8. `AlignMatMulInputRanks` (`74`)
9. `ConvertTileToSeqTiles` (`75`)
10. `ConvertToPowerStatic` (`76`)
11. `ConvertToLeakyRelu` (`77`)
12. `ConvertToSwishCPU` (`78`)
13. `OptimizeSequenceTransposes` (`79`)
14. `MoveReadValueInputsToSubgraph` (`92`)

This stage is post-LPT, but still relevant for quantized graph canonicalization into CPU-specific operators.

## 8. Runtime GraphOptimizer order (low-precision relevant subset)

Entry and sequence: `src/plugins/intel_cpu/src/graph_optimizer.cpp:86-240`.

Low-precision-critical order:

1. `FuseConvolutionAndZeroPoints` (first by design) (`95`)
2. `FuseConvMatmulFCDeconvAndDQScales` and `FuseConvolutionMatMulDeconvAndBias` with ARM/x86 order swap (`98-117`)
3. `FuseClampAndFakeQuantize` (`138-140`)
4. `FusePerformedAsScaleShiftAndFakeQuantize` (`142-144`)
5. `FusePoolingAndFakeQuantize` (`158-160`)

These passes transform the internal CPU graph to executor-facing fused quantization attributes.
