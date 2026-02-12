# 05. x64 vs ARM Comparison

## 1. LPT manager differences (`runLptPasses`)

Source: `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:919-1081`.

| Area | x64 / x86 behavior | ARM / ARM64 behavior | Code refs |
|---|---|---|---|
| Precision restrictions for `LowPrecision` | `Convolution` input0 allows `u8` or `i8` only on AMX/AVX2_VNNI_2; otherwise `u8` only | Fixed conv/matmul support list with activation `u8/i8` and i8 weights | `932-939`, `945-963` vs `923-930` |
| Extra local passes | None in LPT manager | `ConvertConvolutionBias`, `FallbackUnsupportedLPConvToFP16` | `974-975` |
| `ConvolutionTransformation` callback | x64 callback not used here | ARM callback limits by stride/kernel/OC-IC threshold | `976-982` |
| FQ fusion callbacks | x64 uses generic LPT behavior + x64-specific fold/fuse convert callbacks | ARM adds callbacks for `FuseSubtractToFakeQuantizeTransformation`, `FuseMultiplyToFakeQuantizeTransformation`, and disables specific `FakeQuantizeTransformation` cases | `988-999`, `1026-1041` |
| Disabled LPT subpasses | `MultiplyToGroupConvolutionTransformation` disabled globally | Additional disable list for pool/interpolate/groupconv/MVN/normalize/recurrent/reduces | `1000`, `1002-1012` |
| MatMul low-precision behavior | x64 standard LPT matmul transform path | ARM callback restricts INT8 MatMul cases (ACL lowp limitation) | `1014-1024` |
| x64-only callbacks | Asymmetric quantization guard for deconv, fold/fuse convert on decompression path | N/A | `1043-1078` |

## 2. Pre-LPT canonicalization differences

Source: `transformation_pipeline.cpp:667-687`.

| Pass | x64/x86 | ARM/ARM64 |
|---|---|---|
| `ConvertToInteraction`, `ConvertInteractionInt8` | Present | Absent |
| `ConvertReduceNoKeepDims`, `ConvertReduceMultiAxis`, `ConvertConv1D`, `ConvertGroupConv1D`, `ConvertGroupConvolution`, `GridSampleDecomposition`, `Deconv1DDecomposition` | Absent | Present |
| `MishDecomposition` | Absent | ARM32-only |
| `DecomposeIntegerDivide` | x86-family enabled | ARM enabled |
| `PermuteSliceAndInterpolation`, `SwapConvertTranspose`, `InsertConvertAfterExtension` | Present | Present |

## 3. Post-LPT differences

Source: `transformation_pipeline.cpp:1128-1207`.

| Pass | x64 | ARM64 | ARM32 |
|---|---|---|---|
| `FuseFQtoInteraction` | Yes | No | No |
| `ConvertFqRnnToQuantizedRnn` | Yes | No | No |
| `RoPEFusion` | Yes | Yes | No (macro) |
| `CausalMaskPreprocessFusion` | Yes | No | No |
| `MLPFusion`, `QKVProjFusion` | Conditional on AMX + inference precision | No | No |
| `DecomposeRMSNorm` | Yes | No | No |
| `NgramFusion` | Yes (common registration) | Yes (common registration) | Yes |

## 4. Runtime capability gates (x64-specific)

| Gate | Effect | Refs |
|---|---|---|
| `avx512_core_amx` or `avx2_vnni_2` | Allows signed (`i8`) conv input0 in LPT precision restrictions | `transformation_pipeline.cpp:934-939` |
| `avx512_core_amx` + bf16 or `avx512_core_amx_fp16` + f16 | Enables throughput-focused `MLPFusion` + `QKVProjFusion` | `1140-1147`, `1161` |

## 5. Post-snippets FQ decomposition differences

Source: `transformation_pipeline.cpp:1666-1688`.

- x64:
  - Keep FQ when native CPU FQ supports the node (`1667-1673`, uses `node::FakeQuantize::isSupportedOperation`).
- ARM:
  - Keep only conv-fusible patterns with same activation/output low-precision types (`1676-1688` + `transformations/utils.cpp:60-95`).
  - All other FQ nodes are decomposed to generic arithmetic.

## 6. GraphOptimizer differences that affect low precision

Source: `src/plugins/intel_cpu/src/graph_optimizer.cpp`.

| Aspect | x64/x86 | ARM/ARM64 | Refs |
|---|---|---|---|
| `FuseConvolutionAndZeroPoints` | Enabled | Disabled | `938-943` |
| Bias vs DQ-scale fusion order | DQ scales first, then bias | Bias first, then DQ scales | `98-117` |
| DQ-scale parent-edge condition | Requires parent edges size `==2` | Allows `2` or `3` (bias may already be fused) | `279-284` |

## 7. Expected graph/execution impact

### x64/x86

- More aggressive direct quantized kernels and interaction/LLM fusions.
- Higher chance that FQ remains fused in custom nodes (`InteractionNode`, quantized RNN path).
- oneDNN-style zero-point and DQ-scale folding is deeper.

### ARM/AArch64

- More decomposition/canonicalization to ACL-friendly primitives.
- Additional bias/type-fixup for low-precision conv (`ConvertConvolutionBias`).
- Explicit fallback handling for unsupported lowp conv (`FallbackUnsupportedLPConvToFP16`).
- More frequent FQ decomposition in post-snippets except selected conv-compatible patterns.
