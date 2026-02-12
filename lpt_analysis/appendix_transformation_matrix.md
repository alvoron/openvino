# Appendix. Transformation Matrix and Coverage Validation

## A. How this matrix was built

The matrix below is extracted from pass registration macros in:

- `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp`
- `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp`

Included macro types:

- `CPU_REGISTER_PASS_*`
- `CPU_DISABLE_PASS_*`
- `CPU_ENABLE_PASS_*`

Important notes:

- Some passes are defined outside `src/plugins/intel_cpu/src/transformations/**` (OpenVINO common/LPT/snippets libraries). Those are marked as external.
- `NgramFusion` is additionally registered manually via `symbolic_pipeline->get_manager()->register_pass<NgramFusion>();` in `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1205`; it is listed in section C.
- `InsertConvertAfterExtension` is CPU-local (`src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/insert_convert_after_extension.hpp:14`) even though some auto-mapping scripts classify it as external because it is declared in `namespace ov::pass`.

## B. Full pass registration matrix (ordered by source line)

| Phase | Line | Macro | Pass | Local source in `transformations/**` |
|---|---:|---|---|---|
| `cpu_specific_opset` | 45 | `CPU_REGISTER_PASS_X64` | `MoEMatMulsFusion` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/moe_matmuls_fusion.hpp` |
| `cpu_specific_opset` | 46 | `CPU_REGISTER_PASS_X64` | `ov::pass::Validate` | External / not CPU-local class |
| `cpu_specific_opset` | 47 | `CPU_REGISTER_PASS_X64` | `ConvertBatchGatherMatmulToBatchGatherMatmulCompressed` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_batch_gather_matmul_to_compressed.hpp` |
| `cpu_specific_opset` | 59 | `CPU_REGISTER_PASS_COMMON` | `ConvertMatMulToFC` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_matmul_to_fc.hpp` |
| `cpu_specific_opset` | 60 | `CPU_REGISTER_PASS_COMMON` | `FullyConnectedBiasFusion` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/fc_bias_fusion.hpp` |
| `cpu_specific_opset` | 62 | `CPU_REGISTER_PASS_COMMON` | `pass::ConvertFullyConnectedToFullyConnectedCompressed` | External / not CPU-local class |
| `cpu_specific_opset` | 71 | `CPU_REGISTER_PASS_X64` | `pass::ConvertFCToFCQuantizedLegacy` | External / not CPU-local class |
| `cpu_specific_opset` | 72 | `CPU_REGISTER_PASS_COMMON` | `MoveFCReshapeToWeights` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/move_fc_reshape_to_weights.hpp` |
| `cpu_specific_opset` | 73 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::Validate` | External / not CPU-local class |
| `cpu_specific_opset` | 74 | `CPU_REGISTER_PASS_COMMON` | `AlignMatMulInputRanks` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/align_matmul_input_ranks.hpp` |
| `cpu_specific_opset` | 75 | `CPU_REGISTER_PASS_COMMON` | `ConvertTileToSeqTiles` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_tile_to_seq_tiles.hpp` |
| `cpu_specific_opset` | 76 | `CPU_REGISTER_PASS_COMMON` | `ConvertToPowerStatic` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_to_power_static.hpp` |
| `cpu_specific_opset` | 77 | `CPU_REGISTER_PASS_COMMON` | `ConvertToLeakyRelu` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_to_leaky_relu.hpp` |
| `cpu_specific_opset` | 78 | `CPU_REGISTER_PASS_COMMON` | `ConvertToSwishCPU` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_to_swish_cpu.hpp` |
| `cpu_specific_opset` | 79 | `CPU_REGISTER_PASS_COMMON` | `OptimizeSequenceTransposes` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/rnn_sequences_optimization.hpp` |
| `cpu_specific_opset` | 82 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ReshapeSequenceFusion` | External / not CPU-local class |
| `cpu_specific_opset` | 83 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConstantFolding` | External / not CPU-local class |
| `cpu_specific_opset` | 84 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertPrecision` | External / not CPU-local class |
| `cpu_specific_opset` | 90 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::Validate` | External / not CPU-local class |
| `cpu_specific_opset` | 91 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::EliminateConvert` | External / not CPU-local class |
| `cpu_specific_opset` | 92 | `CPU_REGISTER_PASS_COMMON` | `MoveReadValueInputsToSubgraph` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/move_readvalue_inputs_to_subgraph.hpp` |
| `transformation_pipeline` | 469 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::InitNodeInfo` | External / not CPU-local class |
| `transformation_pipeline` | 471 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::CompressedGatherTransformation` | External / not CPU-local class |
| `transformation_pipeline` | 472 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::MarkShapeOfSubgraphs` | External / not CPU-local class |
| `transformation_pipeline` | 474 | `CPU_REGISTER_PASS_X64` | `ov::pass::TransposeMatMul` | External / not CPU-local class |
| `transformation_pipeline` | 475 | `CPU_REGISTER_PASS_ARM` | `ov::pass::TransposeMatMul` | External / not CPU-local class |
| `transformation_pipeline` | 478 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::MarkDequantization` | External / not CPU-local class |
| `transformation_pipeline` | 514 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::MarkDequantization` | External / not CPU-local class |
| `transformation_pipeline` | 551 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertPrecision` | External / not CPU-local class |
| `transformation_pipeline` | 558 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::KeepConstAndDecompression` | External / not CPU-local class |
| `transformation_pipeline` | 573 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::AUGRUCellFusion` | External / not CPU-local class |
| `transformation_pipeline` | 574 | `CPU_REGISTER_PASS_COMMON` | `SDPASubgraphFusion` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/stateful_sdpa_fusion.hpp` |
| `transformation_pipeline` | 615 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertPagedAttnInputs` | External / not CPU-local class |
| `transformation_pipeline` | 616 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::CommonOptimizations` | External / not CPU-local class |
| `transformation_pipeline` | 617 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::KeepConstPrecision` | External / not CPU-local class |
| `transformation_pipeline` | 624 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::WrapInterpolateIntoTransposes` | External / not CPU-local class |
| `transformation_pipeline` | 625 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::TransposeSinking` | External / not CPU-local class |
| `transformation_pipeline` | 626 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertSequenceToTensorIterator` | External / not CPU-local class |
| `transformation_pipeline` | 627 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertOpSet2ToOpSet1` | External / not CPU-local class |
| `transformation_pipeline` | 628 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertShapeOf3` | External / not CPU-local class |
| `transformation_pipeline` | 629 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::LSTMCellDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 630 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::GRUCellDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 631 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::RNNCellDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 632 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertNMS1ToNMS9` | External / not CPU-local class |
| `transformation_pipeline` | 633 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertNMS3ToNMS9` | External / not CPU-local class |
| `transformation_pipeline` | 634 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertNMS4ToNMS9` | External / not CPU-local class |
| `transformation_pipeline` | 635 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertNMS5ToNMS9` | External / not CPU-local class |
| `transformation_pipeline` | 636 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertNMS9ToNMSIEInternal` | External / not CPU-local class |
| `transformation_pipeline` | 637 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::Validate` | External / not CPU-local class |
| `transformation_pipeline` | 638 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertMulticlassNmsToMulticlassNmsIE` | External / not CPU-local class |
| `transformation_pipeline` | 639 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::Validate` | External / not CPU-local class |
| `transformation_pipeline` | 640 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertMatrixNmsToMatrixNmsIE` | External / not CPU-local class |
| `transformation_pipeline` | 641 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::Validate` | External / not CPU-local class |
| `transformation_pipeline` | 642 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::TransposeMatMul` | External / not CPU-local class |
| `transformation_pipeline` | 643 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConstantFolding` | External / not CPU-local class |
| `transformation_pipeline` | 644 | `CPU_REGISTER_PASS_ARM64` | `ov::pass::HardSigmoidDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 648 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::low_precision::ConvertSubtractConstant` | External / not CPU-local class |
| `transformation_pipeline` | 650 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::Validate` | External / not CPU-local class |
| `transformation_pipeline` | 656 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::InsertConvertAfterExtension` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/insert_convert_after_extension.hpp` |
| `transformation_pipeline` | 660 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertPrecision` | External / not CPU-local class |
| `transformation_pipeline` | 667 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::EliminateConvert` | External / not CPU-local class |
| `transformation_pipeline` | 668 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::EliminateIdentityConvert` | External / not CPU-local class |
| `transformation_pipeline` | 669 | `CPU_REGISTER_PASS_COMMON` | `SwapConvertTranspose` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/swap_convert_transpose.hpp` |
| `transformation_pipeline` | 670 | `CPU_REGISTER_PASS_X64` | `ConvertToInteraction` | `src/plugins/intel_cpu/src/transformations/cpu_opset/x64/pass/convert_to_interaction.hpp` |
| `transformation_pipeline` | 671 | `CPU_REGISTER_PASS_X64` | `ConvertInteractionInt8` | `src/plugins/intel_cpu/src/transformations/cpu_opset/x64/pass/convert_to_interaction.hpp` |
| `transformation_pipeline` | 672 | `CPU_REGISTER_PASS_ARM` | `ConvertReduceNoKeepDims` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/convert_reduce_no_keep_dims.hpp` |
| `transformation_pipeline` | 673 | `CPU_REGISTER_PASS_ARM` | `ConvertReduceMultiAxis` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/convert_reduce_multi_axis.hpp` |
| `transformation_pipeline` | 674 | `CPU_REGISTER_PASS_ARM32` | `MishDecomposition` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/mish_decomposition.hpp` |
| `transformation_pipeline` | 675 | `CPU_REGISTER_PASS_ARM` | `ConvertConv1D` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/convert_group_conv1d.hpp` |
| `transformation_pipeline` | 676 | `CPU_REGISTER_PASS_ARM` | `ConvertGroupConv1D` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/convert_group_conv1d.hpp` |
| `transformation_pipeline` | 677 | `CPU_REGISTER_PASS_ARM` | `ConvertGroupConvolution` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/convert_group_conv.hpp` |
| `transformation_pipeline` | 679 | `CPU_REGISTER_PASS_ARM` | `GridSampleDecomposition` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/grid_sample_decomposition.hpp` |
| `transformation_pipeline` | 680 | `CPU_REGISTER_PASS_ARM` | `Deconv1DDecomposition` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/deconv_1d_decomposition.hpp` |
| `transformation_pipeline` | 683 | `CPU_REGISTER_PASS_ARM` | `DecomposeIntegerDivide` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/decompose_integer_divide.hpp` |
| `transformation_pipeline` | 684 | `CPU_REGISTER_PASS_X86` | `DecomposeIntegerDivide` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/decompose_integer_divide.hpp` |
| `transformation_pipeline` | 686 | `CPU_REGISTER_PASS_COMMON` | `PermuteSliceAndInterpolation` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/permute_slice_n_interpolation.hpp` |
| `transformation_pipeline` | 687 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::Validate` | External / not CPU-local class |
| `transformation_pipeline` | 825 | `CPU_ENABLE_PASS_COMMON` | `ov::pass::SoftmaxDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 848 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::DisableDecompressionConvertConstantFolding` | External / not CPU-local class |
| `transformation_pipeline` | 849 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertCompressedOnlyToLegacy` | External / not CPU-local class |
| `transformation_pipeline` | 850 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::EyeDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 851 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertGELU` | External / not CPU-local class |
| `transformation_pipeline` | 852 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::Gelu7Downgrade` | External / not CPU-local class |
| `transformation_pipeline` | 853 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertMod` | External / not CPU-local class |
| `transformation_pipeline` | 854 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::WeightsDequantizeToFakeQuantize` | External / not CPU-local class |
| `transformation_pipeline` | 855 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::SimplifyCTCGreedyDecoderSeqLen` | External / not CPU-local class |
| `transformation_pipeline` | 856 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertGather7ToGather1` | External / not CPU-local class |
| `transformation_pipeline` | 857 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertGather8ToGather7` | External / not CPU-local class |
| `transformation_pipeline` | 858 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertBroadcastToTiles` | External / not CPU-local class |
| `transformation_pipeline` | 859 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertReduceMeanToPooling` | External / not CPU-local class |
| `transformation_pipeline` | 860 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertReduceMaxToPooling` | External / not CPU-local class |
| `transformation_pipeline` | 861 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertReduceSumToPooling` | External / not CPU-local class |
| `transformation_pipeline` | 862 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::SliceToStridedSlice` | External / not CPU-local class |
| `transformation_pipeline` | 863 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertDetectionOutput8ToDetectionOutput1` | External / not CPU-local class |
| `transformation_pipeline` | 864 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertROIAlign9To3` | External / not CPU-local class |
| `transformation_pipeline` | 865 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::SoftSignDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 866 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::UniqueDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 867 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertTopK11ToTopK3` | External / not CPU-local class |
| `transformation_pipeline` | 868 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::HSwishDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 869 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::MatMulConstTransposesExtraction` | External / not CPU-local class |
| `transformation_pipeline` | 870 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertScatterNDUpdate15ToScatterNDUpdate3` | External / not CPU-local class |
| `transformation_pipeline` | 871 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::ConvertSliceScatter` | External / not CPU-local class |
| `transformation_pipeline` | 872 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::SDPAFusion` | External / not CPU-local class |
| `transformation_pipeline` | 873 | `CPU_DISABLE_PASS_X64` | `ov::pass::HSigmoidDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 874 | `CPU_DISABLE_PASS_ARM64` | `ov::pass::HSigmoidDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 876 | `CPU_DISABLE_PASS_X64` | `ov::pass::ReduceL1Decomposition` | External / not CPU-local class |
| `transformation_pipeline` | 877 | `CPU_DISABLE_PASS_X64` | `ov::pass::ReduceL2Decomposition` | External / not CPU-local class |
| `transformation_pipeline` | 879 | `CPU_ENABLE_PASS_COMMON` | `ov::pass::NormalizeL2Decomposition` | External / not CPU-local class |
| `transformation_pipeline` | 880 | `CPU_ENABLE_PASS_COMMON` | `ov::pass::ConvertInterpolate1ToInterpolate4` | External / not CPU-local class |
| `transformation_pipeline` | 881 | `CPU_ENABLE_PASS_COMMON` | `ov::pass::ConvertGather1ToGather7` | External / not CPU-local class |
| `transformation_pipeline` | 882 | `CPU_ENABLE_PASS_COMMON` | `ov::pass::ConvertDetectionOutput1ToDetectionOutput8` | External / not CPU-local class |
| `transformation_pipeline` | 883 | `CPU_ENABLE_PASS_COMMON` | `ov::pass::ConvertROIAlign3To9` | External / not CPU-local class |
| `transformation_pipeline` | 911 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::EnableDecompressionConvertConstantFolding` | External / not CPU-local class |
| `transformation_pipeline` | 912 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::KeepConstAndDecompression` | External / not CPU-local class |
| `transformation_pipeline` | 913 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConstantFolding` | External / not CPU-local class |
| `transformation_pipeline` | 914 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::LoraSubgraphFusion` | External / not CPU-local class |
| `transformation_pipeline` | 968 | `CPU_REGISTER_PASS_COMMON` | `LowPrecision` | External / not CPU-local class |
| `transformation_pipeline` | 974 | `CPU_REGISTER_PASS_ARM` | `ConvertConvolutionBias` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/convert_conv_bias.hpp` |
| `transformation_pipeline` | 975 | `CPU_REGISTER_PASS_ARM` | `FallbackUnsupportedLPConvToFP16` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/fallback_unsupported_lp_conv_to_fp16.hpp` |
| `transformation_pipeline` | 1000 | `CPU_DISABLE_PASS_COMMON` | `MultiplyToGroupConvolutionTransformation` | External / not CPU-local class |
| `transformation_pipeline` | 1002 | `CPU_DISABLE_PASS_ARM` | `AvgPoolTransformation` | External / not CPU-local class |
| `transformation_pipeline` | 1003 | `CPU_DISABLE_PASS_ARM` | `InterpolateTransformation` | External / not CPU-local class |
| `transformation_pipeline` | 1004 | `CPU_DISABLE_PASS_ARM` | `GroupConvolutionTransformation` | External / not CPU-local class |
| `transformation_pipeline` | 1005 | `CPU_DISABLE_PASS_ARM` | `MaxPoolTransformation` | External / not CPU-local class |
| `transformation_pipeline` | 1006 | `CPU_DISABLE_PASS_ARM` | `MVNTransformation` | External / not CPU-local class |
| `transformation_pipeline` | 1007 | `CPU_DISABLE_PASS_ARM` | `NormalizeL2Transformation` | External / not CPU-local class |
| `transformation_pipeline` | 1008 | `CPU_DISABLE_PASS_ARM` | `RecurrentCellTransformation` | External / not CPU-local class |
| `transformation_pipeline` | 1009 | `CPU_DISABLE_PASS_ARM` | `ReduceMaxTransformation` | External / not CPU-local class |
| `transformation_pipeline` | 1010 | `CPU_DISABLE_PASS_ARM` | `ReduceMeanTransformation` | External / not CPU-local class |
| `transformation_pipeline` | 1011 | `CPU_DISABLE_PASS_ARM` | `ReduceMinTransformation` | External / not CPU-local class |
| `transformation_pipeline` | 1012 | `CPU_DISABLE_PASS_ARM` | `ReduceSumTransformation` | External / not CPU-local class |
| `transformation_pipeline` | 1097 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConvertBroadcast3` | External / not CPU-local class |
| `transformation_pipeline` | 1098 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::UnrollTensorIterator` | External / not CPU-local class |
| `transformation_pipeline` | 1099 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ReshapePRelu` | External / not CPU-local class |
| `transformation_pipeline` | 1107 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::MoveEltwiseUpThroughDataMov` | External / not CPU-local class |
| `transformation_pipeline` | 1108 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::MoveEltwiseUpThroughDataMovPerChannel` | External / not CPU-local class |
| `transformation_pipeline` | 1124 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::Validate` | External / not CPU-local class |
| `transformation_pipeline` | 1126 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConstantFolding` | External / not CPU-local class |
| `transformation_pipeline` | 1128 | `CPU_REGISTER_PASS_X64` | `FuseFQtoInteraction` | `src/plugins/intel_cpu/src/transformations/cpu_opset/x64/pass/convert_to_interaction.hpp` |
| `transformation_pipeline` | 1131 | `CPU_REGISTER_PASS_X64` | `ConvertFqRnnToQuantizedRnn` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_fq_rnn_to_quantized_rnn.hpp` |
| `transformation_pipeline` | 1133 | `CPU_REGISTER_PASS_X64` | `ov::pass::RoPEFusion` | External / not CPU-local class |
| `transformation_pipeline` | 1134 | `CPU_REGISTER_PASS_ARM64` | `ov::pass::RoPEFusion` | External / not CPU-local class |
| `transformation_pipeline` | 1135 | `CPU_DISABLE_PASS_COMMON` | `ov::pass::RoPEFusionFlux` | External / not CPU-local class |
| `transformation_pipeline` | 1136 | `CPU_REGISTER_PASS_X64` | `CausalMaskPreprocessFusion` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/causal_mask_preprocess_fusion.hpp` |
| `transformation_pipeline` | 1147 | `CPU_REGISTER_PASS_X64` | `MLPFusion` | `src/plugins/intel_cpu/src/transformations/cpu_opset/x64/pass/mlp_fusion.hpp` |
| `transformation_pipeline` | 1161 | `CPU_REGISTER_PASS_X64` | `QKVProjFusion` | `src/plugins/intel_cpu/src/transformations/cpu_opset/x64/pass/qkv_proj_fusion.hpp` |
| `transformation_pipeline` | 1186 | `CPU_REGISTER_PASS_X64` | `ov::pass::RMSFusion` | External / not CPU-local class |
| `transformation_pipeline` | 1187 | `CPU_REGISTER_PASS_X64` | `ov::intel_cpu::DecomposeRMSNorm` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/decompose_rms_norm.hpp` |
| `transformation_pipeline` | 1198 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::MarkRopeInputsToKeepInMixedPrecision` | External / not CPU-local class |
| `transformation_pipeline` | 1199 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::MarkFloatingPointRange` | External / not CPU-local class |
| `transformation_pipeline` | 1204 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::SymbolicOptimizations` | External / not CPU-local class |
| `transformation_pipeline` | 1321 | `CPU_REGISTER_PASS_ARM64` | `SnippetsMarkSkipped` | `src/plugins/intel_cpu/src/transformations/snippets/aarch64/pass/snippets_mark_skipped.hpp` |
| `transformation_pipeline` | 1322 | `CPU_REGISTER_PASS_X64` | `SnippetsMarkSkipped` | `src/plugins/intel_cpu/src/transformations/snippets/x64/pass/snippets_mark_skipped.hpp` |
| `transformation_pipeline` | 1323 | `CPU_DISABLE_PASS_COMMON` | `TokenizeFCSnippets` | External / not CPU-local class |
| `transformation_pipeline` | 1324 | `CPU_DISABLE_PASS_COMMON` | `TokenizeGatedMLPSnippets` | External / not CPU-local class |
| `transformation_pipeline` | 1326 | `CPU_REGISTER_PASS_COMMON` | `SnippetsTokenization` | External / not CPU-local class |
| `transformation_pipeline` | 1361 | `CPU_DISABLE_PASS_COMMON` | `TokenizeMHASnippets` | External / not CPU-local class |
| `transformation_pipeline` | 1362 | `CPU_DISABLE_PASS_COMMON` | `ExtractReshapesFromMHA` | External / not CPU-local class |
| `transformation_pipeline` | 1376 | `CPU_DISABLE_PASS_COMMON` | `TokenizeMLPSeqSnippets` | External / not CPU-local class |
| `transformation_pipeline` | 1666 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::FakeQuantizeDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 1689 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::FakeConvertDecomposition` | External / not CPU-local class |
| `transformation_pipeline` | 1690 | `CPU_REGISTER_PASS_COMMON` | `ov::pass::ConstantFolding` | External / not CPU-local class |

## C. Manual registrations not captured by macro scan

| Phase | Registration site | Pass | Local source |
|---|---|---|---|
| `post_lpt.symbolic_pipeline` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1205` | `NgramFusion` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/ngram_fusion.hpp:11` |

## D. CPU-local transformation coverage status (LPT-focused deep analysis)

Legend:

- `Deep` = full per-pass analysis in `03_transformations_catalog.md` (intent, guards, rewrite, precision, interactions, Mermaid before/after).
- `Enumerated` = included in full pipeline matrix/order analysis, but not expanded in deep per-pass LPT catalog because it is outside the LPT-critical path.

| CPU-local transformation | Primary source | Registration reference(s) | Coverage |
|---|---|---|---|
| `SDPASubgraphFusion` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/stateful_sdpa_fusion.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:574` | Deep |
| `InsertConvertAfterExtension` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/insert_convert_after_extension.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:656` | Deep |
| `SwapConvertTranspose` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/swap_convert_transpose.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:669` | Deep |
| `ConvertReduceNoKeepDims` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/convert_reduce_no_keep_dims.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:672` | Deep |
| `ConvertReduceMultiAxis` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/convert_reduce_multi_axis.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:673` | Deep |
| `MishDecomposition` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/mish_decomposition.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:674` | Deep |
| `ConvertConv1D` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/convert_group_conv1d.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:675` | Deep |
| `ConvertGroupConv1D` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/convert_group_conv1d.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:676` | Deep |
| `ConvertGroupConvolution` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/convert_group_conv.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:677` | Deep |
| `GridSampleDecomposition` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/grid_sample_decomposition.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:679` | Deep |
| `Deconv1DDecomposition` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/deconv_1d_decomposition.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:680` | Deep |
| `DecomposeIntegerDivide` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/decompose_integer_divide.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:683`, `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:684` | Deep |
| `PermuteSliceAndInterpolation` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/permute_slice_n_interpolation.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:686` | Deep |
| `ConvertToInteraction` | `src/plugins/intel_cpu/src/transformations/cpu_opset/x64/pass/convert_to_interaction.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:670` | Deep |
| `ConvertInteractionInt8` | `src/plugins/intel_cpu/src/transformations/cpu_opset/x64/pass/convert_to_interaction.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:671` | Deep |
| `ConvertConvolutionBias` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/convert_conv_bias.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:974` | Deep |
| `FallbackUnsupportedLPConvToFP16` | `src/plugins/intel_cpu/src/transformations/cpu_opset/arm/pass/fallback_unsupported_lp_conv_to_fp16.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:975` | Deep |
| `FuseFQtoInteraction` | `src/plugins/intel_cpu/src/transformations/cpu_opset/x64/pass/convert_to_interaction.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1128` | Deep |
| `ConvertFqRnnToQuantizedRnn` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_fq_rnn_to_quantized_rnn.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1131` | Deep |
| `CausalMaskPreprocessFusion` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/causal_mask_preprocess_fusion.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1136` | Deep |
| `MLPFusion` / `MLPFusionPass` | `src/plugins/intel_cpu/src/transformations/cpu_opset/x64/pass/mlp_fusion.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1147` | Deep |
| `QKVProjFusion` / `QKVProjFusionPass1/2` | `src/plugins/intel_cpu/src/transformations/cpu_opset/x64/pass/qkv_proj_fusion.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1161` | Deep |
| `DecomposeRMSNorm` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/decompose_rms_norm.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1187` | Deep |
| `NgramFusion` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/ngram_fusion.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1205` | Deep |
| ARM LPT callback helpers (`match_*`) | `src/plugins/intel_cpu/src/transformations/utils.cpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:976-999`, `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1027-1037`, `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1676-1688` | Deep |
| `MoEMatMulsFusion` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/moe_matmuls_fusion.hpp` | `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:45` | Enumerated |
| `ConvertBatchGatherMatmulToBatchGatherMatmulCompressed` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_batch_gather_matmul_to_compressed.hpp` | `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:47` | Enumerated |
| `ConvertMatMulToFC` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_matmul_to_fc.hpp` | `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:59` | Enumerated |
| `FullyConnectedBiasFusion` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/fc_bias_fusion.hpp` | `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:60` | Enumerated |
| `MoveFCReshapeToWeights` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/move_fc_reshape_to_weights.hpp` | `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:72` | Enumerated |
| `AlignMatMulInputRanks` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/align_matmul_input_ranks.hpp` | `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:74` | Enumerated |
| `ConvertTileToSeqTiles` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_tile_to_seq_tiles.hpp` | `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:75` | Enumerated |
| `ConvertToPowerStatic` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_to_power_static.hpp` | `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:76` | Enumerated |
| `ConvertToLeakyRelu` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_to_leaky_relu.hpp` | `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:77` | Enumerated |
| `ConvertToSwishCPU` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/convert_to_swish_cpu.hpp` | `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:78` | Enumerated |
| `OptimizeSequenceTransposes` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/rnn_sequences_optimization.hpp` | `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:79` | Enumerated |
| `MoveReadValueInputsToSubgraph` | `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/move_readvalue_inputs_to_subgraph.hpp` | `src/plugins/intel_cpu/src/transformations/cpu_opset/convert_to_cpu_specific_opset.hpp:92` | Enumerated |
| `SnippetsMarkSkipped` (AArch64) | `src/plugins/intel_cpu/src/transformations/snippets/aarch64/pass/snippets_mark_skipped.hpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1321` | Enumerated |
| `SnippetsMarkSkipped` (x64) | `src/plugins/intel_cpu/src/transformations/snippets/x64/pass/snippets_mark_skipped.hpp` | `src/plugins/intel_cpu/src/transformations/transformation_pipeline.cpp:1322` | Enumerated |

## E. Validation checklist

- [x] `graph_optimizer.cpp` integration analyzed for low-precision-sensitive runtime fusions.
- [x] `transformation_pipeline.cpp` pass order and architecture branches enumerated.
- [x] `src/plugins/intel_cpu/src/transformations/**` local passes referenced by LPT path identified and cataloged.
- [x] x64 vs ARM/AArch64 differences documented in dedicated comparison (`05_x64_vs_arm.md`).
- [x] Before/after Mermaid graphs provided for each deep-analyzed local LPT-related transformation.
- [x] Pipeline registration matrix includes both registered and disabled/enabled pass directives from mandatory pipeline files.
- [x] Ambiguity boundary made explicit: external OpenVINO common/LPT pass internals are listed in matrix/order but not re-implemented from non-CPU-plugin directories.
