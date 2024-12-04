// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "debug_messages.hpp"
#include "implementation_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected.hpp"
#include "nodes/executors/dnnl/dnnl_matmul_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mlas/mlas_gemm.hpp"
#include "nodes/executors/precision_matcher.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "ov_optional.hpp"
#include "utils/cpp/maybe_unused.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

static const MappingNotation dnnlMatMulMappingNotation{ARG_SRC, ARG_WEI, ARG_BIAS, ARG_DST};

using LayoutConfig = std::vector<LayoutType>;
static const LayoutConfig dnnlMatMulLayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp};

// clang-format off
static const TypeMapping dnnlMatMulTypeMapping {
    // {src, wei, bia, dst}                                   pt<src, wei, bias, dst>
    {{_bf16, _bf16 | _f32, _any, _bf16 | _f32},               pt(bypass(), bypass(), use<3>(), bypass())},
    {{_f16, _f16, _any, _f16 | _f32},                         pt(bypass(), bypass(), use<3>(), bypass())},
    // integer precision outputs are not supported for float precision inputs
    {{_f32 | _bf16 | _f16, _any, _any, _i8 | _u8},            pt(bypass(), bypass(), use<0>(), use<0>())},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float, _any, _any | _any},                  pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_bf16, _f16, _any, _any | _any},                        pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_f16, _bf16, _any, _any | _any},                        pt(bypass(), bypass(), use<0>(), use<0>())},
    // quantization configuration
    {{_u8 | _i8, _i8, _u8|_i8|_i32|_bf16|_f16|_f32|_undefined, _u8|_i8|_i32|_bf16|_f16|_f32}, pt(bypass(), bypass(), bypass(),  bypass())},
    {{_u8 | _i8, _i8, _any, _any},                            pt(bypass(), bypass(), just<f32>(), just<f32>())},
    // @todo should we fallback to FPXX instead of _f32?
    {{_any, _any, _any, _any},                                pt(just<f32>(), just<f32>(), just<f32>(), just<f32>())},
    // @todo explicitly cover configuration limitations for oneDNN on ARM
};
// clang-format on

static bool fullyMatchConfiguration(const MemoryDescArgs& currentDescriptors,
                                    const InOutTypes& typeConfig,
                                    const LayoutConfig& layoutConfig,
                                    const MappingNotation& notation) {
    for (size_t i = 0; i < typeConfig.size(); i++) {
        const auto& type = typeConfig[i];
        const auto& desc = currentDescriptors.at(notation[i]);

        if (desc->empty())
            continue;

        if (desc->getPrecision() != type)
            return false; // type mismatch

        if (!desc->hasLayoutType(layoutConfig[i]))
            return false; // layout mismatch
    }

    return true;
}

static MemoryDescArgs createOptimalDescriptors(const MemoryDescArgs& currentDescriptors,
                                               const InOutTypes& typeConfig,
                                               const LayoutConfig& layoutConfig,
                                               const MappingNotation& notation) {
    MemoryDescArgs descs = currentDescriptors;

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < typeConfig.size(); i++) {
        const auto& desc = currentDescriptors.at(notation[i]);
        const auto& descType = desc->getPrecision();
        const auto& type = typeConfig[i];
        const auto& layout = layoutConfig[i];

        if (desc->empty())
            continue;

        if (descType == type && desc->hasLayoutType(layout)) {
            continue;
        }

        descs[notation[i]] = creatorsMap.at(layout)->createSharedDesc(type, desc->getShape());
    }

    return descs;
}

template <typename Attrs>
ov::optional<executor::Config<Attrs>> requiresFallbackCommon(const executor::Config<Attrs>& config,
                                                             const TypeMapping& typeMapping,
                                                             const LayoutConfig& layoutConfig,
                                                             const MappingNotation& notation) {
    const auto typeConfig = getTypeConfiguration(config.descs, typeMapping, notation);

    if (fullyMatchConfiguration(config.descs, typeConfig, layoutConfig, notation)) {
        return {};
    }

    const auto optimalDescriptors = createOptimalDescriptors(config.descs, typeConfig, layoutConfig, notation);

    return ov::optional<executor::Config<Attrs>>(MatMulConfig{optimalDescriptors, config.attrs, config.postOps});
}

template <>
const std::vector<ExecutorImplementation<MatMulAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<MatMulAttrs>> matmulImplementations {
        OV_CPU_INSTANCE_DNNL(
            "matmul_dnnl",
            ExecutorType::Dnnl,
            OperationType::MatMul,
            ShapeTolerance::Dependant,
            // supports
            [](const MatMulConfig& config) -> bool {
                //todo: add checks
                return true;
            },
            // requiresFallback
            [](const MatMulConfig& config) -> ov::optional<executor::Config<MatMulAttrs>> {
                return requiresFallbackCommon(config,
                                              dnnlMatMulTypeMapping,
                                              dnnlMatMulLayoutConfig,
                                              dnnlMatMulMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                return true;
            },
            // create
            [](const MatMulAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               ExecutorContext::CPtr context) -> std::shared_ptr<Executor> {
                struct MatMulInstantiator {
                    std::shared_ptr<DnnlMatMulPrimitive> operator()(
                        const MemoryArgs& memory,
                        const MatMulAttrs& attrs,
                        const ExecutorContext::CPtr context,
                        std::shared_ptr<DnnlShapeAgnosticData> shareAgnosticData) const {
                        MatMulAttrs matMulAttrs{false,
                                                false,
                                                attrs.dequantizationScales};
                        auto primitive =
                            DefaultInstantiator<DnnlMatMulPrimitive, MatMulAttrs, DnnlShapeAgnosticData>{}(
                            memory,
                            matMulAttrs,
                            context,
                            shareAgnosticData);
                        return primitive;
                    }
                };

                return std::make_shared<
                    DnnlFCExecutor<DnnlMatMulPrimitive, MatMulAttrs, DnnlShapeAgnosticData, MatMulInstantiator>>(
                    attrs,
                    postOps,
                    memory,
                    context,
                    false);
            })
    };

    return matmulImplementations;
}
}  // namespace intel_cpu
}  // namespace ov
