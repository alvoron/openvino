// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.h"

#include "memory_desc/cpu_blocked_memory_desc.h"
#include "cpu_types.h"
#include "eltwise.h"

#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include "common/cpu_memcpy.h"
#include "openvino/opsets/opset1.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "fake_quantize.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "dnnl_extension_utils.h"
#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "shape_inference/custom/matmul.hpp"
using namespace dnnl;


namespace ov {
namespace intel_cpu {
namespace node {

bool MatMul::canBeExecutedInInt8() const {
    auto firstInputPrecision = getOriginalInputPrecisionAtPort(0);
    auto secondInputPrecision = getOriginalInputPrecisionAtPort(1);

    return one_of(firstInputPrecision, ov::element::u8, ov::element::i8) && secondInputPrecision == ov::element::i8;
}

bool MatMul::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto matMul = std::dynamic_pointer_cast<const ov::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_partial_shape(i).rank().get_length();
            if (inShapeRank < 2) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_output_partial_shape(0).rank().get_length();
        if (outShapeRank < 2) {
            errorMessage = "Unsupported rank: " + std::to_string(outShapeRank) + " on output";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MatMul::MatMul(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
    Node(op, context, MMShapeInferFactory(op)), withBiases(false) {
    std::string errorMessage;
    errorPrefix = "MatMul node with name '" + getName() + "'";

    if (!isSupportedOperation(op, errorMessage))
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);

    const auto matMul = std::dynamic_pointer_cast<const ov::opset1::MatMul>(op);

    if (!matMul) {
        OPENVINO_THROW_NOT_IMPLEMENTED("Operation with name ",
                                       op->get_friendly_name(),
                                       ":",
                                       op->get_type_name(),
                                       " is not an instance of MatMul from opset1");
    }

    transposeIn[0] = matMul->get_transpose_a();
    transposeIn[1] = matMul->get_transpose_b();
}

bool MatMul::canFuse(const NodePtr& node) const {
    // WA for CVS-84056: oneDNN brgemm impl has problem with per-OC binary-postOps for MatMul with 6D inputs
    if (impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            if (eltwiseNode->getBroadcastingPolicy() == Eltwise::BroadcastingPolicy::PerChannel) {
                auto rank = getInputShapeAtPort(0).getRank();
                if (rank > 4) {
                    DEBUG_LOG("skip fusing non-perTensor Eltwise:", eltwiseNode->getName(), " into 6D MatMul:", getName());
                    return false;
                }
            }
        }
    }

    //  Consider the case when Matmul doesn't support execution in int8, but is getting fused with FQ with int8 output.
    //  Then the Matmul will change its output precision to fp32. If fusing FQ into matmul, there would be reorder inserted
    //  after matmul. In some bert model, this reorder causes great perf degradation.
    //  Todo: Remove this if onednn primitive support U8 output with floating input.
    if (node->getType() == Type::FakeQuantize && one_of(node->getOriginalOutputPrecisionAtPort(0), ov::element::i8, ov::element::u8) &&
        !canBeExecutedInInt8() &&
        getOriginalInputPrecisionAtPort(0) == ov::element::f32 )
        return false;
    return canFuseSimpleOperation(node);
}

void MatMul::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto addSupportedPrimitiveDescriptor = [&](const dnnl::primitive_desc& prim_desc) {
        std::vector<PortConfig> inConfs, outConfs;
        const int inPlaceOutPort = canBeInPlace() ? 0 : -1;

        for (size_t i = 0; i < descInputNumbers(); i++) {
            auto desc = getSrcMemDesc(prim_desc, i);

            inConfs.emplace_back(desc);
        }

        for (size_t i = 0; i < descOutputNumbers(); i++) {
            auto desc = getDstMemDesc(prim_desc, i);

            outConfs.emplace_back(desc, BlockedMemoryDesc::FULL_MASK, inPlaceOutPort);
        }

        const NodeConfig config(inConfs, outConfs);
        const impl_desc_type impl_type = parse_impl_name(prim_desc.impl_info_str());

        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    };
#ifdef CPU_DEBUG_CAPS
    {
       if (!customImplPriorities.empty()) {
            DEBUG_LOG("#", getName(), " customImplPriorities [", 0 , "/", customImplPriorities.size(),
                        "]: ", impl_type_to_string(customImplPriorities[0]));
       }
    }
#endif
    for (auto& desc : descs) {
        auto first_desc = dnnl::primitive_desc(DnnlExtensionUtils::clone_primitive_desc(desc.get()));
        const bool first_match = customImplPriorities.empty();
        DEBUG_LOG("#", getName(),
                ", itpd.impl_info_str(): ", desc.impl_info_str(),
            ", parsed imp_type: ", impl_type_to_string(parse_impl_name(desc.impl_info_str())),
            ", first_match: ", first_match ? "true" : "false");
        DnnlExtensionUtils::for_each_implementation(desc,
                                                    first_match,
                                                    [&](impl_desc_type implType) {
                                                        return contains(getImplPriority(), implType);
                                                    },
                                                    [&](dnnl::primitive_desc& desc) {
                                                        addSupportedPrimitiveDescriptor(desc);
                                                    });

        // fallback. if none of the primitive types is present in the priority list just add first implementation
        // @todo this fallback is not necessary if primitive priority list is filled correctly
        if (supportedPrimitiveDescriptors.empty())
            addSupportedPrimitiveDescriptor(first_desc);
   }
}

MemoryDescPtr MatMul::getSrcMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const {
    auto desc = idx > 0 ? prim_desc.weights_desc(idx - 1): prim_desc.src_desc(idx);

    if (idx < 2) // inputs
        return std::make_shared<CpuBlockedMemoryDesc>(
            DnnlExtensionUtils::DataTypeToElementType(desc.get_data_type()),
            getInputShapeAtPort(idx)); /* provide initial shapes, so hide transpose effect */
    else // bias
        return DnnlExtensionUtils::makeDescriptor(desc);
}

bool MatMul::created() const {
    return getType() == Type::MatMul;
}

ov::element::Type MatMul::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

ExecutorPtr MatMul::createExecutor() {
    const auto& executor = factory->make(memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(executor->implType());

    return executor;
}

void MatMul::getSupportedDescriptors() {}

void MatMul::prepareParams() {
    if (!memory[ARG_DST] || !memory[ARG_DST]->isDefined())
        OPENVINO_THROW(errorPrefix, " has undefined destination memory");
    if (!memory[ARG_SRC_0] || !memory[ARG_SRC_0]->isDefined() ||
        !memory[ARG_SRC_1] || !memory[ARG_SRC_1]->isDefined())
        OPENVINO_THROW(errorPrefix, " has undefined input memory");
    executor = createExecutor();
}

void MatMul::createPrimitive() {
    memory[ARG_SRC_0] = getSrcMemoryAtPort(0);
    memory[ARG_SRC_1] = getSrcMemoryAtPort(1);
    memory[ARG_DST] = getDstMemoryAtPort(0);

    factory->preconfigure(memory);

    Node::createPrimitive();
}

void MatMul::execute(dnnl::stream strm) {
    if (executor) {
        executor->execute(memory);
    } else {
        OPENVINO_THROW(errorPrefix, " doesn't have an initialized executor");
    }
}

void MatMul::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

const std::vector<impl_desc_type>& MatMul::getDefaultImplPriority() {
    static const std::vector<impl_desc_type> priorities = {
        impl_desc_type::unknown,
        impl_desc_type::brgemm_avx512_amx,
        impl_desc_type::brgemm_avx512,
        impl_desc_type::brgemm_avx2,
        impl_desc_type::gemm_acl,
        impl_desc_type::gemm_blas,
        impl_desc_type::gemm_avx512,
        impl_desc_type::gemm_avx2,
        impl_desc_type::gemm_avx,
        impl_desc_type::gemm_sse42,
        impl_desc_type::gemm_any,
        impl_desc_type::gemm,
        impl_desc_type::jit_gemm,
        impl_desc_type::jit_uni_dw,
        impl_desc_type::jit_uni_1x1,
        impl_desc_type::jit_uni,
        impl_desc_type::jit_avx512_dw,
        impl_desc_type::jit_avx512_1x1,
        impl_desc_type::jit_avx512,
        impl_desc_type::jit_avx2_dw,
        impl_desc_type::jit_avx2_1x1,
        impl_desc_type::jit_avx2,
        impl_desc_type::jit_avx_dw,
        impl_desc_type::jit_avx_1x1,
        impl_desc_type::jit_avx,
        impl_desc_type::jit_sse42_dw,
        impl_desc_type::jit_sse42_1x1,
        impl_desc_type::jit_sse42,
        impl_desc_type::ref,
    };

    return priorities;
}

bool MatMul::isExecutable() const {
    return !hasEmptyOutputTensors();
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
