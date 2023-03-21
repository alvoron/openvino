// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/concat.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

class AclConcatExecutor : public ConcatExecutor {
public:
    AclConcatExecutor(const ExecutorContext::CPtr context);

    bool init(const ConcatAttrs& concatAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              std::unordered_map<int, MemoryPtr> postOpsArgs) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    ConcatAttrs concatAttrs;
    impl_desc_type implType = impl_desc_type::gemm_acl;

    arm_compute::Tensor srcTensor;
    std::vector<const arm_compute::ITensor*> srcTensorVectorConst;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NECopy> concatNECopy = nullptr;
    std::unique_ptr<arm_compute::NEConcatenateLayer> concatNEConcatenate = nullptr;

    std::vector<const arm_compute::ITensorInfo*> inputs_vector;
    size_t axis;
};

class AclConcatExecutorBuilder : public ConcatExecutorBuilder {
public:
    bool isSupported(const ConcatAttrs& concatAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        if (srcDescs.size() > 1) {
            std::cout << "AclConcatExecutorBuilder::isSupported - srcDescs.size() > 1" << std::endl;
            bool areAllsrcDescsFP32 = true;
            bool areAllsrcDescsFP16 = true;
            std::string listSrcDescs;
            for (int i = 0; i < srcDescs.size(); i++) {
                if (srcDescs[i]->getPrecision() != InferenceEngine::Precision::FP32) {
                    areAllsrcDescsFP32 = false;
                }
                if (srcDescs[i]->getPrecision() != InferenceEngine::Precision::FP16) {
                    areAllsrcDescsFP16 = false;
                }
                listSrcDescs += " src[" + std::to_string(i) + "]=" + srcDescs[i]->getPrecision().name();
            }
            if ((!areAllsrcDescsFP32 && dstDescs[0]->getPrecision() != InferenceEngine::Precision::FP32) &&
                (!areAllsrcDescsFP16 && dstDescs[0]->getPrecision() != InferenceEngine::Precision::FP16)) {
                    DEBUG_LOG("AclConcatExecutor does not support precisions:" + listSrcDescs + " dst: " + dstDescs[0]->getPrecision().name());
                    return false;
            }
            std::cout << "return true, precisions: " << listSrcDescs << " dst: " << dstDescs[0]->getPrecision() << std::endl;
        } else {
            std::cout << "AclConcatExecutorBuilder::isSupported - srcDescs.size() == 1 return true" << std::endl;
        }
        return true;
    }

    ConcatExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclConcatExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov