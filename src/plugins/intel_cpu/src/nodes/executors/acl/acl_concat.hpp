// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/concat.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

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
    std::vector<arm_compute::ITensor*> srcTensorVector;
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
        /*if (matmulAttrs.transposeA || matmulAttrs.transposeB || matmulAttrs.withBias)
            return false;

        if (srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 ||
            srcDescs[1]->getPrecision() != InferenceEngine::Precision::FP32 ||
            dstDescs[0]->getPrecision() != InferenceEngine::Precision::FP32)
            return false;

        if (!srcDescs[0]->hasLayoutType(LayoutType::ncsp) ||
            !srcDescs[1]->hasLayoutType(LayoutType::ncsp) ||
            !dstDescs[0]->hasLayoutType(LayoutType::ncsp))
            return false;*/

        return true;
    }

    ConcatExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclConcatExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov