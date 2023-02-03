// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// TODO: remove relative path
#include "../reduce.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

arm_compute::TensorShape shapeCast(const VectorDims& dims);

static std::map<ReduceOperation, arm_compute::ReductionOperation> aclReduceOp = {
    {ReduceOperation::MAX, arm_compute::ReductionOperation::MAX},
    {ReduceOperation::MIN, arm_compute::ReductionOperation::MIN},
    {ReduceOperation::PROD, arm_compute::ReductionOperation::PROD},
    {ReduceOperation::SUM, arm_compute::ReductionOperation::SUM}
};

class AclReduceExecutor : public ReduceExecutor {
public:
    AclReduceExecutor();

    bool init(const ReduceAttrs& reduceAttrs,
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
    ReduceAttrs reduceAttrs;
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor weiTensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEReductionOperation> reduce = nullptr;
};

class AclReduceExecutorBuilder : public ReduceExecutorBuilder {
public:
    bool isSupported(const ReduceAttrs& reduceAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs,
                     const dnnl::primitive_attr &attr) const override {
/*        if (matmulAttrs.transposeA || matmulAttrs.transposeB || matmulAttrs.withBias)
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

    ReduceExecutorPtr makeExecutor() const override {
        return std::make_shared<AclReduceExecutor>();
    }
};

}   // namespace intel_cpu
}   // namespace ov