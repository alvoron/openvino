// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// TODO: remove relative path
#include "../mvn.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

class AclMVNExecutor : public MVNExecutor {
public:
    AclMVNExecutor();

    bool init(const MVNAttrs& mvnAttrs,
              const std::vector<MemoryDescCPtr>& srcDescs,
              const std::vector<MemoryDescCPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, 
              const std::vector<MemoryPtr>& dst, 
              const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEMeanStdDevNormalizationLayer> mvn = nullptr;
};

class AclMVNExecutorBuilder : public MVNExecutorBuilder {
public:
    bool isSupported(const MVNAttrs& mvnAttrs, 
                     const std::vector<MemoryDescCPtr>& srcDescs, 
                     const std::vector<MemoryDescCPtr>& dstDescs) const override {
        if ((srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 &&
             srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP16) ||
             srcDescs[0]->getPrecision() != dstDescs[0]->getPrecision())
            return false;

        if (!(srcDescs[0]->hasLayoutType(LayoutType::ncsp) &&
              dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
            !(srcDescs[0]->hasLayoutType(LayoutType::nspc) &&
              dstDescs[0]->hasLayoutType(LayoutType::nspc)))
            return false;

        if (mvnAttrs.epsMode_ == MVNEpsMode::OUTSIDE_SQRT) {
            return false;
        }
        if (!mvnAttrs.normalizeVariance_) {
            return false;
        }

        return true;
    }

    MVNExecutorPtr makeExecutor() const override {
        return std::make_shared<AclMVNExecutor>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
