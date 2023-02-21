// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/conv.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

class AclConvExecutor : public ConvExecutor {
public:
    AclConvExecutor();

    bool init(const ConvAttrs& convAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    ConvAttrs convAttrs;
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor weiTensor;
    arm_compute::Tensor biasTensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEConvolutionLayer> conv = nullptr;
};

class AclConvExecutorBuilder : public ConvExecutorBuilder {
public:
    bool isSupported(const ConvAttrs& convAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        std::cout << "AclConvExecutorBuilder::isSupported" << std::endl;
        /*if (mvnAttrs.epsMode_ == MVNEpsMode::INSIDE_SQRT) {
            std::cout << "AclMVNExecutorBuilder::isSupported - false (mvnAttrs.epsMode_ == MVNEpsMode::INSIDE_SQRT)" << std::endl;
            return false;
        }
        if (!mvnAttrs.normalizeVariance_) {
            std::cout << "AclMVNExecutorBuilder::isSupported - false (!mvnAttrs.normalizeVariance_)" << std::endl;
            return false;
        }
        // ACL supports MVN with 2D inputs only
        if (srcDescs[0]->getShape().getRank() != 2) {
            std::cout << "AclMVNExecutorBuilder::isSupported -false (srcDescs[0]->getShape().getRank() != 2 (" << srcDescs[0]->getShape().getRank() << "))" << std::endl;
            return false;
        }*/

        return true;
    }

    ConvExecutorPtr makeExecutor() const override {
        return std::make_shared<AclConvExecutor>();
    }
};

}   // namespace intel_cpu
}   // namespace ov