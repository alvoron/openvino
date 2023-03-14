// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/gather.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

class AclGatherExecutor : public GatherExecutor {
public:
    AclGatherExecutor(const ExecutorContext::CPtr context);

    bool init(const GatherAttrs& gatherAttrs,
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
    GatherAttrs gatherAttrs;
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor indTensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEGather> gather = nullptr;
};

class AclGatherExecutorBuilder : public GatherExecutorBuilder {
public:
    bool isSupported(const GatherAttrs& gatherAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        //ACL supports all layouts and all precisions
        return true;
    }

    GatherExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclGatherExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov