// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "dnnl_scratch_pad.h"
#include "executor.hpp"

namespace ov {
namespace intel_cpu {

struct GatherAttrs {
    std::vector<int> axes;
    Algorithm operation;
    bool keepDims;
};

class GatherExecutor {
public:
    GatherExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const GatherAttrs& gatherAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) = 0;
    virtual ~GatherExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

protected:
    GatherAttrs gatherAttrs;
    const ExecutorContext::CPtr context;
};

using GatherExecutorPtr = std::shared_ptr<GatherExecutor>;
using GatherExecutorCPtr = std::shared_ptr<const GatherExecutor>;

class GatherExecutorBuilder {
public:
    ~GatherExecutorBuilder() = default;
    virtual bool isSupported(const GatherAttrs& gatherAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual GatherExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using GatherExecutorBuilderPtr = std::shared_ptr<GatherExecutorBuilder>;
using GatherExecutorBuilderCPtr = std::shared_ptr<const GatherExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov