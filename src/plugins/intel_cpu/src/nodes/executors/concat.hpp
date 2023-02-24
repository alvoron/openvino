// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"

namespace ov {
namespace intel_cpu {

struct ConcatAttrs {
    size_t axis;
};

class ConcatExecutor {
public:
    ConcatExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const ConcatAttrs& concatAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) = 0;
    virtual ~ConcatExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

protected:
    ConcatAttrs concatAttrs;
    const ExecutorContext::CPtr context;
};

using ConcatExecutorPtr = std::shared_ptr<ConcatExecutor>;
using ConcatExecutorCPtr = std::shared_ptr<const ConcatExecutor>;

class ConcatExecutorBuilder {
public:
    ~ConcatExecutorBuilder() = default;
    virtual bool isSupported(const ConcatAttrs& ConcatAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual ConcatExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using ConcatExecutorBuilderPtr = std::shared_ptr<ConcatExecutorBuilder>;
using ConcatExecutorBuilderCPtr = std::shared_ptr<const ConcatExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov