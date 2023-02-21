// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"

namespace ov {
namespace intel_cpu {

// Defines way to add epsilon: inside sqrt or outside.
struct ConvAttrs {
    bool withBiases;
    std::vector<size_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;
    InferenceEngine::SizeVector weightDims;
    InferenceEngine::SizeVector biasesDims;
};

class ConvExecutor {
public:
    ConvExecutor();
    virtual bool init(const ConvAttrs& mvnAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) = 0;
    virtual ~ConvExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

protected:
    ConvAttrs convAttrs;
};

using ConvExecutorPtr = std::shared_ptr<ConvExecutor>;
using ConvExecutorCPtr = std::shared_ptr<const ConvExecutor>;

class ConvExecutorBuilder {
public:
    ~ConvExecutorBuilder() = default;
    virtual bool isSupported(const ConvAttrs& convAttrs, const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual ConvExecutorPtr makeExecutor() const = 0;
};

using ConvExecutorBuilderPtr = std::shared_ptr<ConvExecutorBuilder>;
using ConvExecutorBuilderCPtr = std::shared_ptr<const ConvExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov