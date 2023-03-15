// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"

namespace ov {
namespace intel_cpu {

// Defines way to add epsilon: inside sqrt or outside.
struct DeconvAttrs {
    bool withBiases = false;
    std::vector<ptrdiff_t> kernel;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;
    ov::CoordinateDiff outputPadding;
    std::vector<int32_t> lastOutputSpatialDims;
    VectorDims int8WeightDims;
    VectorDims biasesDims;
    //InferenceEngine::SizeVector weightDims;
    //InferenceEngine::SizeVector biasesDims;
};

class DeconvExecutor {
public:
    DeconvExecutor();
    virtual bool init(const DeconvAttrs& deconvAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) = 0;
    virtual ~DeconvExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

protected:
    DeconvAttrs deconvAttrs;
};

using DeconvExecutorPtr = std::shared_ptr<DeconvExecutor>;
using DeconvExecutorCPtr = std::shared_ptr<const DeconvExecutor>;

class DeconvExecutorBuilder {
public:
    ~DeconvExecutorBuilder() = default;
    virtual bool isSupported(const DeconvAttrs& convAttrs, const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual DeconvExecutorPtr makeExecutor() const = 0;
};

using DeconvExecutorBuilderPtr = std::shared_ptr<DeconvExecutorBuilder>;
using DeconvExecutorBuilderCPtr = std::shared_ptr<const DeconvExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov