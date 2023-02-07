// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reduce.hpp"

namespace ov {
namespace intel_cpu {

class RefReduceExecutor : public ReduceExecutor {
public:
    RefReduceExecutor(const ExecutorContext::CPtr context);

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
    void reduce_ref_process(const float *in_ptr, float *out_ptr, float init_value, std::function<float(float, float)> func);

    impl_desc_type implType = impl_desc_type::ref;
};

class RefReduceExecutorBuilder : public ReduceExecutorBuilder {
public:
    bool isSupported(const ReduceAttrs& reduceAttrs, const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }

    ReduceExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<RefReduceExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov