// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reduce.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace ov {
namespace intel_cpu {

enum ReduceLayoutType {
    reduce_ncsp,
    reduce_nspc,
    reduce_blocked
};

struct jit_reduce_config_params {
    ReduceLayoutType layout;
    Algorithm reduce_mode;
    dnnl::memory::data_type src_dt;
    dnnl::memory::data_type dst_dt;
    int src_data_size;
    int dst_data_size;
};

struct jit_reduce_call_args {
    const void *src;
    const int *idx;
    void *dst;
    size_t work_amount;
    size_t work_batch;
    size_t reduce_w = 2;    // only used in planar layout  [1: reduce width dimension]   [0: reduce other dimension] [other value: N/A]
    size_t reduce_stride;   // only used in planar layout while reducing dimensions except for width
};

struct jit_reduce_post_call_args {
    const void *src;
    void *dst;
    size_t work_amount;
    size_t reduce_c = 2;    // only used in blocked layout [1: reduce channel dimension] [0: reduce other dimension] [other value: N/A]
    size_t oc_off;          // offset in byte along channel on output tensor
    size_t channel_size;    // only for post ops fusion of nspc layout
    const float *divisor;   // mean = sum / divisor
    const void** post_op_data;
};

struct jit_uni_reduce_kernel {
    void (*ker_)(const jit_reduce_call_args *);

    void operator()(const jit_reduce_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_reduce_kernel(jit_reduce_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_reduce_kernel() {}

    virtual void create_ker() = 0;

    jit_reduce_config_params jcp_;
};

struct jit_uni_reduce_post_kernel {
    void (*ker_)(const jit_reduce_post_call_args *);

    void operator()(const jit_reduce_post_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_reduce_post_kernel(jit_reduce_config_params jcp, const dnnl_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_reduce_post_kernel() {}

    virtual void create_ker() = 0;

    jit_reduce_config_params jcp_;
    const dnnl_primitive_attr &attr_;
};

class JitReduceExecutor : public ReduceExecutor {
public:
    JitReduceExecutor();

    bool init(const ReduceAttrs& reduceAttrs,
              const std::vector<MemoryDescCPtr>& srcDescs,
              const std::vector<MemoryDescCPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) override;

    void reduce_type(const uint8_t *in_ptr, uint8_t *out_ptr, size_t dst_size);
    void reduce_PLN(const uint8_t *in_ptr, uint8_t *out_ptr);
    void reduce_BLK(const uint8_t *in_ptr, uint8_t *out_ptr);
    void reduce_BLK_concern_padding(const uint8_t *in_ptr, uint8_t *out_ptr);

    impl_desc_type getImplType() const override {
        return implType;
    }

    struct Key {
        jit_reduce_config_params jcp;
        dnnl::post_ops postOps;

        size_t hash() const;
        bool operator==(const Key& rhs) const;
    };

private:
    std::shared_ptr<jit_uni_reduce_kernel_f32> reduce_kernel_f32;
    std::shared_ptr<jit_uni_reduce_post_kernel_f32> reduce_post_kernel_f32;
    std::shared_ptr<jit_uni_reduce_kernel> reduce_kernel;
    std::shared_ptr<jit_uni_reduce_post_kernel> reduce_post_kernel;

    jit_reduce_config_params jcp;

    impl_desc_type implType = impl_desc_type::jit_uni;

    InferenceEngine::SizeVector src_dims;

    size_t reduce_stride;

    bool jit_beyond_5D = false;
    bool is_hybrid_layout = false;
    bool keep_dims = true;
    bool compile_post_kernel = true;

    ReduceLayoutType layout;
};

class JitReduceExecutorBuilder : public ReduceExecutorBuilder {
public:
    bool isSupported(const ReduceAttrs& reduceAttrs, 
                     const std::vector<MemoryDescCPtr>& srcDescs, 
                     const std::vector<MemoryDescCPtr>& dstDescs) const override {
        //TODO: implement
        return false;
    }

    MVNExecutorPtr makeExecutor() const override {
        return std::make_shared<JitReduceExecutor>();
    }
};

}   // namespace intel_cpu
}   // namespace ov