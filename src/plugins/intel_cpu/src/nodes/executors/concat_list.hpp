// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "concat.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_concat.hpp"
#endif

//#include "dnnl/dnnl_concat.hpp"

namespace ov {
namespace intel_cpu {

struct ConcatExecutorDesc {
    ExecutorType executorType;
    ConcatExecutorBuilderCPtr builder;
};

const std::vector<ConcatExecutorDesc>& getConcatExecutorsList();

class ConcatExecutorFactory : public ExecutorFactory {
public:
    ConcatExecutorFactory(const ConcatAttrs& ConcatAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          /*const dnnl::primitive_attr &attr,*/
                          const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getConcatExecutorsList()) {
            if (desc.builder->isSupported(ConcatAttrs, srcDescs, dstDescs/*, attr*/)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~ConcatExecutorFactory() = default;
    virtual ConcatExecutorPtr makeExecutor(const ConcatAttrs& ConcatAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const dnnl::primitive_attr &attr) {
        auto build = [&](const ConcatExecutorDesc* desc) {
            switch (desc->executorType) {
#if defined(OPENVINO_ARCH_X86_64)
                case ExecutorType::x64: {
                    auto builder = [&](const DnnlConcatExecutor::Key& key) -> ConcatExecutorPtr {
                        auto executor = desc->builder->makeExecutor(context);
                        if (executor->init(ConcatAttrs, srcDescs, dstDescs, attr)) {
                            return executor;
                        } else {
                            return nullptr;
                        }
                    };

                    auto key = DnnlConcatExecutor::Key(ConcatAttrs, srcDescs, dstDescs, attr);
                    auto res = context->getRuntimeCache().lock()->getOrCreate(key, builder);
                    return res.first;
                } break;
#endif
                default: {
                    auto executor = desc->builder->makeExecutor(context);

                    if (executor->init(ConcatAttrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            ConcatExecutorPtr ptr = nullptr;
            return ptr;
        };

        if (chosenDesc) {
            if (auto executor = build(chosenDesc)) {
                return executor;
            }
        }

        for (const auto& sd : supportedDescs) {
            if (auto executor = build(&sd)) {
                chosenDesc = &sd;
                return executor;
            }
        }

        IE_THROW() << "Supported executor is not found";
    }

    void setEngine(const dnnl::engine& engine) {
        this->engine = engine;
    }

    void setScratchPad(const DnnlScratchPadPtr& scratchPad) {
        this->scratchPad = scratchPad;
    }

private:
    // TODO: remove dnnl dependency
    dnnl::engine engine;

    DnnlScratchPadPtr scratchPad = nullptr;

    std::vector<ConcatExecutorDesc> supportedDescs;
    const ConcatExecutorDesc* chosenDesc = nullptr;
};

using ConcatExecutorFactoryPtr = std::shared_ptr<ConcatExecutorFactory>;
using ConcatExecutorFactoryCPtr = std::shared_ptr<const ConcatExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov