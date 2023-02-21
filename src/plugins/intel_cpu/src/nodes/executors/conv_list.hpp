// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "conv.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_conv.hpp"
#endif

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct ConvExecutorDesc {
    ExecutorType executorType;
    ConvExecutorBuilderCPtr builder;
};

const std::vector<ConvExecutorDesc>& getConvExecutorsList();

class ConvExecutorFactory : public ExecutorFactory {
public:
    ConvExecutorFactory(const ConvAttrs& convAttrs,
                       const std::vector<MemoryDescPtr>& srcDescs,
                       const std::vector<MemoryDescPtr>& dstDescs,
                       const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getConvExecutorsList()) {
            if (desc.builder->isSupported(convAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~ConvExecutorFactory() = default;
    virtual ConvExecutorPtr makeExecutor(const ConvAttrs& convAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const dnnl::primitive_attr &attr) {
        auto build = [&](const ConvExecutorDesc* desc) {
            switch (desc->executorType) {
#if defined(OPENVINO_ARCH_X86_64)
                case ExecutorType::x64: {
                    auto builder = [&](const DnnlConvExecutor::Key& key) -> ConvExecutorPtr {
                        auto executor = desc->builder->makeExecutor();
                        if (executor->init(convAttrs, srcDescs, dstDescs, attr)) {
                            return executor;
                        } else {
                            return nullptr;
                        }
                    };

                    auto key = DnnlConvExecutor::Key(convAttrs, srcDescs, dstDescs, attr);
                    auto res = runtimeCache->getOrCreate(key, builder);
                    return res.first;
                } break;
#endif
                default: {
                    auto executor = desc->builder->makeExecutor();
                    if (executor->init(convAttrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            ConvExecutorPtr ptr = nullptr;
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

private:
    std::vector<ConvExecutorDesc> supportedDescs;
    const ConvExecutorDesc* chosenDesc = nullptr;
};

using ConvExecutorFactoryPtr = std::shared_ptr<ConvExecutorFactory>;
using ConvExecutorFactoryCPtr = std::shared_ptr<const ConvExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov