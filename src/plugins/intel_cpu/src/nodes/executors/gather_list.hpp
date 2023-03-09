// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "gather.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_gather.hpp"
#endif

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct GatherExecutorDesc {
    ExecutorType executorType;
    GatherExecutorBuilderCPtr builder;
};

const std::vector<GatherExecutorDesc>& getGatherExecutorsList();

class GatherExecutorFactory : public ExecutorFactory {
public:
    GatherExecutorFactory(const GatherAttrs& gatherAttrs,
                       const std::vector<MemoryDescPtr>& srcDescs,
                       const std::vector<MemoryDescPtr>& dstDescs,
                       const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getGatherExecutorsList()) {
            if (desc.builder->isSupported(gatherAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~GatherExecutorFactory() = default;
    virtual GatherExecutorPtr makeExecutor(const GatherAttrs& gatherAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const dnnl::primitive_attr &attr) {
        auto build = [&](const GatherExecutorDesc* desc) {
            switch (desc->executorType) {
                default: {
                    auto executor = desc->builder->makeExecutor(context);
                    if (executor->init(gatherAttrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            GatherExecutorPtr ptr = nullptr;
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
    std::vector<GatherExecutorDesc> supportedDescs;
    const GatherExecutorDesc* chosenDesc = nullptr;
};

using GatherExecutorFactoryPtr = std::shared_ptr<GatherExecutorFactory>;
using GatherExecutorFactoryCPtr = std::shared_ptr<const GatherExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov