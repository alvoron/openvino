// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<GatherExecutorDesc>& getGatherExecutorsList() {
    static std::vector<GatherExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclGatherExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov