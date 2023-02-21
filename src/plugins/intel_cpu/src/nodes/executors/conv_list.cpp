// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<ConvExecutorDesc>& getConvExecutorsList() {
    static std::vector<ConvExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclConvExecutorBuilder>())
        //OV_CPU_INSTANCE_DNNL(ExecutorType::Dnnl, std::make_shared<DnnlConvExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov