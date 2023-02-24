// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concat_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<ConcatExecutorDesc>& getConcatExecutorsList() {
    static std::vector<ConcatExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclConcatExecutorBuilder>())
        //OV_CPU_INSTANCE_DNNL(ExecutorType::Dnnl, std::make_shared<DnnlMatMulExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov