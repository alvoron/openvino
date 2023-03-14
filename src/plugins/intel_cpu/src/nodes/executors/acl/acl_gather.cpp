// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_utils.hpp"
#include "acl_gather.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclGatherExecutor::AclGatherExecutor(const ExecutorContext::CPtr context) : GatherExecutor(context) {}

bool AclGatherExecutor::init(const GatherAttrs& gatherAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    std::cout << "AclGatherExecutor::init" << std::endl;

    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto indDims = srcDescs[1]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    TensorInfo srcTensorInfo = TensorInfo(shapeCast(srcDims), 1,
    precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo indTensorInfo = TensorInfo(shapeCast(indDims), 1,
    precisionToAclDataType(srcDescs[1]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[1]));
    TensorInfo dstTensorInfo = TensorInfo(shapeCast(dstDims), 1,
    precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    auto axis = axisCast(gatherAttrs.axis, srcDims.size());
    std::cout << "axis: " << axis << std::endl;
    Status s = NEGather::validate(&srcTensorInfo, &indTensorInfo, &dstTensorInfo, axis);
    if (!s) {
        DEBUG_LOG("NEGather validation failed: ", s.error_description());
        return false;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    indTensor.allocator()->init(indTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    gather = std::make_unique<NEGather>();
    gather->configure(&srcTensor, &indTensor, &dstTensor, axis);

    return true;
}

void AclGatherExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    std::cout << "AclGatherExecutor::exec" << std::endl;
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    indTensor.allocator()->import_memory(src[1]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    gather->run();

    srcTensor.allocator()->free();
    indTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov