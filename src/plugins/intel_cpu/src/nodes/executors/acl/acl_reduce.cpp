// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_utils.hpp"
#include "acl_reduce.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

arm_compute::ReductionOperation getAclReductionOperationByAlgorithm(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::ReduceMax:  return arm_compute::ReductionOperation::MAX;
        case Algorithm::ReduceMin:  return arm_compute::ReductionOperation::MIN;
        case Algorithm::ReduceSum:  return arm_compute::ReductionOperation::SUM;
        case Algorithm::ReduceProd: return arm_compute::ReductionOperation::PROD;
        default:                    IE_THROW() << "Unsupported reduction operation";
    }
}

AclReduceExecutor::AclReduceExecutor(const ExecutorContext::CPtr context) : ReduceExecutor(context) {}

bool AclReduceExecutor::init(const ReduceAttrs& reduceAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    std::cout << "AclReduceExecutor::init" << std::endl;

    //ACL does not support more than 1 reduction axes
    if (reduceAttrs.axes.size() != 1)
        return false;

    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    auto M = srcDims[srcDims.size() - 2];
    auto K = srcDims[srcDims.size() - 1];

    TensorInfo srcTensorInfo = TensorInfo(TensorShape(K, M), 1, precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo dstTensorInfo = TensorInfo(TensorShape(K, M), 1, precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    unsigned int axis = srcDescs[0]->getShape().getRank() - reduceAttrs.axes[0] - 1;
    if (!arm_compute::NEReductionOperation::validate(&srcTensorInfo, &dstTensorInfo, axis,
                                                     getAclReductionOperationByAlgorithm(reduceAttrs.operation), reduceAttrs.keepDims))
        return false;

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    reduce = std::make_unique<arm_compute::NEReductionOperation>();
    reduce->configure(&srcTensor, &dstTensor, reduceAttrs.axes[0], getAclReductionOperationByAlgorithm(reduceAttrs.operation), reduceAttrs.keepDims);

    return true;
}

void AclReduceExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    std::cout << "AclReduceExecutor::exec" << std::endl;
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    reduce->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov