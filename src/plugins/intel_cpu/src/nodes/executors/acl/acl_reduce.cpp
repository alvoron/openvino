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
        default:                    IE_THROW() << "Unsupported reduction operation: " << static_cast<int>(algorithm);
    }
}

AclReduceExecutor::AclReduceExecutor(const ExecutorContext::CPtr context) : ReduceExecutor(context) {}

bool AclReduceExecutor::init(const ReduceAttrs& reduceAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    if (reduceAttrs.operation != Algorithm::ReduceMax &&
        reduceAttrs.operation != Algorithm::ReduceMin &&
        reduceAttrs.operation != Algorithm::ReduceSum &&
        reduceAttrs.operation != Algorithm::ReduceProd &&
        reduceAttrs.operation != Algorithm::ReduceMean) {
            return false;
        }

    this->reduceAttrs = reduceAttrs;

    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    TensorInfo srcTensorInfo = TensorInfo(shapeCast(srcDims), 1,
    precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo dstTensorInfo = TensorInfo(shapeCast(dstDims), 1,
    precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    arm_compute::Coordinates axesMean;
    if (reduceAttrs.operation == Algorithm::ReduceMean) {
        auto srcDims1 = srcDescs[1]->getShape().getStaticDims();
        for (size_t i = 0; i < reduceAttrs.axes.size(); ++i) {
            auto pos = axisCast(i, reduceAttrs.axes.size());
            axesMean.set(pos, reduceAttrs.axes[i]);
        }
        if (!arm_compute::NEReduceMean::validate(&srcTensorInfo, axesMean, reduceAttrs.keepDims, &dstTensorInfo)) {
            return false;
        }
    } else {
        if (reduceAttrs.axes.size() != 1)
            return false;
        if (!arm_compute::NEReductionOperation::validate(&srcTensorInfo, &dstTensorInfo, axisCast(reduceAttrs.axes[0], srcDims.size()),
                                                         getAclReductionOperationByAlgorithm(reduceAttrs.operation), reduceAttrs.keepDims)) {
            return false;
        }
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    if (reduceAttrs.operation == Algorithm::ReduceMean) {
        reduceMean = std::make_unique<arm_compute::NEReduceMean>();
        reduceMean->configure(&srcTensor, axesMean, reduceAttrs.keepDims, &dstTensor);
    } else {
        reduce = std::make_unique<arm_compute::NEReductionOperation>();
        reduce->configure(&srcTensor, &dstTensor, axisCast(reduceAttrs.axes[0], srcDims.size()),
                          getAclReductionOperationByAlgorithm(reduceAttrs.operation), reduceAttrs.keepDims);
    }

    return true;
}

void AclReduceExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    if (this->reduceAttrs.operation == Algorithm::ReduceMean) {
        reduceMean->run();
    } else {
        reduce->run();
    }

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov