// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_utils.hpp"
#include "acl_roi_align.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclROIAlignExecutor::AclROIAlignExecutor(const ExecutorContext::CPtr context) : ROIAlignExecutor(context) {}

bool AclROIAlignExecutor::init(const ROIAlignAttrs& roialignAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    std::cout << "AclROIAlignExecutor::init" << std::endl;

    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto roiDims = srcDescs[1]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();
    std::cout << "roiDims: "; for (size_t i : roiDims) std::cout << i << " "; std::cout << std::endl;

    TensorInfo srcTensorInfo = TensorInfo(shapeCast(srcDims), 1,
    precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo roiTensorInfo = TensorInfo(shapeCast(roiDims), 1,
    precisionToAclDataType(srcDescs[1]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[1]));
    TensorInfo dstTensorInfo = TensorInfo(shapeCast(dstDims), 1,
    precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    ROIPoolingLayerInfo poolInfo(roialignAttrs.pooledW, roialignAttrs.pooledH, roialignAttrs.spatialScale, roialignAttrs.samplingRatio);

    Status s = arm_compute::NEROIAlignLayer::validate(&srcTensorInfo, &roiTensorInfo, &dstTensorInfo, poolInfo);
    if (!s) {
        std::cout << "AclROIAlignExecutor::init failed: " << s.error_description() << std::endl;
        return false;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    roiTensor.allocator()->init(roiTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    roialign = std::make_unique<arm_compute::NEROIAlignLayer>();
    roialign->configure(&srcTensor, &roiTensor, &dstTensor, poolInfo);

    return true;
}

void AclROIAlignExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    std::cout << "AclROIAlignExecutor::exec" << std::endl;
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    roiTensor.allocator()->import_memory(src[1]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    roialign->run();

    srcTensor.allocator()->free();
    roiTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov