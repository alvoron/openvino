// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_mvn.hpp"
#include "utils/acl_utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclMVNExecutor::AclMVNExecutor() : MVNExecutor() {}

bool AclMVNExecutor::init(const MVNAttrs& mvnAttrs,
                          const std::vector<MemoryDescCPtr>& srcDescs,
                          const std::vector<MemoryDescCPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    size_t X, Y;
    if (getAclDataLayoutByMemoryDesc(srcDescs[0]) == arm_compute::DataLayout::NCHW) {
        if (mvnAttrs.initAcrossChannels_) {
            Y = srcDims[srcDims.size() - 4]; // Y = N
            X = srcDims[srcDims.size() - 3] * srcDims[srcDims.size() - 2] * srcDims[srcDims.size() - 1]; // X = CHW
        } else {
            Y = srcDims[srcDims.size() - 4] * srcDims[srcDims.size() - 3]; // Y = NC
            X = srcDims[srcDims.size() - 2] * srcDims[srcDims.size() - 1]; // X = HW
        }
    } else if (getAclDataLayoutByMemoryDesc(srcDescs[0]) == arm_compute::DataLayout::NHWC) {
        if (mvnAttrs.initAcrossChannels_) {
            Y = srcDims[srcDims.size() - 4]; // Y = N
            X = srcDims[srcDims.size() - 3] * srcDims[srcDims.size() - 2] * srcDims[srcDims.size() - 1]; // X = HWC
        } else {
            return false;
        }
    } else return false; // TODO: support other ranks

    TensorInfo srcTensorInfo = TensorInfo(TensorShape(Y, X), 1, precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo dstTensorInfo = TensorInfo(TensorShape(Y, X), 1, precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    /*auto M = srcDims[srcDims.size() - 2];
    auto K = srcDims[srcDims.size() - 1];

    TensorInfo srcTensorInfo = TensorInfo(TensorShape(K, M), 1, precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo dstTensorInfo = TensorInfo(TensorShape(K, M), 1, precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));*/

    if (!arm_compute::NEMeanStdDevNormalizationLayer::validate(&srcTensorInfo, &dstTensorInfo, mvnAttrs.epsValue_))
        return false;

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    mvn = std::make_unique<arm_compute::NEMeanStdDevNormalizationLayer>();
    mvn->configure(&srcTensor, &dstTensor, mvnAttrs.epsValue_);

    return true;
}

void AclMVNExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());
    std::cout << "AclMVNExecutor::exec" << std::endl;
    mvn->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov
