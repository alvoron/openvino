// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_mvn.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclMVNExecutor::AclMVNExecutor() : MVNExecutor() {}

bool AclMVNExecutor::init(const MVNAttrs& mvnAttrs,
                          const std::vector<MemoryDescCPtr>& srcDescs,
                          const std::vector<MemoryDescCPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    std::cout << "AclMVNExecutor::init" << std::endl;
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    auto M = srcDims[srcDims.size() - 2];
    auto K = srcDims[srcDims.size() - 1];

    TensorInfo srcTensorInfo = TensorInfo(TensorShape(K, M), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo dstTensorInfo = TensorInfo(TensorShape(K, M), 1, DataType::F32, DataLayout::NCHW);

    if (!arm_compute::NEMeanStdDevNormalizationLayer::validate(&srcTensorInfo, &dstTensorInfo, mvnAttrs.epsValue_))
        return false;

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    mvn = std::make_unique<arm_compute::NEMeanStdDevNormalizationLayer>();
    mvn->configure(&srcTensor, &dstTensor, mvnAttrs.epsValue_);

    return true;
}

void AclMVNExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    std::cout << "AclMVNExecutor::exec" << std::endl;
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    mvn->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov
