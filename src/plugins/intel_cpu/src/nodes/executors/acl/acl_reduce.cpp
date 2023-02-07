// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_reduce.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclReduceExecutor::AclReduceExecutor(const ExecutorContext::CPtr context) : ReduceExecutor(context) {}

bool AclReduceExecutor::init(const ReduceAttrs& reduceAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    std::cout << "AclReduceExecutor::init" << std::endl;
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    auto M = srcDims[srcDims.size() - 2];
    auto K = srcDims[srcDims.size() - 1];

    TensorInfo srcTensorInfo = TensorInfo(TensorShape(K, M), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo dstTensorInfo = TensorInfo(TensorShape(K, M), 1, DataType::F32, DataLayout::NCHW);

    if (!arm_compute::NEReductionOperation::validate(&srcTensorInfo, &dstTensorInfo, reduceAttrs.axis, aclReduceOp[reduceAttrs.operation]))
        return false;

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    reduce = std::make_unique<arm_compute::NEReductionOperation>();
    reduce->configure(&srcTensor, &dstTensor, reduceAttrs.axis, aclReduceOp[reduceAttrs.operation]);

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