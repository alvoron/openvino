// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_concat.hpp"
#include "acl_utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclConcatExecutor::AclConcatExecutor(const ExecutorContext::CPtr context) : ConcatExecutor(context) {}

bool AclConcatExecutor::init(const ConcatAttrs& concatAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             const dnnl::primitive_attr &attr) {
    auto dstDims = dstDescs[0]->getShape().getStaticDims();
    TensorInfo dstTensorInfo = TensorInfo(shapeCast(dstDims), 1,
        precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    if (srcDescs.size() == 1) {
        auto srcDims = srcDescs[0]->getShape().getStaticDims();
        TensorInfo srcTensorInfo = TensorInfo(shapeCast(srcDims), 1,
        precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));

        arm_compute::Status s = arm_compute::NECopy::validate(&srcTensorInfo, &dstTensorInfo);
        if (!s) {
            std::cout << "AclConcatExecutor::init validate failed: " << s.error_description() << std::endl;
            return false;
        }
        std::cout << "AclConcatExecutor::init validate OK" << std::endl;

        srcTensor.allocator()->init(srcTensorInfo);
        dstTensor.allocator()->init(dstTensorInfo);

        concatNECopy = std::make_unique<arm_compute::NECopy>();
        concatNECopy->configure(&srcTensor, &dstTensor);
    } else {
        //std::vector<const ITensorInfo*> inputs_vector;
        for (int i = 0; i < srcDescs.size(); i++) {
            auto srcDims = srcDescs[i]->getShape().getStaticDims();
            TensorInfo srcTensorInfo = TensorInfo(shapeCast(srcDims), 1,
            precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
            inputs_vector.push_back(&srcTensorInfo);
        }
        int64_t axis = concatAttrs.axis;
        //if (axis < 0)
        arm_compute::Status s = arm_compute::NEConcatenateLayer::validate(inputs_vector, &dstTensorInfo,
         axisCast(axis, srcDescs[0]->getShape().getStaticDims().size()));
        if (!s) {
            std::cout << "AclConcatExecutor::init validate failed: " << s.error_description() << std::endl;
            return false;
        }
        std::cout << "AclConcatExecutor::init validate OK" << std::endl;

        /*for (int i = 0; i < srcDescs.size(); i++) {
            arm_compute::Tensor tensor;
            tensor.allocator()->init(*inputs_vector[i]);
            srcTensorVectorConst.push_back(&tensor);
            srcTensorVector.push_back(&tensor);
        }*/
        dstTensor.allocator()->init(dstTensorInfo);

        concatNEConcatenate = std::make_unique<arm_compute::NEConcatenateLayer>();
        axis = axisCast(axis, srcDescs[0]->getShape().getStaticDims().size());
        //concatNEConcatenate->configure(srcTensorVectorConst, &dstTensor, axisCast(axis, srcDescs[0]->getShape().getStaticDims().size()));
    }

    return true;
}

void AclConcatExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    std::cout << "AclConcatExecutor::exec" << std::endl;

    dstTensor.allocator()->import_memory(dst[0]->GetPtr());
    if (src.size() == 1) {
        srcTensor.allocator()->import_memory(src[0]->GetPtr());
        concatNECopy->run();
        srcTensor.allocator()->free();
    } else {
        for (int i = 0; i < src.size(); i++) {
            const arm_compute::ITensorInfo *ti = inputs_vector[i];
            arm_compute::Tensor tensor;
            tensor.allocator()->init(*ti);
            tensor.allocator()->import_memory(src[i]->GetPtr());
            srcTensorVectorConst.push_back(&tensor);
            //srcTensorVector.push_back(&tensor);
        }
        concatNEConcatenate->configure(srcTensorVectorConst, &dstTensor, axis);
        concatNEConcatenate->run();
        /*for (int i = 0; i < src.size(); i++) {
            arm_compute::ITensor* tensor = srcTensorVectorConst[i];
            tensor.allocator()->free();
        }*/
        /*for (int i = 0; i < src.size(); i++) {
            const arm_compute::ITensor* t = srcTensorVectorConst[i];
            dynamic_cast<arm_compute::Tensor*>(t)->allocator()->import_memory(src[i]->GetPtr());
        }
        concatNEConcatenate->run();
        for (int i = 0; i < src.size(); i++) {
            arm_compute::ITensor* t = srcTensorVector[i];
            dynamic_cast<arm_compute::Tensor*>(t)->allocator()->free();
        }*/
    }
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov
