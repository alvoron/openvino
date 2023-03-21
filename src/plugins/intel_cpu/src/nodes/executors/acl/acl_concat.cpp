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
            std::cout << "AclConcatExecutor::init NECopy validate failed: " << s.error_description() << std::endl;
            return false;
        }
        std::cout << "AclConcatExecutor::init NECopy validate OK" << std::endl;

        srcTensor.allocator()->init(srcTensorInfo);
        dstTensor.allocator()->init(dstTensorInfo);

        concatNECopy = std::make_unique<arm_compute::NECopy>();
        concatNECopy->configure(&srcTensor, &dstTensor);
    } else {
        for (int i = 0; i < srcDescs.size(); i++) {
            auto srcDims = srcDescs[i]->getShape().getStaticDims();
            inputs_vector.push_back(new TensorInfo(shapeCast(srcDims), 1,
            precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0])));
        }
        axis = axisCast(concatAttrs.axis, srcDescs[0]->getShape().getStaticDims().size());

        std::cout << "Original axis value: " << concatAttrs.axis << std::endl;
        std::cout << "Casted axis: " << axis << std::endl;

        arm_compute::Status s = arm_compute::NEConcatenateLayer::validate(inputs_vector, &dstTensorInfo, axis);
        if (!s) {
            std::cout << "AclConcatExecutor::init NEConcatenateLayer validate failed: " << s.error_description() << std::endl;
            return false;
        }
        std::cout << "AclConcatExecutor::init NEConcatenateLayer validate OK" << std::endl;

        dstTensor.allocator()->init(dstTensorInfo);

        concatNEConcatenate = std::make_unique<arm_compute::NEConcatenateLayer>();
    }

    return true;
}

struct delete_ptr {
    template <typename P>
    void operator () (P p) {
        delete p;
    }
};

void AclConcatExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    std::cout << "AclConcatExecutor::exec" << std::endl;

    dstTensor.allocator()->import_memory(dst[0]->GetPtr());
    if (src.size() == 1) {
        std::cout << "concatNECopy" << std::endl;
        srcTensor.allocator()->import_memory(src[0]->GetPtr());
        concatNECopy->run();
        srcTensor.allocator()->free();
    } else {
        for (int i = 0; i < src.size(); i++) {
            const arm_compute::ITensorInfo *ti = inputs_vector[i];
            arm_compute::Tensor *tensor = new arm_compute::Tensor();
            tensor->allocator()->init(*ti);
            tensor->allocator()->import_memory(src[i]->GetPtr());
            srcTensorVectorConst.push_back(tensor);
        }
        std::cout << "concatNEConcatenate" << std::endl;
        concatNEConcatenate->configure(srcTensorVectorConst, &dstTensor, axis);
        concatNEConcatenate->run();

        std::for_each(srcTensorVectorConst.begin(), srcTensorVectorConst.end(), delete_ptr());
        std::for_each(inputs_vector.begin(), inputs_vector.end(), delete_ptr());
    }
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov
