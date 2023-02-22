// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_utils.hpp"
#include "acl_conv.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclConvExecutor::AclConvExecutor() : ConvExecutor() {}

bool AclConvExecutor::init(const ConvAttrs& convAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    std::cout << "AclConvExecutor::init" << std::endl;
    this->convAttrs = convAttrs;
    //convAttrs.withBias = (srcDescs.size() == 3);
    auto srcDims  = srcDescs[0]->getShape().getStaticDims();
    auto weiDims  = srcDescs[1]->getShape().getStaticDims();
    auto dstDims  = dstDescs[0]->getShape().getStaticDims();

    VectorDims biasDims;
    TensorInfo biasTensorInfo;
    if (convAttrs.withBiases) {
        std::cout << "BIAS mode" << std::endl;
        biasDims = srcDescs[2]->getShape().getStaticDims();
        biasTensorInfo = TensorInfo(shapeCast(biasDims), 1,
        precisionToAclDataType(srcDescs[2]->getPrecision()), /*getAclDataLayoutByMemoryDesc(srcDescs[2])*/arm_compute::DataLayout::NCHW);
    } else {
        std::cout << "non-BIAS mode" << std::endl;
    }

    TensorInfo srcTensorInfo = TensorInfo(shapeCast(srcDims), 1,
    precisionToAclDataType(srcDescs[0]->getPrecision()), /*getAclDataLayoutByMemoryDesc(srcDescs[0])*/arm_compute::DataLayout::NCHW);
    TensorInfo weiTensorInfo = TensorInfo(shapeCast(weiDims), 1,
    precisionToAclDataType(srcDescs[1]->getPrecision()), /*getAclDataLayoutByMemoryDesc(srcDescs[1])*/arm_compute::DataLayout::NCHW);
    TensorInfo dstTensorInfo = TensorInfo(shapeCast(dstDims), 1,
    precisionToAclDataType(dstDescs[0]->getPrecision()), /*getAclDataLayoutByMemoryDesc(dstDescs[0])*/arm_compute::DataLayout::NCHW);

//TODO: understand why padding, stride and dilation vectors may contain only 1 element
    unsigned int pad_l = (convAttrs.paddingL.size() == 1) ? convAttrs.paddingL.at(0) : convAttrs.paddingL.at(1);//convAttrs.paddingL.at(1);
    unsigned int pad_r = (convAttrs.paddingR.size() == 1) ? convAttrs.paddingR.at(0) : convAttrs.paddingR.at(1);//convAttrs.paddingR.at(1);
    unsigned int pad_t = convAttrs.paddingL.at(0);
    unsigned int pad_b = convAttrs.paddingR.at(0);
    unsigned int stride_x = (convAttrs.stride.size() == 1) ? convAttrs.stride.at(0) : convAttrs.stride.at(1);//convAttrs.stride.at(1);
    unsigned int stride_y = convAttrs.stride.at(0);

    arm_compute::PadStrideInfo conv_info(stride_x, stride_y, pad_l, pad_r, pad_t, pad_b, arm_compute::DimensionRoundingType::FLOOR);
    arm_compute::Size2D dilation((convAttrs.dilation.size() == 1) ? convAttrs.dilation.at(0) : convAttrs.dilation.at(1)/*convAttrs.dilation.at(1)*/,
                                                                    convAttrs.dilation.at(0));

    arm_compute::Status status = arm_compute::NEConvolutionLayer::validate(&srcTensorInfo,
                                                                           &weiTensorInfo,
                                                                           convAttrs.withBiases ? &biasTensorInfo : nullptr,
                                                                           &dstTensorInfo,
                                                                           conv_info,
                                                                           arm_compute::WeightsInfo{},
                                                                           dilation);
    if (!status) {
        std::cout << "AclConvExecutor::init validate failed: " << status.error_description() << std::endl;
        return false;
    } else {
        std::cout << "AclConvExecutor::init validate OK" << std::endl;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    weiTensor.allocator()->init(weiTensorInfo);
    if (convAttrs.withBiases) biasTensor.allocator()->init(biasTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    conv = std::make_unique<arm_compute::NEConvolutionLayer>();
    conv->configure(&srcTensor, &weiTensor, convAttrs.withBiases ? &biasTensor : nullptr, &dstTensor, conv_info, arm_compute::WeightsInfo{}, dilation);

    return true;
}

void AclConvExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    std::cout << "AclConvExecutor::exec" << std::endl;
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    weiTensor.allocator()->import_memory(src[1]->GetPtr());
    if (convAttrs.withBiases) biasTensor.allocator()->import_memory(src[2]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    conv->run();

    srcTensor.allocator()->free();
    weiTensor.allocator()->free();
    if (convAttrs.withBiases) biasTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov
