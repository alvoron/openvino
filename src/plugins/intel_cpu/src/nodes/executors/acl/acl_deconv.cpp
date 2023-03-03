// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_deconv.hpp"
#include "acl_utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclDeconvExecutor::AclDeconvExecutor() : DeconvExecutor() {}

bool AclDeconvExecutor::init(const DeconvAttrs& deconvAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    std::cout << "AclDeconvExecutor::init" << std::endl;
    this->deconvAttrs = deconvAttrs;
    //deconvAttrs.withBias = (srcDescs.size() == 3);
    auto srcDims  = srcDescs[0]->getShape().getStaticDims();
    auto weiDims  = srcDescs[1]->getShape().getStaticDims();
    auto dstDims  = dstDescs[0]->getShape().getStaticDims();

    VectorDims biasDims;
    TensorInfo biasTensorInfo;
    if (deconvAttrs.withBiases) {
        std::cout << "BIAS mode" << std::endl;
        biasDims = srcDescs[2]->getShape().getStaticDims();
        biasTensorInfo = TensorInfo(shapeCast(biasDims), 1,
        precisionToAclDataType(srcDescs[2]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[2]));
    } else {
        std::cout << "non-BIAS mode" << std::endl;
    }

    //it's a wierd WA to pass ACL check NEDeconvolutionLayer.cpp:127: Output's depth is invalid
    dstDims[1] *= dstDims[0];
    dstDims[0] = 1;
    TensorInfo srcTensorInfo = TensorInfo(shapeCast(srcDims), 1,
    precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo weiTensorInfo = TensorInfo(shapeCast(weiDims), 1,
    precisionToAclDataType(srcDescs[1]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[1]));
    TensorInfo dstTensorInfo = TensorInfo(shapeCast(dstDims), 1,
    precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    unsigned int pad_l = deconvAttrs.paddingL.at(1);
    unsigned int pad_r = deconvAttrs.paddingR.at(1);
    unsigned int pad_t = deconvAttrs.paddingL.at(0);
    unsigned int pad_b = deconvAttrs.paddingR.at(0);
    unsigned int stride_x = deconvAttrs.stride.at(1);
    unsigned int stride_y = deconvAttrs.stride.at(0);
    unsigned int dilation_x = deconvAttrs.dilation.at(1) + 1;
    unsigned int dilation_y = deconvAttrs.dilation.at(0) + 1;

    std::cout << "pad (l,r): " << pad_l << " " << pad_r << std::endl;
    std::cout << "pad (t,b): " << pad_t << " " << pad_b << std::endl;
    std::cout << "stride (x,y): " << stride_x << " " << stride_y << std::endl;
    std::cout << "dilation (x,y): " << dilation_x << " " << dilation_y << std::endl;

    arm_compute::PadStrideInfo deconv_info(stride_x, stride_y, pad_l, pad_r, pad_t, pad_b, arm_compute::DimensionRoundingType::FLOOR);
    arm_compute::Size2D dilation(dilation_x, dilation_y);

    arm_compute::Status status = arm_compute::NEDeconvolutionLayer::validate(&srcTensorInfo,
                                                                           &weiTensorInfo,
                                                                           deconvAttrs.withBiases ? &biasTensorInfo : nullptr,
                                                                           &dstTensorInfo,
                                                                           deconv_info);
    if (!status) {
        std::cout << "AclDeconvExecutor::init validate failed: " << status.error_description() << std::endl;
        return false;
    } else {
        std::cout << "AclDeconvExecutor::init validate OK" << std::endl;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    weiTensor.allocator()->init(weiTensorInfo);
    if (deconvAttrs.withBiases) biasTensor.allocator()->init(biasTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    deconv = std::make_unique<arm_compute::NEDeconvolutionLayer>();
    deconv->configure(&srcTensor, &weiTensor, deconvAttrs.withBiases ? &biasTensor : nullptr, &dstTensor, deconv_info);

    return true;
}

void AclDeconvExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    std::cout << "AclDeconvExecutor::exec" << std::endl;
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    weiTensor.allocator()->import_memory(src[1]->GetPtr());
    if (deconvAttrs.withBiases) biasTensor.allocator()->import_memory(src[2]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    deconv->run();

    srcTensor.allocator()->free();
    weiTensor.allocator()->free();
    if (deconvAttrs.withBiases) biasTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov