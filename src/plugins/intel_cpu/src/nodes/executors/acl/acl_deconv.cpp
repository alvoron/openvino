// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_deconv.hpp"
#include "acl_utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

//FIXME: add context
AclDeconvExecutor::AclDeconvExecutor() : DeconvExecutor() {}

bool AclDeconvExecutor::init(const DeconvAttrs& deconvAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    std::cout << "AclDeconvExecutor::init" << std::endl;
    this->deconvAttrs = deconvAttrs;
    //this->deconvAttrs.withBiases = (srcDescs.size() == 3);

    auto srcDims  = srcDescs[0]->getShape().getStaticDims();
    auto weiDims  = srcDescs[1]->getShape().getStaticDims();
    //swap input and output channels dimensions to be align with ACL
    //weights tensor shape is changed because ACL expects [W, H, I, O] tensor while OV uses [I, O, H, W] tensor
    std::swap(weiDims[0], weiDims[1]);
    auto dstDims  = dstDescs[0]->getShape().getStaticDims();

    VectorDims biasDims;
    TensorInfo biasTensorInfo;
    if (deconvAttrs.withBiases) {
        std::cout << "with BIAS mode" << std::endl;
        biasDims = srcDescs[2]->getShape().getStaticDims();
        //bias presicion is I32 but ACL requests bias precision as input ones
        biasTensorInfo = TensorInfo(shapeCast(biasDims), 1,
        precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[2]));
    } else {
        std::cout << "non-BIAS mode" << std::endl;
    }

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
    dstTensor.allocator()->init(dstTensorInfo);
    if (deconvAttrs.withBiases)
        biasTensor.allocator()->init(biasTensorInfo);

    deconv = std::make_unique<arm_compute::NEDeconvolutionLayer>();
    deconv->configure(&srcTensor, &weiTensor, deconvAttrs.withBiases ? &biasTensor : nullptr, &dstTensor, deconv_info);

    return true;
}

static void transpose_to_1023(const MemoryCPtr& srcMemPtr, std::vector<float>& dst_data) {
    const auto src_data = reinterpret_cast<float*>(srcMemPtr->GetPtr());

    const int DIM0 = srcMemPtr->getStaticDims()[0];
    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];

    for (int dim0 = 0; dim0 < DIM0; ++dim0)
        for (int dim1 = 0; dim1 < DIM1; ++dim1)
            for (int dim2 = 0; dim2 < DIM2; ++dim2)
                for (int dim3 = 0; dim3 < DIM3; ++dim3) {
                    const int src_off = dim0 * DIM1 * DIM2 * DIM3 +
                                        dim1 * DIM2 * DIM3 +
                                        dim2 * DIM3 +
                                        dim3;
                    const int dst_off = dim1 * DIM0 * DIM2 * DIM3 +
                                        dim0 * DIM2 * DIM3 +
                                        dim2 * DIM3 +
                                        dim3;

                    dst_data[dst_off] = src_data[src_off];
                }
}

void AclDeconvExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    std::cout << "AclDeconvExecutor::exec" << std::endl;

    //weights tensor shape is changed because ACL expects [W, H, I, O] tensor while OV uses [I, O, H, W] tensor
    std::vector<float> weiBuffer(src[1]->getStaticDims()[0] *
                                 src[1]->getStaticDims()[1] *
                                 src[1]->getStaticDims()[2] *
                                 src[1]->getStaticDims()[3]);
    transpose_to_1023(src[1], weiBuffer);

    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());
    weiTensor.allocator()->import_memory(weiBuffer.data());
    if (deconvAttrs.withBiases)
        biasTensor.allocator()->import_memory(src[2]->GetPtr());

    deconv->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
    weiTensor.allocator()->free();
    if (deconvAttrs.withBiases)
        biasTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov