// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce.h"

#include "fake_quantize.h"
#include "eltwise.h"
#include <string>
#include <vector>
#include <set>
#include <onednn/dnnl.h>
#include <dnnl_extension_utils.h>
#include "utils/bfloat16.hpp"
//#include "emitters/x64/jit_bf16_emitters.hpp"
#include "ie_parallel.hpp"
#include <algorithm>


#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <common/primitive_hashing_utils.hpp>

using namespace dnnl;
using namespace InferenceEngine;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define SET_SRC_DIM_VALUE(batch, channel, depth, height, width) IB = batch;   \
                                                                IC = channel; \
                                                                ID = depth;   \
                                                                IH = height;  \
                                                                IW = width;
#define SET_DST_DIM_VALUE(batch, channel, depth, height, width) OB = batch;   \
                                                                OC = channel; \
                                                                OD = depth;   \
                                                                OH = height;  \
                                                                OW = width;

#define GET_OFF(field) offsetof(jit_reduce_call_args, field)
#define GET_OFF_POST(field) offsetof(jit_reduce_post_call_args, field)

#define GET_PTR_N_PLN              const uint8_t    *in_ptr_n      = in_ptr       + src_data_size * ib * IC * ID * IH * IW;               \
                                         uint8_t    *out_ptr_n     = out_ptr      + dst_data_size * ob * OC * OD * OH * OW;
#define GET_PTR_NC_PLN             const uint8_t    *in_ptr_nc     = in_ptr_n     + src_data_size * ic * ID * IH * IW;                    \
                                         uint8_t    *out_ptr_nc    = out_ptr_n    + dst_data_size * oc * OD * OH * OW;
#define GET_PTR_NCD_PLN            const uint8_t    *in_ptr_ncd    = in_ptr_nc    + src_data_size * id * IH * IW;                         \
                                         uint8_t    *out_ptr_ncd   = out_ptr_nc   + dst_data_size * od * OH * OW;
#define GET_PTR_NCDH_PLN           const uint8_t    *in_ptr_ncdh   = in_ptr_ncd   + src_data_size * ih * IW;                              \
                                         uint8_t    *out_ptr_ncdh  = out_ptr_ncd  + dst_data_size * oh * OW;
#define GET_PTR_NCD_BASE_PTR_N_PLN const uint8_t    *in_ptr_ncd    = in_ptr_n     + src_data_size * (ic * ID + id) * IH * IW;             \
                                         uint8_t    *out_ptr_ncd   = out_ptr_n    + dst_data_size * (oc * OD + od) * OH * OW;
#define GET_PTR_N_BLK              const uint8_t    *in_ptr_n      = in_ptr       + src_data_size * ib * ICB * ID * IH * IW * blk_size;   \
                                         uint8_t    *out_ptr_n     = out_ptr      + dst_data_size * ob * OCB * OD * OH * OW * blk_size;
#define GET_PTR_NC_BLK             const uint8_t    *in_ptr_nc     = in_ptr_n     + src_data_size * icb * ID * IH * IW * blk_size;        \
                                         uint8_t    *out_ptr_nc    = out_ptr_n    + dst_data_size * ocb * OD * OH * OW * blk_size;
#define GET_PTR_NCD_BLK            const uint8_t    *in_ptr_ncd    = in_ptr_nc    + src_data_size * id * IH * IW * blk_size;              \
                                         uint8_t    *out_ptr_ncd   = out_ptr_nc   + dst_data_size * od * OH * OW * blk_size;
#define GET_PTR_NCDH_BLK           const uint8_t    *in_ptr_ncdh   = in_ptr_ncd   + src_data_size * ih * IW * blk_size;                   \
                                         uint8_t    *out_ptr_ncdh  = out_ptr_ncd  + dst_data_size * oh * OW * blk_size;
#define GET_PTR_NCDHW_BLK          const uint8_t    *in_ptr_ncdhw  = in_ptr_ncdh  + src_data_size * iw * blk_size;                        \
                                         uint8_t    *out_ptr_ncdhw = out_ptr_ncdh + dst_data_size * ow * blk_size;
#define GET_PTR_NCD_BASE_PTR_N_BLK const uint8_t    *in_ptr_ncd    = in_ptr_n     + src_data_size * (icb * ID + id) * IH * IW * blk_size; \
                                         uint8_t    *out_ptr_ncd   = out_ptr_n    + dst_data_size * (ocb * OD + od) * OH * OW * blk_size;

namespace ov {
namespace intel_cpu {
namespace node {

// some utility functions
static inline bool isFloatCompatible(memory::data_type type) {
    return memory::data_type::f32 == type || memory::data_type::bf16 == type;
}



const std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>&, Reduce&)>> Reduce::initializers = {
    {ngraph::opset4::ReduceL1::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceL1;
    }},
    {ngraph::opset4::ReduceL2::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceL2;
    }},
    {ngraph::opset1::ReduceLogicalAnd::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceAnd;
    }},
    {ngraph::opset1::ReduceLogicalOr::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceOr;
    }},
    {ngraph::opset1::ReduceMax::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceMax;
    }},
    {ngraph::opset1::ReduceMean::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceMean;
    }},
    {ngraph::opset1::ReduceMin::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceMin;
    }},
    {ngraph::opset1::ReduceProd::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceProd;
    }},
    {ngraph::opset1::ReduceSum::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceSum;
    }}
};

bool Reduce::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (std::dynamic_pointer_cast<const ngraph::op::util::ArithmeticReductionKeepDims>(op) == nullptr &&
                std::dynamic_pointer_cast<const ngraph::op::util::LogicalReductionKeepDims>(op) == nullptr) {
            errorMessage = "Reduce node with name " + op->get_friendly_name() + " is not derived from ArithmeticReductionKeepDims or LogicalReductionKeepDims";
            return false;
        }
        if (const auto reduce = std::dynamic_pointer_cast<const ngraph::op::util::ArithmeticReductionKeepDims>(op)) {
            auto reduceConst = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(reduce->get_input_node_shared_ptr(REDUCE_INDEXES));
            if (!reduceConst) {
                errorMessage = "Second tensor is not constant";
                return false;
            }
        }
        if (const auto reduce = std::dynamic_pointer_cast<const ngraph::op::util::LogicalReductionKeepDims>(op)) {
            auto reduceConst = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(reduce->get_input_node_shared_ptr(REDUCE_INDEXES));
            if (!reduceConst) {
                errorMessage = "Second tensor is not constant";
                return false;
            }
        }
        if (initializers.find(op->get_type_info()) == initializers.end()) {
            errorMessage = "Doesn't support Reduce algorithm: " +  std::string(op->get_type_info().name);
            return false;
        }
        if (std::dynamic_pointer_cast<ngraph::opset1::Constant>(op->get_input_node_shared_ptr(REDUCE_INDEXES)) == nullptr) {
            errorMessage = "Only const 'reduce_indexes' input is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Reduce::Reduce(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, PortMask(REDUCE_INDEXES))) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Reduce node with name '" + getName() + "'";
        initializers.at(op->get_type_info())(op, *this);
        if (const auto reduce = std::dynamic_pointer_cast<ngraph::op::util::ArithmeticReductionKeepDims>(op)) {
            keep_dims = reduce->get_keep_dims();
            auto reduceConst = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(reduce->get_input_node_shared_ptr(REDUCE_INDEXES));
            if (!reduceConst)
                IE_THROW() << errorPrefix << " second tensor is not constant!";
            raw_axes = reduceConst->cast_vector<int>();
        } else if (const auto reduce = std::dynamic_pointer_cast<ngraph::op::util::LogicalReductionKeepDims>(op)) {
            keep_dims = reduce->get_keep_dims();
            auto reduceConst = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(reduce->get_input_node_shared_ptr(REDUCE_INDEXES));
            if (!reduceConst)
                IE_THROW() << errorPrefix << " second tensor is not constant!";
            raw_axes = reduceConst->cast_vector<int>();
        }
        vec_reduceDH_prc.clear();
        setJITBeyond5D();
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void Reduce::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 2)
        IE_THROW() << errorPrefix << " gets incorrect number of input edges!";
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << " gets incorrect number of output edges!";

    if (getInputShapeAtPort(REDUCE_INDEXES).getRank() != 1) {
        IE_THROW() << errorPrefix << " gets incorrect index vector dimension! Index vector should be 1 dimension.";
    }

    if (keep_dims) {
        if (getInputShapeAtPort(REDUCE_DATA).getRank() != getOutputShapeAtPort(0).getRank())
            IE_THROW() << errorPrefix << " gets incorrect number of input/output dimensions!";
    } else {
        // In fact, after the Reduce operation, the shape must be a scalar if the previous one was 1d.
        // But for now, 0d tensor (scalar) is emulated as 1d tensor. Skip checking in such cases.
        bool is_emulated_0d_as_1d = getInputShapeAtPort(REDUCE_DATA).getRank() == 1 && getOutputShapeAtPort(0).getRank() == 1;
        if (getInputShapeAtPort(REDUCE_DATA).getRank() <= getOutputShapeAtPort(0).getRank() && !is_emulated_0d_as_1d)
            IE_THROW() << errorPrefix << "gets incorrect number of input/output dimensions!";
    }
}

void Reduce::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    input_prec = getOriginalInputPrecisionAtPort(REDUCE_DATA);
    output_prec = getOriginalOutputPrecisionAtPort(0);

    if (!fusedWith.empty()) {
        output_prec = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    jit_mode = canApplyJIT(input_prec, output_prec);

    if (jit_mode) {
        // Since in jit mode we use the output memory as an intermediate accumulator for certain reduce modes, we can't use BF16 output precision due to
        // the possible accuracy loss. Therefore, for such mods, we will change the output precision to FP32.
        if (Precision::BF16 == output_prec) {
            if (!mayiuse(avx512_core)) {
                    output_prec = Precision::FP32;
            } else if (algorithm != Algorithm::ReduceAnd && algorithm != Algorithm::ReduceOr &&
                       algorithm != Algorithm::ReduceMin && algorithm != Algorithm::ReduceMax) {
                            output_prec = Precision::FP32;
            }
        }
    }

    support_split = algorithm != Algorithm::ReduceL2 && algorithm != Algorithm::ReduceLogSumExp &&
                    algorithm != Algorithm::ReduceSumSquare && input_prec == output_prec;

    src_data_size = input_prec.size();
    dst_data_size = output_prec.size();

    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.inConfs[REDUCE_DATA].constant(false);
    config.inConfs[REDUCE_INDEXES].constant(false);
    config.outConfs[0].constant(false);
    config.inConfs[REDUCE_DATA].inPlace(-1);
    config.inConfs[REDUCE_INDEXES].inPlace(-1);
    config.outConfs[0].inPlace(-1);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    auto pushDesc = [&](LayoutType inFormat, LayoutType outFormat, InferenceEngine::Precision inPrecision,
            InferenceEngine::Precision outPrecision, impl_desc_type impl_type) {
        config.inConfs[REDUCE_DATA].setMemDesc(creatorsMap.at(inFormat)->createSharedDesc(inPrecision, getInputShapeAtPort(REDUCE_DATA)));
        config.inConfs[REDUCE_INDEXES].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(InferenceEngine::Precision::I32,
                                                                                                 getInputShapeAtPort(REDUCE_INDEXES)));
        config.outConfs[0].setMemDesc(creatorsMap.at(outFormat)->createSharedDesc(outPrecision, getOutputShapeAtPort(0)));

        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (int i = 0; i < config.inConfs.size(); i++) {
            srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        for (int i = 0; i < config.outConfs.size(); i++) {
            dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
        }
        auto factory = std::make_shared<ReduceExecutorFactory>(reduceAttrs, srcMemoryDescs, dstMemoryDescs);
        supportedPrimitiveDescriptors.push_back({config, impl_type, factory});
    };

    if (jit_mode) {
        impl_desc_type impl_type = impl_desc_type::jit_sse42;
        if (mayiuse(cpu::x64::avx512_core)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (mayiuse(cpu::x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        }

        pushDesc(LayoutType::ncsp, LayoutType::ncsp, input_prec, output_prec, impl_type);
        if ((getInputShapeAtPort(REDUCE_DATA).getRank() == 4 || getInputShapeAtPort(REDUCE_DATA).getRank() == 5) &&
                getInputShapeAtPort(REDUCE_DATA).getMinDims()[1] > 1) {
            if (keep_dims) {
                if (mayiuse(cpu::x64::avx512_core)) {
                    pushDesc(LayoutType::nspc, LayoutType::nspc, input_prec, output_prec, impl_type);
                    pushDesc(LayoutType::nCsp16c, LayoutType::nCsp16c, input_prec, output_prec, impl_type);
                } else if (mayiuse(cpu::x64::avx2) || mayiuse(cpu::x64::sse41)) {
                    pushDesc(LayoutType::nspc, LayoutType::nspc, input_prec, output_prec, impl_type);
                    pushDesc(LayoutType::nCsp8c, LayoutType::nCsp8c, input_prec, output_prec, impl_type);
                }
            } else {
                if (mayiuse(cpu::x64::avx512_core)) {
                    pushDesc(LayoutType::nspc, LayoutType::ncsp, input_prec, output_prec, impl_type);
                    pushDesc(LayoutType::nCsp16c, LayoutType::ncsp, input_prec, output_prec, impl_type);
                } else if (mayiuse(cpu::x64::avx2) || mayiuse(cpu::x64::sse41)) {
                    pushDesc(LayoutType::nspc, LayoutType::ncsp, input_prec, output_prec, impl_type);
                    pushDesc(LayoutType::nCsp8c, LayoutType::ncsp, input_prec, output_prec, impl_type);
                }
            }
        }
    } else {
        pushDesc(LayoutType::ncsp, LayoutType::ncsp, InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32, impl_desc_type::ref);
    }
}

bool Reduce::isExecutable() const {
    return !isInputTensorAtPortEmpty(REDUCE_DATA);
}

void Reduce::prepareParams() {
    /*src_dims = getParentEdgesAtPort(REDUCE_DATA)[0]->getMemory().getDesc().getShape().getDims();
    std::vector<int> reduce_axes;
    if (jit_mode && jit_beyond_5D) {
        reduce_axes = update_src_dims();
    } else {
        reduce_axes = raw_axes;
    }*/

    /*auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    const SizeVector &dst_dims = dstMemPtr->getDesc().getShape().getDims();
    dst_size = dstMemPtr->GetSize();
    calc_process_dst_dims(reduce_axes, dst_dims);
    if (jit_mode) {
        set_reduce_dim_flags();
    }*/

    /*auto builder = [&](const ReduceKey& key) -> std::shared_ptr<jit_uni_reduce_post_kernel> {
        std::shared_ptr<jit_uni_reduce_post_kernel> post_kernel;

        if (mayiuse(cpu::x64::avx512_core)) {
            post_kernel.reset(new jit_uni_reduce_post_kernel_f32<cpu::x64::avx512_core>(key.jcp, *attr.get()));
        } else if (mayiuse(cpu::x64::avx2)) {
            post_kernel.reset(new jit_uni_reduce_post_kernel_f32<cpu::x64::avx2>(key.jcp, *attr.get()));
        } else if (mayiuse(cpu::x64::sse41)) {
            post_kernel.reset(new jit_uni_reduce_post_kernel_f32<cpu::x64::sse41>(key.jcp, *attr.get()));
        }
        if (post_kernel)
            post_kernel->create_ker();

        return post_kernel;
    };*/

    /*if (compile_post_kernel) {
        setPostOps(attr, dst_dims, true);

        ReduceKey key = {jcp, attr.get_post_ops()};
        auto cache = context->getParamsCache();
        auto result = cache->getOrCreate(key, builder);
        if (!result.first) {
            IE_THROW() << errorPrefix << " has not found jit_uni_reduce_post_kernel_f32.";
        }

        reduce_post_kernel = result.first;
        jit_mode = jit_mode && reduce_post_kernel;

        if (!isDynamicNode()) {
            compile_post_kernel = false;
        }
    }*/
}

void Reduce::createPrimitive() {
    if (!isExecutable()) {
        return;
    }
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(REDUCE_DATA)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << errorPrefix << " has not allocated destination memory.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << errorPrefix << " has not allocate input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << errorPrefix << " has nullable preferable primitive descriptor";

    /*if (srcMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
        layout = ReduceLayoutType::reduce_ncsp;
    } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nspc)) {
        layout = ReduceLayoutType::reduce_nspc;
    } else {
        layout = ReduceLayoutType::reduce_blocked;
    }*/



    auto selectedPD = getSelectedPrimitiveDescriptor();
/*    jcp = jit_reduce_config_params();
    jcp.src_dt = DnnlExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().inConfs[REDUCE_DATA].getMemDesc()->getPrecision());
    jcp.dst_dt = DnnlExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().outConfs[0].getMemDesc()->getPrecision());
    jcp.src_data_size = DnnlExtensionUtils::sizeOfDataType(jcp.src_dt);
    jcp.dst_data_size = DnnlExtensionUtils::sizeOfDataType(jcp.dst_dt);
    jcp.layout = layout;
    jcp.reduce_mode = getAlgorithm();*/

    /*compile_post_kernel = true;

    if (mayiuse(cpu::x64::avx512_core)) {
        blk_size = 16;
    } else {
        blk_size = 8;
    }*/

    //Question: what to do with the code below?
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }

    /*if (mayiuse(cpu::x64::avx512_core)) {
        reduce_kernel.reset(new jit_uni_reduce_kernel_f32<cpu::x64::avx512_core>(jcp));
    } else if (mayiuse(cpu::x64::avx2)) {
        reduce_kernel.reset(new jit_uni_reduce_kernel_f32<cpu::x64::avx2>(jcp));
    } else if (mayiuse(cpu::x64::sse41)) {
        reduce_kernel.reset(new jit_uni_reduce_kernel_f32<cpu::x64::sse41>(jcp));
    }
    if (reduce_kernel)
        reduce_kernel->create_ker();
    jit_mode = jit_mode && reduce_kernel;*/
}

void Reduce::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Reduce::execute(dnnl::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(REDUCE_DATA)->getMemoryPtr();

    const uint8_t *src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetPtr());
    uint8_t *dst_data = reinterpret_cast<uint8_t *>(dstMemPtr->GetPtr());

    if (jit_mode) {
        if (is_hybrid_layout) {
            dst_data = reinterpret_cast<uint8_t *>(prc_mem->get_data_handle());
        }
        reduce_type(src_data, dst_data, dst_size);
    } else {
        if (layout == ReduceLayoutType::reduce_ncsp) {
            auto in_ptr = reinterpret_cast<const float *>(src_data);
            auto out_ptr = reinterpret_cast<float *>(dst_data);
            reduce_ref(in_ptr, out_ptr);
        } else {
            IE_THROW() << errorPrefix << " supports only plain layout on machine w/o sse42.";
        }
    }
}





inline void Reduce::create_working_memory() {
    auto rank = getInputShapeAtPort(REDUCE_DATA).getRank();
    memory::format_tag format = (layout == ReduceLayoutType::reduce_nspc) ? (rank == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc)
                                        : (rank == 4 ? (mayiuse(cpu::x64::avx512_core) ? memory::format_tag::nChw16c : memory::format_tag::nChw8c)
                                                     : (mayiuse(cpu::x64::avx512_core) ? memory::format_tag::nCdhw16c : memory::format_tag::nCdhw8c));
    auto prc_dims = rank == 4 ? std::vector<size_t>{OB, OC, OH, OW} : std::vector<size_t>{OB, OC, OD, OH, OW};
    auto desc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(prc_dims), DnnlExtensionUtils::IEPrecisionToDataType(output_prec), format);
    prc_mem = std::make_shared<dnnl::memory>(desc, getEngine());
    dst_size = desc.get_size();
}

inline void Reduce::create_DH_working_memory() {
    ReduceDH_opt = layout == ReduceLayoutType::reduce_nspc && !isDynamicNode() && support_split &&
                   !ReduceC && ReduceD && ReduceH && !ReduceW && IC == 1 && ID > 1;
    if (ReduceDH_opt) {
        PD = ID;
        PW = IW / blk_size * blk_size;
        prc_data_size = src_data_size;
        prc_size = PD * PW * src_data_size;
        if (prc_size > vec_reduceDH_prc.size()) {
            vec_reduceDH_prc.resize(prc_size);
        }
    }
}

inline void Reduce::calc_process_dst_dims(std::vector<int> &reduce_axes, const SizeVector &dst_dims) {
    std::set<size_t> axes;
    SizeVector out_dims;
    process_dst_dims.clear();
    axes_for_reduction.clear();
    for (auto &axis : reduce_axes) {
        if (axis < 0)
            axis += src_dims.size();
        if (static_cast<size_t>(axis) > src_dims.size())
            IE_THROW() << errorPrefix << " exceeds data tensor dimension on index to reduce";
        axes.insert(static_cast<size_t>(axis));
    }
    for (size_t i = 0; i < src_dims.size(); i++) {
        bool found = false;
        for (auto axis : axes) {
            if (i == axis) {
                found = true;
                break;
            }
        }
        if (found) {
            if (keep_dims) out_dims.push_back(1);
            process_dst_dims.push_back(1);
            axes_for_reduction.push_back(i);
        } else {
            out_dims.push_back(src_dims[i]);
            process_dst_dims.push_back(src_dims[i]);
        }
    }
    if (jit_mode && jit_beyond_5D) {
        if (std::accumulate(out_dims.begin(), out_dims.end(), size_t(1), std::multiplies<size_t>()) !=
            std::accumulate(dst_dims.begin(), dst_dims.end(), size_t(1), std::multiplies<size_t>()))
            IE_THROW() << errorPrefix << "gets incorrect number of output dimensions!";
    } else {
        for (size_t i = 0; i < std::min(out_dims.size(), dst_dims.size()); i++) {
            if (out_dims[i] != dst_dims[i])
                IE_THROW() << errorPrefix << "gets incorrect number of output dimensions!";
        }
    }
}

inline void Reduce::set_reduce_dim_flags() {
    size_t dims_size = src_dims.size();
    if (dims_size == 5) {
        SET_SRC_DIM_VALUE(src_dims[0], src_dims[1], src_dims[2], src_dims[3], src_dims[4]);
        SET_DST_DIM_VALUE(process_dst_dims[0], process_dst_dims[1], process_dst_dims[2], process_dst_dims[3], process_dst_dims[4]);
    } else if (dims_size == 4) {
        SET_SRC_DIM_VALUE(src_dims[0], src_dims[1], 1, src_dims[2], src_dims[3]);
        SET_DST_DIM_VALUE(process_dst_dims[0], process_dst_dims[1], 1, process_dst_dims[2], process_dst_dims[3]);
    } else if (dims_size == 3) {
        SET_SRC_DIM_VALUE(1, src_dims[0], 1, src_dims[1], src_dims[2]);
        SET_DST_DIM_VALUE(1, process_dst_dims[0], 1, process_dst_dims[1], process_dst_dims[2]);
    } else if (dims_size == 2) {
        SET_SRC_DIM_VALUE(1, 1, 1, src_dims[0], src_dims[1]);
        SET_DST_DIM_VALUE(1, 1, 1, process_dst_dims[0], process_dst_dims[1]);
    } else {
        SET_SRC_DIM_VALUE(1, src_dims[0], 1, 1, 1);
        SET_DST_DIM_VALUE(1, process_dst_dims[0], 1, 1, 1);
    }

    // must be done before the following dimension change
    if (is_hybrid_layout) {
        create_working_memory();
    }

    // Reducing a dimesion in nspc layout can be treated as reducing another dimension in ncsp layout,
    // eg. reducing C in nspc can be treated as reducing W in ncsp layout, so that the routine reduce_PLN can be reused.
    // nspc -- ncsp
    //    D -- C
    //    H -- D
    //    W -- H
    //    C -- W
    if (layout == ReduceLayoutType::reduce_nspc) {
        size_t ITmp = IC; IC = ID; ID = IH; IH = IW; IW = ITmp;
        size_t OTmp = OC; OC = OD; OD = OH; OH = OW; OW = OTmp;
    }

    ReduceN = IB != OB && OB == 1;
    ReduceC = IC != OC && OC == 1;
    ReduceD = ID != OD && OD == 1;
    ReduceH = IH != OH && OH == 1;
    ReduceW = IW != OW && OW == 1;

    // must be done before the above dimension change
    create_DH_working_memory();

    // suit for parallel
    if (ReduceH && IW == 1) {
        ReduceW = true;
    }
    if (ReduceC && ReduceH && ID == 1) {
        ReduceD = true;
    }
}

inline void Reduce::reduce_ref(const float *in_ptr, float *out_ptr) {
    switch (algorithm) {
        case Algorithm::ReduceAnd:
            reduce_ref_process(in_ptr, out_ptr, 1, [](float x, float y)->float { return x && y; });
            break;
        case Algorithm::ReduceL1:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + (y >= 0 ? y : -y); });
            break;
        case Algorithm::ReduceL2:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + y * y; });
            break;
        case Algorithm::ReduceLogSum:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x + y; });
            break;
        case Algorithm::ReduceLogSumExp:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + expf(y); });
            break;
        case Algorithm::ReduceMax:
            reduce_ref_process(in_ptr, out_ptr, std::numeric_limits<float>::lowest(),
                                                    [](float x, float y)->float { return x > y ? x : y; });
            break;
        case Algorithm::ReduceMean:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x + y; });
            break;
        case Algorithm::ReduceMin:
            reduce_ref_process(in_ptr, out_ptr, std::numeric_limits<float>::max(),
                                                    [](float x, float y)->float { return x < y ? x : y; });
            break;
        case Algorithm::ReduceOr:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x || y; });
            break;
        case Algorithm::ReduceProd:
            reduce_ref_process(in_ptr, out_ptr, 1, [](float x, float y)->float { return x * y; });
            break;
        case Algorithm::ReduceSum:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x + y; });
            break;
        case Algorithm::ReduceSumSquare:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + y * y; });
            break;
    default:
        IE_THROW() << errorPrefix << "gets unsupported reduce mode.";
    }
}

void Reduce::reduce_ref_process(const float *in_ptr, float *out_ptr, float init_value, std::function<float(float, float)> func) {
    size_t work_amount_dst = 1, reduced_dims_work_amount = 1;
    for (size_t i = 0; i < process_dst_dims.size(); i++)
        work_amount_dst *= process_dst_dims[i];
    for (size_t i = 0; i < src_dims.size(); i++)
        reduced_dims_work_amount *= src_dims[i];
    reduced_dims_work_amount /= work_amount_dst;

    SizeVector src_strides = getParentEdgeAt(REDUCE_DATA)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    parallel_nt(0, [&](const int ithr, const int nthr) {
        int j;
        size_t i, start = 0, end = 0;
        SizeVector dst_counters(process_dst_dims.size(), 0);
        splitter(work_amount_dst, nthr, ithr, start, end);
        for (j = process_dst_dims.size() - 1, i = start; j >= 0; j--) {
            dst_counters[j] = i % process_dst_dims[j];
            i /= process_dst_dims[j];
        }
        for (size_t src_idx = 0, dst_idx = start; dst_idx < end; ++dst_idx) {
            float reduce_prod = init_value;
            bool update_idx = true;
            SizeVector src_counters = dst_counters;
            for (i = 0; i < reduced_dims_work_amount; ++i) {
                if (update_idx) {
                    src_idx = 0;
                    for (j = 0; j < static_cast<int>(src_dims.size()); ++j)
                        src_idx += (src_counters[j] % src_dims[j]) * src_strides[j];
                    update_idx = false;
                }
                reduce_prod = func(reduce_prod, in_ptr[src_idx]);
                for (j = axes_for_reduction.size() - 1; j >= 0; j--) {
                    src_counters[axes_for_reduction[j]]++;
                    if (src_counters[axes_for_reduction[j]] < src_dims[axes_for_reduction[j]]) {
                        src_idx += src_strides[axes_for_reduction[j]];
                        break;
                    } else {
                        src_counters[axes_for_reduction[j]] = 0;
                        update_idx = true;
                    }
                }
            }
            out_ptr[dst_idx] = reduce_prod;
            for (j = process_dst_dims.size() - 1; j >= 0; j--) {
                dst_counters[j]++;
                if (dst_counters[j] < process_dst_dims[j])
                    break;
                else
                    dst_counters[j] = 0;
            }
        }
    });

    reduce_ref_map(out_ptr, work_amount_dst, reduced_dims_work_amount);
}

inline void Reduce::reduce_ref_map(float *out_ptr, size_t work_amount_dst, size_t reduced_dims_work_amount) {
    switch (algorithm) {
        case Algorithm::ReduceAnd:
        case Algorithm::ReduceL1:
        case Algorithm::ReduceMax:
        case Algorithm::ReduceMin:
        case Algorithm::ReduceOr:
        case Algorithm::ReduceProd:
        case Algorithm::ReduceSum:
        case Algorithm::ReduceSumSquare:
            break;
        case Algorithm::ReduceL2:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] = std::sqrt(out_ptr[i]);
            });
            break;
        case Algorithm::ReduceLogSum:
        case Algorithm::ReduceLogSumExp:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] = logf(out_ptr[i]);
            });
            break;
        case Algorithm::ReduceMean:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] /= reduced_dims_work_amount;
            });
            break;
        default:
            IE_THROW() << errorPrefix << "gets unsupported reduce mode.";
    }
}

void Reduce::setPostOps(dnnl::primitive_attr &attr, const VectorDims &postOpDims, bool initWeights) {
    dnnl::post_ops ops;
    postOpsDataPtrs.clear();
    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, {}, postOpsDataPtrs);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<Eltwise *>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops, postOpDims, postOpsDataPtrs, getFusingAxis());
            continue;
        }
        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

void Reduce::setJITBeyond5D() {
    jit_beyond_5D = false;
    if (getInputShapeAtPort(REDUCE_DATA).getRank() > 5) {
        for (auto &axis : raw_axes) {
            if (axis < 0)
                axis += static_cast<int>(getInputShapeAtPort(REDUCE_DATA).getRank());
        }

        if (raw_axes.size() <= 1) {
            jit_beyond_5D = true;
        } else {
            for (size_t i = 1; i < raw_axes.size(); i++) {
                if (raw_axes[i] != raw_axes[i - 1] + 1) {
                    jit_beyond_5D = false;
                    break;
                }
                jit_beyond_5D = true;
            }
        }
    }
}

std::vector<int> Reduce::update_src_dims() {
    std::vector<int> reduce_axes = raw_axes;

    if (reduce_axes.size() < 1)
        return reduce_axes;

    size_t axis_dim = 1;
    size_t outer_dim = 1;
    size_t inner_dim = 1;
    int outer_end = reduce_axes[0];
    int inner_start = reduce_axes[reduce_axes.size() - 1];
    for (size_t i = 0; i < src_dims.size(); i++) {
        if (i < outer_end) {
            outer_dim *= src_dims[i];
        } else if (i > inner_start) {
            inner_dim *= src_dims[i];
        } else {
            axis_dim *= src_dims[i];
        }
    }

    reduce_axes.clear();
    reduce_axes.push_back(1);

    src_dims.clear();
    src_dims.push_back(outer_dim);
    src_dims.push_back(axis_dim);
    src_dims.push_back(inner_dim);

    return reduce_axes;
}

bool Reduce::canApplyJIT(const Precision &input_prec, const Precision &output_prec) const {
    static const Precision supportedPrecisions[] = {
            Precision::FP32,
            Precision::BF16,
            Precision::I32,
            Precision::I8,
            Precision::U8
    };

    return (mayiuse(cpu::x64::sse41)) && (getInputShapeAtPort(REDUCE_DATA).getRank() <= 5 || jit_beyond_5D) &&
           std::find(std::begin(supportedPrecisions), std::end(supportedPrecisions), input_prec) != std::end(supportedPrecisions) &&
           std::find(std::begin(supportedPrecisions), std::end(supportedPrecisions), output_prec) != std::end(supportedPrecisions);
}

int Reduce::getFusingAxis() const {
    int channelAxis = 1;
    if (!keep_dims) {
        for (auto &raw_axis : raw_axes) {
            int axis = raw_axis >= 0 ? raw_axis : raw_axis + static_cast<int>(getInputShapeAtPort(REDUCE_DATA).getRank());
            if (axis == 1) {
                // channel axis has been reduced and doesn't exist any more
                channelAxis = -1;
                break;
            } else if (axis == 0) {
                channelAxis = 0;
            }
        }
    }
    return channelAxis;
}

bool Reduce::canFuse(const NodePtr& node) const {
    Precision input_prec = getOriginalInputPrecisionAtPort(REDUCE_DATA);
    Precision output_prec = getOriginalOutputPrecisionAtPort(0);
    if (!canApplyJIT(input_prec, output_prec) || jit_beyond_5D || algorithm == Algorithm::ReduceAnd || algorithm == Algorithm::ReduceOr) {
        return false;
    }

    // In jit mode we use the output memory as an intermediate accumulator for certain reduce modes.
    // If the post ops node has a lower precision for such modes, post ops fusing won't be supposted, in order to avoid accuracy loss.
    if (output_prec == Precision::FP32 &&
        !node->getOriginalOutputPrecisions().empty() && node->getOriginalOutputPrecisionAtPort(0) != Precision::FP32) {
        if (algorithm != Algorithm::ReduceAnd && algorithm != Algorithm::ReduceOr &&
            algorithm != Algorithm::ReduceMin && algorithm != Algorithm::ReduceMax) {
            return false;
        }
    }

    return canFuseSimpleOperation(node);
}

bool Reduce::created() const {
    return getType() == Type::Reduce;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
