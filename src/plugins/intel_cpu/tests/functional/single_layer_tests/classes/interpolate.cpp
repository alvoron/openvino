// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate.hpp"

#include "gtest/gtest.h"
#include "test_utils/cpu_test_utils.hpp"
#include <transformations/op_conversions/convert_interpolate11_downgrade.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

std::string InterpolateLayerCPUTest::getTestCaseName(
    testing::TestParamInfo<InterpolateLayerCPUTestParamsSet> obj) {
    InterpolateSpecificParams specificParams;
    ShapeParams shapeParams;
    ElementType prec;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    std::map<std::string, std::string> additionalConfig;
    std::tie(specificParams, shapeParams, prec, cpuParams, fusingParams, additionalConfig) = obj.param;

    ov::op::v11::Interpolate::InterpolateMode mode;
    ov::op::v11::Interpolate::CoordinateTransformMode transfMode;
    ov::op::v11::Interpolate::NearestMode nearMode;
    bool antiAlias;
    std::vector<size_t> padBegin;
    std::vector<size_t> padEnd;
    double cubeCoef;
    std::tie(mode, transfMode, nearMode, antiAlias, padBegin, padEnd, cubeCoef) = specificParams;

    ov::op::v11::Interpolate::ShapeCalcMode shapeCalcMode;
    InputShape inputShapes;
    ngraph::helpers::InputLayerType shapeInputType;
    std::vector<std::vector<float>> shapeDataForInput;
    std::vector<int64_t> axes;
    std::tie(shapeCalcMode, inputShapes, shapeInputType, shapeDataForInput, axes) = shapeParams;

    std::ostringstream result;
    result << "ShapeCalcMode=" << shapeCalcMode << "_";
    result << "IS=";
    result << CommonTestUtils::partialShape2str({inputShapes.first}) << "_";
    result << "TS=";
    for (const auto& shape : inputShapes.second) {
        result << CommonTestUtils::vec2str(shape) << "_";
    }
    if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
        result << "Scales=";
    } else {
        result << "Sizes=";
    }
    for (const auto& data : shapeDataForInput) {
        result << CommonTestUtils::vec2str(data) << "_";
    }
    result << shapeInputType << "_";
    result << "InterpolateMode=" << mode << "_";
    result << "CoordinateTransformMode=" << transfMode << "_";
    result << "NearestMode=" << nearMode << "_";
    result << "CubeCoef=" << cubeCoef << "_";
    result << "Antialias=" << antiAlias << "_";
    result << "PB=" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE=" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "Axes=" << CommonTestUtils::vec2str(axes) << "_";
    result << "PRC=" << prec << "_";

    result << CPUTestsBase::getTestCaseName(cpuParams);
    result << CpuTestWithFusing::getTestCaseName(fusingParams);

    if (!additionalConfig.empty()) {
        result << "_PluginConf";
        for (auto& item : additionalConfig) {
            result << "_" << item.first << "=" << item.second;
        }
    }

    return result.str();
}

void InterpolateLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;
        if (i == 1) {
            if (shapeCalcMode == ov::op::v4::Interpolate::ShapeCalcMode::SIZES) {
                tensor =
                    ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], sizes[inferRequestNum].data());
            } else {
                tensor = ov::Tensor(funcInput.get_element_type(),
                                    targetInputStaticShapes[i],
                                    scales[inferRequestNum].data());
            }
        } else {
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                             targetInputStaticShapes[i],
                                                             2560,
                                                             0,
                                                             256);
        }

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
    inferRequestNum++;
}

void InterpolateLayerCPUTest::configure_model() {
    ov::preprocess::PrePostProcessor p(function);
    {
        auto& params = function->get_parameters();
        for (size_t i = 0; i < params.size(); i++) {
            if (i > 0) {
                continue;
            }
            if (inType != ov::element::Type_t::undefined) {
                p.input(i).tensor().set_element_type(inType);
            }
        }
    }
    {
        auto results = function->get_results();
        for (size_t i = 0; i < results.size(); i++) {
            if (outType != ov::element::Type_t::undefined) {
                p.output(i).tensor().set_element_type(outType);
            }
        }
    }
    function = p.build();
}

void InterpolateLayerCPUTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_CPU;

    InterpolateSpecificParams specificParams;
    ShapeParams shapeParams;
    ElementType ngPrc;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    std::map<std::string, std::string> additionalConfig;
    std::tie(specificParams, shapeParams, ngPrc, cpuParams, fusingParams, additionalConfig) = this->GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    std::tie(mode, transfMode, nearMode, antiAlias, padBegin, padEnd, cubeCoef) = specificParams;

    InputShape dataShape;
    //ngraph::helpers::InputLayerType shapeInputType;
    std::vector<std::vector<float>> shapeDataForInput;
    std::vector<int64_t> axes;
    std::tie(shapeCalcMode, dataShape, shapeInputType, shapeDataForInput, axes) = shapeParams;

    if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
        scales = shapeDataForInput;
    } else {
        sizes.resize(shapeDataForInput.size());
        for (size_t i = 0; i < shapeDataForInput.size(); i++) {
            for (size_t j = 0; j < shapeDataForInput[i].size(); j++) {
                sizes[i].push_back(shapeDataForInput[i][j]);
            }
        }
    }

    std::vector<InputShape> inputShapes;
    inputShapes.push_back(dataShape);
    if (shapeInputType == ngraph::helpers::InputLayerType::PARAMETER) {
        inputShapes.push_back(InputShape({static_cast<int64_t>(axes.size())},
                                         std::vector<ov::Shape>(dataShape.second.size(), {axes.size()})));
    }

    if (additionalConfig[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] ==
        InferenceEngine::PluginConfigParams::YES) {
        inType = outType = ngPrc = ElementType::bf16;
        rel_threshold = 1e-2f;
    } else {
        inType = outType = ngPrc;
    }

    init_input_shapes(inputShapes);

    auto params = ngraph::builder::makeDynamicParams(ngPrc, {inputDynamicShapes.front()});

    std::shared_ptr<ov::Node> sizesInput, scalesInput;
    //float scale_h, scale_w;
    if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
        if (shapeInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            auto paramNode =
                std::make_shared<ov::op::v0::Parameter>(ElementType::f32, ov::Shape{scales.front().size()});
            params.push_back(paramNode);
            scalesInput = paramNode;
        } else {
            //scale_h = static_cast<float>(scales.front()[2]);
        //scale_w = static_cast<float>(scales.front()[3]);
            scalesInput = std::make_shared<ov::op::v0::Constant>(ElementType::f32,
                                                                 ov::Shape{scales.front().size()},
                                                                 scales.front());
        }
    } else {
        if (shapeInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            auto paramNode = std::make_shared<ov::op::v0::Parameter>(ElementType::i32, ov::Shape{sizes.front().size()});
            params.push_back(paramNode);
            sizesInput = paramNode;
        } else {
            //scale_h = static_cast<float>(sizes.front()[2]) / params[0]->get_output_shape(0)[2];
        //scale_w = static_cast<float>(sizes.front()[3]) / params[0]->get_output_shape(0)[3];
            sizesInput = std::make_shared<ov::op::v0::Constant>(ElementType::i32,
                                                                ov::Shape{sizes.front().size()},
                                                                sizes.front());
        }
    }
    auto axesInput = std::make_shared<ov::op::v0::Constant>(ElementType::i64, ov::Shape{axes.size()}, axes);

    for (size_t i = 0; i < params.size(); i++) {
        params[i]->set_friendly_name(std::string("param_") + std::to_string(i));
    }

    ov::op::v11::Interpolate::InterpolateAttrs
        interpAttr{mode, shapeCalcMode, padBegin, padEnd, transfMode, nearMode, antiAlias, cubeCoef};

    std::shared_ptr<ov::op::v11::Interpolate> interp = nullptr;
    if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
        interp = std::make_shared<ov::op::v11::Interpolate>(params[0], scalesInput, axesInput, interpAttr);
    } else {
        interp = std::make_shared<ov::op::v11::Interpolate>(params[0], sizesInput, axesInput, interpAttr);
    }

    function = makeNgraphFunction(ngPrc, params, interp, "InterpolateCPU");

    ov::pass::Manager m;
    m.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    m.run_passes(function);

// ACL supported check - begin
    /*float scale_h = static_cast<float>(out_shape[index_h]) / inp_shape[index_h];
    float scale_w = static_cast<float>(out_shape[index_w]) / inp_shape[index_w];
    bool is_upsample = scale_h > 1 && scale_w > 1;
    if ()*/
/*
    bool isACLsupported = true;
    bool is_upsample = scale_h > 1 && scale_w > 1;
    std::cout << "(SetUp) is_upsample = " << is_upsample << std::endl;

    if (!std::all_of(padBegin.begin(), padBegin.end(), [](int i){return i == 0;}) ||
        !std::all_of(padEnd.begin(), padEnd.end(), [](int i){return i == 0;})) {
        std::cout << "isACLsupported - false (paddings are not null)" << std::endl;
        isACLsupported = false;
        goto check_exit;
    }

    if (antiAlias ||
        transfMode == ov::op::v11::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN ||
        nearMode == ov::op::v11::Interpolate::NearestMode::CEIL) {
        std::cout << "isACLsupported - false - exit 3" << std::endl;
        isACLsupported = false;
        goto check_exit;
    }

    if (mode == ov::op::v11::Interpolate::InterpolateMode::CUBIC ||
        mode == ov::op::v11::Interpolate::InterpolateMode::BILINEAR_PILLOW ||
        mode == ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW) {
        std::cout << "isACLsupported - false - exit 4" << std::endl;
        isACLsupported = false;
        goto check_exit;
    }

    if (transfMode == ov::op::v11::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL) {
        std::cout << "isACLsupported - false - exit 6" << std::endl;
        isACLsupported = false;
        goto check_exit;
    }


    if (mode == ov::op::v11::Interpolate::InterpolateMode::NEAREST) {
        //simulate ACLInterpolateExecutorBuilder::isSupportedConfiguration inside this if
        isACLsupported = false;
        if (transfMode == ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC &&
            (nearMode == ov::op::v11::Interpolate::NearestMode::SIMPLE || nearMode == ov::op::v11::Interpolate::NearestMode::FLOOR)) {
                isACLsupported = is_upsample;
                if (is_upsample) std::cout << "isACLsupported - TRUE" << std::endl;
                goto check_exit;
            }
    }
    check_exit:
    if (isACLsupported) {
        std::cout << "selectedType has been changed to ACL" << std::endl;
        selectedType = "acl";
    } else {
        std::cout << "selectedType has NOT been changed: " << selectedType << std::endl;
    }*/

    if (isACLSupported(params[0])) {
        std::cout << "selectedType has been changed to ACL" << std::endl;
        selectedType = "acl";
    } else {
        std::cout << "selectedType has NOT been changed: " << selectedType << std::endl;
    }

// ACL supported check - end

    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }
    selectedType = makeSelectedTypeStr(selectedType, ngPrc);
}

bool InterpolateLayerCPUTest::isACLSupported(std::shared_ptr<ov::op::v0::Parameter> params) {
    float scale_h, scale_w;
    //if (shapeInputType != ngraph::helpers::InputLayerType::PARAMETER) {
        if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
            std::cout << "SCALES: " << scales.front()[2] << std::endl;
            scale_h = static_cast<float>(scales.front()[2]);
            scale_w = static_cast<float>(scales.front()[3]);
            std::cout << "scale_h = " << scale_h << std::endl;
            std::cout << "scale_w = " << scale_w << std::endl;
        } else {
            auto in_shape = params->get_output_partial_shape(0);
            if (in_shape[2].is_dynamic() || in_shape[3].is_dynamic()) {
                return false;
            }
            scale_h = static_cast<float>(sizes.front()[2]) / params->get_output_partial_shape(0).get_shape()[2];
            scale_w = static_cast<float>(sizes.front()[3]) / params->get_output_partial_shape(0).get_shape()[3];
            //std::cout << "scale_h = " << scale_h << " ( " << sizes.front()[2] << " / " << params->get_output_shape(0)[2] << " )" << std::endl;
            //std::cout << "scale_w = " << scale_w << " ( " << sizes.front()[3] << " / " << params->get_output_shape(0)[3] << " )" << std::endl;
        }
    //}//else {
    //    scale_h = static_cast<float>(scales.front()[2]);
    //    scale_w = static_cast<float>(scales.front()[3]);
    //}
    bool is_upsample = scale_h > 1 && scale_w > 1;
    std::cout << "(SetUp) is_upsample = " << is_upsample << std::endl;

    if (!std::all_of(padBegin.begin(), padBegin.end(), [](int i){return i == 0;}) ||
        !std::all_of(padEnd.begin(), padEnd.end(), [](int i){return i == 0;})) {
        std::cout << "isACLsupported - false (paddings are not null)" << std::endl;
        return false;
    }

    if (antiAlias ||
        transfMode == ov::op::v11::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN ||
        nearMode == ov::op::v11::Interpolate::NearestMode::CEIL) {
        std::cout << "isACLsupported - false - exit 3" << std::endl;
        return false;
    }

    if (mode == ov::op::v11::Interpolate::InterpolateMode::CUBIC ||
        mode == ov::op::v11::Interpolate::InterpolateMode::BILINEAR_PILLOW ||
        mode == ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW) {
        std::cout << "isACLsupported - false - exit 4" << std::endl;
        return false;
    }

    if (transfMode == ov::op::v11::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL) {
        std::cout << "isACLsupported - false - exit 6" << std::endl;
        return false;
    }

    if (mode == ov::op::v11::Interpolate::InterpolateMode::NEAREST) {
        //simulate ACLInterpolateExecutorBuilder::isSupportedConfiguration inside this if
        if (transfMode == ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC &&
            (nearMode == ov::op::v11::Interpolate::NearestMode::SIMPLE || nearMode == ov::op::v11::Interpolate::NearestMode::FLOOR)) {
                return is_upsample;
            }
        return false;
    }

    return false;
}

TEST_P(InterpolateLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Interpolate");
}

namespace Interpolate {

std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, x, x, x}, {nChw16c}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x, x}, {nChw8c}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"jit_avx2"}, "jit_avx2"});
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x, x}, {nChw8c}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_sse42"}, "jit_sse42"});
    } else {
//    #if defined(OV_CPU_WITH_ACL)
//        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"acl"}, "acl"});
//    #else
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"ref"}, "ref"});
//    #endif
    }



    return resCPUParams;
}

std::vector<std::map<std::string, std::string>> filterAdditionalConfig() {
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        return {
            {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES}}
        };
    } else {
        return {
            // default config as an stub for target without avx512, otherwise all tests with BF16 in its name are skipped
            {{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO}}
        };
    }
}

} // namespace Interpolate
} // namespace CPULayerTestsDefinitions