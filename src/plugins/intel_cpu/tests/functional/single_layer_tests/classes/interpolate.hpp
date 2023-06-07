// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/interpolate.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions  {

using InterpolateSpecificParams = std::tuple<ov::op::v11::Interpolate::InterpolateMode,          // InterpolateMode
                                             ov::op::v11::Interpolate::CoordinateTransformMode,  // CoordinateTransformMode
                                             ov::op::v11::Interpolate::NearestMode,              // NearestMode
                                             bool,                                                  // AntiAlias
                                             std::vector<size_t>,                                   // PadBegin
                                             std::vector<size_t>,                                   // PadEnd
                                             double>;                                               // Cube coef

using ShapeParams = std::tuple<ov::op::v11::Interpolate::ShapeCalcMode, // ShapeCalculationMode
                               InputShape,                                 // Input shapes
                               // params describing input, choice of which depends on ShapeCalcMode
                               ngraph::helpers::InputLayerType,            // input type
                               std::vector<std::vector<float>>,            // scales or sizes values
                               std::vector<int64_t>>;                      // axes

using InterpolateLayerCPUTestParamsSet = std::tuple<InterpolateSpecificParams,
                                                    ShapeParams,
                                                    ElementType,
                                                    CPUSpecificParams,
                                                    fusingSpecificParams,
                                                    std::map<std::string, std::string>>;

class InterpolateLayerCPUTest : public testing::WithParamInterface<InterpolateLayerCPUTestParamsSet>,
                                virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateLayerCPUTestParamsSet> obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void configure_model() override;

protected:
    std::vector<std::vector<float>> scales;
    std::vector<std::vector<int32_t>> sizes;
    ov::op::v4::Interpolate::ShapeCalcMode shapeCalcMode;
    size_t inferRequestNum = 0;
    //ngraph::ParameterVector params;
    ov::op::v11::Interpolate::InterpolateMode mode;
    ov::op::v11::Interpolate::CoordinateTransformMode transfMode;
    ov::op::v11::Interpolate::NearestMode nearMode;
    bool antiAlias;
    std::vector<size_t> padBegin;
    std::vector<size_t> padEnd;
    double cubeCoef;
    ngraph::helpers::InputLayerType shapeInputType;

    void SetUp() override;
    bool isACLSupported(std::shared_ptr<ov::op::v0::Parameter> params);
};

namespace Interpolate {
    std::vector<CPUSpecificParams> filterCPUInfoForDevice();
    std::vector<std::map<std::string, std::string>> filterAdditionalConfig();

} // namespace Interpolate
} // namespace CPULayerTestsDefinitions