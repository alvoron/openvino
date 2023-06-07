// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/interpolate.hpp"
#include "shared_test_classes/single_layer/interpolate.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include <ngraph_functions/builders.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;
using ngraph::helpers::operator<<;

namespace CPULayerTestsDefinitions {
namespace Interpolate {
namespace {

const std::vector<fusingSpecificParams> interpolateFusingParamsSet{
        fusingSwish,
        fusingFakeQuantizePerTensorRelu
};

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Fusing_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
             interpolateCasesNN_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Fusing_Layout_Test, InterpolateLayerCPUTest,
         ::testing::Combine(
             interpolateCasesNN_Full,
             ::testing::ValuesIn(shapeParams4D_Full),
             ::testing::Values(ElementType::f32),
             ::testing::ValuesIn(filterCPUInfoForDevice()),
             ::testing::ValuesIn(interpolateFusingParamsSet),
             ::testing::ValuesIn(filterAdditionalConfig())),
     InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinearOnnx_Fusing_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx_Fusing_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinear_Fusing_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinear_Fusing_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateCubic_Fusing_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesCubic_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateCubic_Fusing_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesCubic_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinearOnnx5D_Fusing_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx5D_Smoke,
            ::testing::ValuesIn(shapeParams5D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx5D_Fusing_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx5D_Full,
            ::testing::ValuesIn(shapeParams5D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN5D_Fusing_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN5D_Smoke,
            ::testing::ValuesIn(shapeParams5D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN5D_Fusing_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN5D_Full,
            ::testing::ValuesIn(shapeParams5D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_corner_Fusing_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCornerCases,
            ::testing::ValuesIn(shapeParams4D_corner),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_PerChannelFuse_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN_Smoke,
            ::testing::ValuesIn(shapeParams4D_fixed_C),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::Values(fusingFakeQuantizePerChannelRelu),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_PerChannelFuse_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN_Full,
            ::testing::ValuesIn(shapeParams4D_fixed_C),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::Values(fusingFakeQuantizePerChannelRelu),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

} // namespace
} // namespace Interpolate
} // namespace CPULayerTestsDefinitions