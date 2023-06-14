// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/single_layer/convolution_backprop_data.hpp"

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/convolution_params.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using DeconvSpecParams = LayerTestsDefinitions::convBackpropDataSpecificParams;

using DeconvInputData = std::tuple<InputShape,                           // data shape
                                   ngraph::helpers::InputLayerType,      // 'output_shape' input type
                                   std::vector<std::vector<int32_t>>>;   // values for 'output_shape'

using DeconvLayerCPUTestParamsSet = std::tuple<DeconvSpecParams,
                                               DeconvInputData,
                                               ElementType,
                                               fusingSpecificParams,
                                               CPUSpecificParams,
                                               std::map<std::string, std::string>>;

class DeconvolutionLayerCPUTest : public testing::WithParamInterface<DeconvLayerCPUTestParamsSet>,
                                  virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DeconvLayerCPUTestParamsSet> obj);
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;
    void init_ref_function(std::shared_ptr<ov::Model> &funcRef, const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void validate() override;
    void configure_model() override;
    std::shared_ptr<ov::Model> createGraph(const std::vector<ov::PartialShape>& inShapes, ngraph::helpers::InputLayerType outShapeType);

protected:
    InferenceEngine::SizeVector kernel, stride;
    void SetUp() override;

private:
    ElementType prec;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels;
    std::vector<std::vector<int32_t>> outShapeData;
    size_t inferRequestNum = 0;
};
} // namespace CPULayerTestsDefinitions