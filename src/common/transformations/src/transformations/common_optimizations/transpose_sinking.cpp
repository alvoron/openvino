// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_sinking.hpp"

#include <exception>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/logical_reduction_keep_dims.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/print_model.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

using ov::pass::pattern::any_input;
using ov::pass::pattern::consumers_count;
using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace op_util = ov::op::util;
namespace {

constexpr const char* transpose_fq_log_prefix = "[TransposeFQ]";

template <typename T>
std::string to_string(const std::vector<T>& values) {
    std::ostringstream oss;
    oss << "[";
    const char* sep = "";
    for (const auto& value : values) {
        oss << sep << value;
        sep = ", ";
    }
    oss << "]";
    return oss.str();
}

std::string to_string(const Shape& shape) {
    std::ostringstream oss;
    oss << "[";
    const char* sep = "";
    for (const auto& dim : shape) {
        oss << sep << dim;
        sep = ", ";
    }
    oss << "]";
    return oss.str();
}

std::string describe_output(const Output<Node>& output) {
    std::ostringstream oss;
    const auto source_node = output.get_node_shared_ptr();
    oss << source_node->get_friendly_name() << ":" << output.get_index() << " type=" << source_node->get_type_name()
        << " et=" << output.get_element_type().get_type_name() << " ps=" << output.get_partial_shape().to_string();
    return oss.str();
}

std::string describe_targets(const Output<Node>& output) {
    std::ostringstream oss;
    oss << "[";
    const char* sep = "";
    for (const auto& target_input : output.get_target_inputs()) {
        oss << sep << target_input.get_node()->get_friendly_name() << ":input(" << target_input.get_index() << ")";
        sep = ", ";
    }
    oss << "]";
    return oss.str();
}

std::string describe_attributes(const std::shared_ptr<Node>& node) {
    std::ostringstream oss;
    ov::pass::detail::OstreamAttributeVisitor visitor(oss);
    node->visit_attributes(visitor);
    return oss.str();
}

void log_message(const std::string& message) {
    std::cout << transpose_fq_log_prefix << " " << message << std::endl;
}

void log_node(const std::string& label, const std::shared_ptr<Node>& node) {
    if (!node) {
        log_message(label + ": <null>");
        return;
    }

    std::ostringstream header;
    header << label << ": name=" << node->get_friendly_name() << ", type=" << node->get_type_name()
           << ", inputs=" << node->get_input_size() << ", outputs=" << node->get_output_size();
    log_message(header.str());

    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        log_message("  input[" + std::to_string(input_idx) + "] <- " + describe_output(node->input_value(input_idx)));
    }

    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        const auto output = node->output(output_idx);
        log_message("  output[" + std::to_string(output_idx) + "] -> " + describe_output(output) +
                    ", consumers=" + describe_targets(output));
    }

    if (const auto constant = ov::as_type_ptr<v0::Constant>(node)) {
        log_message("  constant shape=" + to_string(constant->get_shape()) + ", values=" +
                    ov::pass::detail::to_code(constant, true));
        return;
    }

    if (const auto fake_quantize = ov::as_type_ptr<v0::FakeQuantize>(node)) {
        std::ostringstream fq_info;
        const auto& autob = fake_quantize->get_auto_broadcast();
        fq_info << "  fake_quantize levels=" << fake_quantize->get_levels() << ", auto_broadcast.type="
                << autob.m_type << ", auto_broadcast.axis=" << autob.m_axis;
        log_message(fq_info.str());
    }

    const auto attributes = describe_attributes(node);
    if (!attributes.empty()) {
        log_message("  attrs={" + attributes + "}");
    }
}

void append_unique_node(std::vector<std::shared_ptr<Node>>& nodes, const std::shared_ptr<Node>& node) {
    if (!node) {
        return;
    }

    const auto same_node = [&node](const std::shared_ptr<Node>& existing_node) {
        return existing_node.get() == node.get();
    };
    if (std::find_if(nodes.begin(), nodes.end(), same_node) == nodes.end()) {
        nodes.push_back(node);
    }
}

void log_transpose_fq_subgraph(const std::string& stage,
                               const std::shared_ptr<Node>& transpose,
                               const std::shared_ptr<Node>& fq,
                               const NodeVector& extra_nodes = {}) {
    std::vector<std::shared_ptr<Node>> nodes;
    if (transpose) {
        for (const auto& input : transpose->input_values()) {
            append_unique_node(nodes, input.get_node_shared_ptr());
        }
        append_unique_node(nodes, transpose);
    }

    if (fq) {
        for (const auto& input : fq->input_values()) {
            append_unique_node(nodes, input.get_node_shared_ptr());
        }
        append_unique_node(nodes, fq);
    }

    for (const auto& node : extra_nodes) {
        append_unique_node(nodes, node);
    }

    log_message(stage + " subgraph snapshot begin");
    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        log_node(stage + " node[" + std::to_string(idx) + "]", nodes[idx]);
    }
    log_message(stage + " subgraph snapshot end");
}

std::shared_ptr<v0::Constant> get_reduced_order_constant(const std::shared_ptr<v0::Constant>& axes_const,
                                                         const std::shared_ptr<v0::Constant>& order_const) {
    auto order = order_const->cast_vector<int64_t>();

    auto axes = axes_const->cast_vector<int64_t>();
    if (!axes.empty()) {
        std::sort(axes.rbegin(), axes.rend());
        for (const auto& i : axes)
            order.erase(order.begin() + i);
    } else {
        // if 2nd input for Squeeze op is not provided, we should remove all 1 dims
        // this case will be supported in new TSGeneral transformation.
        return nullptr;
    }

    const auto& updated_order_size = static_cast<int64_t>(order.size());

    auto order_sorted = order;
    sort(order_sorted.begin(), order_sorted.end());
    for (int64_t i = 0; i < updated_order_size; ++i) {
        auto lowest_greater_eq_i = std::lower_bound(order_sorted.begin(), order_sorted.end(), i);
        std::replace(order.begin(), order.end(), *lowest_greater_eq_i, i);
        std::replace(order_sorted.begin(), order_sorted.end(), *lowest_greater_eq_i, i);
    }
    return std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{order.size()}, order);
}

std::shared_ptr<v0::Constant> get_reversed_order_constant(const std::shared_ptr<v0::Constant>& order_const) {
    const auto& order = order_const->cast_vector<size_t>();
    const auto& rank = order.size();
    AxisVector default_order(rank);
    std::iota(begin(default_order), end(default_order), 0);
    std::vector<size_t> reverse_order(rank);
    for (size_t i = 0; i < rank; ++i)
        reverse_order[order[i]] = default_order[i];

    return std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{reverse_order.size()}, reverse_order);
}

}  // namespace

ov::pass::TransposeFQ::TransposeFQ() {
    MATCHER_SCOPE(TransposeFQ);

    auto transpose_order_m = wrap_type<v0::Constant>();
    auto transpose_label = wrap_type<v1::Transpose>({any_input(pattern::has_static_rank()), transpose_order_m});
    auto fq_label = wrap_type<v0::FakeQuantize>({transpose_label,
                                                 any_input(ov::pass::pattern::has_static_rank()),
                                                 any_input(ov::pass::pattern::has_static_rank()),
                                                 any_input(ov::pass::pattern::has_static_rank()),
                                                 any_input(ov::pass::pattern::has_static_rank())},
                                                consumers_count(1));

    matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();

        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto fq = pattern_to_output.at(fq_label).get_node_shared_ptr();
        auto transpose_order =
            ov::as_type_ptr<v0::Constant>(pattern_to_output.at(transpose_order_m).get_node_shared_ptr());
        log_message("matcher callback triggered");
        log_node("matched transpose", transpose);
        log_node("matched fake_quantize", fq);
        log_node("matched transpose order", transpose_order);
        if (!transpose_order || !fq) {
            log_message("mandatory match objects are missing; exiting without rewrite");
            return false;
        }

        log_transpose_fq_subgraph("before", transpose, fq);

        ov::NodeVector new_ops;
        bool expanded_any_range_input = false;
        bool saw_non_static_range_input = false;

        const auto& reverse_order_constant = get_reversed_order_constant(transpose_order);
        log_message("reverse transpose order computed: " +
                    ov::pass::detail::to_code(reverse_order_constant, true));
        log_node("reverse order constant", reverse_order_constant);
        new_ops.push_back(reverse_order_constant);

        const auto& input_rank = fq->get_input_partial_shape(0).rank().get_length();
        log_message("fake_quantize data input rank=" + std::to_string(input_rank));
        ov::OutputVector fq_inputs = {transpose->input_value(0)};
        log_message("new fake_quantize input[0] will reuse transpose parent: " + describe_output(transpose->input_value(0)));
        for (size_t i = 1; i < fq->inputs().size(); ++i) {
            auto input = fq->input_value(i);
            log_message("processing fake_quantize range input[" + std::to_string(i) + "]");
            log_message("  original input=" + describe_output(input));
            log_node("  original range node", input.get_node_shared_ptr());
            log_message("  static shape available=" + std::string(input.get_partial_shape().is_static() ? "true" : "false"));
            if (!input.get_partial_shape().is_static()) {
                saw_non_static_range_input = true;
                log_message("  input shape is dynamic; existing code will still query get_shape() for scalar fast path");
            }

            size_t input_shape_size = 0;
            try {
                input_shape_size = ov::shape_size(input.get_shape());
            } catch (const std::exception& ex) {
                log_message("  input.get_shape() threw: " + std::string(ex.what()));
                log_message("  this directly tests the partially-dynamic range-shape hypothesis");
                throw;
            }
            log_message("  shape_size=" + std::to_string(input_shape_size));
            if (input_shape_size == 1) {
                log_message("  scalar-or-broadcast-scalar fast path taken; transpose/unsqueeze skipped");
                fq_inputs.push_back(input);
                continue;
            }

            const auto& range_rank = input.get_partial_shape().rank().get_length();
            log_message("  range rank=" + std::to_string(range_rank));
            if (range_rank > input_rank) {
                log_message("  range rank is greater than data input rank; exiting without rewrite");
                return false;
            }

            const auto& ranks_diff = input_rank - range_rank;
            log_message("  rank difference relative to data input=" + std::to_string(ranks_diff));
            if (ranks_diff > 0) {
                expanded_any_range_input = true;
                log_message("  unsqueeze branch taken");
                std::vector<int64_t> axes(ranks_diff);
                std::iota(axes.begin(), axes.end(), 0);
                log_message("  unsqueeze axes=" + to_string(axes));
                const auto& axes_const = v0::Constant::create(element::i64, Shape{axes.size()}, axes);
                new_ops.push_back(axes_const);
                log_node("  unsqueeze axes constant", axes_const);
                const auto& unsqueezed_input = op_util::make_try_fold<v0::Unsqueeze>(input, axes_const);
                new_ops.push_back(unsqueezed_input);
                log_node("  unsqueezed range input", unsqueezed_input);
                input = unsqueezed_input->output(0);
            } else {
                log_message("  unsqueeze branch skipped");
            }
            const auto& transposed_input = op_util::make_try_fold<v1::Transpose>(input, reverse_order_constant);
            new_ops.push_back(transposed_input);
            log_node("  transposed range input", transposed_input);
            fq_inputs.push_back(transposed_input);
            log_message("  new fake_quantize range input[" + std::to_string(i) + "]=" +
                        describe_output(transposed_input->output(0)));
        }

        log_message("candidate fake_quantize inputs summary before clone begin");
        for (size_t i = 0; i < fq_inputs.size(); ++i) {
            log_message("  candidate fq_inputs[" + std::to_string(i) + "]=" + describe_output(fq_inputs[i]));
        }
        log_message("candidate fake_quantize inputs summary before clone end");

        if (const auto fake_quantize = ov::as_type_ptr<v0::FakeQuantize>(fq)) {
            const auto& autob = fake_quantize->get_auto_broadcast();
            std::ostringstream autob_summary;
            autob_summary << "fake_quantize clone precheck: auto_broadcast.type=" << autob.m_type
                          << ", auto_broadcast.axis=" << autob.m_axis
                          << ", expanded_any_range_input=" << (expanded_any_range_input ? "true" : "false")
                          << ", saw_non_static_range_input=" << (saw_non_static_range_input ? "true" : "false");
            log_message(autob_summary.str());
            if (autob.m_type == ov::op::AutoBroadcastType::PDPD && expanded_any_range_input) {
                log_message("PDPD broadcast with rank expansion detected; this is the primary broadcast-axis risk scenario");
            }
        }

        std::shared_ptr<Node> new_fq;
        try {
            new_fq = fq->clone_with_new_inputs(fq_inputs);
        } catch (const std::exception& ex) {
            log_message("FakeQuantize::clone_with_new_inputs threw: " + std::string(ex.what()));
            log_message("this directly tests the broadcast-mode / axis preservation hypothesis");
            throw;
        }
        new_ops.push_back(new_fq);
        log_node("new fake_quantize", new_fq);

        auto new_transpose = register_new_node<v1::Transpose>(new_fq, transpose_order);
        new_ops.push_back(new_transpose);
        new_transpose->set_friendly_name(fq->get_friendly_name());
        log_node("new transpose", new_transpose);

        log_message("new fake_quantize inputs summary begin");
        for (size_t i = 0; i < fq_inputs.size(); ++i) {
            log_message("  fq_inputs[" + std::to_string(i) + "]=" + describe_output(fq_inputs[i]));
        }
        log_message("new fake_quantize inputs summary end");

        log_transpose_fq_subgraph("after", new_transpose, new_fq, new_ops);

        ov::copy_runtime_info({fq, transpose}, new_ops);
        log_message("runtime info copied from original transpose/fake_quantize to new nodes");
        log_message("replacing original fake_quantize '" + fq->get_friendly_name() + "' with new transpose '" +
                    new_transpose->get_friendly_name() + "'");
        ov::replace_node(fq, new_transpose);
        log_message("rewrite completed successfully");
        return true;
    };

    auto m = std::make_shared<Matcher>(fq_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::TransposeEltwise::TransposeEltwise() {
    MATCHER_SCOPE(TransposeEltwise);

    auto eltwise_data_input_p = any_input();
    auto eltwise_const_input_p = wrap_type<v0::Constant>();
    auto eltwise_p = wrap_type<op_util::BinaryElementwiseArithmetic>(
        {eltwise_data_input_p, eltwise_const_input_p},
        [](const Output<Node>& output) {
            return ov::is_preprocesing_node(output.get_node_shared_ptr());
        });
    auto transpose_p = wrap_type<v1::Transpose>({eltwise_p, wrap_type<v0::Constant>()}, consumers_count(1));

    auto callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto eltwise = pattern_to_output.at(eltwise_p).get_node_shared_ptr();
        auto eltwise_const_input = pattern_to_output.at(eltwise_const_input_p);
        auto eltwise_data_input = pattern_to_output.at(eltwise_data_input_p);
        auto transpose = pattern_to_output.at(transpose_p).get_node_shared_ptr();

        const auto& order_size = transpose->get_input_shape(1).at(0);
        const auto& shape = eltwise_const_input.get_shape();
        if (shape.size() != order_size && ov::shape_size(shape) != 1) {
            // TODO: temporary restrictions
            return false;
        }

        if (ov::shape_size(shape) != 1) {
            eltwise_const_input = std::make_shared<v1::Transpose>(eltwise_const_input, transpose->input_value(1));
            if (auto const_node = ov::util::get_constant_from_source(eltwise_const_input)) {
                eltwise_const_input = const_node;
            }
        }

        auto new_transpose = transpose->clone_with_new_inputs({eltwise_data_input, transpose->input_value(1)});
        auto new_eltwise = eltwise->clone_with_new_inputs({new_transpose, eltwise_const_input});
        register_new_node(new_transpose);

        new_transpose->set_friendly_name(eltwise->get_friendly_name());
        copy_runtime_info({eltwise, transpose}, {new_transpose, new_eltwise});
        replace_node(transpose, new_eltwise);
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_p, matcher_name);
    register_matcher(m, callback);
}

ov::pass::TransposeConvert::TransposeConvert() {
    MATCHER_SCOPE(TransposeConvert);

    auto transpose_label = wrap_type<v1::Transpose>({any_input(), wrap_type<v0::Constant>()}, consumers_count(1));
    auto convert_label = wrap_type<v0::Convert>({transpose_label});

    matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto convert = pattern_to_output.at(convert_label).get_node_shared_ptr();

        auto new_convert = convert->clone_with_new_inputs({transpose->input_value(0)});
        auto new_transpose = transpose->clone_with_new_inputs({new_convert, transpose->input_value(1)});
        register_new_node(new_transpose);

        new_transpose->set_friendly_name(convert->get_friendly_name());
        copy_runtime_info({transpose, convert}, {new_convert, new_transpose});
        replace_node(convert, new_transpose);
        return true;
    };

    auto m = std::make_shared<Matcher>(convert_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::TransposeReduction::TransposeReduction() {
    MATCHER_SCOPE(TransposeReduction);

    auto transpose_label = wrap_type<v1::Transpose>({any_input(), wrap_type<v0::Constant>()}, consumers_count(1));
    auto reduce_or_squeeze_label =
        wrap_type<op_util::ArithmeticReductionKeepDims, op_util::LogicalReductionKeepDims, v0::Squeeze>(
            {transpose_label, wrap_type<v0::Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto reduction = pattern_to_output.at(reduce_or_squeeze_label).get_node_shared_ptr();
        auto arithmetic_reduce = ov::as_type_ptr<op_util::ArithmeticReductionKeepDims>(reduction);
        auto logical_reduce = ov::as_type_ptr<op_util::LogicalReductionKeepDims>(reduction);
        auto squeeze = ov::as_type_ptr<v0::Squeeze>(reduction);
        if (!transpose || !(arithmetic_reduce || logical_reduce || squeeze))
            return false;

        bool keep_dims = false;  // squeeze always reduces number of output dimensions
        if (logical_reduce)
            keep_dims = logical_reduce->get_keep_dims();
        else if (arithmetic_reduce)
            keep_dims = arithmetic_reduce->get_keep_dims();

        auto transpose_order = ov::as_type_ptr<v0::Constant>(transpose->get_input_node_shared_ptr(1));
        auto reduction_axes = ov::as_type_ptr<v0::Constant>(reduction->get_input_node_shared_ptr(1));
        if (!transpose_order || !reduction_axes)
            return false;

        const auto non_negative_axes =
            util::try_get_normalized_axis_vector(reduction_axes->get_tensor_view(),
                                                 reduction->get_input_partial_shape(0).rank(),
                                                 *reduction);
        reduction_axes = v0::Constant::create(ov::element::i64, {non_negative_axes.size()}, non_negative_axes);

        ov::NodeVector new_ops;
        auto new_axes = op_util::make_try_fold<v1::Gather>(transpose_order,
                                                           reduction_axes,
                                                           v0::Constant::create(ov::element::i64, {}, {0}));
        new_ops.push_back(new_axes);
        auto new_reduce = reduction->clone_with_new_inputs({transpose->input_value(0), new_axes});
        new_ops.push_back(new_reduce);

        auto updated_order = transpose_order;
        if (!keep_dims) {
            updated_order = get_reduced_order_constant(reduction_axes, transpose_order);
            new_ops.push_back(updated_order);
        }

        if (!updated_order) {
            return false;
        }
        auto new_transpose = register_new_node<v1::Transpose>(new_reduce, updated_order);
        new_ops.push_back(new_transpose);
        new_transpose->set_friendly_name(reduction->get_friendly_name());

        ov::copy_runtime_info({reduction, transpose}, new_ops);
        ov::replace_node(reduction, new_transpose);

        return true;
    };

    auto m = std::make_shared<Matcher>(reduce_or_squeeze_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::TransposeFQReduction::TransposeFQReduction() {
    MATCHER_SCOPE(TransposeFQReduction);

    auto transpose_label = wrap_type<v1::Transpose>({any_input(), wrap_type<v0::Constant>()});
    auto fq_label = wrap_type<v0::FakeQuantize>({transpose_label,
                                                 any_input(ov::pass::pattern::has_static_rank()),
                                                 any_input(ov::pass::pattern::has_static_rank()),
                                                 any_input(ov::pass::pattern::has_static_rank()),
                                                 any_input(ov::pass::pattern::has_static_rank())});
    auto reduce_or_squeeze_label =
        wrap_type<op_util::ArithmeticReductionKeepDims, op_util::LogicalReductionKeepDims, v0::Squeeze>(
            {fq_label, wrap_type<v0::Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();

        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        if (!transpose)
            return false;

        auto transpose_order = ov::as_type_ptr<v0::Constant>(transpose->get_input_node_shared_ptr(1));
        auto fq = pattern_to_output.at(fq_label).get_node_shared_ptr();
        if (!transpose_order || !fq)
            return false;

        ov::NodeVector new_ops;

        const auto& reverse_order_constant = get_reversed_order_constant(transpose_order);
        new_ops.push_back(reverse_order_constant);

        const auto& input_rank = fq->get_input_partial_shape(0).rank().get_length();
        ov::OutputVector fq_inputs = {transpose->input_value(0)};
        for (size_t i = 1; i < fq->inputs().size(); ++i) {
            auto input = fq->input_value(i);
            const auto& ranks_diff = input_rank - input.get_partial_shape().rank().get_length();
            OPENVINO_ASSERT(ranks_diff >= 0);
            if (ranks_diff > 0) {
                std::vector<int64_t> axes(ranks_diff);
                std::iota(axes.begin(), axes.end(), 0);
                const auto& axes_const = v0::Constant::create(element::i64, Shape{axes.size()}, axes);
                new_ops.push_back(axes_const);
                const auto& unsqueezed_input = op_util::make_try_fold<v0::Unsqueeze>(input, axes_const);
                new_ops.push_back(unsqueezed_input);
                input = unsqueezed_input->output(0);
            }
            const auto& transposed_input = op_util::make_try_fold<v1::Transpose>(input, reverse_order_constant);
            new_ops.push_back(transposed_input);
            fq_inputs.push_back(transposed_input);
        }
        auto new_fq = fq->clone_with_new_inputs(fq_inputs);
        new_ops.push_back(new_fq);

        auto new_transpose = register_new_node<v1::Transpose>(new_fq, transpose_order);
        new_ops.push_back(new_transpose);
        new_transpose->set_friendly_name(fq->get_friendly_name());

        ov::copy_runtime_info({fq, transpose}, new_ops);
        ov::replace_node(fq, new_transpose);
        // The root node (reduction) left unchanged during current matcher pass.
        // We return false here for further MatcherPasses to be applicable for this node as a root node
        return false;
    };

    auto m = std::make_shared<Matcher>(reduce_or_squeeze_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::TransposeFuse::TransposeFuse() {
    MATCHER_SCOPE(TransposeFuse);

    auto transpose_1 = wrap_type<v1::Transpose>({any_input(), wrap_type<v0::Constant>()}, consumers_count(1));
    auto transpose_2 = wrap_type<v1::Transpose>({transpose_1, wrap_type<v0::Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto transpose1 = pattern_to_output.at(transpose_1).get_node_shared_ptr();
        auto transpose2 = pattern_to_output.at(transpose_2).get_node_shared_ptr();
        auto input = transpose1->input_value(0);

        auto transpose1_order = ov::as_type_ptr<v0::Constant>(transpose1->get_input_node_shared_ptr(1));
        auto transpose2_order = ov::as_type_ptr<v0::Constant>(transpose2->get_input_node_shared_ptr(1));
        if (!transpose1_order || !transpose2_order)
            return false;

        auto order1 = transpose1_order->cast_vector<int64_t>();
        auto order2 = transpose2_order->cast_vector<int64_t>();
        if (order1.size() != order2.size())
            return false;

        bool is_ordered = true;
        for (size_t i = 0; i < order1.size(); i++) {
            order2[i] = order1[order2[i]];
            if (order2[i] != (int64_t)i)
                is_ordered = false;
        }

        auto transpose_order_type = transpose1_order->get_element_type();
        if (transpose_order_type != transpose2_order->get_element_type())
            transpose_order_type = element::i64;

        if (is_ordered) {
            return ov::replace_output_update_name(transpose2->output(0), input);
        } else {
            auto new_order = v0::Constant::create(transpose_order_type, {order2.size()}, order2);
            auto new_transpose = register_new_node<v1::Transpose>(input, new_order);

            new_transpose->set_friendly_name(m.get_match_root()->get_friendly_name());
            ov::copy_runtime_info({transpose1, transpose2}, new_transpose);
            ov::replace_node(m.get_match_root(), new_transpose);
        }

        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_2, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
