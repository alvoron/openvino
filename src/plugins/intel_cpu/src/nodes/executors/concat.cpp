// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concat.hpp"

namespace ov {
namespace intel_cpu {

using namespace InferenceEngine;

ConcatExecutor::ConcatExecutor(const ExecutorContext::CPtr context) : context(context) {}

}   // namespace intel_cpu
}   // namespace ov