// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include <string>
#include <vector>
#include <iostream>

#include <arm_neon.h>
#include "ggml/ggml.h"
#include "openvino/core/type/float16.hpp"


namespace ov {
namespace intel_cpu {

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline void print_elements(const char* label, const struct ggml_tensor * t) {
    if (!t) {
        printf("%s: %s = null\n", __func__, label);
        return;
    }
    const int nelements = ggml_nelements(t);
    printf("%s: %s = [", __func__, label);
    for (int k = 0; k < nelements; ++k) {
        if (k > 0) { printf(", "); }
        printf("%.5f", ggml_get_f32_1d(t, k));
    }
    printf("] shape: [");
    for (int k = 0; k < t->n_dims; ++k) {
        if (k > 0) { printf(", "); }
        printf("%d", static_cast<int>(t->ne[k]));
    }
    printf("] (%d)\n", t->type);//0 - f32; 1 - f16
}

static void ggml_vec_dot_f32(const int n, float * s, const float * x, const float * y) {
    float sumf = 0.0f;
    const int np = (n & ~(16 - 1));

    float32x4_t sum[(16/4)] = { vdupq_n_f32(0.0f) };

    float32x4_t ax[(16/4)];
    float32x4_t ay[(16/4)];

    for (int i = 0; i < np; i += 16) {
        for (int j = 0; j < (16/4); j++) {
            ax[j] = __extension__({ float32x4_t __ret; __ret = (float32x4_t) __builtin_neon_vld1q_v(x + i + j*4, 41); __ret; });
            ay[j] = __extension__({ float32x4_t __ret; __ret = (float32x4_t) __builtin_neon_vld1q_v(y + i + j*4, 41); __ret; });

            sum[j] = vfmaq_f32(sum[j], ax[j], ay[j]);
        }
    }

    { int offset = (16/4) >> 1; for (int i = 0; i < offset; ++i) { sum[i] = vaddq_f32(sum[i], sum[offset+i]); } offset >>= 1;
    for (int i = 0; i < offset; ++i) { sum[i] = vaddq_f32(sum[i], sum[offset+i]); } offset >>= 1;
    for (int i = 0; i < offset; ++i) { sum[i] = vaddq_f32(sum[i], sum[offset+i]); } sumf = vaddvq_f32(sum[0]); };

    for (int i = np; i < n; ++i) {
        sumf += x[i]*y[i];
    }

    *s = sumf;
}

static const ggml_type_traits_t type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = sizeof(float),
        .is_quantized             = false,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
    }
};

inline static void * ggml_aligned_malloc(size_t size) {
    if (size == 0) {
        //GGML_PRINT("WARNING: Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        return NULL;
    }
    void * aligned_memory = NULL;
    int result = posix_memalign(&aligned_memory, 16, size);
    if (result != 0) {
        // Handle allocation failure
        const char *error_desc = "unknown allocation error";
        switch (result) {
            case EINVAL:
                error_desc = "invalid alignment value";
                break;
            case ENOMEM:
                error_desc = "insufficient memory";
                break;
        }
        //GGML_PRINT("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        return NULL;
    }
    return aligned_memory;
}

template <typename SrcType>
void ggml_mul_mat(int64_t M,
                  int64_t N,
                  int64_t K,
                  float* A_ptr,
                  SrcType* B_ptr,
                  float* dst_ptr,
                  const SrcType* bias_ptr) {
    const size_t mem_size = GGML_PAD(256 * 1024 * 1024, GGML_MEM_ALIGN);
    void* mem_buffer = ggml_aligned_malloc(mem_size);

    ggml_type ggmlDataType;
    if (std::is_same<SrcType, float>::value) {
        ggmlDataType = GGML_TYPE_F32;
    } else if (std::is_same<SrcType, float16>::value) {
        ggmlDataType = GGML_TYPE_F16;
    } else {
        std::cout << "data type is not supported: " << typeid(SrcType).name() << std::endl;
        return;
    }

    const int64_t ne[4] = {N, M, 1, 1};
    size_t nb[4] = {0, 0, 0, 0};
    nb[0] = ggml_type_size(GGML_TYPE_F32);
    nb[1] = nb[0] * (ne[0] / ggml_blck_size(GGML_TYPE_F32));
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }

    struct ggml_object * obj_cur = NULL;
    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;
    size_t offs = cur_end + sizeof(struct ggml_object);
    uint8_t * work_data = reinterpret_cast<uint8_t *>(mem_buffer) + offs;

    const int64_t ne00 = K;
    const int64_t ne01 = N;
    const int64_t ne02 = 1;
    const int64_t ne03 = 1;

    const size_t nb00 = ggml_type_size(ggmlDataType);
    const size_t nb01 = ggml_type_size(ggmlDataType) * (K / ggml_blck_size(ggmlDataType));
    const size_t nb02 = nb01 * ne01;
    const size_t nb03 = nb02 * ne02;

    const int64_t ne10 = K;
    const int64_t ne11 = M;
    const int64_t ne12 = 1;
    const int64_t ne13 = 1;

    const size_t nb10 = ggml_type_size(GGML_TYPE_F32);
    const size_t nb11 = ggml_type_size(GGML_TYPE_F32) * (K / ggml_blck_size(GGML_TYPE_F32));
    const size_t nb12 = nb11 * ne11;
    const size_t nb13 = nb12 * ne12;

    const int64_t ne0 = ne[0];
    const int64_t ne1 = ne[1];
    const int64_t ne2 = ne[2];
    const int64_t ne3 = ne[3];

    const size_t nb0 = nb[0];
    const size_t nb1 = nb[1];
    const size_t nb2 = nb[2];
    const size_t nb3 = nb[3];

    const int ith = 0;//thread index
    const int nth = 1;//number of threads

    const enum ggml_type type = ggmlDataType;

    const bool src1_cont = nb10 == type_traits[GGML_TYPE_F32].type_size &&
        nb11 == (nb10 * ne10) / type_traits[GGML_TYPE_F32].blck_size &&
        nb12 == nb11 * ne11 &&
        nb13 == nb12 * ne12;

    ggml_vec_dot_t    const vec_dot               = type_traits[type].vec_dot;
    enum ggml_type    const vec_dot_type          = type_traits[type].vec_dot_type;
    ggml_from_float_t const from_float_to_vec_dot = type_traits[vec_dot_type].from_float;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == type_traits[type].type_size);
    GGML_ASSERT(nb10 == sizeof(float));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

        if (GGML_TYPE_F32 != vec_dot_type) {
            char * wdata = reinterpret_cast<char *>(work_data);
            const size_t row_size = ne10*type_traits[vec_dot_type].type_size/type_traits[vec_dot_type].blck_size;

            for (int64_t i13 = 0; i13 < ne13; ++i13) {
                for (int64_t i12 = 0; i12 < ne12; ++i12) {
                    for (int64_t i11 = 0; i11 < ne11; ++i11) {
                        from_float_to_vec_dot(reinterpret_cast<float *>(reinterpret_cast<char *>(A_ptr) + i13*nb13 + i12*nb12 + i11*nb11),
                                              reinterpret_cast<void *>(wdata), ne10);
                        wdata += row_size;
                    }
                }
            }
        }

    const void * wdata    = (GGML_TYPE_F32 == vec_dot_type) ? reinterpret_cast<void* >(A_ptr) : work_data;
    const size_t row_size = ne10*type_traits[vec_dot_type].type_size/type_traits[vec_dot_type].blck_size;

    const int64_t nr0 = ne01;           // src0 rows
    const int64_t nr1 = ne11*ne12*ne13; // src1 rows

    // distribute the thread work across the inner or outer loop based on which one is larger

    const int64_t nth0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
    const int64_t nth1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows

    const int64_t ith0 = ith % nth0;
    const int64_t ith1 = ith / nth0;

    const int64_t dr0 = (nr0 + nth0 - 1)/nth0;
    const int64_t dr1 = (nr1 + nth1 - 1)/nth1;

    const int64_t ir010 = dr0*ith0;
    const int64_t ir011 = MIN(ir010 + dr0, nr0);

    const int64_t ir110 = dr1*ith1;
    const int64_t ir111 = MIN(ir110 + dr1, nr1);

    // threads with no work simply yield (not sure if it helps)
    if (ir010 >= ir011 || ir110 >= ir111) {
        sched_yield();
        return;
    }

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    // attempt to reduce false-sharing (does not seem to make a difference)
    float tmp[16];

    for (int64_t iir1 = ir110; iir1 < ir111; iir1 += blck_1) {
        for (int64_t iir0 = ir010; iir0 < ir011; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir111; ++ir1) {
                const int64_t i13 = (ir1/(ne12*ne11));
                const int64_t i12 = (ir1 - i13*ne12*ne11)/ne11;
                const int64_t i11 = (ir1 - i13*ne12*ne11 - i12*ne11);

                // broadcast src0 into src1
                const int64_t i03 = i13/r3;
                const int64_t i02 = i12/r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char *) B_ptr + (0 + i02*nb02 + i03*nb03);
                const char * src1_col = (const char *) wdata +
                    (src1_cont || GGML_TYPE_F32 != vec_dot_type
                     ? (i11      + i12*ne11 + i13*ne12*ne11)*row_size
                     : (i11*nb11 + i12*nb12 + i13*nb13));

                float * dst_col = reinterpret_cast<float *>(reinterpret_cast<char *>(dst_ptr) + (i1*nb1 + i2*nb2 + i3*nb3));
                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], src0_row + ir0*nb01, src1_col);
                }
                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir011) - iir0)*sizeof(float));
            }
        }
    }
}
}  // namespace intel_cpu
}  // namespace ov