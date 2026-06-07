#include "common.h"
#include "turboquant.h"

// Standalone TurboQuant kernels.  Operation-specific templates stay in their
// upstream split sources; only kernels with no upstream template are here.

kernel void kernel_turbo4_dequant_f16(
        device const block_turbo4_0 * src [[buffer(0)]],
        device       half           * dst [[buffer(1)]],
        constant     uint           & n_blocks [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const uint block = tgpig * ntg + tiitg;
    if (block >= n_blocks) return;
    device const block_turbo4_0 & in = src[block];
    device half * out = dst + block * QK_TURBO4;
    for (int i = 0; i < QK_TURBO4; i += 2) {
        const uint8_t q = in.qs[i / 2];
        out[i] = turbo_centroids_4bit_h[q & 15] * in.norm;
        out[i + 1] = turbo_centroids_4bit_h[q >> 4] * in.norm;
    }
}

kernel void kernel_turbo_wht(
        constant ggml_metal_kargs_turbo_wht & args,
        device const float * src [[buffer(1)]],
        device       float * dst [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const int64_t group = tgpig * ntg + tiitg;
    if (group >= args.n_elements / 128) return;
    device const float * in = src + group * 128;
    device float * out = dst + group * 128;
    const bool inverse = args.direction == 1;
    float4 v[32];
    for (int i = 0; i < 32; ++i) {
        const float4 x = float4(in[i * 4], in[i * 4 + 1], in[i * 4 + 2], in[i * 4 + 3]);
        v[i] = x * (inverse ? float4(turbo_wht_signs2[i * 4], turbo_wht_signs2[i * 4 + 1], turbo_wht_signs2[i * 4 + 2], turbo_wht_signs2[i * 4 + 3]) : float4(turbo_wht_signs1[i * 4], turbo_wht_signs1[i * 4 + 1], turbo_wht_signs1[i * 4 + 2], turbo_wht_signs1[i * 4 + 3]));
    }
    for (int i = 0; i < 32; ++i) {
        const float4 x = v[i];
        v[i] = float4(x.x + x.y, x.x - x.y, x.z + x.w, x.z - x.w);
    }
    for (int i = 0; i < 32; ++i) {
        const float4 x = v[i];
        v[i] = float4(x.x + x.z, x.y + x.w, x.x - x.z, x.y - x.w);
    }
    for (int h = 4; h < 128; h *= 2) {
        const int stride = h / 4;
        for (int i = 0; i < 32; ++i) {
            if (i % (2 * stride) < stride) {
                const int j = i + stride;
                const float4 a = v[i], b = v[j];
                v[i] = a + b;
                v[j] = a - b;
            }
        }
    }
    const float4 scale = float4(0.08838834764831845f);
    for (int i = 0; i < 32; ++i) {
        const float4 signs = inverse
            ? float4(turbo_wht_signs1[i * 4], turbo_wht_signs1[i * 4 + 1], turbo_wht_signs1[i * 4 + 2], turbo_wht_signs1[i * 4 + 3])
            : float4(turbo_wht_signs2[i * 4], turbo_wht_signs2[i * 4 + 1], turbo_wht_signs2[i * 4 + 2], turbo_wht_signs2[i * 4 + 3]);
        const float4 x = v[i] * scale * signs;
        out[i * 4] = x.x; out[i * 4 + 1] = x.y;
        out[i * 4 + 2] = x.z; out[i * 4 + 3] = x.w;
    }
}

// TQ3_1S and TQ4_1S use a 32-element RHT for the mul_mm path.  The same
// kernels are shared by both formats because the transform is format-neutral.
kernel void kernel_tq3_rotate_act(
        device float * x [[buffer(0)]],
        constant int64_t & n [[buffer(1)]],
        uint tpig [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int64_t base = (int64_t(tpig) / 32) * 32;
    if (base >= n) return;
    float value = x[base + tiisg] * tq3_signs[tiisg];
    for (ushort step = 1; step < 32; step <<= 1) {
        const float other = simd_shuffle_xor(value, step);
        value = (tiisg & step) ? other - value : other + value;
    }
    x[base + tiisg] = value * tq3_inv_sqrt32;
}

kernel void kernel_tq3_unrotate_act(
        device float * x [[buffer(0)]],
        constant int64_t & n [[buffer(1)]],
        uint tpig [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int64_t base = (int64_t(tpig) / 32) * 32;
    if (base >= n) return;
    float value = x[base + tiisg];
    for (ushort step = 1; step < 32; step <<= 1) {
        const float other = simd_shuffle_xor(value, step);
        value = (tiisg & step) ? other - value : other + value;
    }
    x[base + tiisg] = value * tq3_inv_sqrt32 * tq3_signs[tiisg];
}

template <typename TI, typename block_q, int QK, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_turbo(
        constant ggml_metal_kargs_set_rows & args,
        device const void * src0, device const void * src1, device float * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint3 tptg [[threads_per_threadgroup]]) {
    const int i03 = tgpig.z, i02 = tgpig.y;
    const int i12 = i03 % args.ne12, i11 = i02 % args.ne11;
    const int i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;
    const TI i1 = ((device const TI *) ((device const char *) src1 + i01*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];
    device block_q * dst_row = (device block_q *) ((device char *) dst + i1*args.nb1 + i02*args.nb2 + i03*args.nb3);
    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    for (int ind = tiitg % tptg.x; ind < args.nk0; ind += tptg.x) {
        const device float * group = src_row + QK*ind;
        float norm_sq = 0.0f;
        for (int j = 0; j < QK; ++j) norm_sq += group[j] * group[j];
        const float norm = sqrt(norm_sq);
        const float inv_norm = norm > 1e-10f ? 1.0f/norm : 0.0f;
        float rotated[QK];
        for (int j = 0; j < QK; ++j) rotated[j] = group[j] * inv_norm;
        turbo_rotate_forward(rotated, turbo_wht_signs1, turbo_wht_signs2);
        quantize_func(rotated, dst_row[ind]);
        float reconstruction = 0.0f;
        for (int j = 0; j < QK; ++j) {
            const uint8_t q = turbo_index_3bit(rotated[j]);
            reconstruction += turbo_centroids_3bit[q] * turbo_centroids_3bit[q];
        }
        dst_row[ind].norm = half(sqrt(reconstruction) > 1e-10f ? norm/sqrt(reconstruction) : norm);
    }
}

template <typename TI>
kernel void kernel_set_rows_turbo2(
        constant ggml_metal_kargs_set_rows & args,
        device const void * src0, device const void * src1, device float * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint3 tptg [[threads_per_threadgroup]]) {
    const int i03 = tgpig.z, i02 = tgpig.y;
    const int i12 = i03 % args.ne12, i11 = i02 % args.ne11;
    const int i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;
    const TI i1 = ((device const TI *) ((device const char *) src1 + i01*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];
    device block_turbo2_0 * dst_row = (device block_turbo2_0 *) ((device char *) dst + i1*args.nb1 + i02*args.nb2 + i03*args.nb3);
    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    for (int ind = tiitg % tptg.x; ind < args.nk0; ind += tptg.x) {
        const device float * group = src_row + QK_TURBO2*ind;
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO2; ++j) norm_sq += group[j] * group[j];
        const float norm = sqrt(norm_sq), inv_norm = norm > 1e-10f ? 1.0f/norm : 0.0f;
        float rotated[QK_TURBO2];
        for (int j = 0; j < QK_TURBO2; ++j) rotated[j] = group[j] * inv_norm;
        turbo_rotate_forward(rotated, turbo_wht_signs1, turbo_wht_signs2);
        device block_turbo2_0 & block = dst_row[ind];
        for (int j = 0; j < QK_TURBO2/4; ++j) block.qs[j] = 0;
        float reconstruction = 0.0f;
        for (int j = 0; j < QK_TURBO2; ++j) {
            const uint8_t q = turbo_index_2bit(rotated[j]);
            block.qs[j/4] |= q << ((j & 3)*2);
            reconstruction += turbo_centroids_2bit[q] * turbo_centroids_2bit[q];
        }
        block.norm = half(sqrt(reconstruction) > 1e-10f ? norm/sqrt(reconstruction) : norm);
    }
}

template <typename TI>
kernel void kernel_set_rows_turbo4(
        constant ggml_metal_kargs_set_rows & args,
        device const void * src0, device const void * src1, device float * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint3 tptg [[threads_per_threadgroup]]) {
    const int i03 = tgpig.z, i02 = tgpig.y;
    const int i12 = i03 % args.ne12, i11 = i02 % args.ne11;
    const int i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;
    const TI i1 = ((device const TI *) ((device const char *) src1 + i01*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];
    device block_turbo4_0 * dst_row = (device block_turbo4_0 *) ((device char *) dst + i1*args.nb1 + i02*args.nb2 + i03*args.nb3);
    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    for (int ind = tiitg % tptg.x; ind < args.nk0; ind += tptg.x) {
        const device float * group = src_row + QK_TURBO4*ind;
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4; ++j) norm_sq += group[j] * group[j];
        const float norm = sqrt(norm_sq), inv_norm = norm > 1e-10f ? 1.0f/norm : 0.0f;
        float rotated[QK_TURBO4];
        for (int j = 0; j < QK_TURBO4; ++j) rotated[j] = group[j] * inv_norm;
        turbo_rotate_forward(rotated, turbo_wht_signs1, turbo_wht_signs2);
        device block_turbo4_0 & block = dst_row[ind];
        for (int j = 0; j < QK_TURBO4/2; ++j) block.qs[j] = 0;
        float reconstruction = 0.0f;
        for (int j = 0; j < QK_TURBO4; ++j) {
            const uint8_t q = turbo_index_4bit(rotated[j]);
            block.qs[j/2] |= q << ((j & 1)*4);
            reconstruction += turbo_centroids_4bit[q] * turbo_centroids_4bit[q];
        }
        block.rnorm = half(0.0f);
        block.norm = half(sqrt(reconstruction) > 1e-10f ? norm/sqrt(reconstruction) : norm);
    }
}

typedef decltype(kernel_set_rows_turbo<int64_t, block_turbo3_0, QK_TURBO3, quantize_turbo3_0>) set_rows_turbo3_t;
template [[host_name("kernel_set_rows_turbo3_i64")]] kernel set_rows_turbo3_t kernel_set_rows_turbo<int64_t, block_turbo3_0, QK_TURBO3, quantize_turbo3_0>;
template [[host_name("kernel_set_rows_turbo3_i32")]] kernel set_rows_turbo3_t kernel_set_rows_turbo<int32_t, block_turbo3_0, QK_TURBO3, quantize_turbo3_0>;
typedef decltype(kernel_set_rows_turbo2<int64_t>) set_rows_turbo2_t;
template [[host_name("kernel_set_rows_turbo2_i64")]] kernel set_rows_turbo2_t kernel_set_rows_turbo2<int64_t>;
template [[host_name("kernel_set_rows_turbo2_i32")]] kernel set_rows_turbo2_t kernel_set_rows_turbo2<int32_t>;
typedef decltype(kernel_set_rows_turbo4<int64_t>) set_rows_turbo4_t;
template [[host_name("kernel_set_rows_turbo4_i64")]] kernel set_rows_turbo4_t kernel_set_rows_turbo4<int64_t>;
template [[host_name("kernel_set_rows_turbo4_i32")]] kernel set_rows_turbo4_t kernel_set_rows_turbo4<int32_t>;

// TurboFlash: two-pass fused attention for q8_0/TurboQuant KV caches.
#define TURBO_FLASH_BLOCK_SIZE 64

constant int32_t FC_turbo_flash_p1_dk [[function_constant(FC_TURBO_FLASH_P1 + 0)]];
constant int32_t FC_turbo_flash_p1_dv [[function_constant(FC_TURBO_FLASH_P1 + 1)]];
constant bool FC_turbo_flash_p1_has_mask [[function_constant(FC_TURBO_FLASH_P1 + 2)]];
constant bool FC_turbo_flash_p1_k_is_turbo3 [[function_constant(FC_TURBO_FLASH_P1 + 3)]];
constant int32_t FC_turbo_flash_p2_dv [[function_constant(FC_TURBO_FLASH_P2 + 0)]];

template <short DK, short DV>
kernel void kernel_turbo_flash_p1(
        constant ggml_metal_kargs_turbo_flash_p1 & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       float * partial_out,
        device       float * partial_ms,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiitg [[thread_index_in_threadgroup]]) {
    (void) shmem;

    constexpr short DK_PER_LANE = DK / 32;
    constexpr short DV_PER_LANE = DV / 32;

    const uint lane = tiitg % 32;
    const uint bh_idx = tgpig[0];
    const uint block_id = tgpig[1];
    const int n_blocks = args.n_blocks;
    const int t_start = int(block_id * TURBO_FLASH_BLOCK_SIZE);
    const int t_end = min(t_start + TURBO_FLASH_BLOCK_SIZE, args.ne11);

    const uint iq1 = bh_idx % args.ne01;
    const uint iq2 = (bh_idx / args.ne01) % args.ne02;
    const uint iq3 = bh_idx / (args.ne01 * args.ne02);
    const uint ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const uint ikv3 = iq3 / (args.ne03 / args.ne_12_3);

    device const float * q_ptr = (device const float *) ((device const char *) q +
        iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03);
    float q_vals[DK_PER_LANE];
    for (short i = 0; i < DK_PER_LANE; ++i) {
        const int d = int(lane) + i*32;
        q_vals[i] = d < DK ? q_ptr[d] : 0.0f;
    }

    float cb[8];
    for (int i = 0; i < 8; ++i) {
        cb[i] = float(turbo_centroids_3bit_h[i]);
    }

    float m_state = -INFINITY;
    float l_state = 0.0f;
    float o_state[DV_PER_LANE];
    for (short i = 0; i < DV_PER_LANE; ++i) {
        o_state[i] = 0.0f;
    }

    device const char * k_base = k + ikv2*args.nb12 + ikv3*args.nb13;
    device const char * v_base = v + ikv2*args.nb22 + ikv3*args.nb23;
    device const half * mask_ptr = nullptr;
    if (FC_turbo_flash_p1_has_mask) {
        mask_ptr = (device const half *) (mask + iq1*args.nb31 +
            (iq2 % args.ne32)*args.nb32 + (iq3 % args.ne33)*args.nb33);
    }

    for (int t = t_start; t < t_end; ++t) {
        float mask_val = 0.0f;
        if (FC_turbo_flash_p1_has_mask) {
            mask_val = float(mask_ptr[t]);
            if (mask_val <= -MAXHALF) {
                continue;
            }
        }

        float dot_partial = 0.0f;
        if (FC_turbo_flash_p1_k_is_turbo3) {
            device const block_turbo3_0 * k_row =
                (device const block_turbo3_0 *) (k_base + t*args.nb11);
            const float k_norm = float(k_row[0].norm);
            for (short i = 0; i < DK_PER_LANE; ++i) {
                const int d = int(lane) + i*32;
                if (d >= DK) break;
                const uint8_t q_idx = (k_row[0].qs[d/4] >> ((d % 4)*2)) & 0x03;
                const uint8_t s_bit = (k_row[0].signs[d/8] >> (d % 8)) & 1;
                dot_partial += q_vals[i] * cb[q_idx | (s_bit << 2)] * k_norm;
            }
        } else {
            device const block_q8_0 * k_row =
                (device const block_q8_0 *) (k_base + t*args.nb11);
            for (short i = 0; i < DK_PER_LANE; ++i) {
                const int d = int(lane) + i*32;
                if (d >= DK) break;
                dot_partial += q_vals[i] * float(k_row[d/32].qs[d % 32]) * float(k_row[d/32].d);
            }
        }
        const float score = simd_sum(dot_partial) * args.scale + mask_val;

        device const block_turbo3_0 * v_row =
            (device const block_turbo3_0 *) (v_base + t*args.nb21);
        const float v_norm = float(v_row[0].norm);
        float v_decoded[DV_PER_LANE];
        for (short i = 0; i < DV_PER_LANE; ++i) {
            const int d = int(lane) + i*32;
            if (d >= DV) {
                v_decoded[i] = 0.0f;
                continue;
            }
            const uint8_t q_idx = (v_row[0].qs[d/4] >> ((d % 4)*2)) & 0x03;
            const uint8_t s_bit = (v_row[0].signs[d/8] >> (d % 8)) & 1;
            v_decoded[i] = cb[q_idx | (s_bit << 2)] * v_norm;
        }

        const float new_m = max(m_state, score);
        const float exp_diff = exp(m_state - new_m);
        const float exp_score = exp(score - new_m);
        for (short i = 0; i < DV_PER_LANE; ++i) {
            o_state[i] = o_state[i]*exp_diff + exp_score*v_decoded[i];
        }
        l_state = l_state*exp_diff + exp_score;
        m_state = new_m;
    }

    for (short i = 0; i < DV_PER_LANE; ++i) {
        const int d = int(lane) + i*32;
        if (d < DV) {
            partial_out[bh_idx*(uint64_t)n_blocks*DV + block_id*DV + d] = o_state[i];
        }
    }
    if (lane == 0) {
        partial_ms[bh_idx*(uint64_t)n_blocks*2 + block_id*2 + 0] = m_state;
        partial_ms[bh_idx*(uint64_t)n_blocks*2 + block_id*2 + 1] = l_state;
    }
}

template <short DV>
kernel void kernel_turbo_flash_p2(
        constant ggml_metal_kargs_turbo_flash_p2 & args,
        device const float * partial_out,
        device const float * partial_ms,
        device       float * dst,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiitg [[thread_index_in_threadgroup]]) {
    const uint tid = tiitg;
    const uint bh_idx = tgpig[0];
    const int n_blocks = args.n_blocks;
    threadgroup float * shared_out = shmem;
    threadgroup float * global_max = shmem + DV;
    threadgroup float * global_sum = shmem + DV + 1;

    if (tid == 0) {
        float gmax = -INFINITY;
        for (int b = 0; b < n_blocks; ++b) {
            gmax = max(gmax, partial_ms[bh_idx*(uint64_t)n_blocks*2 + b*2]);
        }
        global_max[0] = gmax;
        float gsum = 0.0f;
        for (int b = 0; b < n_blocks; ++b) {
            const float bmax = partial_ms[bh_idx*(uint64_t)n_blocks*2 + b*2];
            const float bsum = partial_ms[bh_idx*(uint64_t)n_blocks*2 + b*2 + 1];
            gsum += exp(bmax - gmax)*bsum;
        }
        global_sum[0] = gsum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (int(tid) < DV) {
        float accum = 0.0f;
        for (int b = 0; b < n_blocks; ++b) {
            const float bmax = partial_ms[bh_idx*(uint64_t)n_blocks*2 + b*2];
            accum += exp(bmax - global_max[0]) *
                partial_out[bh_idx*(uint64_t)n_blocks*DV + b*DV + tid];
        }
        shared_out[tid] = global_sum[0] > 0.0f ? accum/global_sum[0] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (int(tid) < DV && int(tid) < 128) {
        shared_out[tid] *= turbo_wht_signs2[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int dim_wht = min(int(DV), 128);
        float value = shared_out[tid];
        const uint lane = tid % 32;
        const int log2_dim = dim_wht >= 128 ? 7 : dim_wht >= 64 ? 6 : 5;
        for (int s = 0; s < min(5, log2_dim); ++s) {
            const uint step = 1u << s;
            const float other = simd_shuffle_xor(value, (ushort) step);
            value = (lane & step) ? other - value : other + value;
        }
        if (dim_wht > 32) {
            shared_out[tid] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int half_block = 32; half_block < dim_wht; half_block <<= 1) {
                const int bfly_size = half_block << 1;
                const int base = (int(tid) / bfly_size)*bfly_size;
                const int local = int(tid) % bfly_size;
                const float a = shared_out[base + (local % half_block)];
                const float b = shared_out[base + (local % half_block) + half_block];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                shared_out[tid] = local < half_block ? a + b : a - b;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            value = shared_out[tid];
        }
        dst[bh_idx*DV + tid] = value * rsqrt(float(dim_wht)) * turbo_wht_signs1[tid];
    }
}

typedef decltype(kernel_turbo_flash_p1<128, 128>) turbo_flash_p1_t;
template [[host_name("kernel_turbo_flash_p1_dk64_dv64")]] kernel turbo_flash_p1_t kernel_turbo_flash_p1<64, 64>;
template [[host_name("kernel_turbo_flash_p1_dk96_dv96")]] kernel turbo_flash_p1_t kernel_turbo_flash_p1<96, 96>;
template [[host_name("kernel_turbo_flash_p1_dk128_dv128")]] kernel turbo_flash_p1_t kernel_turbo_flash_p1<128, 128>;

typedef decltype(kernel_turbo_flash_p2<128>) turbo_flash_p2_t;
template [[host_name("kernel_turbo_flash_p2_dv64")]] kernel turbo_flash_p2_t kernel_turbo_flash_p2<64>;
template [[host_name("kernel_turbo_flash_p2_dv96")]] kernel turbo_flash_p2_t kernel_turbo_flash_p2<96>;
template [[host_name("kernel_turbo_flash_p2_dv128")]] kernel turbo_flash_p2_t kernel_turbo_flash_p2<128>;
