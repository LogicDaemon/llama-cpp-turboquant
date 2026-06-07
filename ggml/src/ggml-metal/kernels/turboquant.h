#pragma once

#include "../turbo-wht.h"

// TurboQuant-specific Metal definitions.  Keep these outside the upstream
// operation sources so rebases only touch this overlay and registrations.

constant float turbo_centroids_2bit[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};
constant float turbo_mid_2bit[3] = {
    -0.086728f, 0.0f, 0.086728f
};
constant half turbo_centroids_2bit_h[4] = {
    -0.133462h, -0.039994h, 0.039994h, 0.133462h
};

constant float turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
constant float turbo_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};
constant half turbo_centroids_3bit_h[8] = {
    -0.190685h, -0.117832h, -0.065717h, -0.021460h,
     0.021460h,  0.065717h,  0.117832h,  0.190685h
};
constant half turbo_mag_3bit_h[4] = {
    0.021460h, 0.065717h, 0.117832h, 0.190685h
};

constant float turbo_centroids_4bit[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};
constant float turbo_mid_4bit[15] = {
    -0.145560f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013963f, 0.0f,
     0.013963f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145560f
};
constant half turbo_centroids_4bit_h[16] = {
    -0.173926h, -0.117195h, -0.089527h, -0.068756h,
    -0.051262h, -0.035597h, -0.020989h, -0.006938h,
     0.006938h,  0.020989h,  0.035597h,  0.051262h,
     0.068756h,  0.089527h,  0.117195h,  0.173926h
};

constant float tq3_centroids[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};
constant float tq4_centroids[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f
};
constant float tq3_signs[32] = {
     1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f
};
constant float tq3_inv_sqrt32 = 0.17677669529663688f;

inline uint8_t turbo_index_2bit(float x) {
    return x < turbo_mid_2bit[0] ? 0 : x < turbo_mid_2bit[1] ? 1 : x < turbo_mid_2bit[2] ? 2 : 3;
}
inline uint8_t turbo_index_3bit(float x) {
    if (x < turbo_mid_3bit[0]) return 0;
    if (x < turbo_mid_3bit[1]) return 1;
    if (x < turbo_mid_3bit[2]) return 2;
    if (x < turbo_mid_3bit[3]) return 3;
    if (x < turbo_mid_3bit[4]) return 4;
    if (x < turbo_mid_3bit[5]) return 5;
    if (x < turbo_mid_3bit[6]) return 6;
    return 7;
}
inline uint8_t turbo_index_4bit(float x) {
    for (uint8_t i = 0; i < 15; ++i) {
        if (x < turbo_mid_4bit[i]) return i;
    }
    return 15;
}

void quantize_turbo2_0(device const float * src, device block_turbo2_0 & dst) {
    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO2; ++i) norm_sq += src[i] * src[i];
    const float norm = sqrt(norm_sq);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    dst.norm = half(norm);
    for (int i = 0; i < QK_TURBO2 / 4; ++i) dst.qs[i] = 0;
    for (int i = 0; i < QK_TURBO2; ++i) {
        const uint8_t q = turbo_index_2bit(src[i] * inv_norm);
        dst.qs[i / 4] |= q << ((i & 3) * 2);
    }
}

void quantize_turbo3_0(device const float * src, device block_turbo3_0 & dst) {
    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO3; ++i) norm_sq += src[i] * src[i];
    const float norm = sqrt(norm_sq);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    dst.norm = half(norm);
    for (int i = 0; i < QK_TURBO3 / 4; ++i) dst.qs[i] = 0;
    for (int i = 0; i < QK_TURBO3 / 8; ++i) dst.signs[i] = 0;
    for (int i = 0; i < QK_TURBO3; ++i) {
        const uint8_t q = turbo_index_3bit(src[i] * inv_norm);
        dst.qs[i / 4] |= (q & 3) << ((i & 3) * 2);
        dst.signs[i / 8] |= (q >> 2) << (i & 7);
    }
}

void quantize_turbo4_0(device const float * src, device block_turbo4_0 & dst) {
    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO4; ++i) norm_sq += src[i] * src[i];
    const float norm = sqrt(norm_sq);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    float rotated[QK_TURBO4];
    for (int i = 0; i < QK_TURBO4; ++i) rotated[i] = src[i] * inv_norm;
    turbo_rotate_forward(rotated, turbo_wht_signs1, turbo_wht_signs2);
    for (int i = 0; i < QK_TURBO4 / 2; ++i) dst.qs[i] = 0;
    float recon_sq = 0.0f;
    for (int i = 0; i < QK_TURBO4; ++i) {
        const uint8_t q = turbo_index_4bit(rotated[i]);
        dst.qs[i / 2] |= q << ((i & 1) * 4);
        recon_sq += turbo_centroids_4bit[q] * turbo_centroids_4bit[q];
    }
    dst.rnorm = half(0.0f);
    const float recon = sqrt(recon_sq);
    dst.norm = half(recon > 1e-10f ? norm / recon : norm);
}

template <typename type4x4>
void dequantize_turbo2_0(device const block_turbo2_0 * xb, short il, thread type4x4 & reg) {
    const float d = float(xb->norm);
    const int off = il * 4;
    float4x4 out;
    for (int i = 0; i < 4; ++i) {
        const uint8_t q = xb->qs[off + i];
        out[i] = float4(turbo_centroids_2bit[q & 3], turbo_centroids_2bit[(q >> 2) & 3],
                        turbo_centroids_2bit[(q >> 4) & 3], turbo_centroids_2bit[q >> 6]) * d;
    }
    reg = (type4x4) out;
}
template <typename type4>
void dequantize_turbo2_0_t4(device const block_turbo2_0 * xb, short il, thread type4 & reg) {
    const uint8_t q = xb->qs[il];
    const float d = float(xb->norm);
    reg = (type4) (float4(turbo_centroids_2bit_h[q & 3], turbo_centroids_2bit_h[(q >> 2) & 3],
                          turbo_centroids_2bit_h[(q >> 4) & 3], turbo_centroids_2bit_h[q >> 6]) * d);
}

template <typename type4x4>
void dequantize_turbo3_0(device const block_turbo3_0 * xb, short il, thread type4x4 & reg) {
    const float d = float(xb->norm);
    const int off = il * 16;
    float4x4 out;
    for (int g = 0; g < 4; ++g) {
        const int j = off + g * 4;
        const uint8_t q = xb->qs[j / 4];
        const uint8_t s = xb->signs[j / 8];
        const int shift = (j & 7);
        out[g] = float4(
            turbo_centroids_3bit[(q >> ((j & 3) * 2)) & 3 | (((s >> (shift    )) & 1) << 2)] * d,
            turbo_centroids_3bit[(q >> (((j + 1) & 3) * 2)) & 3 | (((s >> (shift + 1)) & 1) << 2)] * d,
            turbo_centroids_3bit[(q >> (((j + 2) & 3) * 2)) & 3 | (((s >> (shift + 2)) & 1) << 2)] * d,
            turbo_centroids_3bit[(q >> (((j + 3) & 3) * 2)) & 3 | (((s >> (shift + 3)) & 1) << 2)] * d);
    }
    reg = (type4x4) out;
}
template <typename type4>
void dequantize_turbo3_0_t4(device const block_turbo3_0 * xb, short il, thread type4 & reg) {
#ifndef TURBO_PROFILE_MODE
#define TURBO_PROFILE_MODE 0
#endif
#if TURBO_PROFILE_MODE == 1
    reg = type4(0.0f);
#elif TURBO_PROFILE_MODE == 2
    reg = type4(float(xb->norm));
#elif TURBO_PROFILE_MODE == 3
    const float d = float(xb->norm);
    const uint8_t q = xb->qs[il];
    reg = type4(float4(float(turbo_centroids_3bit_h[(q      ) & 3 | 4]),
                      float(turbo_centroids_3bit_h[(q >> 2) & 3 | 4]),
                      float(turbo_centroids_3bit_h[(q >> 4) & 3 | 4]),
                      float(turbo_centroids_3bit_h[(q >> 6)     | 4])) * d);
#elif TURBO_PROFILE_MODE == 4
    (void) xb->signs[il >> 1];
    reg = type4(float(turbo_centroids_3bit_h[0]) * float(xb->norm));
#else
    const uint8_t q = xb->qs[il];
    const uint8_t s = xb->signs[il / 2];
    const int shift = (il & 1) * 4;
    const float d = float(xb->norm);
    const uint8_t q0 = q & 3, q1 = (q >> 2) & 3, q2 = (q >> 4) & 3, q3 = q >> 6;
#if TURBO_USE_4MAG
    const uint8_t s0 = (s >> shift) & 1, s1 = (s >> (shift + 1)) & 1;
    const uint8_t s2 = (s >> (shift + 2)) & 1, s3 = (s >> (shift + 3)) & 1;
    const float v0 = float(turbo_mag_3bit_h[q0 ^ (s0 ? 0u : 3u)]) * d;
    const float v1 = float(turbo_mag_3bit_h[q1 ^ (s1 ? 0u : 3u)]) * d;
    const float v2 = float(turbo_mag_3bit_h[q2 ^ (s2 ? 0u : 3u)]) * d;
    const float v3 = float(turbo_mag_3bit_h[q3 ^ (s3 ? 0u : 3u)]) * d;
    reg = (type4) float4(s0 ? v0 : -v0, s1 ? v1 : -v1, s2 ? v2 : -v2, s3 ? v3 : -v3);
#else
    reg = (type4) (float4(turbo_centroids_3bit[q0 | (((s >> shift) & 1) << 2)],
                          turbo_centroids_3bit[q1 | (((s >> (shift + 1)) & 1) << 2)],
                          turbo_centroids_3bit[q2 | (((s >> (shift + 2)) & 1) << 2)],
                          turbo_centroids_3bit[q3 | (((s >> (shift + 3)) & 1) << 2)]) * d);
#endif
#endif
}

template <typename type4x4>
void dequantize_turbo4_0(device const block_turbo4_0 * xb, short il, thread type4x4 & reg) {
    const float d = float(xb->norm);
    const int off = il * 16;
    float4x4 out;
    for (int g = 0; g < 4; ++g) {
        for (int k = 0; k < 4; ++k) {
            const int j = off + g * 4 + k;
            out[g][k] = turbo_centroids_4bit[(xb->qs[j / 2] >> ((j & 1) * 4)) & 15] * d;
        }
    }
    reg = (type4x4) out;
}
template <typename type4>
void dequantize_turbo4_0_t4(device const block_turbo4_0 * xb, short il, thread type4 & reg) {
    const device uint8_t * q = xb->qs + il * 2;
    const float d = float(xb->norm);
    reg = (type4) (float4(turbo_centroids_4bit_h[q[0] & 15], turbo_centroids_4bit_h[q[0] >> 4],
                          turbo_centroids_4bit_h[q[1] & 15], turbo_centroids_4bit_h[q[1] >> 4]) * d);
}

inline void turbo_unpack_tq3(device const uint8_t * q, thread uint8_t * out) {
    for (int g = 0; g < 4; ++g) {
        const device uint8_t * p = q + g * 3;
        thread uint8_t * o = out + g * 8;
        o[0] = p[0] & 7; o[1] = (p[0] >> 3) & 7;
        o[2] = ((p[0] >> 6) | (p[1] << 2)) & 7; o[3] = (p[1] >> 1) & 7;
        o[4] = (p[1] >> 4) & 7; o[5] = ((p[1] >> 7) | (p[2] << 1)) & 7;
        o[6] = (p[2] >> 2) & 7; o[7] = (p[2] >> 5) & 7;
    }
}
inline uint8_t turbo_tq3_index(device const uint8_t * q, uint i) {
    const device uint8_t * p = q + (i / 8) * 3;
    switch (i & 7) {
        case 0: return p[0] & 7;
        case 1: return (p[0] >> 3) & 7;
        case 2: return ((p[0] >> 6) | (p[1] << 2)) & 7;
        case 3: return (p[1] >> 1) & 7;
        case 4: return (p[1] >> 4) & 7;
        case 5: return ((p[1] >> 7) | (p[2] << 1)) & 7;
        case 6: return (p[2] >> 2) & 7;
        default: return (p[2] >> 5) & 7;
    }
}
inline void turbo_tq3_inverse(thread float * x) {
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; ++j) {
                const float a = x[j], b = x[j + step];
                x[j] = a + b; x[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < 32; ++i) x[i] *= tq3_inv_sqrt32 * tq3_signs[i];
}

template <typename type4x4>
void dequantize_tq3_1s(device const block_tq3_1s * xb, short il, thread type4x4 & reg) {
    uint8_t q[32]; turbo_unpack_tq3(xb->qs, q);
    float x[32];
    for (int i = 0; i < 32; ++i) x[i] = tq3_centroids[q[i]] * float(i < 16 ? xb->d0 : xb->d1);
    turbo_tq3_inverse(x);
    float4x4 out;
    for (int g = 0; g < 4; ++g) for (int k = 0; k < 4; ++k) out[g][k] = x[il * 16 + g * 4 + k];
    reg = (type4x4) out;
}
template <typename type4>
void dequantize_tq3_1s_t4(device const block_tq3_1s * xb, short il, thread type4 & reg) {
    uint8_t q[32]; turbo_unpack_tq3(xb->qs, q);
    float x[32];
    for (int i = 0; i < 32; ++i) x[i] = tq3_centroids[q[i]] * float(i < 16 ? xb->d0 : xb->d1);
    turbo_tq3_inverse(x);
    const int off = il * 4;
    reg = (type4) float4(x[off], x[off + 1], x[off + 2], x[off + 3]);
}
template <typename type4x4>
void dequantize_tq3_1s_rotated(device const block_tq3_1s * xb, short il, thread type4x4 & reg) {
    uint8_t q[32]; turbo_unpack_tq3(xb->qs, q); float4x4 out;
    for (int g = 0; g < 4; ++g) for (int k = 0; k < 4; ++k) {
        const int i = il * 16 + g * 4 + k;
        out[g][k] = tq3_centroids[q[i]] * float(i < 16 ? xb->d0 : xb->d1);
    }
    reg = (type4x4) out;
}
template <typename type4>
void dequantize_tq3_1s_rotated_t4(device const block_tq3_1s * xb, short il, thread type4 & reg) {
    uint8_t q[32]; turbo_unpack_tq3(xb->qs, q); const int off = il * 4;
    const float d = float(off < 16 ? xb->d0 : xb->d1);
    reg = (type4) float4(tq3_centroids[q[off]], tq3_centroids[q[off + 1]], tq3_centroids[q[off + 2]], tq3_centroids[q[off + 3]]) * d;
}

template <typename type4x4>
void dequantize_tq4_1s(device const block_tq4_1s * xb, short il, thread type4x4 & reg) {
    float x[32];
    for (int i = 0; i < 32; ++i) x[i] = tq4_centroids[(xb->qs[i / 2] >> ((i & 1) * 4)) & 15] * float(i < 16 ? xb->d0 : xb->d1);
    turbo_tq3_inverse(x); float4x4 out;
    for (int g = 0; g < 4; ++g) for (int k = 0; k < 4; ++k) out[g][k] = x[il * 16 + g * 4 + k];
    reg = (type4x4) out;
}
template <typename type4>
void dequantize_tq4_1s_t4(device const block_tq4_1s * xb, short il, thread type4 & reg) {
    float x[32];
    for (int i = 0; i < 32; ++i) x[i] = tq4_centroids[(xb->qs[i / 2] >> ((i & 1) * 4)) & 15] * float(i < 16 ? xb->d0 : xb->d1);
    turbo_tq3_inverse(x); const int off = il * 4;
    reg = (type4) float4(x[off], x[off + 1], x[off + 2], x[off + 3]);
}
template <typename type4x4>
void dequantize_tq4_1s_rotated(device const block_tq4_1s * xb, short il, thread type4x4 & reg) {
    float4x4 out;
    for (int g = 0; g < 4; ++g) for (int k = 0; k < 4; ++k) {
        const int i = il * 16 + g * 4 + k;
        out[g][k] = tq4_centroids[(xb->qs[i / 2] >> ((i & 1) * 4)) & 15] * float(i < 16 ? xb->d0 : xb->d1);
    }
    reg = (type4x4) out;
}
template <typename type4>
void dequantize_tq4_1s_rotated_t4(device const block_tq4_1s * xb, short il, thread type4 & reg) {
    const int off = il * 4; float4 out;
    for (int i = 0; i < 4; ++i) {
        const int j = off + i;
        out[i] = tq4_centroids[(xb->qs[j / 2] >> ((j & 1) * 4)) & 15] * float(j < 16 ? xb->d0 : xb->d1);
    }
    reg = (type4) out;
}
