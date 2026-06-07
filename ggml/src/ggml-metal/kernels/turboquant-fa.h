// TurboQuant Flash Attention registrations kept outside the upstream FA source.
// This file is included at the two registration points in fa.metal.

#if !defined(TURBOQUANT_FA_VEC)
#error TURBOQUANT_FA_VEC must be defined before including turboquant-fa.h
#endif

#if TURBOQUANT_FA_VEC

#define TQ_FA_REG(name, kt, knl, kd, vt, vnl, vd, dk, dv, ne) \
template [[host_name(name)]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, kt, knl, kd, vt, vnl, vd, dk, dv, ne>;

// Explicit host-name wrappers keep names stable, including names containing
// underscores such as kq8_0_vturbo3.
#define TQ_FA_REG_T2_T2(n, dk, dv, ne) TQ_FA_REG(n, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, dk, dv, ne)
#define TQ_FA_REG_T3_T3(n, dk, dv, ne) TQ_FA_REG(n, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, dk, dv, ne)
#define TQ_FA_REG_T4_T4(n, dk, dv, ne) TQ_FA_REG(n, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, dk, dv, ne)
#define TQ_FA_REG_T2_T3(n, dk, dv, ne) TQ_FA_REG(n, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, dk, dv, ne)
#define TQ_FA_REG_T3_T2(n, dk, dv, ne) TQ_FA_REG(n, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, dk, dv, ne)
#define TQ_FA_REG_T2_T4(n, dk, dv, ne) TQ_FA_REG(n, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, dk, dv, ne)
#define TQ_FA_REG_T4_T2(n, dk, dv, ne) TQ_FA_REG(n, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, dk, dv, ne)
#define TQ_FA_REG_T3_T4(n, dk, dv, ne) TQ_FA_REG(n, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, dk, dv, ne)
#define TQ_FA_REG_T4_T3(n, dk, dv, ne) TQ_FA_REG(n, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, dk, dv, ne)
#define TQ_FA_REG_Q8_T2(n, dk, dv, ne) TQ_FA_REG(n, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, dk, dv, ne)
#define TQ_FA_REG_T2_Q8(n, dk, dv, ne) TQ_FA_REG(n, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, dk, dv, ne)
#define TQ_FA_REG_Q8_T3(n, dk, dv, ne) TQ_FA_REG(n, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, dk, dv, ne)
#define TQ_FA_REG_T3_Q8(n, dk, dv, ne) TQ_FA_REG(n, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, dk, dv, ne)
#define TQ_FA_REG_Q8_T4(n, dk, dv, ne) TQ_FA_REG(n, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, dk, dv, ne)
#define TQ_FA_REG_T4_Q8(n, dk, dv, ne) TQ_FA_REG(n, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, dk, dv, ne)

// Explicit registrations keep host names stable and avoid relying on token
// pasting to spell names such as kq8_0_vturbo3.
TQ_FA_REG_T2_T2("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk128_dv128", 128, 128, 1)
TQ_FA_REG_T2_T2("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk192_dv192", 192, 192, 2)
TQ_FA_REG_T2_T2("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk192_dv128", 192, 128, 2)
TQ_FA_REG_T2_T2("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk256_dv256", 256, 256, 1)
TQ_FA_REG_T2_T2("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk320_dv256", 320, 256, 2)
TQ_FA_REG_T2_T2("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk512_dv512", 512, 512, 1)
TQ_FA_REG_T2_T2("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk576_dv512", 576, 512, 2)
TQ_FA_REG_T3_T3("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk512_dv512", 512, 512, 1)
TQ_FA_REG_T3_T3("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk576_dv512", 576, 512, 2)
TQ_FA_REG_T3_T3("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk128_dv128", 128, 128, 1)
TQ_FA_REG_T3_T3("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk192_dv192", 192, 192, 2)
TQ_FA_REG_T3_T3("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk192_dv128", 192, 128, 2)
TQ_FA_REG_T3_T3("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk256_dv256", 256, 256, 1)
TQ_FA_REG_T3_T3("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk320_dv256", 320, 256, 2)
TQ_FA_REG_T4_T4("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk512_dv512", 512, 512, 1)
TQ_FA_REG_T4_T4("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk576_dv512", 576, 512, 2)
TQ_FA_REG_T4_T4("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk128_dv128", 128, 128, 1)
TQ_FA_REG_T4_T4("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk192_dv192", 192, 192, 2)
TQ_FA_REG_T4_T4("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk192_dv128", 192, 128, 2)
TQ_FA_REG_T4_T4("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk256_dv256", 256, 256, 1)
TQ_FA_REG_T4_T4("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk320_dv256", 320, 256, 2)

#define TQ_FA_VEC_ASYM(reg, label) \
reg("kernel_flash_attn_ext_vec_" label "_dk128_dv128", 128, 128, 1) \
reg("kernel_flash_attn_ext_vec_" label "_dk192_dv192", 192, 192, 2) \
reg("kernel_flash_attn_ext_vec_" label "_dk192_dv128", 192, 128, 2) \
reg("kernel_flash_attn_ext_vec_" label "_dk256_dv256", 256, 256, 1) \
reg("kernel_flash_attn_ext_vec_" label "_dk320_dv256", 320, 256, 2) \
reg("kernel_flash_attn_ext_vec_" label "_dk512_dv512", 512, 512, 1) \
reg("kernel_flash_attn_ext_vec_" label "_dk576_dv512", 576, 512, 2)

#define TQ_FA_VEC_Q8(reg, label) \
reg("kernel_flash_attn_ext_vec_" label "_dk32_dv32", 32, 32, 4) \
reg("kernel_flash_attn_ext_vec_" label "_dk64_dv64", 64, 64, 2) \
reg("kernel_flash_attn_ext_vec_" label "_dk96_dv96", 96, 96, 4) \
reg("kernel_flash_attn_ext_vec_" label "_dk128_dv128", 128, 128, 1) \
reg("kernel_flash_attn_ext_vec_" label "_dk192_dv192", 192, 192, 2) \
reg("kernel_flash_attn_ext_vec_" label "_dk192_dv128", 192, 128, 2) \
reg("kernel_flash_attn_ext_vec_" label "_dk256_dv256", 256, 256, 1) \
reg("kernel_flash_attn_ext_vec_" label "_dk320_dv256", 320, 256, 2) \
reg("kernel_flash_attn_ext_vec_" label "_dk512_dv512", 512, 512, 1) \
reg("kernel_flash_attn_ext_vec_" label "_dk576_dv512", 576, 512, 2)

TQ_FA_VEC_ASYM(TQ_FA_REG_T2_T3, "kturbo2_vturbo3")
TQ_FA_VEC_ASYM(TQ_FA_REG_T3_T2, "kturbo3_vturbo2")
TQ_FA_VEC_ASYM(TQ_FA_REG_T2_T4, "kturbo2_vturbo4")
TQ_FA_VEC_ASYM(TQ_FA_REG_T4_T2, "kturbo4_vturbo2")
TQ_FA_VEC_ASYM(TQ_FA_REG_T3_T4, "kturbo3_vturbo4")
TQ_FA_VEC_ASYM(TQ_FA_REG_T4_T3, "kturbo4_vturbo3")
TQ_FA_VEC_Q8(TQ_FA_REG_Q8_T2, "kq8_0_vturbo2")
TQ_FA_VEC_Q8(TQ_FA_REG_T2_Q8, "kturbo2_vq8_0")
TQ_FA_VEC_Q8(TQ_FA_REG_Q8_T3, "kq8_0_vturbo3")
TQ_FA_VEC_Q8(TQ_FA_REG_T3_Q8, "kturbo3_vq8_0")
TQ_FA_VEC_Q8(TQ_FA_REG_Q8_T4, "kq8_0_vturbo4")
TQ_FA_VEC_Q8(TQ_FA_REG_T4_Q8, "kturbo4_vq8_0")

#undef TQ_FA_VEC_Q8
#undef TQ_FA_VEC_ASYM
#undef TQ_FA_REG_T4_Q8
#undef TQ_FA_REG_Q8_T4
#undef TQ_FA_REG_T3_Q8
#undef TQ_FA_REG_Q8_T3
#undef TQ_FA_REG_T2_Q8
#undef TQ_FA_REG_Q8_T2
#undef TQ_FA_REG_T4_T3
#undef TQ_FA_REG_T3_T4
#undef TQ_FA_REG_T4_T2
#undef TQ_FA_REG_T2_T4
#undef TQ_FA_REG_T3_T2
#undef TQ_FA_REG_T2_T3
#undef TQ_FA_REG_T4_T4
#undef TQ_FA_REG_T3_T3
#undef TQ_FA_REG_T2_T2
#undef TQ_FA_REG

#else

#define TQ_FA_REG(name, kt, knl, kd, vt, vnl, vd, dk, dv) \
template [[host_name(name)]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, kt, knl, kd, vt, vnl, vd, dk, dv>;

#define TQ_FA_DIMS(reg, label, kt, knl, kd, vt, vnl, vd) \
reg("kernel_flash_attn_ext_" label "_dk32_dv32", kt, knl, kd, vt, vnl, vd, 32, 32) \
reg("kernel_flash_attn_ext_" label "_dk64_dv64", kt, knl, kd, vt, vnl, vd, 64, 64) \
reg("kernel_flash_attn_ext_" label "_dk96_dv96", kt, knl, kd, vt, vnl, vd, 96, 96) \
reg("kernel_flash_attn_ext_" label "_dk128_dv128", kt, knl, kd, vt, vnl, vd, 128, 128) \
reg("kernel_flash_attn_ext_" label "_dk192_dv192", kt, knl, kd, vt, vnl, vd, 192, 192) \
reg("kernel_flash_attn_ext_" label "_dk192_dv128", kt, knl, kd, vt, vnl, vd, 192, 128) \
reg("kernel_flash_attn_ext_" label "_dk256_dv256", kt, knl, kd, vt, vnl, vd, 256, 256) \
reg("kernel_flash_attn_ext_" label "_dk320_dv256", kt, knl, kd, vt, vnl, vd, 320, 256) \
reg("kernel_flash_attn_ext_" label "_dk512_dv512", kt, knl, kd, vt, vnl, vd, 512, 512) \
reg("kernel_flash_attn_ext_" label "_dk576_dv512", kt, knl, kd, vt, vnl, vd, 576, 512)

#define TQ_FA_DIMS_Q8(reg, label, kt, knl, kd, vt, vnl, vd) \
reg("kernel_flash_attn_ext_" label "_dk32_dv32", kt, knl, kd, vt, vnl, vd, 32, 32) \
reg("kernel_flash_attn_ext_" label "_dk40_dv40", kt, knl, kd, vt, vnl, vd, 40, 40) \
reg("kernel_flash_attn_ext_" label "_dk48_dv48", kt, knl, kd, vt, vnl, vd, 48, 48) \
reg("kernel_flash_attn_ext_" label "_dk64_dv64", kt, knl, kd, vt, vnl, vd, 64, 64) \
reg("kernel_flash_attn_ext_" label "_dk72_dv72", kt, knl, kd, vt, vnl, vd, 72, 72) \
reg("kernel_flash_attn_ext_" label "_dk80_dv80", kt, knl, kd, vt, vnl, vd, 80, 80) \
reg("kernel_flash_attn_ext_" label "_dk96_dv96", kt, knl, kd, vt, vnl, vd, 96, 96) \
reg("kernel_flash_attn_ext_" label "_dk112_dv112", kt, knl, kd, vt, vnl, vd, 112, 112) \
reg("kernel_flash_attn_ext_" label "_dk128_dv128", kt, knl, kd, vt, vnl, vd, 128, 128) \
reg("kernel_flash_attn_ext_" label "_dk192_dv192", kt, knl, kd, vt, vnl, vd, 192, 192) \
reg("kernel_flash_attn_ext_" label "_dk192_dv128", kt, knl, kd, vt, vnl, vd, 192, 128) \
reg("kernel_flash_attn_ext_" label "_dk256_dv256", kt, knl, kd, vt, vnl, vd, 256, 256) \
reg("kernel_flash_attn_ext_" label "_dk320_dv256", kt, knl, kd, vt, vnl, vd, 320, 256) \
reg("kernel_flash_attn_ext_" label "_dk512_dv512", kt, knl, kd, vt, vnl, vd, 512, 512) \
reg("kernel_flash_attn_ext_" label "_dk576_dv512", kt, knl, kd, vt, vnl, vd, 576, 512)

#define TQ_NV(reg, label, kt, knl, kd, vt, vnl, vd) TQ_FA_DIMS(reg, label, kt, knl, kd, vt, vnl, vd)
#define TQ_NV_Q8(reg, label, kt, knl, kd, vt, vnl, vd) TQ_FA_DIMS_Q8(reg, label, kt, knl, kd, vt, vnl, vd)

TQ_NV(TQ_FA_REG, "kturbo2_vturbo2", block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0)
TQ_NV(TQ_FA_REG, "kturbo3_vturbo3", block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0)
TQ_NV(TQ_FA_REG, "kturbo4_vturbo4", block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0)
TQ_NV(TQ_FA_REG, "kturbo2_vturbo3", block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0)
TQ_NV(TQ_FA_REG, "kturbo3_vturbo2", block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0)
TQ_NV(TQ_FA_REG, "kturbo2_vturbo4", block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo4_0, 8, dequantize_turbo4_0)
TQ_NV(TQ_FA_REG, "kturbo4_vturbo2", block_turbo4_0, 8, dequantize_turbo4_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0)
TQ_NV(TQ_FA_REG, "kturbo3_vturbo4", block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo4_0, 8, dequantize_turbo4_0)
TQ_NV(TQ_FA_REG, "kturbo4_vturbo3", block_turbo4_0, 8, dequantize_turbo4_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0)
TQ_NV_Q8(TQ_FA_REG, "kq8_0_vturbo2", block_q8_0, 2, dequantize_q8_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0)
TQ_NV_Q8(TQ_FA_REG, "kturbo2_vq8_0", block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_q8_0, 2, dequantize_q8_0)
TQ_NV_Q8(TQ_FA_REG, "kq8_0_vturbo3", block_q8_0, 2, dequantize_q8_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0)
TQ_NV_Q8(TQ_FA_REG, "kturbo3_vq8_0", block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_q8_0, 2, dequantize_q8_0)
TQ_NV_Q8(TQ_FA_REG, "kq8_0_vturbo4", block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0)
TQ_NV_Q8(TQ_FA_REG, "kturbo4_vq8_0", block_turbo4_0, 8, dequantize_turbo4_0, block_q8_0, 2, dequantize_q8_0)

#undef TQ_NV_Q8
#undef TQ_NV
#undef TQ_FA_DIMS_Q8
#undef TQ_FA_DIMS
#undef TQ_FA_REG

#endif
