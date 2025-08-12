#ifndef BACKEND_UTILS_COMMON_CL
#define BACKEND_UTILS_COMMON_CL

#ifdef __COMPILER_RELATIVE_PATH__
#include <constants.device.h>
#include <types.device.h>
#else
#include "glsc2/src/constants.device.h"
#include "glsc2/src/types.device.h"
#endif

//-----------------------------------------------------------------------------

// Those are always mapped into kernel dimensions.
inline uint get_sub_group_local_id()    { return get_local_id(0);   }
inline uint get_sub_group_id()          { return get_local_id(1);   }
inline uint get_sub_group_size()        { return get_local_size(0); }
inline uint get_num_sub_groups()        { return get_local_size(1); }

// Utility functions

inline uint     ugetLo           (ulong a)              { return a & 0x00000000FFFFFFFFu; }
inline int      getLo            (long a)               { return a & 0x00000000FFFFFFFF; }
inline uint     ugetHi           (ulong a)              { return a >> 32; }
inline int      getHi            (long a)               { return a >> 32; }
inline ulong    ucombineLoHi     (uint lo, uint hi)     { return ((ulong)hi << 32) | (ulong)lo; }
inline long     combineLoHi      (int lo, int hi)       { return ((long)hi << 32) | (long)lo; }
inline void     add_add_carry    (uint* rlo, uint alo, uint blo, uint* rhi, uint ahi, uint bhi) { ulong r = ucombineLoHi(alo, ahi) + ucombineLoHi(blo, bhi); *rlo = ugetLo(r); *rhi = ugetHi(r); }

// ISA dependancy
//#define CUDA
#ifdef CUDA
inline int      findLeadingOne      (uint v)                    { int r; asm("bfind.u32 %0, %1;" : "=r"(r) : "r"(v)); return r; }
inline uint     getLaneMaskLt       (void)                      { uint r; asm("mov.u32 %0, %%lanemask_lt;" : "=r"(r)); return r; }
inline uint     getLaneMaskLe       (void)                      { uint r; asm("mov.u32 %0, %%lanemask_le;" : "=r"(r)); return r; }

inline int      f32_to_s32_sat          (float a)                 { int v; asm("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
inline uint     f32_to_u32_sat          (float a)                 { uint v; asm("cvt.rni.sat.u32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
inline uint     f32_to_u32_sat_rmi  (float a)                   { uint v; asm("cvt.rmi.sat.u32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
inline long   f32_to_s64              (float a)                 { long v; asm("cvt.rni.s64.f32 %0, %1;" : "=l"(v) : "f"(a)); return v; }
inline int      add_s16lo_s16lo     (int a, int b)              { int v; asm("vadd.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      add_s16hi_s16lo     (int a, int b)              { int v; asm("vadd.s32.s32.s32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      sub_s16lo_s16lo     (int a, int b)              { int v; asm("vsub.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      sub_s16hi_s16lo     (int a, int b)			    { int v; asm("vsub.s32.s32.s32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      sub_s16hi_s16hi     (int a, int b)              { int v; asm("vsub.s32.s32.s32 %0, %1.h1, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      max_max             (int a, int b, int c)       { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
inline int      min_min             (int a, int b, int c)       { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
inline uint     add_sub             (uint a, uint b, uint c)    { uint v; asm("vsub.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(c), "r"(b)); return v; }
inline uint     add_add				(uint a, uint b, uint c)	{ uint v; asm("vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
inline int      add_clamp_0_x       (int a, int b, int c)       { int v; asm("vadd.u32.s32.s32.sat.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
inline uint     prmt				(uint a, uint b, uint c)    { uint v; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
inline uint     slct_ui             (uint a, uint b, int c)   { uint v; asm("slct.u32.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
inline int      slct_i              (int a, int b, int c)   { int v; asm("slct.s32.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
inline float    slct_f              (float a, float b, int c)   { float v; asm("slct.f32.s32 %0, %1, %2, %3;" : "=f"(v) : "f"(a), "f"(b), "r"(c)); return v; }

inline uint     get_max_sub_group_size(void) { return 32; }
inline uint     sub_group_activemask()    { uint r; asm volatile("activemask.b32 %0;" : "=r"(r)); return r; }
// inline uint     sub_group_ballot(int p) { 
//     uint r; asm volatile(
//         "{"
//         ".reg .pred p;"
//         "setp.ne.u32 p, %1, 0;"
//         ".reg .b32 mask;"
//         "activemask.b32 mask;"
//         "vote.sync.ballot.b32 %0, p, mask;"
//         "}"
//         : "=r"(r) : "r"(p)
//     ); 
//     return r; }
inline uint     sub_group_masked_ballot(int p, uint mask) { uint r; asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 0; vote.sync.ballot.b32 %0, p, %2; }" : "=r"(r) : "r"(p), "r"(mask)); return r; }
inline uint     sub_group_any(int p)                        { uint r; asm volatile("{ .reg .pred pi; setp.ne.u32 pi, %1, 0; vote.sync.any.pred  pi, pi, 0xffffffff; selp.u32 %0, 1, 0, pi; }" : "=r"(r) : "r"(p)); return r; }
inline uint     sub_group_masked_any(int p, uint mask)      { uint r; asm volatile("{ .reg .pred pi; setp.ne.u32 pi, %1, 0; vote.sync.any.pred  pi, pi, %2;         selp.u32 %0, 1, 0, pi; }" : "=r"(r) : "r"(p), "r"(mask)); return r; }
inline uint     sub_group_all(int p)                        { uint r; asm volatile("{ .reg .pred pi; setp.ne.u32 pi, %1, 0; vote.sync.all.pred  pi, pi, 0xffffffff; selp.u32 %0, 1, 0, pi; }" : "=r"(r) : "r"(p)); return r; }
inline uint     sub_group_masked_all(int p, uint mask)      { uint r; asm volatile("{ .reg .pred pi; setp.ne.u32 pi, %1, 0; vote.sync.all.pred  pi, pi, %2;         selp.u32 %0, 1, 0, pi; }" : "=r"(r) : "r"(p), "r"(mask)); return r; }
#else
inline int      findLeadingOne      (uint v)                    { 
    for (int bit = 31; bit >= 0; --bit) {
        if (v >= (1 << bit)) return bit;
    }
    return -1;
}
inline uint     getLaneMaskLt       (void)                      { return (1 << get_local_id(0)) - 1; }
inline uint     getLaneMaskLe       (void)                      { return (2 << get_local_id(0)) - 1; }

inline int      f32_to_s32_sat      (float a)                   { return (int)a; }
inline uint     f32_to_u32_sat      (float a)                   { return (uint)a; }
inline uint     f32_to_u32_sat_rmi  (float a)                   { return (uint)a; }
inline long   f32_to_s64              (float a)                 { return (long)a; }

inline int      add_s16lo_s16lo     (int a, int b)              { return (short)a + (short)b; }
inline int      add_s16hi_s16lo     (int a, int b)              { return (a >> 16) + (short)b; }
inline int      sub_s16lo_s16lo     (int a, int b)              { return (short)a - (short)b; }
inline int      sub_s16hi_s16lo     (int a, int b)			    { return (a >> 16) - (short)b; }
inline int      sub_s16hi_s16hi     (int a, int b)              { return (a >> 16) - (b >> 16); }

inline int      max_max             (int a, int b, int c)       { return max(a, max(b, c)); }
inline int      min_min             (int a, int b, int c)       { return min(a, min(b, c)); }
inline uint     add_sub             (uint a, uint b, uint c)    { return a+b-c; }
inline uint     add_add             (uint a, uint b, uint c)    { return a+b+c; }
inline int      add_clamp_0_x       (int a, int b, int c)       { return clamp(a+b,0,c); }
inline uint     prmt				(uint a, uint b, uint c)    { 
    ulong tmp = (ulong)b << 32 | a;
    uint4 masks;
    uint v = 0;
    masks.x = (c >>  0) & 0xf;
    masks.y = (c >>  4) & 0xf;
    masks.z = (c >>  8) & 0xf;
    masks.w = (c >> 12) & 0xf;
    
    v |= ((tmp >> (masks.x*8)) & 0xFF) <<  0;
    v |= ((tmp >> (masks.y*8)) & 0xFF) <<  8;
    v |= ((tmp >> (masks.z*8)) & 0xFF) << 16;
    v |= ((tmp >> (masks.w*8)) & 0xFF) << 24;
    
    return v;
}
inline uint     slct_ui             (uint a, uint b, int c)   { return (c >= 0) ? a : b; }
inline int      slct_i              (int a, int b, int c)   { return (c >= 0) ? a : b; }
inline float    slct_f              (float a, float b, int c)   { return (c >= 0) ? a : b; }

#endif 

/*
    The multiple dimensions is mapped to sub groups and work groups on the base OpenCL language.
*/



// TODO add asm extension as attachable header file at compile time
#ifdef cl_khr_subgroups
    #pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif
#ifdef __opencl_c_subgroups
    #pragma OPENCL EXTENSION __opencl_c_subgroups : enable
#else
    #ifdef CUDA
    

    inline uint sub_group_scan_inclusive_add_ui (uint x) {
        uint r;
        asm volatile(
            "{"
            ".reg .pred p;"
            ".reg .b32 dst;"
            "mov.b32 %0, %1;"
            "shfl.sync.up.b32  dst|p, %0, 0x1, 0x0, 0xffffffff;"
            "@p add.u32        %0, dst, %0;"
            "shfl.sync.up.b32  dst|p, %0, 0x2, 0x0, 0xffffffff;"
            "@p add.u32        %0, dst, %0;"
            "shfl.sync.up.b32  dst|p, %0, 0x4, 0x0, 0xffffffff;"
            "@p add.u32        %0, dst, %0;"
            "shfl.sync.up.b32  dst|p, %0, 0x8, 0x0, 0xffffffff;"
            "@p add.u32        %0, dst, %0;"
            "shfl.sync.up.b32  dst|p, %0, 0x10,0x0, 0xffffffff;"
            "@p add.u32        %0, dst, %0;"
            "}"
            : "=r"(r) : "r"(x));
        return r;
    }

    inline uint sub_group_scan_inclusive_min_ui (uint x) {
        uint r;
        asm volatile(
            "{"
            ".reg .pred p;"
            ".reg .b32 dst;"
            "mov.b32 %0, %1;"
            "shfl.sync.up.b32  dst|p, %0, 0x1, 0x0, 0xffffffff;"
            "@p min.u32        %0, dst, %0;"
            "shfl.sync.up.b32  dst|p, %0, 0x2, 0x0, 0xffffffff;"
            "@p min.u32        %0, dst, %0;"
            "shfl.sync.up.b32  dst|p, %0, 0x4, 0x0, 0xffffffff;"
            "@p min.u32        %0, dst, %0;"
            "shfl.sync.up.b32  dst|p, %0, 0x8, 0x0, 0xffffffff;"
            "@p min.u32        %0, dst, %0;"
            "shfl.sync.up.b32  dst|p, %0, 0x10,0x0, 0xffffffff;"
            "@p min.u32        %0, dst, %0;"
            "}"
            : "=r"(r) : "r"(x));
        return r;
    }

    inline uint sub_group_scan_inclusive_max_ui (uint x) {
        uint r;
        asm volatile(
            "{"
            ".reg .b32 Ry, Rx;"
            "shfl.sync.up.b32  Ry, %1, 0x1, 0x0, 0xffffffff;"
            "max.u32             Rx, Ry, %1;"
            "shfl.sync.up.b32  Ry, Rx, 0x2, 0x0, 0xffffffff;"
            "max.u32             Rx, Ry, Rx;"
            "shfl.sync.up.b32  Ry, Rx, 0x4, 0x0, 0xffffffff;"
            "max.u32             Rx, Ry, Rx;"
            "shfl.sync.up.b32  Ry, Rx, 0x8, 0x0, 0xffffffff;"
            "max.u32             Rx, Ry, Rx;"
            "shfl.sync.up.b32  Ry, Rx, 0x10,0x0, 0xffffffff;"
            "max.u32             %0, Ry, Rx;"
            "}"
            : "=r"(r) : "r"(x));
        return r;
    }

    

    inline uint sub_group_broadcast_ui (uint x, uint sub_group_local_id) {
        uint r;
        asm volatile(
            "shfl.sync.idx.b32  %0, %1, %2, 0x1f, 0xffffffff;"
            : "=r"(r) : "r"(x), "r"(sub_group_local_id));
        return r;
    }

    inline uint sub_group_non_uniform_broadcast_ui (uint x, uint sub_group_local_id) {
        uint r;
        asm volatile(
            ".reg .b32 mask;"
            "activemask.b32 mask;"
            "shfl.sync.idx.b32  %0, %1, %2, 0x1f, mask;"
            : "=r"(r) : "r"(x), "r"(sub_group_local_id));
        return r;
    }

    inline uint sub_group_masked_broadcast_ui (uint x, uint sub_group_local_id, uint mask) {
        uint r;
        asm volatile(
            "shfl.sync.idx.b32  %0, %1, %2, 0x1f, %3;"
            : "=r"(r) : "r"(x), "r"(sub_group_local_id), "r"(mask));
        return r;
    }

    inline void sub_group_barrier() {
        asm volatile("bar.warp.sync 0xffffffff;");
    }

    inline uint sub_group_reduce_max_ui (uint x) {
        uint r;
        asm volatile(
            "redux.sync.max.u32 %0, %1, 0xffffffff;"
            : "=r"(r) : "r"(x));
        return r;
    }

    inline uint sub_group_reduce_min_ui (uint x) {
        uint r;
        asm volatile(
            "redux.sync.min.u32 %0, %1, 0xffffffff;"
            : "=r"(r) : "r"(x));
        return r;
    }

    inline uint sub_group_reduce_or_ui (uint x) {
        uint r;
        asm volatile(
            "redux.sync.or.b32 %0, %1, 0xffffffff;"
            : "=r"(r) : "r"(x));
        return r;
    }

    inline uint sub_group_broadcast_first_ui (uint x) {
        uint r;
        uint mask = sub_group_activemask();
        uint sub_group_local_id = findLeadingOne(mask & -mask);
        asm volatile(
            "shfl.sync.idx.b32  %0, %1, %2, 0x1f, %3;"
            : "=r"(r) : "r"(x), "r"(sub_group_local_id), "r"(mask));
        return r;
    }
    #endif
#endif
#ifdef cl_khr_subgroup_ballot
#pragma OPENCL EXTENSION cl_khr_subgroup_ballot : enable
#endif


inline size_t get_local_linear_size() { return get_local_size(0) * get_local_size(1) * get_local_size(2); }


// #define CONF_DEBUG_KERNEL
#ifdef CONF_DEBUG_KERNEL
#define DEBUG(...) __VA_ARGS__
#else
#define DEBUG(...)
#endif 

// UTILS

//------------------------------------------------------------------------

inline uint cover8x8_selectFlips(int dx, int dy) // 10 instr
{
    uint flips = 0;
    if (dy > 0 || (dy == 0 && dx <= 0))
        flips ^= (1 << CR_FLIPBIT_FLIP_X) ^ (1 << CR_FLIPBIT_FLIP_Y) ^ (1 << CR_FLIPBIT_COMPL);
    if (dx > 0)
        flips ^= (1 << CR_FLIPBIT_FLIP_X) ^ (1 << CR_FLIPBIT_FLIP_Y);
    if (abs(dx) < abs(dy))
        flips ^= (1 << CR_FLIPBIT_SWAP_XY) ^ (1 << CR_FLIPBIT_FLIP_Y);
    return flips;
}

inline ulong cover8x8_lookup_mask(long yinit, uint yinc, uint flips, volatile const ulong* lut)
{
    // First half.

    uint yfrac = ugetLo(yinit);
    uint shape = add_clamp_0_x(ugetHi(yinit) + 4, 0, 11);
    add_add_carry(&yfrac, yfrac, yinc, &shape, shape, shape);
    add_add_carry(&yfrac, yfrac, yinc, &shape, shape, shape);
    add_add_carry(&yfrac, yfrac, yinc, &shape, shape, shape);
    int oct = flips & ((1 << CR_FLIPBIT_FLIP_X) | (1 << CR_FLIPBIT_SWAP_XY));
    ulong mask = *(ulong*)((uchar*)lut + oct + (shape << 5));

    // Second half.

    add_add_carry(&yfrac, yfrac, yinc, &shape, shape, shape);
    shape = add_clamp_0_x(ugetHi(yinit) + 4, popcount(shape & 15), 11);
    add_add_carry(&yfrac, yfrac, yinc, &shape, shape, shape);
    add_add_carry(&yfrac, yfrac, yinc, &shape, shape, shape);
    add_add_carry(&yfrac, yfrac, yinc, &shape, shape, shape);
    mask |= *(ulong*)((uchar*)lut + oct + (shape << 5) + (12 << 8));
    return (flips >= (1 << CR_FLIPBIT_COMPL)) ? ~mask : mask;
}

inline void cover8x8_setupLUT(volatile ulong* lut)
{
    for (int lutIdx = get_local_linear_id(); lutIdx < CR_COVER8X8_LUT_SIZE; lutIdx += get_local_linear_size())
    {
        int _half       = (lutIdx < (CR_COVER8X8_LUT_SIZE/2)) ? 0 : 1;
        int yint       = (lutIdx >> 5) - _half * 12 - 3;
        uint shape      = ((lutIdx >> 2) & 7) << (31 - 2);
        int slctSwapXY = lutIdx << (31 - 1);
        int slctNegX   = lutIdx << (31 - 0);
        int slctCompl  = slctSwapXY ^ slctNegX;

        ulong mask = 0;
        int xlo = _half * 4;
        int xhi = xlo + 4;
        for (int x = xlo; x < xhi; x++)
        {
            int ylo = slct_i(0, max(yint, 0), slctCompl);
            int yhi = slct_i(min(yint, 8), 8, slctCompl);
            for (int y = ylo; y < yhi; y++)
            {
                int xx = slct_i(x, y, slctSwapXY);
                int yy = slct_i(y, x, slctSwapXY);
                xx = slct_i(xx, 7 - xx, slctNegX);
                mask |= (ulong)1 << (xx + yy * 8);
            }
            yint += shape >> 31;
            shape <<= 1;
        }
        lut[lutIdx] = mask;
    }
}

inline ulong cover8x8_conservative_fast(int ox, int oy, int dx, int dy, uint flips, volatile const ulong* lut) // 54 instr
{
    float  halfPixel  = (float)(1 << (CR_SUBPIXEL_LOG2 - 1));
    float  yinitBias  = (float)(1 << (31 - CR_MAXVIEWPORT_LOG2 - CR_SUBPIXEL_LOG2 * 2));
    float  yinitScale = (float)(1 << (32 - CR_SUBPIXEL_LOG2));
    float  yincScale  = 65536.0f * 65536.0f;

    int  slctFlipY  = flips << (31 - CR_FLIPBIT_FLIP_Y);
    int  slctFlipX  = flips << (31 - CR_FLIPBIT_FLIP_X);
    int  slctSwapXY = flips << (31 - CR_FLIPBIT_SWAP_XY);

    // Evaluate cross product.

    int t = ox * dy - oy * dx;
    float det = (float)slct_i(t, t - dy * (7 << CR_SUBPIXEL_LOG2), slctFlipX);

    float xabs = (float)abs(slct_i(dx, dy, slctSwapXY));
    float yabs = (float)abs(slct_i(dy, dx, slctSwapXY));
    det = det + xabs * halfPixel + yabs * halfPixel;

    if (flips >= (1 << CR_FLIPBIT_COMPL))
        det = -det;

    // Represent Y as a function of X.

    float xrcp  = 1.0f / xabs;
    float yzero = det * yinitScale * xrcp + yinitBias;
    long yinit = f32_to_s64(slct_f(yzero, -yzero, slctFlipY));
    uint yinc  = f32_to_u32_sat(yabs * xrcp * yincScale);

    // Lookup.

    return cover8x8_lookup_mask(yinit, yinc, flips, lut);
}

inline ulong cover8x8_exact_fast(int ox, int oy, int dx, int dy, uint flips, volatile const ulong* lut) // 52 instr
{
    float  yinitBias  = (float)(1 << (31 - CR_MAXVIEWPORT_LOG2 - CR_SUBPIXEL_LOG2 * 2));
    float  yinitScale = (float)(1 << (32 - CR_SUBPIXEL_LOG2));
    float  yincScale  = 65536.0f * 65536.0f;

    int  slctFlipY  = flips << (31 - CR_FLIPBIT_FLIP_Y);
    int  slctFlipX  = flips << (31 - CR_FLIPBIT_FLIP_X);
    int  slctSwapXY = flips << (31 - CR_FLIPBIT_SWAP_XY);

    // Evaluate cross product.

    int t = ox * dy - oy * dx;
    float det = (float)slct_i(t, t - dy * (7 << CR_SUBPIXEL_LOG2), slctFlipX);
    if (flips >= (1 << CR_FLIPBIT_COMPL))
        det = -det;

    // Represent Y as a function of X.

    float xrcp  = 1.0f / (float) abs(slct_i(dx, dy, slctSwapXY));
    float yzero = det * yinitScale * xrcp + yinitBias;
    long yinit = f32_to_s64(slct_f(yzero, -yzero, slctFlipY));
    uint yinc  = f32_to_u32_sat((float)abs(slct_i(dy, dx, slctSwapXY)) * xrcp * yincScale);

    // Lookup.

    return cover8x8_lookup_mask(yinit, yinc, flips, lut);
}

inline uint idiv_fast(uint a, uint b)
{
    return f32_to_u32_sat_rmi(((float)a + 0.5f) / (float)b);
}

//------------------------------------------------------------------------
// v0 = subpixels relative to the bottom-left sampling point

inline uint3 setupPleq(float3 values, int2 v0, int2 d1, int2 d2, float areaRcp, int samplesLog2)
{
    float mx = fmax(fmax(values.x, values.y), values.z);
    int sh = min(max((__float_as_int(mx) >> 23) - (127 + 22), 0), 8);
    int t0 = (uint)values.x >> sh;
    int t1 = ((uint)values.y >> sh) - t0;
    int t2 = ((uint)values.z >> sh) - t0;
    
    uint rcpMant = (__float_as_int(areaRcp) & 0x007FFFFF) | 0x00800000;
    int rcpShift = (23 + 127) - (__float_as_int(areaRcp) >> 23);

    uint3 pleq;
    long xc = ((long)t1 * d2.y - (long)t2 * d1.y) * rcpMant;
    long yc = ((long)t2 * d1.x - (long)t1 * d2.x) * rcpMant;
    pleq.x = (uint)(xc >> (rcpShift - (sh + CR_SUBPIXEL_LOG2 - samplesLog2)));
    pleq.y = (uint)(yc >> (rcpShift - (sh + CR_SUBPIXEL_LOG2 - samplesLog2)));

    int centerX = (v0.x * 2 + min_min(d1.x, d2.x, 0) + max_max(d1.x, d2.x, 0)) >> (CR_SUBPIXEL_LOG2 - samplesLog2 + 1);
    int centerY = (v0.y * 2 + min_min(d1.y, d2.y, 0) + max_max(d1.y, d2.y, 0)) >> (CR_SUBPIXEL_LOG2 - samplesLog2 + 1);
    int vcx = v0.x - (centerX << (CR_SUBPIXEL_LOG2 - samplesLog2));
    int vcy = v0.y - (centerY << (CR_SUBPIXEL_LOG2 - samplesLog2));

    pleq.z = t0 << sh;
    pleq.z -= (uint)(((xc >> 13) * vcx + (yc >> 13) * vcy) >> (rcpShift - (sh + 13)));
    pleq.z -= pleq.x * centerX + pleq.y * centerY;
    return pleq;
}

//------------------------------------------------------------------------

inline int clipPolygonWithPlane(float* baryOut, const float* baryIn, int numIn, float v0, float v1, float v2)
{
    int numOut = 0;
    if (numIn >= 3)
    {
        int ai = (numIn - 1) * 2;
        float av = v0 + v1 * baryIn[ai + 0] + v2 * baryIn[ai + 1];
        for (int bi = 0; bi < numIn * 2; bi += 2)
        {
            float bv = v0 + v1 * baryIn[bi + 0] + v2 * baryIn[bi + 1];
            if (av * bv < 0.0f)
            {
                float bc = av / (av - bv);
                float ac = 1.0f - bc;
                baryOut[numOut + 0] = baryIn[ai + 0] * ac + baryIn[bi + 0] * bc;
                baryOut[numOut + 1] = baryIn[ai + 1] * ac + baryIn[bi + 1] * bc;
                numOut += 2;
            }
            if (bv >= 0.0f)
            {
                baryOut[numOut + 0] = baryIn[bi + 0];
                baryOut[numOut + 1] = baryIn[bi + 1];
                numOut += 2;
            }
            ai = bi;
            av = bv;
        }
    }
    return (numOut >> 1);
}
//------------------------------------------------------------------------
// bary = &Vec2f[9] (output)
// v0 = &Vec4f(clipPos0)
// v1 = &Vec4f(clipPos1)
// v2 = &Vec4f(clipPos2)
// d1 = &Vec4f(clipPos1 - clipPos0)
// d2 = &Vec4f(clipPos2 - clipPos0)

inline int clipTriangleWithFrustum(float* bary, const float* v0, const float* v1, const float* v2, const float* d1, const float* d2)
{
    int num = 3;
    bary[0] = 0.0f, bary[1] = 0.0f;
    bary[2] = 1.0f, bary[3] = 0.0f;
    bary[4] = 0.0f, bary[5] = 1.0f;

    if ((v0[3] < fabs(v0[0])) | (v1[3] < fabs(v1[0])) | (v2[3] < fabs(v2[0])))
    {
        float temp[18];
        num = clipPolygonWithPlane(temp, bary, num, v0[3] + v0[0], d1[3] + d1[0], d2[3] + d2[0]);
        num = clipPolygonWithPlane(bary, temp, num, v0[3] - v0[0], d1[3] - d1[0], d2[3] - d2[0]);
    }
    if ((v0[3] < fabs(v0[1])) | (v1[3] < fabs(v1[1])) | (v2[3] < fabs(v2[1])))
    {
        float temp[18];
        num = clipPolygonWithPlane(temp, bary, num, v0[3] + v0[1], d1[3] + d1[1], d2[3] + d2[1]);
        num = clipPolygonWithPlane(bary, temp, num, v0[3] - v0[1], d1[3] - d1[1], d2[3] - d2[1]);
    }
    if ((v0[3] < fabs(v0[2])) | (v1[3] < fabs(v1[2])) | (v2[3] < fabs(v2[2])))
    {
        float temp[18];
        num = clipPolygonWithPlane(temp, bary, num, v0[3] + v0[2], d1[3] + d1[2], d2[3] + d2[2]);
        num = clipPolygonWithPlane(bary, temp, num, v0[3] - v0[2], d1[3] - d1[2], d2[3] - d2[2]);
    }
    return num;
}

#endif