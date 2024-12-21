// Constants

#define CR_MAXVIEWPORT_LOG2     11      // ViewportSize / PixelSize.
#define CR_SUBPIXEL_LOG2        4       // PixelSize / SubpixelSize.

#define CR_MAXBINS_LOG2         4       // ViewportSize / BinSize.
#define CR_BIN_LOG2             4       // BinSize / TileSize.
#define CR_TILE_LOG2            3       // TileSize / PixelSize.

#define CR_COVER8X8_LUT_SIZE    768     // 64-bit entries.
#define CR_FLIPBIT_FLIP_Y       2
#define CR_FLIPBIT_FLIP_X       3
#define CR_FLIPBIT_SWAP_XY      4
#define CR_FLIPBIT_COMPL        5

#define CR_BIN_STREAMS_LOG2     4
#define CR_BIN_SEG_LOG2         9       // 32-bit entries.
#define CR_TILE_SEG_LOG2        5       // 32-bit entries.

#define CR_MAXSUBTRIS_LOG2      24      // Triangle structs. Dictated by CoarseRaster.
#define CR_COARSE_QUEUE_LOG2    10      // Triangles.

#define CR_SETUP_WARPS          2
#define CR_SETUP_OPT_BLOCKS     8
#define CR_BIN_WARPS            16
#define CR_COARSE_WARPS         16      // Must be a power of two.
#define CR_FINE_MAX_WARPS       20      // Absolute maximum for 48KB of shared mem.
#define CR_FINE_OPT_WARPS       20      // Preferred value.

//------------------------------------------------------------------------

#define CR_MAXVIEWPORT_SIZE     (1 << CR_MAXVIEWPORT_LOG2)
#define CR_SUBPIXEL_SIZE        (1 << CR_SUBPIXEL_LOG2)
#define CR_SUBPIXEL_SQR         (1 << (CR_SUBPIXEL_LOG2 * 2))

#define CR_MAXBINS_SIZE         (1 << CR_MAXBINS_LOG2)
#define CR_MAXBINS_SQR          (1 << (CR_MAXBINS_LOG2 * 2))
#define CR_BIN_SIZE             (1 << CR_BIN_LOG2)
#define CR_BIN_SQR              (1 << (CR_BIN_LOG2 * 2))

#define CR_MAXTILES_LOG2        (CR_MAXBINS_LOG2 + CR_BIN_LOG2)
#define CR_MAXTILES_SIZE        (1 << CR_MAXTILES_LOG2)
#define CR_MAXTILES_SQR         (1 << (CR_MAXTILES_LOG2 * 2))
#define CR_TILE_SIZE            (1 << CR_TILE_LOG2)
#define CR_TILE_SQR             (1 << (CR_TILE_LOG2 * 2))

#define CR_BIN_STREAMS_SIZE     (1 << CR_BIN_STREAMS_LOG2)
#define CR_BIN_SEG_SIZE         (1 << CR_BIN_SEG_LOG2)
#define CR_TILE_SEG_SIZE        (1 << CR_TILE_SEG_LOG2)

#define CR_MAXSUBTRIS_SIZE      (1 << CR_MAXSUBTRIS_LOG2)
#define CR_COARSE_QUEUE_SIZE    (1 << CR_COARSE_QUEUE_LOG2)

//-----------------------------------------------------------------------------
// Types
// GLSL compiler must set up this ones
typedef struct
{
    short v0x;    // Subpixels relative to viewport center. Valid if triSubtris = 1.
    short v0y;
    short v1x;
    short v1y;
    short v2x;
    short v2y;

    uint misc;   // triSubtris=1: (zmin:20, f01:4, f12:4, f20:4), triSubtris>=2: (subtriBase)
} CRTriangleHeader;

typedef struct 
{
    U32 zx;     // zx * sampleX + zy * sampleY + zb = lerp(CR_DEPTH_MIN, CR_DEPTH_MAX, (clipZ / clipW + 1) / 2)
    U32 zy;
    U32 zb;
    U32 zslope; // (abs(zx) + abs(zy)) * (samplesPerPixel / 2)

    S32 wx;     // wx * (sampleX * 2 + 1) + wy * (sampleY * 2 + 1) + wb = minClipW / clipW * CR_BARY_MAX
    S32 wy;
    S32 wb;

    S32 ux;     // ux * (sampleX * 2 + 1) + uy * (sampleY * 2 + 1) + ub = baryU * minClipW / clipW * CR_BARY_MAX
    S32 uy;
    S32 ub;

    S32 vx;     // vx * (sampleX * 2 + 1) + vy * (sampleY * 2 + 1) + vb = baryV * minClipW / clipW * CR_BARY_MAX
    S32 vy;
    S32 vb;

    U32 vi0;    // Vertex indices.
    U32 vi1;
    U32 vi2;
} CRTriangleData; 

// Extensions

#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION __opencl_c_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_ballot : enable

// Utility functions

inline uint   getLo                   (ulong a)                 { return a & 0x00000000FFFFFFFFu; }
inline int   getLo                   (long a)                 { return a & 0x00000000FFFFFFFF; }
inline uint   getHi                   (ulong a)                 { return a >> 32; }
inline int   getHi                   (long a)                 { return a >> 32; }
inline ulong   combineLoHi             (uint lo, uint hi)        { return ((ulong)hi << 32) + (ulong)lo; }
inline long   combineLoHi             (int lo, int hi)        { return ((long)hi << 32) + (long)lo; }

// ISA dependancy

#ifdef CUDA
inline int      findLeadingOne      (uint v)                    { uint r; asm("bfind.u32 %0, %1;" : "=r"(r) : "r"(v)); return r; }
inline uint     getLaneMaskLt       (void)                      { uint r; asm("mov.u32 %0, %%lanemask_lt;" : "=r"(r)); return r; }
inline uint     getLaneMaskLe       (void)                      { uint r; asm("mov.u32 %0, %%lanemask_le;" : "=r"(r)); return r; }

inline void  add_add_carry           (uint* rlo, uint alo, uint blo, uint* rhi, uint ahi, uint bhi) { ulong r = combineLoHi(alo, ahi) + combineLoHi(blo, bhi); *rlo = getLo(r); *rhi = getHi(r); }
inline uint     f32_to_u32_sat_rmi  (float a)                   { uint v; asm("cvt.rmi.sat.u32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
inline int      add_s16lo_s16lo     (int a, int b)              { int v; asm("vadd.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      add_s16hi_s16lo     (int a, int b)              { int v; asm("vadd.s32.s32.s32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      sub_s16lo_s16lo     (int a, int b)              { int v; asm("vsub.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      sub_s16hi_s16lo     (int a, int b)			    { int v; asm("vsub.s32.s32.s32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      sub_s16hi_s16hi     (int a, int b)              { int v; asm("vsub.s32.s32.s32 %0, %1.h1, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      max_max             (int a, int b, int c)       { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
inline int      min_min             (int a, int b, int c)       { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
inline uint     add_sub             (uint a, uint b, uint c)    { uint v; asm("vsub.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(c), "r"(b)); return v; }
inline int      add_clamp_0_x       (int a, int b, int c)       { int v; asm("vadd.u32.s32.s32.sat.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }

inline uint     get_max_sub_group_size(void) { return 32; }
inline uint     sub_group_ballot(int p) { uint r; asm("vote.sync.ballot.b32  r,p,0xffffffff" : "=r"(r) : "=p"(p)); return r; }
inline uint     sub_group_any(int p)    { uint r; asm("vote.sync.any.b32  r,p,0xffffffff" : "=r"(r) : "=p"(p)); return r; }
inline uint     sub_group_all(int p)    { uint r; asm("vote.sync.all.b32  r,p,0xffffffff" : "=r"(r) : "=p"(p)); return r; }
#else
inline uint     f32_to_u32_sat_rmi  (float a)                   { return (uint)a; }
inline int      add_s16lo_s16lo     (int a, int b)              { return (a & 0xFFFF) + (b & 0xFFFF); }
inline int      add_s16hi_s16lo     (int a, int b)              { return (a >> 16) + (b & 0xFFFF); }
inline int      sub_s16lo_s16lo     (int a, int b)              { return (a & 0xFFFF) - (b & 0xFFFF); }
inline int      sub_s16hi_s16lo     (int a, int b)			    { return (a >> 16) - (b & 0xFFFF); }
inline int      sub_s16hi_s16hi     (int a, int b)              { return (a >> 16) - (b >> 16); }
inline int      max_max             (int a, int b, int c)       { return max(a, max(b, c)); }
inline int      min_min             (int a, int b, int c)       { return min(a, min(b, c)); }
inline uint     add_sub             (uint a, uint b, uint c)    { return a+b-c; }
inline int      add_clamp_0_x       (int a, int b, int c)       { return clamp(a+b,0,c); }
inline uint     getLaneMaskLt       (void) {
    uint mask;
    for(uint warp = 0; warp < get_max_sub_group_size(); ++warp) mask |= 1 << warp;
    return mask >> (get_max_sub_group_size() - get_local_id(0));
}
#endif

inline uint idiv_fast(uint a, uint b)
{
    return f32_to_u32_sat_rmi(((float)a + 0.5f) / (float)b);
}


// Render flags
#define RENDER_MODE_FLAG_ENABLE_QUADS   (1 << 0)
#define RENDER_MODE_FLAG_ENABLE_DEPTH   (1 << 1)
#define RENDER_MODE_FLAG_ENABLE_LERP    (1 << 2)

// Blender data
#define BLENDER_FUNC_ADD                     0
#define BLENDER_FUNC_SUBTRACT                1
#define BLENDER_FUNC_REVERSE_SUBTRACT        2

#define BLENDER_OP_ZERO                      0
#define BLENDER_OP_ONE                       1
#define BLENDER_OP_SRC_COLOR                 2
#define BLENDER_OP_ONE_MINUS_SRC_COLOR       3
#define BLENDER_OP_SRC_ALPHA                 4
#define BLENDER_OP_ONE_MINUS_SRC_ALPHA       5
#define BLENDER_OP_DST_ALPHA                 6
#define BLENDER_OP_ONE_MINUS_DST_ALPHA       7
#define BLENDER_OP_DST_COLOR                 8
#define BLENDER_OP_ONE_MINUS_DST_COLOR       9
#define BLENDER_OP_SRC_ALPHA_SATURATE        10
#define BLENDER_OP_CONSTANT_COLOR            11
#define BLENDER_OP_ONE_MINUS_CONSTANT_COLOR  12
#define BLENDER_OP_CONSTANT_ALPHA            13
#define BLENDER_OP_ONE_MINUS_CONSTANT_ALPHA  14

// Texture types
#define TEX_R8                               0
#define TEX_RG8                              1
#define TEX_RGB8                             2
#define TEX_RGBA8                            3
#define TEX_RGBA4                            4
#define TEX_RGB5_A1                          5
#define TEX_RGB565                           6

//----------------------------------------------
// Transpiler objects
//----------------------------------------------

typedef struct {
    float4 gl_FragColor;
    float gl_FragDepth;
    uint color;
    bool discard;
} fragment_shader_output_t;

typedef struct {
    float3 color;
} fragment_shader_input_t;

typedef struct {
    float4 color;
} vertex_shader_output_t;

inline void fragment_shader(
    fragment_shader_input_t* input, 
    fragment_shader_output_t* output
) {
    output->gl_FragColor = (float4){input->color, 1.f};
    output->discard = false;
}

//------------------------------------------------------------------------
// Shader wrappers.
//------------------------------------------------------------------------

inline float4 get_varying_at_vertex(
    int varying_idx, int vert_idx, 
    image1d_t t_vertex_buffer
) {
    return read_imagef(t_vertex_buffer, vert_idx * (sizeof(vertex_shader_output_t) / sizeof(float4)) + varying_idx + 1);
}

inline float4 interpolate_varying(
    int varying_idx, const uint3 vert_idx, const float3 bary, 
    image1d_t t_vertex_buffer
) {
    float4 v0 = get_varying_at_vertex(varying_idx, vert_idx.x, t_vertex_buffer);
    float4 v1 = get_varying_at_vertex(varying_idx, vert_idx.y, t_vertex_buffer);
    float4 v2 = get_varying_at_vertex(varying_idx, vert_idx.z, t_vertex_buffer);
    return v0 * bary.x + v1 * bary.y + v2 * bary.z; 
}

inline float3 compute_bary(
    const int3* wpleq, const int3* upleq, const int3* vpleq,
    int sample_x, int sample_y)
{
    float w = 1.0f / (float){wpleq->x * sample_x + wpleq->y * sample_y + wpleq->z};
    float u = w * (float){upleq->x * sample_x + upleq->y * sample_y + upleq->z};
    float v = w * (float){vpleq->x * sample_x + vpleq->y * sample_y + vpleq->z};
    return (float3){1.0f - u - v, u, v};
}

//------------------------------------------------------------------------

inline void run_fragment_shader(
    fragment_shader_output_t* output,
    int tri_idx, int data_idx, int pixel_x, int pixel_y, uint centroid, local volatile uint* shared,
    image1d_t t_tri_data,
    image1d_t t_vertex_buffer,
    )
{
    rasterize_output_t input;
    
    // Fetch primitive data.
    uint4 t1 = read_imageui(t_tri_data, data_idx * 4 + 1); // wx, wy, wb, ux
    uint4 t2 = read_imageui(t_tri_data, data_idx * 4 + 2); // uy, ub, vx, vy
    uint4 t3 = read_imageui(t_tri_data, data_idx * 4 + 3); // vb, vi0, vi1, vi2

    // Pixel varying data.
    int3 wpleq = (int3){t1.x, t1.y, t1.z};
    int3 upleq = (int3){t1.w, t2.x, t2.y};
    int3 vpleq = (int3){t2.z, t2.w, t3.x};
    int4 vert_idx = (int3){t3.y, t3.z, t3.w};
    float3 bary = compute_bary(wpleq, upleq, vpleq, (pixel_x * 2 + 1), (pixel_y * 2 + 1));

    // Transpiler dependant
    input.color = interpolate_varying(0, &vert_idx, &bary, t_vertex_buffer);

    fragment_shader(&input, output);
}

//------------------------------------------------------------------------

inline void run_blend_shader(
    blend_shader_input_t* input, blend_shader_output_t* output,
    int tri_idx, int pixel_x, int pixel_y, int sampleIdx, uint src, uint dst)
{
    output->wirte_color = true;
    output->color = src;
}

//------------------------------------------------------------------------
// Utility funcs.
//------------------------------------------------------------------------

inline ulong cover8x8_lookup_mask(long yinit, uint yinc, uint flips, volatile const ulong* lut)
{
    // First half.

    uint yfrac = getLo(yinit);
    uint shape = add_clamp_0_x(getHi(yinit) + 4, 0, 11);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    int oct = flips & ((1 << CR_FLIPBIT_FLIP_X) | (1 << CR_FLIPBIT_SWAP_XY));
    ulong mask = *(ulong*)((uchar*)lut + oct + (shape << 5));

    // Second half.

    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    shape = add_clamp_0_x(getHi(yinit) + 4, popcount(shape & 15), 11);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    add_add_carry(yfrac, yfrac, yinc, shape, shape, shape);
    mask |= *(ulong*)((uchar*)lut + oct + (shape << 5) + (12 << 8));
    return (flips >= (1 << CR_FLIPBIT_COMPL)) ? ~mask : mask;
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
    float det = (float)slct(t, t - dy * (7 << CR_SUBPIXEL_LOG2), slctFlipX);
    if (flips >= (1 << CR_FLIPBIT_COMPL))
        det = -det;

    // Represent Y as a function of X.

    float xrcp  = 1.0f / fabs(slct(dx, dy, slctSwapXY));
    float yzero = det * yinitScale * xrcp + yinitBias;
    S64 yinit = f32_to_s64(slct(yzero, -yzero, slctFlipY));
    U32 yinc  = f32_to_u32_sat(fabs(slct(dy, dx, slctSwapXY)) * xrcp * yincScale);

    // Lookup.

    return cover8x8_lookup_mask(yinit, yinc, flips, lut);
}

//------------------------------------------------------------------------

inline void init_tile_z_max(uint* tile_z_max, bool* tile_z_upd, local volatile uint* w_tile_depth)
{
    *tile_z_max = CR_DEPTH_MAX;
    *tile_z_upd = (min(w_tile_depth[get_local_id(0)], w_tile_depth[get_local_id(0) + 32]) < *tile_z_max);
}

inline void update_tile_z_max(uint render_mode_flags, uint* tile_z_max, bool* tile_z_upd, local volatile uint* w_tile_depth, local volatile uint* temp)
{
    if ((render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0 && sub_group_any(*tile_z_upd))
    {
        uint z = max(w_tile_depth[get_local_id(0)], w_tile_depth[get_local_id(0) + 32]);
        temp[get_local_id(0) + 16] = z;
        z = max(z, temp[get_local_id(0) + 16 -  1]); temp[get_local_id(0) + 16] = z;
        z = max(z, temp[get_local_id(0) + 16 -  2]); temp[get_local_id(0) + 16] = z;
        z = max(z, temp[get_local_id(0) + 16 -  4]); temp[get_local_id(0) + 16] = z;
        z = max(z, temp[get_local_id(0) + 16 -  8]); temp[get_local_id(0) + 16] = z;
        z = max(z, temp[get_local_id(0) + 16 - 16]); temp[get_local_id(0) + 16] = z;
        *tile_z_max = temp[47];
        *tile_z_upd = false;
    }
}

//------------------------------------------------------------------------

inline void get_triangle(
    int* tri_idx, int* *data_idx, uint4* tri_header, int* *segment, 
    global const CRTriangleHeader* g_tri_header,
    global const int* g_tile_seg_data,
    global const int* g_tile_seg_next,
    global const int* g_tile_seg_count,
    image2d_t img_tri_header,
)
{

    if (get_local_id(0) >= g_tile_seg_count[*segment])
    {
        *tri_idx = -1;
        *data_idx = -1;
    }
    else
    {
        int subtri_idx = g_tile_seg_data[*segment * CR_TILE_SEG_SIZE + get_local_id(0)];
        *tri_idx = subtri_idx >> 3;
        *data_idx = *tri_idx;
        subtri_idx &= 7;
        if (subtri_idx != 7)
            *data_idx = g_tri_header[*tri_idx].misc + subtri_idx;
        *tri_header = read_imageui(img_tri_header, *data_idx);
    }

    // advance to next segment
    *segment = g_tile_seg_next[*segment];
}

//------------------------------------------------------------------------

inline bool early_z_cull(uint render_mode_flags, uint4 tri_header, uint tile_z_max)
{
    if ((render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0)
    {
        uint zmin = tri_header.w & 0xFFFFF000;
        if (zmin >= tile_z_max)
            return true;
    }
    return false;
}

//------------------------------------------------------------------------

inline uint triangle_pixel_coverage(const int samples_log_2, const uint4 tri_header, int tile_x, int tile_y, local volatile ulong* s_cover8x8_lut, 
    int p_viewport_width, int p_viewport_height)
{
    int base_x = (tile_x << (CR_TILE_LOG2 + CR_SUBPIXEL_LOG2)) - ((p_viewport_width  - 1) << (CR_SUBPIXEL_LOG2 - 1));
    int base_y = (tile_y << (CR_TILE_LOG2 + CR_SUBPIXEL_LOG2)) - ((p_viewport_height - 1) << (CR_SUBPIXEL_LOG2 - 1));

    // extract S16 vertex positions while subtracting tile coordinates
    int v0x  = sub_s16lo_s16lo(tri_header.x, base_x);
    int v0y  = sub_s16hi_s16lo(tri_header.x, base_y);
    int v01x = sub_s16lo_s16lo(tri_header.y, tri_header.x);
    int v01y = sub_s16hi_s16hi(tri_header.y, tri_header.x);
    int v20x = sub_s16lo_s16lo(tri_header.x, tri_header.z);
    int v20y = sub_s16hi_s16hi(tri_header.x, tri_header.z);

    // extract flipbits
    uint f01 = (tri_header.w >> 6) & 0x3C;
    uint f12 = (tri_header.w >> 2) & 0x3C;
    uint f20 = (tri_header.w << 2) & 0x3C;

    // compute per-edge coverage masks
    ulong c01, c12, c20;
    if (samples_log_2 == 0)
    {
        c01 = cover8x8_exact_fast(v0x, v0y, v01x, v01y, f01, s_cover8x8_lut);
        c12 = cover8x8_exact_fast(v0x + v01x, v0y + v01y, -v01x - v20x, -v01y - v20y, f12, s_cover8x8_lut);
        c20 = cover8x8_exact_fast(v0x, v0y, v20x, v20y, f20, s_cover8x8_lut);
    }
    else
    {
        c01 = cover8x8_conservative_fast(v0x, v0y, v01x, v01y, f01, s_cover8x8_lut);
        c12 = cover8x8_conservative_fast(v0x + v01x, v0y + v01y, -v01x - v20x, -v01y - v20y, f12, s_cover8x8_lut);
        c20 = cover8x8_conservative_fast(v0x, v0y, v20x, v20y, f20, s_cover8x8_lut);
    }

    // combine masks
    return c01 & c12 & c20;
}

//------------------------------------------------------------------------

inline uint scan32_value(uint value, local volatile uint* temp)
{
    temp[get_local_id(0) + 16] = value;
    value += temp[get_local_id(0) + 16 -  1], temp[get_local_id(0) + 16] = value;
    value += temp[get_local_id(0) + 16 -  2], temp[get_local_id(0) + 16] = value;
    value += temp[get_local_id(0) + 16 -  4], temp[get_local_id(0) + 16] = value;
    value += temp[get_local_id(0) + 16 -  8], temp[get_local_id(0) + 16] = value;
    value += temp[get_local_id(0) + 16 - 16], temp[get_local_id(0) + 16] = value;
    return value;
}

inline const uint scan32_total(volatile uint* temp)
{
    return temp[47];
}

//------------------------------------------------------------------------

template <class BlendShaderClass>
inline uint determine_ROP_lane_mask(uint render_mode_flags, local volatile uint* warpTemp) // mask of lanes that should process an earlier fragment than this lane
{
    bool reverse_lanes = true;
    if ((render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) == 0)
    {
        BlendShaderClass bs;
        if (!bs.needsDst())
            reverse_lanes = false;
    }

    uint mask = (reverse_lanes) ? (1u << get_local_id(0)) : ~0u;
    do
    {
        *warpTemp = get_local_id(0);
        mask ^= 1u << *warpTemp;
    }
    while (*warpTemp != get_local_id(0));
    return mask;
}

inline int find_bit(uint render_mode_flags, ulong mask, int idx)
{
    uint x = getLo(mask);
    int  pop = popcount(x);
    bool p   = (pop <= idx);
    if (p) x = getHi(mask);
    if (p) idx -= pop;
    int bit = p ? 32 : 0;

    pop = popcount(x & 0x0000ffffu);
    p   = (pop <= idx);
    if (p) x >>= 16;
    if (p) bit += 16;
    if (p) idx -= pop;

    if ((render_mode_flags & RENDER_MODE_FLAG_ENABLE_QUADS) == 0)
    {
        // Optimized variant.
        // Assumes that scanlines do not contain holes, and doesn't thus support quad rendering.
        // Counts scanlines LSB->MSB, but bits within them MSB->LSB.
        // 21 instructions.

        U32 tmp = x & 0x000000ffu;
        pop = popcount(tmp);
        p   = (pop <= idx);
        if (p) tmp = x & 0x0000ff00u;
        if (p) idx -= pop;

        return findLeadingOne(tmp) + bit - idx;
    }
    else
    {
        // Generic variant. Counts bits LSB->MSB.
        // 33 instructions.

        pop = popcount(x & 0x000000ffu);
        p   = (pop <= idx);
        if (p) x >>= 8;
        if (p) bit += 8;
        if (p) idx -= pop;

        pop = popcount(x & 0x0000000fu);
        p   = (pop <= idx);
        if (p) x >>= 4;
        if (p) bit += 4;
        if (p) idx -= pop;

        pop = popcount(x & 0x00000003u);
        p   = (pop <= idx);
        if (p) x >>= 2;
        if (p) bit += 2;
        if (p) idx -= pop;

        if (idx >= (x & 1))
            bit++;
        return bit;
    }
}

inline ulong quad_coverage(ulong mask)
{
    mask |= mask >> 1;
    mask |= mask >> 8;
    return mask & 0x0055005500550055;
}


inline int num_fragments(uint render_mode_flags, ulong coverage)
{
    if ((render_mode_flags & RENDER_MODE_FLAG_ENABLE_QUADS) == 0)
        return popcount(coverage);
    else
        return popcount(quad_coverage(coverage)) << 2;
}

inline int find_fragment(uint render_mode_flags, ulong coverage, int frag_idx)
{
    if ((render_mode_flags & RENDER_MODE_FLAG_ENABLE_QUADS) == 0)
        return find_bit(render_mode_flags, coverage, frag_idx);
    else
    {
        int t = find_bit(render_mode_flags, quad_coverage(coverage), frag_idx >> 2);
        return t + (get_local_id(0) & 1) + ((get_local_id(0) & 2) << 2);
    }
}

//------------------------------------------------------------------------
// Single-sample implementation.
//------------------------------------------------------------------------

typedef struct {
    bool needs_dst = true; 
} blend_shader_input_t;

typedef struct {
    bool write_color;
    uint color;
} blend_shader_output_t;

inline void execute_ROP_single_sample(
    uint render_mode_flags,
    int tri_idx, int pixel_x, int pixel_y,
    uint color, uint depth, local volatile uint* ptr_color, local volatile uint* ptr_depth)
{
    blend_shader_input_t blend_shader_input;
    blend_shader_output_t blend_shader_output;

    int rounds = 0;

    if ((render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0)
    {
        do
        {
            rounds++;
            *ptr_depth = depth;
			uint s_color = *ptr_color;
			run_blend_shader(&blend_shader_input, &blend_shader_output, tri_idx, pixel_x, pixel_y, 0, color, s_color);
            if (blend_shader_output.write_color)
                *ptr_color = blend_shader_output.color;
        }
        while (depth < *ptr_depth);
    }
    else if (!blend_shader_input.needs_dst)
    {
        rounds++;
        run_blend_shader(&blend_shader_input, &blend_shader_output, tri_idx, pixel_x, pixel_y, 0, color, 0);
        if (blend_shader_output.write_color)
            *ptr_color = blend_shader_output.color;
    }
    else
    {
        do
        {
            rounds++;
            *ptr_depth = get_local_id(0);
			uint s_color = *ptr_color;
			run_blend_shader(&blend_shader_input, &blend_shader_output, tri_idx, pixel_x, pixel_y, 0, color, s_color);
            if (blend_shader_output.write_color)
                *ptr_color = blend_shader_output.color;
        }
        while (*ptr_depth != get_local_id(0));
    }
}

//------------------------------------------------------------------------

//template <class VertexClass, class FragmentShaderClass, class BlendShaderClass, U32 RenderModeFlags>
kernel void fine_raster_single_sample(
    // global atomics
    global   int* g_num_subtris,        // = numTris
    global   int* g_bin_counter,        // = 0
    global   int* g_num_bin_segs,       // = 0
    global   int* g_coarse_counter;     // = 0
    global   int* g_num_tile_segs;      // = 0
    global   int* g_num_active_tiles;   // = 0
    global   int* g_fine_counter;       // = 0
    
    // common params
    private  int    p_num_tris,
    global   void*  g_vertex_buffer,
    private  uint   p_vertex_item_sz,
    global   int*   g_index_buffer,

    private  int    p_viewport_width,
    private  int    p_viewport_height,
    private  int    p_width_pixels,
    private  int    p_height_pixels,

    private  int    p_width_bins,
    private  int    p_height_bins,
    private  int    p_num_bins,

    private  int    p_width_tiles,
    private  int    p_height_tiles,
    private  int    p_num_tiles,

    private  int    p_bin_batch_sz,

    private  int    p_deferred_clear,
    private  uint   p_clear_color,
    private  uint   p_clear_depth,
    // Setup output / bin input
    private  int    p_max_subtris,
    global   uchar* g_tri_subtris,
    global   CRTriangleHeader*  g_tri_header,
    global   CRTriangleData*  g_tri_data, // unused
    // Bin output / coarse input.
    private  int    p_max_bin_segs,
    global   int*   g_bin_first_seg,    
    global   int*   g_bin_total,
    global   int*   g_bin_seg_data,
    global   int*   g_bin_seg_next,
    global   int*   g_bin_seg_count,
    // Coarse output / fine input.
    private  int    p_max_tile_segs,
    global   int*   g_active_tiles,        // CR_MAXTILES_SQR * (S32 tile_idx)
    global   int*   g_tile_first_seg,       // CR_MAXTILES_SQR * (S32 seg_idx), -1 = none
    global   int*   g_tile_seg_data,        // p_max_tile_segs * CR_TILE_SEG_SIZE * (S32 tri_idx)
    global   int*   g_tile_seg_next,        // p_max_tile_segs * (S32 seg_idx), -1 = none
    global   int*   g_tile_seg_count,
    private  uint   p_render_mode_flags,

    image2d_t img_color_buffer,
    image2d_t img_depth_buffer,
    image1d_t t_tri_data,
    image1d_t t_vertex_buffer,
)
{
                                                                            // for 20 warps:
    local volatile ulong    s_cover8x8_lut      [CR_COVER8X8_LUT_SIZE];           // 6KB
    local volatile uint     s_tile_color        [CR_FINE_MAX_WARPS][CR_TILE_SQR]; // 5KB
    local volatile uint     s_tile_depth        [CR_FINE_MAX_WARPS][CR_TILE_SQR]; // 5KB
    local volatile uint     s_triangle_idx      [CR_FINE_MAX_WARPS][64];          // 5KB  original triangle index
    local volatile uint     s_tri_data_idx      [CR_FINE_MAX_WARPS][64];          // 5KB  CRTriangleData index
    local volatile ulong    s_triangle_cov      [CR_FINE_MAX_WARPS][64];          // 10KB coverage mask
    local volatile uint     s_triangle_frag     [CR_FINE_MAX_WARPS][64];          // 5KB  fragment index
    local volatile uint     s_temp              [CR_FINE_MAX_WARPS][80];          // 6.25KB
                                                                            // = 47.25KB total
    // Warp Space
    local volatile uint*   w_tile_color         = &s_tile_color[get_local_id(1)];
    local volatile uint*   w_tile_depth         = &s_tile_depth[get_local_id(1)];
    local volatile uint*   w_triangle_idx       = &s_triangle_idx[get_local_id(1)];
    local volatile uint*   w_tri_data_idx       = &s_tri_data_idx[get_local_id(1)];
    local volatile ulong*  w_triangle_cov       = &s_triangle_cov[get_local_id(1)];
    local volatile uint*   w_triangle_frag      = &s_triangle_frag[get_local_id(1)];
    local volatile uint*   w_temp               = &s_temp[get_local_id(1)];

    if (g_num_subtris > p_max_subtris || g_num_bin_segs > p_max_bin_segs || g_num_tile_segs > p_max_tile_segs)
        return;

    uint rop_lane_mask = determine_ROP_lane_mask<BlendShaderClass>(p_render_mode_flags, temp[0]);
    temp[get_local_id(1)] = 0; // first 16 elements of temp are always zero
    cover8x8_setupLUT(s_cover8x8_lut);
    barrier(CLK_LOCAL_MEM_FENCE);

    // loop over tiles
    for (;;)
    {
        // pick a tile
        if (get_local_id(0) == 0)
            temp[16] = atomic_add(g_fine_counter, 1);
        int active_idx = temp[16];
        if (active_idx >= g_num_active_tiles)
        {
            break;
        }

        int tile_idx = g_active_tiles[active_idx];
        int segment = g_tile_first_seg[tile_idx];
        int tile_y = idiv_fast(tile_idx, p_width_tiles);
        int tile_x = tile_idx - tile_y * p_width_tiles;

        // initialize per-tile state
        int tri_read = 0, tri_write = 0;
        int frag_read = 0, frag_write = 0;
        w_triangle_frag[63] = 0; // "previous triangle"

        // deferred clear => clear tile
        if (p_deferred_clear)
        {
			w_tile_color[get_local_id(0)] = p_clear_color;
            w_tile_depth[get_local_id(0)] = p_clear_depth;
            w_tile_color[get_local_id(0) + 32] = p_clear_color;
            w_tile_depth[get_local_id(0) + 32] = p_clear_depth;
        }

        // otherwise => read tile from framebuffer
        else
        {
            int surf_x = (tile_x << (CR_TILE_LOG2 + 2)) + ((get_local_id(0) & (CR_TILE_SIZE - 1)) << 2);
            int surf_y = (tile_y << CR_TILE_LOG2) + (get_local_id(0) >> CR_TILE_LOG2);
			w_tile_color[get_local_id(0)] = read_imageui(img_color_buffer,(int2){surf_x,surf_y});
            w_tile_depth[get_local_id(0)] = read_imageui(img_depth_buffer,(int2){surf_x,surf_y});
            w_tile_color[get_local_id(0) + 32] = read_imageui(img_color_buffer,(int2){surf_x,surf_y+4});
            w_tile_depth[get_local_id(0) + 32] = read_imageui(img_depth_buffer,(int2){surf_x,surf_y+4});
        }

        uint tile_z_max;
        bool tile_z_upd;
        init_tile_z_max(&tile_z_max, &tile_z_upd, w_tile_depth);

        // process fragments
        for(;;)
        {
            // need to queue more fragments?
            if (frag_write - frag_read < 32 && segment >= 0)
            {
                // update tile z
                update_tile_z_max(render_mode_flags, &tile_z_max, &tile_z_upd, w_tile_depth, temp);

                // read triangles
                do
                {
                    // read triangle index and data, advance to next segment
                    int tri_idx, data_idx;
                    uint4 tri_header;
                    get_triangle(&tri_idx, &data_idx, &tri_header, &segment, 
                        g_tri_header, g_tile_seg_data, g_tile_seg_next, g_tile_seg_count, img_tri_header);

                    // early z cull
                    if (tri_idx >= 0 && early_z_cull(p_render_mode_flags, tri_header, tile_z_max))
                        tri_idx = -1;

                    // determine coverage
                    ulong coverage = triangle_pixel_coverage(0, tri_header, tile_x, tile_y, s_cover8x8_lut,
                        p_viewport_width, p_viewport_height);
                    int pop = (tri_idx == -1) ? 0 : num_fragments(p_render_mode_flags, coverage);

                    // fragment count scan
                    uint frag = scan32_value(pop, temp);
                    frag += frag_write; // frag now holds cumulative fragment count
                    frag_write += scan32_total(temp);

                    // queue non-empty triangles
                    uint good_mask = sub_group_ballot(pop != 0);
                    if (pop != 0)
                    {
                        int idx = (tri_write + popcount(good_mask & getLaneMaskLt())) & 63;
                        w_triangle_idx  [idx] = tri_idx;
                        w_tri_data_idx  [idx] = data_idx;
                        w_triangle_frag [idx] = frag;
                        w_triangle_cov  [idx] = coverage;
                    }
                    tri_write += popcount(good_mask);
                }
                while (frag_write - frag_read < 32 && segment >= 0);
            }

            // end of segment?
            if (frag_read == frag_write)
                break;

            // tag triangle boundaries
            temp[get_local_id(0) + 16] = 0;
            if (tri_read + get_local_id(0) < tri_write)
            {
                int idx = w_triangle_frag[(tri_read + get_local_id(0)) & 63] - frag_read;
                if (idx <= 32)
                    temp[idx + 16 - 1] = 1;
            }

            int rop_lane_idx = popcount(rop_lane_mask);
            uint boundary_mask = sub_group_ballot(temp[rop_lane_idx + 16]);

            // distribute fragments
            if (rop_lane_idx < frag_write - frag_read)
            {
                int tri_buf_idx = (tri_read + popcount(boundary_mask & rop_lane_mask)) & 63;
                int frag_idx = add_sub(frag_read, rop_lane_idx, w_triangle_frag[(tri_buf_idx - 1) & 63]);
                ulong coverage = w_triangle_cov[tri_buf_idx];
                int pixel_in_tile = find_fragment(p_render_mode_flags, coverage, frag_idx);
                int tri_idx = w_triangle_idx[tri_buf_idx];
                int data_idx = w_tri_data_idx[tri_buf_idx];

                // determine pixel position
                uint pixel_x = (tile_x << CR_TILE_LOG2) + (pixel_in_tile & 7);
                uint pixel_y = (tile_y << CR_TILE_LOG2) + (pixel_in_tile >> 3);

                // depth test
                uint depth = 0;
                bool zkill = false;
                if ((p_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0)
                {
                    uint4 zdata = read_imageui(t_tri_data, data_idx * 4);
                    depth = zdata.x * pixel_x + zdata.y * pixel_y + zdata.z;
                    uint old_depth = w_tile_depth[pixel_in_tile];
                    if (depth >= old_depth)
                        zkill = true;
                    else if (old_depth == tile_z_max)
                        tile_z_upd = true; // we are replacing previous zmax => need to update
                }

                if (!zkill)
                {
                    // run fragment shader
                    fragment_shader_output_t fragment_shader_output;
                    run_fragment_shader(
                        0, render_mode_flags,
                        &fragment_shader_output,
                        tri_idx, data_idx, pixel_x, pixel_y, 0x11, &temp[16]
                        t_tri_data,
                        t_vertex_buffer
                        );

                    // run ROP
                    if (!fragment_shader_output.discard)
                    {
					    execute_ROP_single_sample(
                            p_render_mode_flags,
                            tri_idx, pixel_x, pixel_yY, fragment_shader_output.color, depth,
                            &w_tile_color[pixel_in_tile], &w_tile_depth[pixel_in_tile]
                        );
                    }
                }
            }

            // update counters
            frag_read = min(frag_read + 32, frag_write);
            tri_read += popcount(boundary_mask);
        }

        // Write tile back to the framebuffer.

        {
            int surf_x = (tile_x << (CR_TILE_LOG2 + 2)) + ((get_local_id(0) & (CR_TILE_SIZE - 1)) << 2);
            int surf_y = (tile_y << CR_TILE_LOG2) + (get_local_id(0) >> CR_TILE_LOG2);
            write_imageui(img_color_buffer, (int2){surf_x, surf_y}, w_tile_color[get_local_id(0)]);
            write_imageui(img_depth_buffer, (int2){surf_x, surf_y}, w_tile_depth[get_local_id(0)]);
            write_imageui(img_color_buffer, (int2){surf_x, surf_y + 4}, w_tile_color[get_local_id(0) + 32]);
            write_imageui(img_depth_buffer, (int2){surf_x, surf_y + 4}, w_tile_depth[get_local_id(0) + 32]);
        }
    }
}
