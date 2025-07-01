#ifndef SHADER

#ifdef __COMPILER_RELATIVE_PATH__
#include "shaders/color.cl"
#else
#include "glsc2/src/kernels/shaders/color.cl"
#endif

#endif


#ifdef __COMPILER_RELATIVE_PATH__
#include "blending.cl"
#include "common.cl"
#include "depth.cl"
#include "stencil.cl"
#include "shaders/common.cl"
#else
#include "glsc2/src/kernels/blending.cl"
#include "glsc2/src/kernels/common.cl"
#include "glsc2/src/kernels/depth.cl"
#include "glsc2/src/kernels/stencil.cl"
#include "glsc2/src/kernels/shaders/common.cl"
#endif


inline uint uint4_to_int(const uint4 data) {
    return data.x | (data.y << 8) | (data.z << 16) | (data.w << 24);
}

inline uint4 uint_to_uint4(const int data, int mode) {
    switch (mode) {
        case TEX_R8:
            return (uint4) { data, 0, 0, 1};
        case TEX_RG8:
            return (uint4) { data, data >> 8, 0 ,1};
        case TEX_RGB8:
            return (uint4) { data, data >> 8, data >> 16, 1};
        case TEX_RGBA8:
        case TEX_RGBA4:
        case TEX_RGB5_A1:
        case TEX_RGB565:
        default:
            return (uint4) { data & 0xFF, data >> 8 & 0xFF, data >> 16 & 0xFF, data >> 24 & 0xFF};
    }
}

inline uint read_tex_from_buffer(global const void* g_color_buffer, size_t position, uint tex_mode) {
    switch(tex_mode) {
        default:
        case TEX_R8:
            return *((global const uchar*) g_color_buffer + position);
        case TEX_RG8:
            return *((global const ushort*) g_color_buffer + position);
        case TEX_RGB8:
            return *((global const uint*) (g_color_buffer + position*3)) & 0x00FFFFFF;
        case TEX_RGBA8:
            return *((global const uint*) g_color_buffer + position);
        case TEX_RGBA4:
            {
            uint tex = *((global const ushort*) g_color_buffer + position);
            return 
                (tex & 0x000F) << 0 |
                (tex & 0x00F0) << 4 |
                (tex & 0x0F00) << 8 |
                (tex & 0xF000) << 12;
            }
        case TEX_RGB5_A1:
            {
            uint tex = *((global const ushort*) g_color_buffer + position);
            return 
                (tex & 0x001F) << 0 |
                (tex & 0x03E0) << 5 |
                (tex & 0x7C00) << 10|
                (tex & 0x8000) << 15;
            }
        case TEX_RGB565:
            {
            uint tex = *((global const ushort*) g_color_buffer + position);
            return 
                (tex & 0x001F) << 0 |
                (tex & 0x07E0) << 5 |
                (tex & 0xF800) << 11;
            }
    }
}

inline void write_tex_to_buffer(global void* g_color_buffer, size_t position, int tex_mode, uint tex_data) {
    switch(tex_mode) {
        case TEX_R8:
            *((global uchar*) g_color_buffer + position) = tex_data;
            break;
        case TEX_RG8:
            *((global ushort*) g_color_buffer + position) = tex_data;
            break;
        case TEX_RGB8:
            {
            global void* buffer = g_color_buffer + position*3;
            *((global ushort*) buffer)      = tex_data;
            *((global uchar*)  buffer + 2)  = tex_data >> 16;
            }
            break;
        case TEX_RGBA8:
            *(((global uint*) g_color_buffer) + position) = tex_data;
            break;
        case TEX_RGBA4:
            *((global ushort*) g_color_buffer + position) = 
                (tex_data & 0x0000000F) >> 0 |
                (tex_data & 0x00000F00) >> 4 |
                (tex_data & 0x000F0000) >> 8 |
                (tex_data & 0x0F000000) >> 12;
            break;
        case TEX_RGB5_A1:
            *((global ushort*) g_color_buffer + position) = 
                (tex_data & 0x0000001F) >> 0 |
                (tex_data & 0x00001F00) >> 3 |
                (tex_data & 0x001F0000) >> 6 |
                (tex_data & 0x01000000) >> 9 ;
            break;
        case TEX_RGB565:
            *((global ushort*) g_color_buffer + position) = 
                (tex_data & 0x0000001F) >> 0 |
                (tex_data & 0x00003F00) >> 3 |
                (tex_data & 0x001F0000) >> 5 ;
            break;
    }
}

//----------------------------------------------
// Transpiler objects
//----------------------------------------------

/*
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
    float4 pos;
    float4 color;
} vertex_shader_output_t;

inline void fragment_shader(
    fragment_shader_input_t* input, 
    fragment_shader_output_t* output
) {
    output->gl_FragColor = (float4){input->color, 0.5f};
    output->discard = false;
    output->color = 
        ((int)(output->gl_FragColor.x * 255) << 8*0) |
        ((int)(output->gl_FragColor.y * 255) << 8*1) |
        ((int)(output->gl_FragColor.z * 255) << 8*2) |
        ((int)(output->gl_FragColor.w * 255) << 8*3);
}
*/
typedef struct {
    bool needs_dst; 
} blend_shader_input_t;

typedef struct {
    bool write_color;
    uint color;
} blend_shader_output_t;

//------------------------------------------------------------------------
// Shader wrappers.
//------------------------------------------------------------------------


inline float3 compute_barys(
    const int3* wpleq, const int3* upleq, const int3* vpleq,
    int sample_x, int sample_y)
{
    float w = 1.0f / (float){wpleq->x * sample_x + wpleq->y * sample_y + wpleq->z};
    float u = w * (float){upleq->x * sample_x + upleq->y * sample_y + upleq->z};
    float v = w * (float){vpleq->x * sample_x + vpleq->y * sample_y + vpleq->z};
    return (float3){1.0f - u - v, u, v};
}

//------------------------------------------------------------------------

inline bool run_fragment_shader(
    KERNEL_FRAGMENT_INPUT

    fragment_shader_output_t* output,
    int data_idx, int pixel_x, int pixel_y,
    #ifdef CONF_FINE_IMAGE_ENABLED
    image1d_buffer_t t_tri_data,
    image1d_buffer_t vertex_buffer
    #else
    global const CRTriangleData* g_tri_data, 
    global const float4* vertex_buffer
    #endif
) {
    fragment_shader_input_t input;
    
    // Fetch primitive data.
    uint4 t1, t2, t3;
    #ifdef CONF_FINE_IMAGE_ENABLED
    t1 = read_imageui(t_tri_data, data_idx * 4 + 1); // wx, wy, wb, ux
    t2 = read_imageui(t_tri_data, data_idx * 4 + 2); // uy, ub, vx, vy
    t3 = read_imageui(t_tri_data, data_idx * 4 + 3); // vb, vi0, vi1, vi2
    #else
    t1 = ((global const uint4*) g_tri_data)[data_idx * 4 + 1];
    t2 = ((global const uint4*) g_tri_data)[data_idx * 4 + 2];
    t3 = ((global const uint4*) g_tri_data)[data_idx * 4 + 3];
    #endif

    // Pixel varying data.
    int3 wpleq = (int3){t1.x, t1.y, t1.z};
    int3 upleq = (int3){t1.w, t2.x, t2.y};
    int3 vpleq = (int3){t2.z, t2.w, t3.x};
    uint3 vert_idx = (uint3){t3.y, t3.z, t3.w};
    float3 bary = compute_barys(&wpleq, &upleq, &vpleq, (pixel_x * 2 + 1), (pixel_y * 2 + 1));

    return gl_fragment_shader(
        KERNEL_FRAGMENT_INPUT_NAMES
        output, vertex_buffer, vert_idx, bary 
        );
}

//------------------------------------------------------------------------

inline bool bs_needs_dst(uint c_render_mode_flags, uint c_blender_op) {
    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_BLENDER) != 0) {
        return true;
    }
    return false; 
}

inline void run_blend_shader(
    blend_shader_input_t* input, blend_shader_output_t* output,
    int tri_idx, int pixel_x, int pixel_y, int sampleIdx, uint src, uint dst,
    uint c_blending_color, uint c_blending_data, uint c_render_mode_flags)
{
    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_BLENDER) != 0) {
        output->color = blend(src, dst, c_blending_color, c_blending_data);
    } else {
        output->color = src;
    }
    output->write_color = true;
}

//------------------------------------------------------------------------

/*
inline bool depth_test(ushort z, ushort depth) { return z < depth; }
inline bool update_z_max(ushort depth, ushort tile_z) { return depth == tile_z; };
inline void init_tile_z_max(ushort* tile_z_max, bool* tile_z_upd, local volatile ushort* w_tile_depth)
{
    *tile_z_max = CR_DEPTH_MAX >> 16;
    *tile_z_upd = (min(w_tile_depth[get_local_id(0)], w_tile_depth[get_local_id(0) + 32]) < *tile_z_max);
}
#ifdef CONF_FINE_SUB_GROUP_ENABLED
inline void sub_group_update_tile_z_max(uint c_render_mode_flags, ushort* tile_z_max, bool* tile_z_upd, local volatile ushort* w_tile_depth)
{
    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0 && sub_group_any(*tile_z_upd))
    {
        ushort z = max(w_tile_depth[get_local_id(0)], w_tile_depth[get_local_id(0) + 32]);
        *tile_z_max = sub_group_reduce_max_ui(z);
        *tile_z_upd = false;
    }
}
#endif
inline void local_1dim_update_tile_z_max(uint c_render_mode_flags, ushort* tile_z_max, bool* tile_z_upd, local volatile ushort* w_tile_depth, local volatile uint* l_temp)
{
    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0) {
        ushort z = max(w_tile_depth[get_local_id(0)], w_tile_depth[get_local_id(0) + 32]);
        *tile_z_max = local_reduce_max_1dim_ui(z, l_temp);
        *tile_z_upd = false;
    }
}
*/

//------------------------------------------------------------------------

inline void get_triangle(
    int* tri_idx, int* data_idx, uint4* tri_header, int* segment, 
    global const CRTriangleHeader* g_tri_header,
    global const int* g_tile_seg_data,
    global const int* g_tile_seg_next,
    global const int* g_tile_seg_count
    #ifdef CONF_FINE_IMAGE_ENABLED
    , image1d_buffer_t t_tri_header
    #endif
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
        #ifdef CONF_FINE_IMAGE_ENABLED
        *tri_header = read_imageui(t_tri_header, *data_idx);
        #else
        *tri_header = ((global const uint4*) g_tri_header)[*data_idx];
        #endif
    }

    // advance to next segment
    *segment = g_tile_seg_next[*segment];
}

//------------------------------------------------------------------------


inline bool early_z_cull(uint render_mode_flags, uint4 tri_header, ushort tile_z_min, ushort tile_z_max, uint c_depth_data)
{
    if ((render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0)
    {
        ushort z = (tri_header.w & 0xFFFFF000) >> 16;
        return !range_depth_test(z, tile_z_min, tile_z_max, c_depth_data);
    }
    return false;
}
/*
inline bool early_z_cull(uint render_mode_flags, uint4 tri_header, ushort tile_z_min, ushort tile_z_max, uint c_depth_data)
{
    if ((render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0)
    {
        ushort zmin = (tri_header.w & 0xFFFFF000) >> 16;
        if (!depth_test(zmin, tile_z_max, c_depth_data))
            return true;
    }
    return false;
}*/

//------------------------------------------------------------------------

inline ulong triangle_pixel_coverage(const int samples_log_2, const uint4 tri_header, int tile_x, int tile_y, local volatile ulong* s_cover8x8_lut, 
    int c_viewport_width, int c_viewport_height)
{
    int base_x = (tile_x << (CR_TILE_LOG2 + CR_SUBPIXEL_LOG2)) - ((c_viewport_width  - 1) << (CR_SUBPIXEL_LOG2 - 1));
    int base_y = (tile_y << (CR_TILE_LOG2 + CR_SUBPIXEL_LOG2)) - ((c_viewport_height - 1) << (CR_SUBPIXEL_LOG2 - 1));

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

// template <class BlendShaderClass>
inline uint determine_ROP_lane_mask(uint c_render_mode_flags) //, local volatile uint* warp_temp) mask of lanes that should process an earlier fragment than this lane
{
    bool reverse_lanes = true;
    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) == 0)
    {
        if (!bs_needs_dst(c_render_mode_flags, 0))
            reverse_lanes = false;
    }

    // TODO: Analyze this behaviour 
    // uint mask = (reverse_lanes) ? (1u << get_local_id(0)) : ~0u;
    // do
    // {
    //     *warp_temp = get_local_id(0);
    //     mask ^= 1u << *warp_temp;
    // }
    // while (*warp_temp != get_local_id(0));
    // return mask;

    return reverse_lanes ? getLaneMaskLt() : ~getLaneMaskLe();
}

inline int find_bit(uint render_mode_flags, ulong mask, int idx)
{
    uint x = ugetLo(mask);
    int  pop = popcount(x);
    bool p   = (pop <= idx);
    if (p) x = ugetHi(mask);
    if (p) idx -= pop;
    int bit = p ? 32 : 0;

    pop = popcount(x & 0x0000ffffu);
    p   = (pop <= idx);
    if (p) x >>= 16;
    if (p) bit += 16;
    if (p) idx -= pop;

    uint tmp = x & 0x000000ffu;
    pop = popcount(tmp);
    p   = (pop <= idx);
    if (p) tmp = x & 0x0000ff00u;
    if (p) idx -= pop;

    return findLeadingOne(tmp) + bit - idx;
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



inline void execute_ROP_single_sample(
    uint render_mode_flags,
    int tri_idx, int pixel_x, int pixel_y,
    fragment_shader_output_t* fs_out, ushort depth, 
    local volatile uint* ptr_color, local volatile ushort* ptr_depth, local volatile uchar* ptr_stencil,
    local volatile uint* w_temp,
    uint c_blending_color, uint c_blending_data,
    uint c_depth_data,
    uint c_render_mode_flags,
    uint c_stencil_data
)
{
    blend_shader_input_t blend_shader_input;
    blend_shader_output_t blend_shader_output;

    uint color = float4_to_uint(&fs_out->gl_FragColor);
    bool discarded = false;
    // blend_shader_input.needs_dst = false;

    #ifdef CONF_FINE_SUB_GROUP_RAW_ENABLED

    // if (fs_out->discard) return;

    // TODO: Optimize, ordering on triangle is not always required.  
    do
    {
        *w_temp = 0;
        atomic_or(w_temp,1 << get_local_id(0));

        if ((*w_temp & getLaneMaskLt()) == 0) {
            
            // stencil test
            if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_STENCIL) != 0) {
                if (!stencil_test(*ptr_stencil, c_stencil_data)) {
                    stencil_operation(ptr_stencil, c_stencil_data, get_stencil_operation_sfail(c_stencil_data));
                    return;
                }
            }

            // depth test
            if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0) {
                discarded = !depth_test(depth, *ptr_depth, c_depth_data);
                if (discarded) {
                    stencil_operation(ptr_stencil, c_stencil_data, get_stencil_operation_dfail(c_stencil_data));
                    return;
                }
                *ptr_depth = depth;
                stencil_operation(ptr_stencil, c_stencil_data, get_stencil_operation_dpass(c_stencil_data));
            }

            // blending
            run_blend_shader(&blend_shader_input, &blend_shader_output, tri_idx, pixel_x, pixel_y, 0, color, *ptr_color,
                c_blending_color, c_blending_data, c_render_mode_flags);
            if (blend_shader_output.write_color)
                *ptr_color = blend_shader_output.color;
            
            break;
        }
    }
    while (true);

    /*
    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0)
    {
        // TODO: maybe a more OpenCL way to communicate
        do
        {
            rounds++;
            *ptr_depth = depth;
			uint s_color = *ptr_color;
			run_blend_shader(&blend_shader_input, &blend_shader_output, tri_idx, pixel_x, pixel_y, 0, color, s_color,
                c_blending_color, c_blending_data, c_render_mode_flags);
            if (blend_shader_output.write_color)
                *ptr_color = blend_shader_output.color;
        }
        while (depth_test(depth, *ptr_depth, c_depth_data));
    }
    else if (!blend_shader_input.needs_dst)
    {
        rounds++;
        run_blend_shader(&blend_shader_input, &blend_shader_output, tri_idx, pixel_x, pixel_y, 0, color, 0,
        c_blending_color, c_blending_data, c_render_mode_flags);
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
			run_blend_shader(&blend_shader_input, &blend_shader_output, tri_idx, pixel_x, pixel_y, 0, color, s_color,
            c_blending_color, c_blending_data, c_render_mode_flags);
            if (blend_shader_output.write_color)
                *ptr_color = blend_shader_output.color;
        }
        while (*ptr_depth != get_local_id(0));
    }
    */
    #endif
}

//------------------------------------------------------------------------

kernel void fine_raster_single_sample(
    KERNEL_FRAGMENT_INPUT

    global int* a_fine_counter,
    global const int* a_num_active_tiles,
    global const int* a_num_bin_segs,
    global const int* a_num_subtris,
    global const int* a_num_tile_segs,
    
    global const int* g_active_tiles,
    global const int* g_tile_first_seg,
    global const int* g_tile_seg_count,
    global const int* g_tile_seg_data,
    global const int* g_tile_seg_next,
    global const CRTriangleHeader* g_tri_header,

    #ifdef CONF_FINE_IMAGE_ENABLED
    read_write image2d_t t_color_buffer,
    read_write image2d_t t_depth_buffer,
    read_write image2d_t t_stencil_buffer,
    read_only image1d_buffer_t t_tri_data,
    read_only image1d_buffer_t t_tri_header,
    read_only image1d_buffer_t t_vertex_buffer,
    #else
    global void* g_color_buffer,
    global ushort* g_depth_buffer,
    global uchar* g_stencil_buffer,
    global const CRTriangleData* g_tri_data,
    global const float4* g_vertex_buffer,
    #endif

    const uint   c_blending_color,
    const uint   c_blending_data,
    const uint   c_clear_color,
    const ushort c_clear_depth,
    const uchar  c_clear_stencil,
    const uint   c_color_buffer_mode,
    const int    c_deferred_clear,
    const uint   c_depth_data,
    const int    c_max_bin_segs,
    const int    c_max_subtris,
    const int    c_max_tile_segs,
    const uint   c_render_mode_flags,
    const uint   c_stencil_data,
    const int    c_viewport_height,
    const int    c_viewport_width,
    const int    c_width_tiles

)
{
                                                                            // for 20 warps:
    local volatile ulong    s_cover8x8_lut      [CR_COVER8X8_LUT_SIZE];           // 6KB
    local volatile uint     s_tile_color        [CONF_FINE_SUB_GROUPS][CR_TILE_SQR]; // 5KB
    local volatile ushort   s_tile_depth        [CONF_FINE_SUB_GROUPS][CR_TILE_SQR]; // 2.5KB
    local volatile uchar    s_tile_stencil      [CONF_FINE_SUB_GROUPS][CR_TILE_SQR]; // 1.25KB
    local volatile uint     s_triangle_idx      [CONF_FINE_SUB_GROUPS][64];          // 5KB  original triangle index
    local volatile uint     s_tri_data_idx      [CONF_FINE_SUB_GROUPS][64];          // 5KB  CRTriangleData index
    local volatile ulong    s_triangle_cov      [CONF_FINE_SUB_GROUPS][64];          // 10KB coverage mask
    local volatile uint     s_triangle_frag     [CONF_FINE_SUB_GROUPS][64];          // 5KB  fragment index
    local volatile uint     s_temp              [CONF_FINE_SUB_GROUPS*80];          // 6.25KB
                                                                            // = 47.25KB total
    // Warp Space
    local volatile uint*    w_tile_color        = (local volatile uint*)    &s_tile_color[get_local_id(1)];
    local volatile ushort*  w_tile_depth        = (local volatile ushort*)  &s_tile_depth[get_local_id(1)];
    local volatile uchar*   w_tile_stencil      = (local volatile uchar*)   &s_tile_stencil[get_local_id(1)];
    local volatile uint*    w_triangle_idx      = (local volatile uint*)    &s_triangle_idx[get_local_id(1)];
    local volatile uint*    w_tri_data_idx      = (local volatile uint*)    &s_tri_data_idx[get_local_id(1)];
    local volatile ulong*   w_triangle_cov      = (local volatile ulong*)   &s_triangle_cov[get_local_id(1)];
    local volatile uint*    w_triangle_frag     = (local volatile uint*)    &s_triangle_frag[get_local_id(1)];
    local volatile uint*    w_temp              = (local volatile uint*)    &s_temp[get_local_id(1)*80];

    if (*a_num_subtris > c_max_subtris || *a_num_bin_segs > c_max_bin_segs || *a_num_tile_segs > c_max_tile_segs)
        return;

    uint rop_lane_mask = determine_ROP_lane_mask(c_render_mode_flags); //, &w_temp[0]); TODO: solved??, erase local mem
    w_temp[get_local_id(1)] = 0; // first 16 elements of temp are always zero, TODO: carefull with size, depends on thread size y
    cover8x8_setupLUT(s_cover8x8_lut);
    barrier(CLK_LOCAL_MEM_FENCE);

    // loop over tiles
    for (;;)
    {
        // each warp pick a tile
        int active_idx;
        if (get_local_id(0) == 0)
            active_idx = atomic_add(a_fine_counter, 1);

        #ifdef CONF_FINE_SUB_GROUP_ENABLED
        {
            active_idx = sub_group_broadcast_ui(active_idx, 0);
        }
        #else
        {
            if (get_local_id(0) == 0) 
                w_temp[16] = active_idx;
            barrier(CLK_LOCAL_MEM_FENCE);
            active_idx = w_temp[16];
        }
        #endif

        bool is_not_active = active_idx >= *a_num_active_tiles;
        #ifdef CONF_FINE_SUB_GROUP_ENABLED
        {
            if (is_not_active)
                break;
        }
        #else
        {
            if (local_reduce_and_ui(is_not_active ? 1 : 0, s_temp))
                break;
        }
        #endif

        int tile_idx, segment, tile_y, tile_x;
        #ifndef CONF_FINE_SUB_GROUP_ENABLED
        if (!is_not_active)
        #endif
        {
            tile_idx = g_active_tiles[active_idx];
            segment = g_tile_first_seg[tile_idx];
            tile_y = idiv_fast(tile_idx, c_width_tiles);
            tile_x = tile_idx - tile_y * c_width_tiles;
        }

        // initialize per-tile state
        int tri_read = 0, tri_write = 0;
        int frag_read = 0, frag_write = 0;
        w_triangle_frag[63] = 0; // "previous triangle"

        #ifndef CONF_FINE_SUB_GROUP_ENABLED
        if (!is_not_active)
        #endif
        {
            // deferred clear => clear tile
            if (c_deferred_clear)
            {
                w_tile_color[get_local_id(0)] = c_clear_color;
                w_tile_depth[get_local_id(0)] = c_clear_depth;
                w_tile_color[get_local_id(0) + 32] = c_clear_color;
                w_tile_depth[get_local_id(0) + 32] = c_clear_depth;
                w_tile_stencil[get_local_id(0)] = c_clear_stencil;
                w_tile_stencil[get_local_id(0) + 32] = c_clear_stencil;
            }

            // otherwise => read tile from framebuffer
            else
            {
                int surf_x = (tile_x << (CR_TILE_LOG2 + 2)) + ((get_local_id(0) & (CR_TILE_SIZE - 1)) << 2);
                int surf_y = (tile_y << CR_TILE_LOG2) + (get_local_id(0) >> CR_TILE_LOG2);
                #ifdef CONF_FINE_IMAGE_ENABLED
                w_tile_color[get_local_id(0)]       = uint4_to_int(read_imageui(t_color_buffer,(int2){surf_x/4,surf_y}));
                w_tile_color[get_local_id(0) + 32]  = uint4_to_int(read_imageui(t_color_buffer,(int2){surf_x/4,surf_y+4}));
                w_tile_depth[get_local_id(0)]       = read_imageui(t_depth_buffer,(int2){surf_x/4,surf_y}).x;
                w_tile_depth[get_local_id(0) + 32]  = read_imageui(t_depth_buffer,(int2){surf_x/4,surf_y+4}).x;
                w_tile_stencil[get_local_id(0)]       = read_imageui(t_stencil_buffer,(int2){surf_x/4,surf_y}).x;
                w_tile_stencil[get_local_id(0) + 32]  = read_imageui(t_stencil_buffer,(int2){surf_x/4,surf_y+4}).x;
                #else
                w_tile_color[get_local_id(0)]       = read_tex_from_buffer(g_color_buffer, surf_x/4 + surf_y*c_viewport_width, c_color_buffer_mode);
                w_tile_color[get_local_id(0) + 32]  = read_tex_from_buffer(g_color_buffer, surf_x/4 + (surf_y+4)*c_viewport_width, c_color_buffer_mode);
                w_tile_depth[get_local_id(0)]       = g_depth_buffer[surf_x/4 + surf_y*c_viewport_width];
                w_tile_depth[get_local_id(0) + 32]  = g_depth_buffer[surf_x/4 + (surf_y + 4)*c_viewport_width];
                w_tile_stencil[get_local_id(0)]       = g_stencil_buffer[surf_x/4 + surf_y*c_viewport_width];
                w_tile_stencil[get_local_id(0) + 32]  = g_stencil_buffer[surf_x/4 + (surf_y + 4)*c_viewport_width];
                #endif
            }
        }

        // bound tile z
        ushort tile_z_max, tile_z_min;
        bool tile_z_upd_max, tile_z_upd_min;
        init_tile_z_max(&tile_z_max, &tile_z_upd_max, w_tile_depth);
        init_tile_z_min(&tile_z_min, &tile_z_upd_min, w_tile_depth);
        
        // process fragments
        for(;;)
        {
            // need to queue more fragments?
            bool need_fragments = frag_write - frag_read < 32 && segment >= 0;

            bool local_need_fragments = 0;
            #ifndef CONF_FINE_SUB_GROUP_ENABLED
            {
                need_fragments = need_fragments && !is_not_active;
                local_need_fragments = local_reduce_or_ui(need_fragments, s_temp);
            }
            #endif
            
            // This is done to allow 1dim comunication between local group
            if (need_fragments || local_need_fragments)
            {
                // update tile z

                #ifdef CONF_FINE_SUB_GROUP_ENABLED
                sub_group_update_tile_z_max(c_render_mode_flags, &tile_z_max, &tile_z_upd_max, w_tile_depth);
                sub_group_update_tile_z_min(c_render_mode_flags, &tile_z_min, &tile_z_upd_min, w_tile_depth);
                #else
                local_1dim_update_tile_z_max(c_render_mode_flags, &tile_z_max, &tile_z_upd_max, w_tile_depth, s_temp);
                local_1dim_update_tile_z_min(c_render_mode_flags, &tile_z_min, &tile_z_upd_min, w_tile_depth, s_temp);
                #endif
                
                // read triangles
                do
                {
                    // read triangle index and data, advance to next segment
                    int tri_idx, data_idx;
                    uint4 tri_header;
                    #ifndef CONF_FINE_SUB_GROUP_ENABLED
                    if (need_fragments)
                    #endif
                    {
                        get_triangle(&tri_idx, &data_idx, &tri_header, &segment, 
                            g_tri_header, g_tile_seg_data, g_tile_seg_next, g_tile_seg_count
                            #ifdef CONF_FINE_IMAGE_ENABLED
                            , t_tri_header
                            #endif
                            );

                        // early z cull
                        if (tri_idx >= 0 && early_z_cull(c_render_mode_flags, tri_header, tile_z_min, tile_z_max, c_depth_data))
                            tri_idx = -1;
                    }

                    // determine coverage
                    ulong coverage;
                    int pop;
                    #ifndef CONF_FINE_SUB_GROUP_ENABLED
                    if (need_fragments)
                    #endif
                    {
                        coverage = triangle_pixel_coverage(0, tri_header, tile_x, tile_y, s_cover8x8_lut, c_viewport_width, c_viewport_height);
                        pop = (tri_idx == -1) ? 0 : num_fragments(c_render_mode_flags, coverage);
                    }

                    // fragment count scan
                    uint frag;
                    #ifdef CONF_FINE_SUB_GROUP_ENABLED
                    {
                        frag = sub_group_scan_inclusive_add_ui(pop);
                        uint temp_frag = frag;
                        frag += frag_write; // frag now holds cumulative fragment count
                        frag_write += sub_group_broadcast_ui(temp_frag, get_local_size(0) - 1);
                    }
                    #else
                    {
                        frag = local_scan_inclusive_add_1dim_ui(pop, s_temp);
                        barrier(CLK_LOCAL_MEM_FENCE);
                        frag += frag_write; // frag now holds cumulative fragment count
                        size_t sub_group_id = get_local_linear_id() / get_local_size(0);
                        size_t last_sub_group_member = (sub_group_id+1) * get_local_size(0) - 1;
                        if (need_fragments)
                            frag_write += *((local volatile uint*)s_temp + last_sub_group_member);
                    }
                    #endif

                    // queue non-empty triangles
                    uint good_mask;
                    #ifdef CONF_FINE_SUB_GROUP_ENABLED
                    good_mask = sub_group_ballot(pop != 0);
                    #else
                    good_mask = local_reduce_or_1dim_ui((pop != 0) << get_local_id(0), s_temp);
                    #endif

                    #ifndef CONF_FINE_SUB_GROUP_ENABLED
                    if (need_fragments)
                    #endif
                    {
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

                    need_fragments = frag_write - frag_read < 32 && segment >= 0;
                    #ifndef CONF_FINE_SUB_GROUP_ENABLED
                    need_fragments = need_fragments && !is_not_active;
                    local_need_fragments = local_reduce_or_ui(need_fragments, s_temp);
                    #endif
                }
                while (need_fragments || local_need_fragments);
            }

            // end of segment?
            bool end_of_segment = frag_read == frag_write;
            bool local_end_of_segment = 1;
            #ifndef CONF_FINE_SUB_GROUP_ENABLED
            end_of_segment = end_of_segment || is_not_active;
            local_end_of_segment = local_reduce_and_ui(end_of_segment ? 1 : 0, s_temp);
            #endif
            if (end_of_segment && local_end_of_segment)
                break;
            
            // tag triangle boundaries
            #ifndef CONF_FINE_SUB_GROUP_ENABLED
            if (!end_of_segment)
            #endif
            {
                w_temp[get_local_id(0) + 16] = 0;
                if (tri_read + get_local_id(0) < tri_write)
                {
                    int idx = w_triangle_frag[(tri_read + get_local_id(0)) & 63] - frag_read;
                    if (idx <= 32)
                        w_temp[idx + 16 - 1] = 1;
                }
            }
            
            //sub_group_barrier();

            int rop_lane_idx = popcount(rop_lane_mask);
            uint boundary_mask;
            #ifdef CONF_FINE_SUB_GROUP_ENABLED
            // TODO subgroup barrier w_temp
            boundary_mask = sub_group_ballot(w_temp[rop_lane_idx + 16]);
            #else
            barrier(CLK_LOCAL_MEM_FENCE);
            boundary_mask = local_reduce_or_1dim_ui((w_temp[rop_lane_idx + 16] ? 1 : 0) << get_local_id(0), s_temp);
            #endif
            // distribute fragments
            
            #ifndef CONF_FINE_SUB_GROUP_ENABLED
            if (!end_of_segment)
            #endif
            if (rop_lane_idx < frag_write - frag_read)
            {
                int tri_buf_idx = (tri_read + popcount(boundary_mask & rop_lane_mask)) & 63;
                int frag_idx = add_sub(frag_read, rop_lane_idx, w_triangle_frag[(tri_buf_idx - 1) & 63]);
                ulong coverage = w_triangle_cov[tri_buf_idx];
                int pixel_in_tile = find_fragment(c_render_mode_flags, coverage, frag_idx);
                int tri_idx = w_triangle_idx[tri_buf_idx];
                int data_idx = w_tri_data_idx[tri_buf_idx];

                // determine pixel position
                uint pixel_x = (tile_x << CR_TILE_LOG2) + (pixel_in_tile & 7);
                uint pixel_y = (tile_y << CR_TILE_LOG2) + (pixel_in_tile >> 3);

                // PRE ROP tests
                // TODO: Optimize

                // stencil test
                uchar stencil;
                bool skill = false;
                if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_STENCIL) != 0) {
                    uchar old_stencil = w_tile_stencil[pixel_in_tile];
                    if(!stencil_test(old_stencil, c_stencil_data)) {
                        skill = true;
                    }
                }

                // depth test
                ushort depth;
                bool zkill = false;
                if (!skill && (c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0)
                {
                    uint4 zdata;
                    #ifdef CONF_FINE_IMAGE_ENABLED
                    zdata = read_imageui(t_tri_data, data_idx * 4);
                    #else
                    zdata = *((uint4*) &g_tri_data[data_idx]);
                    #endif
                    depth = (zdata.x * pixel_x + zdata.y * pixel_y + zdata.z) >> 16;
                    ushort old_depth = w_tile_depth[pixel_in_tile];
                    if (!depth_test(depth, old_depth, c_depth_data))
                        zkill = true;
                    else {
                        // TODO: Checkout this
                        tile_z_upd_max = update_tile_z(old_depth, tile_z_max, c_depth_data); // we are replacing previous zmax => need to update
                        tile_z_upd_min = update_tile_z(old_depth, tile_z_min, c_depth_data); // we are replacing previous zmax => need to update
                    }
                }
                
                // TODO: This pipeline does not support discarding from fragment shader
                fragment_shader_output_t fragment_shader_output;
                bool fragment_pass = true;
                // fragment_shader_output.discard = false;

                if (!skill && !zkill)
                {
                    // run fragment shader
                    fragment_pass = run_fragment_shader(
                        KERNEL_FRAGMENT_INPUT_NAMES

                        &fragment_shader_output,
                        data_idx, pixel_x, pixel_y,
                        #ifdef CONF_FINE_IMAGE_ENABLED
                        t_tri_data,
                        t_vertex_buffer
                        #else
                        g_tri_data,
                        g_vertex_buffer
                        #endif
                        // 0, c_render_mode_flags
                        );
                }

                // run ROP
                if (fragment_pass)
                {
                    execute_ROP_single_sample(
                        c_render_mode_flags,
                        tri_idx, pixel_x, pixel_y, &fragment_shader_output, depth,
                        &w_tile_color[pixel_in_tile], &w_tile_depth[pixel_in_tile], &w_tile_stencil[pixel_in_tile],
                        &w_temp[pixel_in_tile],
                        c_blending_color, c_blending_data,
                        c_depth_data,
                        c_render_mode_flags,
                        c_stencil_data
                    );
                }

            }
            
            // update counters
            frag_read = min(frag_read + 32, frag_write);
            tri_read += popcount(boundary_mask);
        }

        // Write tile back to the framebuffer.
        #ifndef CONF_FINE_SUB_GROUP_ENABLED
        if (!is_not_active)
        #endif
        {
            int surf_x = (tile_x << (CR_TILE_LOG2 + 2)) + ((get_local_id(0) & (CR_TILE_SIZE - 1)) << 2);
            int surf_y = (tile_y << CR_TILE_LOG2) + (get_local_id(0) >> CR_TILE_LOG2);
            #ifdef CONF_FINE_IMAGE_ENABLED
            write_imageui(t_color_buffer, (int2){surf_x/4, surf_y}, uint_to_uint4(w_tile_color[get_local_id(0)], TEX_RGBA8));
            write_imageui(t_color_buffer, (int2){surf_x/4, surf_y + 4}, uint_to_uint4(w_tile_color[get_local_id(0) + 32], TEX_RGBA8));
            write_imageui(t_depth_buffer, (int2){surf_x/4, surf_y}, w_tile_depth[get_local_id(0)]);
            write_imageui(t_depth_buffer, (int2){surf_x/4, surf_y + 4}, w_tile_depth[get_local_id(0) + 32]);
            write_imageui(t_stencil_buffer, (int2){surf_x/4, surf_y}, w_tile_stencil[get_local_id(0)]);
            write_imageui(t_stencil_buffer, (int2){surf_x/4, surf_y + 4}, w_tile_stencil[get_local_id(0) + 32]);
            #else
            write_tex_to_buffer(g_color_buffer, surf_x/4 + surf_y*c_viewport_width, c_color_buffer_mode, w_tile_color[get_local_id(0)]);
            write_tex_to_buffer(g_color_buffer, surf_x/4 + (surf_y+4)*c_viewport_width, c_color_buffer_mode, w_tile_color[get_local_id(0) + 32]);
            g_depth_buffer[surf_x/4 + surf_y*c_viewport_width] = w_tile_depth[get_local_id(0)];
            g_depth_buffer[surf_x/4 + (surf_y+4)*c_viewport_width] = w_tile_depth[get_local_id(0) + 32];
            g_stencil_buffer[surf_x/4 + surf_y*c_viewport_width] = w_tile_stencil[get_local_id(0)];
            g_stencil_buffer[surf_x/4 + (surf_y+4)*c_viewport_width] = w_tile_stencil[get_local_id(0) + 32];
            #endif
        }
        
    }
}
