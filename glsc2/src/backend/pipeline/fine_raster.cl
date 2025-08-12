#ifdef __COMPILER_RELATIVE_PATH__
    #include <backend/types.cl>
    #include <backend/utils/sub_group_mask.cl>
    #include <backend/utils/sync.cl>
    #include <backend/pipeline/gl/blending.cl>
    #include <backend/pipeline/gl/depth.cl>
    #include <backend/pipeline/gl/stencil.cl>
    #include <backend/pipeline/gl/clear.cl>
#else
    #include "glsc2/src/backend/types.cl"
    #include "glsc2/src/backend/utils/sub_group_mask.cl"
    #include "glsc2/src/backend/utils/sync.cl"
    #include "glsc2/src/backend/pipeline/gl/blending.cl"
    #include "glsc2/src/backend/pipeline/gl/depth.cl"
    #include "glsc2/src/backend/pipeline/gl/stencil.cl"
    #include "glsc2/src/backend/pipeline/gl/clear.cl"
#endif

#ifndef SHADER
    #ifdef __COMPILER_RELATIVE_PATH__
        #include <backend/pipeline/gl/base.cl>
    #else
        #include "glsc2/src/backend/pipeline/gl/base.cl"
    #endif
#endif

//-----------------------
// Framebuffer operations
//-----------------------

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



//------------------------------------------------------------------------
// Fragment Shader Interface Wrapper.
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
    FS_KERNEL_PARAMS

    fragment_shader_output_t* output,
    int data_idx, int pixel_x, int pixel_y,
    #ifdef DEVICE_IMAGE_ENABLED
    image1d_buffer_t t_tri_data,
    #else
    global const triangle_data_t* g_tri_data, 
    #endif
    ro_vertex_buffer_t vertex_buffer
) {
    // Fetch primitive data.
    uint4 t1, t2, t3;
    #ifdef DEVICE_IMAGE_ENABLED
    {
        t1 = read_imageui(t_tri_data, data_idx * 4 + 1); // wx, wy, wb, ux
        t2 = read_imageui(t_tri_data, data_idx * 4 + 2); // uy, ub, vx, vy
        t3 = read_imageui(t_tri_data, data_idx * 4 + 3); // vb, vi0, vi1, vi2
    }
    #else
    {
        t1 = ((global const uint4*) g_tri_data)[data_idx * 4 + 1];
        t2 = ((global const uint4*) g_tri_data)[data_idx * 4 + 2];
        t3 = ((global const uint4*) g_tri_data)[data_idx * 4 + 3];
    }
    #endif

    // Pixel varying data.
    int3 wpleq = (int3){t1.x, t1.y, t1.z};
    int3 upleq = (int3){t1.w, t2.x, t2.y};
    int3 vpleq = (int3){t2.z, t2.w, t3.x};
    uint3 vert_idx = (uint3){t3.y, t3.z, t3.w};
    float3 bary = compute_barys(&wpleq, &upleq, &vpleq, (pixel_x * 2 + 1), (pixel_y * 2 + 1));

    return gl_fragment_shader(
        FS_KERNEL_ARGS
        output, vertex_buffer, vert_idx, bary 
        );
}

//----------------------------------------------
// Blending Shader
//----------------------------------------------


typedef struct {
    bool needs_dst; 
} blend_shader_input_t;

typedef struct {
    bool write_color;
    uint color;
} blend_shader_output_t;

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

inline void get_triangle(
    int* tri_idx, int* data_idx, uint4* tri_header, int* segment, 
    global const triangle_header_t* g_tri_header,
    global const int* g_tile_seg_data,
    global const int* g_tile_seg_next,
    global const int* g_tile_seg_count
    #ifdef DEVICE_IMAGE_ENABLED
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
        #ifdef DEVICE_IMAGE_ENABLED
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
    if ((render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0 && (render_mode_flags & RENDER_MODE_FLAG_ENABLE_STENCIL) != 0)
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
inline sub_group_mask_t determine_ROP_lane_mask(uint c_render_mode_flags) //, local volatile uint* warp_temp) mask of lanes that should process an earlier fragment than this lane
{
    bool reverse_lanes = true;
    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) == 0)
    {
        if (!bs_needs_dst(c_render_mode_flags, 0))
            reverse_lanes = false;
    }

    return reverse_lanes ? get_lane_sub_group_mask_lt() : not_sub_group_mask(get_lane_sub_group_mask_le());
}

inline int find_bit(ulong mask, int idx)
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

inline int find_fragment(ulong coverage, int frag_idx)
{
    return find_bit(coverage, frag_idx);
}

//------------------------------------------------------------------------
// Single-sample implementation.
//------------------------------------------------------------------------

#if (DEVICE_SUB_GROUP_RAW == 1)
inline void execute_ROP_single_sample(
    uint render_mode_flags,
    int tri_idx, int pixel_x, int pixel_y,
    fragment_shader_output_t* restrict fs_out, ushort depth, 
    local volatile uint* restrict ptr_color, 
    local volatile ushort* restrict ptr_depth, 
    local volatile uchar* restrict ptr_stencil,
    local volatile sub_group_mask_t* restrict ptr_temp,
    uint c_blending_color, uint c_blending_data,
    uint c_depth_data,
    uint c_render_mode_flags,
    uint c_stencil_data
)
{
    blend_shader_input_t blend_shader_input;
    blend_shader_output_t blend_shader_output;

    uint color = float4_to_uint(&fs_out->gl_FragColor);

    // per-fragment enabled operations
    bool stencil_test_enabled = (c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_STENCIL) != 0;
    bool depth_test_enabled = (c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0;

    // blend_shader_input.needs_dst = false;

    sub_group_mask_t sub_group_mask;
    clear_sub_group_mask(&sub_group_mask);
    set_bit_sub_group_mask(&sub_group_mask, get_sub_group_local_id());

    sub_group_mask_t lt_mask = get_lane_sub_group_mask_lt();
    // TODO: Optimize, ordering on primitive is not always required.
    do
    {
        clear_sub_group_mask(ptr_temp);
        atomic_or_sub_group_mask(ptr_temp, sub_group_mask);

        // check if this lane is processing an earlier fragment
        if (!any_sub_group_mask(and_sub_group_mask(*ptr_temp, lt_mask))) {
            
            // stencil test
            uchar stencil = *ptr_stencil;
            
            if (stencil_test_enabled) {
                if (!stencil_test(stencil, c_stencil_data)) {
                    stencil_operation(ptr_stencil, c_stencil_data, get_stencil_operation_sfail(c_stencil_data));
                    return;
                }
            }

            // depth test
            if (depth_test_enabled) {
                if (!depth_test(depth, *ptr_depth, c_depth_data)) {
                    if (stencil_test_enabled)
                        stencil_operation(ptr_stencil, c_stencil_data, get_stencil_operation_dfail(c_stencil_data));
                    return;
                }
                *ptr_depth = depth;
            }

            // stencil op post depth test
            if (stencil_test_enabled) {
                stencil_operation(ptr_stencil, c_stencil_data, get_stencil_operation_dpass(c_stencil_data));
            }
            
            // blending
            run_blend_shader(&blend_shader_input, &blend_shader_output, tri_idx, pixel_x, pixel_y, 0, color, *ptr_color,
                c_blending_color, c_blending_data, c_render_mode_flags);
            if (blend_shader_output.write_color)
                *ptr_color = blend_shader_output.color;
            
            return;
        }
    }
    while (true);
    
}
#endif // DEVICE_SUB_GROUP_RAW_ENABLED

//------------------------------------------------------------------------

#ifndef FS_KERNEL_ARGS
#define FS_KERNEL_ARGS
#endif // FS_KERNEL_ARGS

#ifndef FS_KERNEL_PARAMS
#define FS_KERNEL_PARAMS
#endif // FS_KERNEL_PARAMS

/**
 * Single-sample rasterization kernel.
 *
 * If device sub-group is enabled, multiple tiles are processed in parallel.
 * Otherwise, only one tile is processed per work group.
 */
kernel 
/*
#ifdef DEVICE_SUB_GROUP_ENABLED
#ifndef DEVICE_SUB_GROUP_RAW_ENABLED
    #error "Sub-group raw mode must be enabled to use sub-group."
#endif
__attribute__((reqd_work_group_size(DEVICE_SUB_GROUP_THREADS, DEVICE_FINE_SUB_GROUPS, 1)))
#else
    #error "Sub-group mode must be enabled to use fine rasterization."
__attribute__((reqd_work_group_size(DEVICE_FINE_THREADS*DEVICE_FINE_SUB_GROUPS, 1, 1)))
#endif
*/
void fine_raster_single_sample(
    FS_KERNEL_PARAMS

    global int* restrict a_fine_counter,
    global const int* restrict a_num_active_tiles,
    global const int* restrict a_num_bin_segs,
    global const int* restrict a_num_subtris,
    global const int* restrict a_num_tile_segs,
    
    global const int* restrict g_active_tiles,
    global const int* restrict g_tile_first_seg,
    global const int* restrict g_tile_seg_count,
    global const int* restrict g_tile_seg_data,
    global const int* restrict g_tile_seg_next,
    global const triangle_header_t* restrict g_tri_header,

    #ifdef DEVICE_IMAGE_ENABLED
    read_write image2d_t t_color_buffer,
    read_write image2d_t t_depth_buffer,
    read_write image2d_t t_stencil_buffer,
    read_only image1d_buffer_t t_tri_data,
    read_only image1d_buffer_t t_tri_header,
    #else
    global void* restrict g_color_buffer,
    global ushort* restrict g_depth_buffer,
    global uchar* restrict g_stencil_buffer,
    global const triangle_data_t* restrict g_tri_data,
    #endif
    ro_vertex_buffer_t vertex_buffer,

    const uint   c_blending_color,
    const uint   c_blending_data,
    const ulong  c_clear_write_values, 
    const ushort c_clear_enabled_data,
    const ushort c_enabled_data,
    const uint   c_color_buffer_mode,
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
    local volatile uint     s_tile_color        [DEVICE_FINE_SUB_GROUPS][CR_TILE_SQR]; // 5KB
    local volatile ushort   s_tile_depth        [DEVICE_FINE_SUB_GROUPS][CR_TILE_SQR]; // 2.5KB
    local volatile uchar    s_tile_stencil      [DEVICE_FINE_SUB_GROUPS][CR_TILE_SQR]; // 1.25KB
    local volatile uint     s_triangle_idx      [DEVICE_FINE_SUB_GROUPS][64];          // 5KB  original triangle index
    local volatile uint     s_tri_data_idx      [DEVICE_FINE_SUB_GROUPS][64];          // 5KB  CRTriangleData index
    local volatile ulong    s_triangle_cov      [DEVICE_FINE_SUB_GROUPS][64];          // 10KB coverage mask
    local volatile uint     s_triangle_frag     [DEVICE_FINE_SUB_GROUPS][64];          // 5KB  fragment index
    local volatile uint     s_temp              [DEVICE_FINE_SUB_GROUPS*64];          // 6.25KB
                                                                            // = 47.25KB total
    // Warp Space
    local volatile uint*    w_tile_color        = (local volatile uint*)    &s_tile_color[get_local_id(1)];
    local volatile ushort*  w_tile_depth        = (local volatile ushort*)  &s_tile_depth[get_local_id(1)];
    local volatile uchar*   w_tile_stencil      = (local volatile uchar*)   &s_tile_stencil[get_local_id(1)];
    local volatile uint*    w_triangle_idx      = (local volatile uint*)    &s_triangle_idx[get_local_id(1)];
    local volatile uint*    w_tri_data_idx      = (local volatile uint*)    &s_tri_data_idx[get_local_id(1)];
    local volatile ulong*   w_triangle_cov      = (local volatile ulong*)   &s_triangle_cov[get_local_id(1)];
    local volatile uint*    w_triangle_frag     = (local volatile uint*)    &s_triangle_frag[get_local_id(1)];
    local volatile uint*    w_temp              = (local volatile uint*)    &s_temp[get_local_id(1)*64];

    local volatile uint l_temp[DEVICE_FINE_SUB_GROUPS][DEVICE_SUB_GROUP_THREADS];
    local volatile uint (*sg_temp)[DEVICE_SUB_GROUP_THREADS] = &l_temp[get_sub_group_id()];

    if (*a_num_subtris > c_max_subtris || *a_num_bin_segs > c_max_bin_segs || *a_num_tile_segs > c_max_tile_segs)
        return;

    sub_group_mask_t rop_lane_mask = determine_ROP_lane_mask(c_render_mode_flags); //, &w_temp[0]); TODO: solved??, erase local mem
    w_temp[get_local_id(1)] = 0; // first 16 elements of temp are always zero, TODO: carefull with size, depends on thread size y
    cover8x8_setupLUT(s_cover8x8_lut);
    barrier(CLK_LOCAL_MEM_FENCE);

    // common logic
    bool colorbuffer_clear = 
        (c_clear_enabled_data & CLEAR_ENABLED_COLOR_CHANNEL_MASK) != 0;
    bool depthbuffer_clear = 
        (c_clear_enabled_data & CLEAR_ENABLED_DEPTH_CHANNEL) != 0;
    bool stencilbuffer_clear = 
        (c_clear_enabled_data & CLEAR_ENABLED_STENCIL_CHANNEL_MASK) != 0;

    bool colorbuffer_full_clear = 
        (c_clear_enabled_data & CLEAR_ENABLED_COLOR_CHANNEL_MASK) == CLEAR_ENABLED_COLOR_CHANNEL_MASK;
    bool stencilbuffer_full_clear = 
        (c_clear_enabled_data & CLEAR_ENABLED_STENCIL_CHANNEL_MASK) == CLEAR_ENABLED_STENCIL_CHANNEL_MASK;

    bool depth_test_enabled = (c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0;
    bool stencil_test_enabled = (c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_STENCIL) != 0;

    // loop over tiles
    for (;;)
    {
        // each warp pick a tile
        int active_idx;
        if (get_local_id(0) == 0)
            active_idx = atomic_add(a_fine_counter, 1);

        #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
        {
            active_idx = sub_group_broadcast(active_idx, 0);
        }
        #else
        {
            (*sg_temp)[get_sub_group_local_id()] = active_idx;
            #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
                barrier(CLK_LOCAL_MEM_FENCE);
            #endif
            active_idx = (*sg_temp)[0];
        }
        #endif

        if (active_idx >= *a_num_active_tiles)
            break;

        int tile_idx = g_active_tiles[active_idx];
        int segment = g_tile_first_seg[tile_idx];
        int tile_y = idiv_fast(tile_idx, c_width_tiles);
        int tile_x = tile_idx - tile_y * c_width_tiles;

        // initialize per-tile state
        int tri_read = 0, tri_write = 0;
        int frag_read = 0, frag_write = 0;
        w_triangle_frag[63] = 0; // "previous triangle"

        {
            // Copy and clear if required framebuffers
            // TODO: Scissor test and dithering.
            bool colorbuffer_needs_load = !colorbuffer_full_clear;
            bool depthbuffer_needs_load = depth_test_enabled && !depthbuffer_clear;
            bool stencilbuffer_needs_load = stencil_test_enabled && !stencilbuffer_full_clear;

            #pragma unroll
            for (int pixel = get_local_id(0); pixel < CR_TILE_SQR; pixel += get_local_size(0)) {
                uint color;
                ushort depth;
                uchar stencil;
                
                int surf_x = (tile_x << CR_TILE_LOG2) + (pixel & (CR_TILE_SIZE - 1));
                int surf_y = (tile_y << CR_TILE_LOG2) + (pixel >> CR_TILE_LOG2);

                if (surf_x >= c_viewport_width || surf_y >= c_viewport_height) continue;
                
                if (colorbuffer_needs_load) {
                    #ifdef DEVICE_IMAGE_ENABLED
                        color = uint4_to_int(read_imageui(t_color_buffer,(int2){surf_x,surf_y}));
                    #else
                        color = read_tex_from_buffer(g_color_buffer, surf_x + surf_y*c_viewport_width, c_color_buffer_mode);
                    #endif
                }
                color = clear_color(color, c_clear_write_values, c_clear_enabled_data);

                if (depthbuffer_needs_load) {
                    #ifdef DEVICE_IMAGE_ENABLED
                        depth = read_imageui(t_depth_buffer,(int2){surf_x,surf_y}).x;
                    #else
                        depth = g_depth_buffer[surf_x + surf_y*c_viewport_width];
                    #endif
                } else {
                    depth = clear_depth(c_clear_write_values, c_clear_enabled_data);
                }

                if (stencilbuffer_needs_load) {
                    #ifdef DEVICE_IMAGE_ENABLED
                        stencil = read_imageui(t_stencil_buffer,(int2){surf_x,surf_y}).x;
                    #else
                        stencil = g_stencil_buffer[surf_x + surf_y*c_viewport_width];
                    #endif
                }
                stencil = clear_stencil(stencil, c_clear_write_values, c_clear_enabled_data);

                w_tile_color[pixel] = color; 
                w_tile_depth[pixel] = depth;
                w_tile_stencil[pixel] = stencil;
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
            
            if (frag_write - frag_read < get_sub_group_size() && segment >= 0)
            {
                // update tile z

                // TODO: Optimize for non ordering req primitives.
                // #ifdef DEVICE_SUB_GROUP_ENABLED
                // sub_group_update_tile_z_max(c_render_mode_flags, &tile_z_max, &tile_z_upd_max, w_tile_depth);
                // sub_group_update_tile_z_min(c_render_mode_flags, &tile_z_min, &tile_z_upd_min, w_tile_depth);
                // #else
                // local_1dim_update_tile_z_max(c_render_mode_flags, &tile_z_max, &tile_z_upd_max, w_tile_depth, s_temp);
                // local_1dim_update_tile_z_min(c_render_mode_flags, &tile_z_min, &tile_z_upd_min, w_tile_depth, s_temp);
                // #endif
                
                // read triangles
                do
                {
                    // read triangle index and data, advance to next segment
                    int tri_idx, data_idx;
                    uint4 tri_header;
     
                    {
                        get_triangle(&tri_idx, &data_idx, &tri_header, &segment, 
                            g_tri_header, g_tile_seg_data, g_tile_seg_next, g_tile_seg_count
                            #ifdef DEVICE_IMAGE_ENABLED
                            , t_tri_header
                            #endif
                            );

                        // early z cull
                        // if stencil is enabled or fragment can be discarded
                        //if (tri_idx >= 0 && early_z_cull(c_render_mode_flags, tri_header, tile_z_min, tile_z_max, c_depth_data))
                        //    tri_idx = -1;
                        
                    }

                    // determine coverage
                    ulong coverage = triangle_pixel_coverage(0, tri_header, tile_x, tile_y, s_cover8x8_lut, c_viewport_width, c_viewport_height);
                    int pop = (tri_idx == -1) ? 0 : popcount(coverage);

                    // fragment count scan
                    uint frag = local_1dim_scan_inclusive_add(pop, *l_temp);
                    uint tmp_frag = frag;
                    frag += frag_write; // frag now holds cumulative fragment count

                    #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
                    {
                        frag_write += sub_group_broadcast(tmp_frag, get_sub_group_size() - 1);
                    }
                    #else
                    {
                        #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
                            barrier(CLK_LOCAL_MEM_FENCE);
                        #endif
                        frag_write += (*sg_temp)[get_sub_group_size()-1];
                    }
                    #endif

                    // queue non-empty triangles
                    sub_group_mask_t good_mask = local_1dim_ballot(pop != 0, (local volatile sub_group_mask_t*) l_temp);

                    if (pop != 0)
                    {
                        sub_group_mask_t lt_mask = get_lane_sub_group_mask_lt();
                        int idx = popcount_sub_group_mask(and_sub_group_mask(good_mask, lt_mask));
                        idx = (tri_write + idx) & 63; // wrap index
                        w_triangle_idx  [idx] = tri_idx;
                        w_tri_data_idx  [idx] = data_idx;
                        w_triangle_frag [idx] = frag;
                        w_triangle_cov  [idx] = coverage;
                    }
                    tri_write += popcount_sub_group_mask(good_mask);

                }
                while (frag_write - frag_read < get_sub_group_size() && segment >= 0);
            }
            
            // end of segment?
            if (frag_read == frag_write)
                break;
            
            
            // tag triangle boundaries
            (*sg_temp)[get_sub_group_local_id()] = 0;
            if (tri_read + get_sub_group_local_id() < tri_write)
            {
                int idx = w_triangle_frag[(tri_read + get_sub_group_local_id()) & 63] - frag_read;
                if (idx <= get_sub_group_size())
                    (*sg_temp)[idx - 1] = 1;
            }
            #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
                barrier(CLK_LOCAL_MEM_FENCE);
            #endif
            
            int rop_lane_idx = popcount_sub_group_mask(rop_lane_mask);
            sub_group_mask_t boundary_mask = local_1dim_ballot((*sg_temp)[rop_lane_idx], (local volatile sub_group_mask_t*) l_temp);
            
            // distribute fragments
            if (rop_lane_idx < frag_write - frag_read)
            {
                int tri_buf_idx = (tri_read + popcount_sub_group_mask(and_sub_group_mask(boundary_mask,rop_lane_mask))) & 63;
                int frag_idx = add_sub(frag_read, rop_lane_idx, w_triangle_frag[(tri_buf_idx - 1) & 63]);
                ulong coverage = w_triangle_cov[tri_buf_idx];
                int pixel_in_tile = find_fragment(coverage, frag_idx);
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
                
                // depth test
                ushort depth;
                bool zkill = false;
                
                
                if (!skill && (c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0)
                {
                
                    uint4 zdata;
                    #ifdef DEVICE_IMAGE_ENABLED
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

                fragment_shader_output_t fragment_shader_output;
                bool fragment_pass = true;
                // fragment_shader_output.discard = false;

                if (!skill && !zkill)
                {
                    // run fragment shader
                    
                    fragment_pass = run_fragment_shader(
                        FS_KERNEL_ARGS

                        &fragment_shader_output,
                        data_idx, pixel_x, pixel_y,
                        #ifdef DEVICE_IMAGE_ENABLED
                        t_tri_data,
                        #else
                        g_tri_data,
                        #endif
                        vertex_buffer
                        );
                        
                }

                // run ROP
                if (fragment_pass)
                {
                    
                    execute_ROP_single_sample(
                        c_render_mode_flags,
                        tri_idx, pixel_x, pixel_y, &fragment_shader_output, depth,
                        &w_tile_color[pixel_in_tile], &w_tile_depth[pixel_in_tile], &w_tile_stencil[pixel_in_tile],
                        (local volatile sub_group_mask_t*) &w_temp[pixel_in_tile],
                        c_blending_color, c_blending_data,
                        c_depth_data,
                        c_render_mode_flags,
                        c_stencil_data
                    );
                    
                }
                

            }
            // update counters
            frag_read = min(frag_read + 32, frag_write);
            tri_read += popcount_sub_group_mask(boundary_mask);
        }

        // Write tile back to the framebuffer.
        {
            bool colorbuffer_full_masked = (c_enabled_data & ENABLED_COLOR_CHANNEL_MASK) == 0;
            bool depthbuffer_masked = (c_enabled_data & ENABLED_DEPTH_CHANNEL) == 0;
            bool stencilbuffer_full_masked = (c_enabled_data & ENABLED_STENCIL_CHANNEL_MASK) == 0;

            bool colorbuffer_needs_store = colorbuffer_clear || !colorbuffer_full_masked;
            bool depthbuffer_needs_store = depthbuffer_clear || (depth_test_enabled && !depthbuffer_masked);
            bool stencilbuffer_needs_store = stencilbuffer_clear || (stencil_test_enabled && !stencilbuffer_full_masked);
            
            #pragma unroll
            for (int pixel = get_local_id(0); pixel < CR_TILE_SQR; pixel += get_local_size(0)) {
                
                int surf_x = (tile_x << CR_TILE_LOG2) + (pixel & (CR_TILE_SIZE - 1));
                int surf_y = (tile_y << CR_TILE_LOG2) + (pixel >> CR_TILE_LOG2);

                if (surf_x >= c_viewport_width || surf_y >= c_viewport_height) continue;

                if (colorbuffer_needs_store) {
                    #ifdef DEVICE_IMAGE_ENABLED
                        write_imageui(t_color_buffer, (int2){surf_x, surf_y}, uint_to_uint4(w_tile_color[pixel], c_color_buffer_mode));
                    #else
                        write_tex_to_buffer(g_color_buffer, surf_x + surf_y*c_viewport_width, c_color_buffer_mode, (uint)w_tile_color[pixel]); //); // (uint)w_tile_stencil[pixel]*255); //
                    #endif
                }

                if (depthbuffer_needs_store) {
                    #ifdef DEVICE_IMAGE_ENABLED
                        write_imageui(t_depth_buffer, (int2){surf_x, surf_y}, w_tile_depth[pixel]);
                    #else
                        g_depth_buffer[surf_x + surf_y*c_viewport_width] = w_tile_depth[pixel];
                    #endif
                }

                if (stencilbuffer_needs_store) {
                    #ifdef DEVICE_IMAGE_ENABLED
                        write_imageui(t_stencil_buffer, (int2){surf_x, surf_y}, w_tile_stencil[pixel]);
                    #else
                        g_stencil_buffer[surf_x + surf_y*c_viewport_width] = w_tile_stencil[pixel];
                    #endif
                }
                

            }
            
        }
    }

}
