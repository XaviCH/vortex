//#include "glsc2/src/kernels/common/headers.cl"
#include "common/headers.cl"

// Render flags
// #define RENDER_MODE_FLAG_ENABLE_QUADS   (1 << 0)
// #define RENDER_MODE_FLAG_ENABLE_DEPTH   (1 << 1)
// #define RENDER_MODE_FLAG_ENABLE_LERP    (1 << 2)
// #define RENDER_MODE_FLAG_ENABLE_BLENDER    (1 << 3)

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
    float4 pos;
    float4 color;
} vertex_shader_output_t;

inline void fragment_shader(
    fragment_shader_input_t* input, 
    fragment_shader_output_t* output
) {
    output->gl_FragColor = (float4){input->color, 1.f};
    output->discard = false;
    output->color = 
        ((int)(output->gl_FragColor.x * 255) << 8*0) |
        ((int)(output->gl_FragColor.y * 255) << 8*1) |
        ((int)(output->gl_FragColor.z * 255) << 8*2) |
        ((int)(output->gl_FragColor.w * 255) << 8*3);
}

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

inline float4 get_varying_at_vertex(
    int varying_idx, int vert_idx, 
    image1d_buffer_t t_vertex_buffer
) {
    return read_imagef(t_vertex_buffer, vert_idx * (sizeof(vertex_shader_output_t) / sizeof(float4)) + varying_idx + 1);
}

inline float4 interpolate_varying(
    int varying_idx, const uint3 vert_idx, const float3 bary, 
    image1d_buffer_t t_vertex_buffer
) {
    float4 v0 = get_varying_at_vertex(varying_idx, vert_idx.x, t_vertex_buffer);
    float4 v1 = get_varying_at_vertex(varying_idx, vert_idx.y, t_vertex_buffer);
    float4 v2 = get_varying_at_vertex(varying_idx, vert_idx.z, t_vertex_buffer);
    return v0 * bary.x + v1 * bary.y + v2 * bary.z; 
}

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

inline void run_fragment_shader(
    fragment_shader_output_t* output,
    int tri_idx, int data_idx, int pixel_x, int pixel_y, uint centroid, local volatile uint* shared,
    image1d_buffer_t t_tri_data,
    image1d_buffer_t t_vertex_buffer
) {
    fragment_shader_input_t input;
    
    // Fetch primitive data.
    uint4 t1 = read_imageui(t_tri_data, data_idx * 4 + 1); // wx, wy, wb, ux
    uint4 t2 = read_imageui(t_tri_data, data_idx * 4 + 2); // uy, ub, vx, vy
    uint4 t3 = read_imageui(t_tri_data, data_idx * 4 + 3); // vb, vi0, vi1, vi2

    // Pixel varying data.
    int3 wpleq = (int3){t1.x, t1.y, t1.z};
    int3 upleq = (int3){t1.w, t2.x, t2.y};
    int3 vpleq = (int3){t2.z, t2.w, t3.x};
    uint3 vert_idx = (uint3){t3.y, t3.z, t3.w};
    float3 bary = compute_barys(&wpleq, &upleq, &vpleq, (pixel_x * 2 + 1), (pixel_y * 2 + 1));

    // Transpiler dependant
    input.color = interpolate_varying(0, vert_idx, bary, t_vertex_buffer).xyz;
    // input.color = (float3) {1,1,1};
    fragment_shader(&input, output);
}

//------------------------------------------------------------------------

inline bool bs_needs_dst(uint c_render_mode_flags, uint c_blender_op) {
    if (c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_BLENDER) {
        return c_blender_op;
    }
    return false; 
}

inline void run_blend_shader(
    blend_shader_input_t* input, blend_shader_output_t* output,
    int tri_idx, int pixel_x, int pixel_y, int sampleIdx, uint src, uint dst)
{
    output->write_color = true;
    output->color = src;
}

//------------------------------------------------------------------------

inline void init_tile_z_max(uint* tile_z_max, bool* tile_z_upd, local volatile uint* w_tile_depth)
{
    *tile_z_max = CR_DEPTH_MAX;
    *tile_z_upd = (min(w_tile_depth[get_local_id(0)], w_tile_depth[get_local_id(0) + 32]) < *tile_z_max);
}

inline void update_tile_z_max(uint c_render_mode_flags, uint* tile_z_max, bool* tile_z_upd, local volatile uint* w_tile_depth) // , local volatile uint* temp)
{
    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0 && sub_group_any(*tile_z_upd))
    {
        uint z = max(w_tile_depth[get_local_id(0)], w_tile_depth[get_local_id(0) + 32]);
        // temp[get_local_id(0) + 16] = z;
        // z = max(z, temp[get_local_id(0) + 16 -  1]); temp[get_local_id(0) + 16] = z;
        // z = max(z, temp[get_local_id(0) + 16 -  2]); temp[get_local_id(0) + 16] = z;
        // z = max(z, temp[get_local_id(0) + 16 -  4]); temp[get_local_id(0) + 16] = z;
        // z = max(z, temp[get_local_id(0) + 16 -  8]); temp[get_local_id(0) + 16] = z;
        // z = max(z, temp[get_local_id(0) + 16 - 16]); temp[get_local_id(0) + 16] = z;
        // *tile_z_max = temp[47];
        *tile_z_max = sub_group_reduce_max_ui(z);
        *tile_z_upd = false;
    }
}

//------------------------------------------------------------------------

inline void get_triangle(
    int* tri_idx, int* data_idx, uint4* tri_header, int* segment, 
    global const CRTriangleHeader* g_tri_header,
    global const int* g_tile_seg_data,
    global const int* g_tile_seg_next,
    global const int* g_tile_seg_count,
    image1d_buffer_t t_tri_header
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
        *tri_header = read_imageui(t_tri_header, *data_idx);
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



inline uint scan32_value(uint value, local volatile uint* temp)
{
    temp[get_local_id(0) + 16] = value;
    value += temp[get_local_id(0) + 16 -  1]; 
    temp[get_local_id(0) + 16] = value;
    value += temp[get_local_id(0) + 16 -  2]; 
    temp[get_local_id(0) + 16] = value;
    value += temp[get_local_id(0) + 16 -  4]; 
    temp[get_local_id(0) + 16] = value;
    value += temp[get_local_id(0) + 16 -  8]; 
    temp[get_local_id(0) + 16] = value;
    value += temp[get_local_id(0) + 16 - 16]; 
    temp[get_local_id(0) + 16] = value;
    return value;
}

inline uint scan32_total(local volatile uint* temp)
{
    return temp[47];
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
    uint color, uint depth, local volatile uint* ptr_color, local volatile uint* ptr_depth)
{
    blend_shader_input_t blend_shader_input;
    blend_shader_output_t blend_shader_output;
    blend_shader_input.needs_dst = false;
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

    // #ifdef __IMAGE_SUPPORT__
    read_write image2d_t t_color_buffer,
    read_write image2d_t t_depth_buffer,
    read_only image1d_buffer_t t_tri_data,
    read_only image1d_buffer_t t_tri_header,
    read_only image1d_buffer_t t_vertex_buffer,

    const uint   c_clear_color,
    const uint   c_clear_depth,
    const int    c_deferred_clear,
    const int    c_max_bin_segs,
    const int    c_max_subtris,
    const int    c_max_tile_segs,
    const uint   c_render_mode_flags,
    const int    c_viewport_height,
    const int    c_viewport_width,
    const int    c_width_tiles
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
    local volatile uint*   w_tile_color         = (local volatile uint*) &s_tile_color[get_local_id(1)];
    local volatile uint*   w_tile_depth         = (local volatile uint*) &s_tile_depth[get_local_id(1)];
    local volatile uint*   w_triangle_idx       = (local volatile uint*) &s_triangle_idx[get_local_id(1)];
    local volatile uint*   w_tri_data_idx       = (local volatile uint*) &s_tri_data_idx[get_local_id(1)];
    local volatile ulong*  w_triangle_cov       = (local volatile ulong*) &s_triangle_cov[get_local_id(1)];
    local volatile uint*   w_triangle_frag      = (local volatile uint*) &s_triangle_frag[get_local_id(1)];
    local volatile uint*   w_temp               = (local volatile uint*) &s_temp[get_local_id(1)];

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

        active_idx = sub_group_broadcast_ui(active_idx, 0);
        // barrier(CLK_LOCAL_MEM_FENCE); // added for unexpected behaviour
        // sub_group_barrier();
        // int active_idx = w_temp[16];
        if (active_idx >= *a_num_active_tiles)
        {
            break;
        }

        int tile_idx = g_active_tiles[active_idx];
        int segment = g_tile_first_seg[tile_idx];
        int tile_y = idiv_fast(tile_idx, c_width_tiles);
        int tile_x = tile_idx - tile_y * c_width_tiles;

        // initialize per-tile state
        int tri_read = 0, tri_write = 0;
        int frag_read = 0, frag_write = 0;
        w_triangle_frag[63] = 0; // "previous triangle"

        // deferred clear => clear tile
        if (c_deferred_clear)
        {
			w_tile_color[get_local_id(0)] = c_clear_color;
            w_tile_depth[get_local_id(0)] = c_clear_depth;
            w_tile_color[get_local_id(0) + 32] = c_clear_color;
            w_tile_depth[get_local_id(0) + 32] = c_clear_depth;
        }

        // otherwise => read tile from framebuffer
        else
        {
            int surf_x = (tile_x << (CR_TILE_LOG2 + 2)) + ((get_local_id(0) & (CR_TILE_SIZE - 1)) << 2);
            int surf_y = (tile_y << CR_TILE_LOG2) + (get_local_id(0) >> CR_TILE_LOG2);
            // TODO now only one component is access
			w_tile_color[get_local_id(0)] = read_imageui(t_color_buffer,(int2){surf_x/4,surf_y})[surf_x%4];
            w_tile_depth[get_local_id(0)] = read_imageui(t_depth_buffer,(int2){surf_x/4,surf_y})[surf_x%4];
            w_tile_color[get_local_id(0) + 32] = read_imageui(t_color_buffer,(int2){surf_x/4,surf_y+4})[surf_x%4];
            w_tile_depth[get_local_id(0) + 32] = read_imageui(t_depth_buffer,(int2){surf_x/4,surf_y+4})[surf_x%4];
        }

        uint tile_z_max;
        bool tile_z_upd;
        init_tile_z_max(&tile_z_max, &tile_z_upd, w_tile_depth);
        //w_tile_color[get_local_id(0)] = 0xFFFFFFFFu;
        //w_tile_color[get_local_id(0) + 32] = 0xFFFFFFFFu;
        
        // process fragments
        for(;;)
        {
            // need to queue more fragments?
            
            if (frag_write - frag_read < 32 && segment >= 0)
            {
                // update tile z
                update_tile_z_max(c_render_mode_flags, &tile_z_max, &tile_z_upd, w_tile_depth); // , w_temp);
                
                // read triangles
                do
                {
                    // read triangle index and data, advance to next segment
                    int tri_idx, data_idx;
                    uint4 tri_header;
                    get_triangle(&tri_idx, &data_idx, &tri_header, &segment, 
                        g_tri_header, g_tile_seg_data, g_tile_seg_next, g_tile_seg_count, t_tri_header);

                    // early z cull
                    if (tri_idx >= 0 && early_z_cull(c_render_mode_flags, tri_header, tile_z_max))
                        tri_idx = -1;

                    // determine coverage
                    ulong coverage = triangle_pixel_coverage(0, tri_header, tile_x, tile_y, s_cover8x8_lut, c_viewport_width, c_viewport_height);
                    int pop = (tri_idx == -1) ? 0 : num_fragments(c_render_mode_flags, coverage);

                    // fragment count scan
                    uint frag = sub_group_scan_inclusive_add_ui(pop); // scan32_value(pop, w_temp);
                    uint temp_frag = frag;
                    frag += frag_write; // frag now holds cumulative fragment count
                    frag_write += sub_group_broadcast_ui(temp_frag, 31); // scan32_total(w_temp);

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
            w_temp[get_local_id(0) + 16] = 0;
            if (tri_read + get_local_id(0) < tri_write)
            {
                int idx = w_triangle_frag[(tri_read + get_local_id(0)) & 63] - frag_read;
                if (idx <= 32)
                    w_temp[idx + 16 - 1] = 1;
            }
            
            //sub_group_barrier();

            int rop_lane_idx = popcount(rop_lane_mask);
            uint boundary_mask = sub_group_ballot(w_temp[rop_lane_idx + 16]);
            // distribute fragments
            
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

                // depth test
                uint depth = 0;
                bool zkill = false;
                if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0)
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
                        &fragment_shader_output,
                        tri_idx, data_idx, pixel_x, pixel_y, 0x11, &w_temp[16],
                        t_tri_data,
                        t_vertex_buffer
                        // 0, c_render_mode_flags
                        );
                    
                    // run ROP
                    if (!fragment_shader_output.discard)
                    {
					    execute_ROP_single_sample(
                            c_render_mode_flags,
                            tri_idx, pixel_x, pixel_y, fragment_shader_output.color, depth,
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
            uint4 tile_color_0 = {
                (w_tile_color[get_local_id(0)] >> 8*0) & 0xFF,
                (w_tile_color[get_local_id(0)] >> 8*1) & 0xFF,
                (w_tile_color[get_local_id(0)] >> 8*2) & 0xFF,
                (w_tile_color[get_local_id(0)] >> 8*3) & 0xFF,
            };
            uint4 tile_color_1 = {
                (w_tile_color[get_local_id(0) + 32] >> 8*0) & 0xFF,
                (w_tile_color[get_local_id(0) + 32] >> 8*1) & 0xFF,
                (w_tile_color[get_local_id(0) + 32] >> 8*2) & 0xFF,
                (w_tile_color[get_local_id(0) + 32] >> 8*3) & 0xFF,
            };
            write_imageui(t_color_buffer, (int2){surf_x/4, surf_y}, tile_color_0); // w_tile_color[get_local_id(0)]);
            write_imageui(t_depth_buffer, (int2){surf_x/4, surf_y}, w_tile_depth[get_local_id(0)]);
            write_imageui(t_color_buffer, (int2){surf_x/4, surf_y + 4}, tile_color_1); // w_tile_color[get_local_id(0) + 32]);
            write_imageui(t_depth_buffer, (int2){surf_x/4, surf_y + 4}, w_tile_depth[get_local_id(0) + 32]);
        }
        
    }
}
