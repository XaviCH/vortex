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

// Types

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

// Extensions

#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION __opencl_c_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_ballot : enable

// ISA dependancy

#ifdef CUDA
inline int      findLeadingOne      (uint v)                    { uint r; asm("bfind.u32 %0, %1;" : "=r"(r) : "r"(v)); return r; }
inline uint     getLaneMaskLt       (void)                      { uint r; asm("mov.u32 %0, %%lanemask_lt;" : "=r"(r)); return r; }
inline uint     getLaneMaskLe       (void)                      { uint r; asm("mov.u32 %0, %%lanemask_le;" : "=r"(r)); return r; }

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

inline void sort_shared(local uint* ptr, int num_items)
{
    int thread_local_id = get_local_id(0) + get_local_id(1) * get_local_size(0);
    int range = 16;

    // Use transposition sort within each 16-wide subrange.

    int base = thread_local_id * 2;
    if (base < num_items - 1)
    {
        bool try_odd = (base < num_items - 2 && (~base & (range - 2)) != 0);
        uint mid = ptr[base + 1];

        for (int iter = 0; iter < range; iter += 2)
        {
            // Evens.

            uint tmp = ptr[base + 0];
            if (tmp > mid)
            {
                ptr[base + 0] = mid;
                mid = tmp;
            }

            // Odds.

            if (try_odd)
            {
                tmp = ptr[base + 2];
                if (mid > tmp)
                {
                    ptr[base + 2] = mid;
                    mid = tmp;
                }
            }
        }
        ptr[base + 1] = mid;
    }

    // Multiple subranges => Merge hierarchically.

    for (; range < num_items; range <<= 1)
    {
        // Assuming that we would insert the current item into the other
        // subrange, use binary search to find the appropriate slot.

        barrier(CLK_LOCAL_MEM_FENCE);

        uint item;
        int slot;
        if (thread_local_id < num_items)
        {
            item = ptr[thread_local_id];
            slot = (thread_local_id & -range) ^ range;
            if (slot < num_items)
            {
                uint tmp = ptr[slot];
                bool inclusive = ((thread_local_id & range) != 0);
                if (tmp < item || (inclusive && tmp == item))
                {
                    for (int step = (range >> 1); step != 0; step >>= 1)
                    {
                        int probe = slot + step;
                        if (probe < num_items)
                        {
                            tmp = ptr[probe];
                            if (tmp < item || (inclusive && tmp == item))
                                slot = probe;
                        }
                    }
                    slot++;
                }
            }
        }

        // Store the item at an appropriate place.

        barrier(CLK_LOCAL_MEM_FENCE);

        if (thread_local_id < num_items)
            ptr[slot + (thread_local_id & (range * 2 - 1)) - range] = item;
    }
}


//---

inline int global_tile_idx(int tile_in_bin, int width_tiles)
{
    int tile_x = tile_in_bin & (CR_BIN_SIZE - 1);
    int tile_y = tile_in_bin >> CR_BIN_LOG2;
    return tile_x + tile_y * width_tiles;
}

//----------------------------------------------------------------------------------------

kernel void coarse_raster(
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
    global   void*  g_tri_data, // unused
    private  uint   p_tri_data_item_sz, // unused
    // Bin output / coarse input.
    private  int    p_max_bin_segs,
    global   int*   g_bin_first_seg,    
    global   int*   g_bin_total,
    global   int*   g_bin_seg_data,
    global   int*   g_bin_seg_next,
    global   int*   g_bin_seg_count,
    // Coarse output / fine input.
    private  int    p_max_tile_segs,
    global   int*   g_active_tiles,        // CR_MAXTILES_SQR * (S32 tileIdx)
    global   int*   g_tile_first_seg,       // CR_MAXTILES_SQR * (S32 seg_idx), -1 = none
    global   int*   g_g_tile_seg_data,        // p_max_tile_segs * CR_TILE_SEG_SIZE * (S32 tri_idx)
    global   int*   g_tile_seg_next,        // p_max_tile_segs * (S32 seg_idx), -1 = none
    global   int*   g_tile_seg_count,
    // Cuda support
    image1d_t g_t_tri_header,
    sampler_t p_s_tri_header
) {

    // Common.

    local volatile uint s_work_counter;
    local volatile uint* s_scan_temp          [CR_COARSE_WARPS][48];              // 3KB

    // Input.

    local volatile uint* s_bin_order           [CR_MAXBINS_SQR];                   // 1KB
    local volatile int* s_bin_stream_curr_seg  [CR_BIN_STREAMS_SIZE];              // 0KB
    local volatile int* s_bin_stream_first_tri [CR_BIN_STREAMS_SIZE];              // 0KB
    local volatile int* s_tri_queue            [CR_COARSE_QUEUE_SIZE];             // 4KB
    local volatile int s_tri_queue_write_pos;
    local volatile uint s_bin_stream_selected_ofs;
    local volatile uint s_bin_stream_selected_size;

    // Output.

    local volatile uint* s_warp_emit_mask      [CR_COARSE_WARPS][CR_BIN_SQR + 1];  // 16KB, +1 to avoid bank collisions
    local volatile uint* s_warp_emit_prefix_sum [CR_COARSE_WARPS][CR_BIN_SQR + 1];  // 16KB, +1 to avoid bank collisions
    local volatile uint* s_tile_emit_prefix_sum [CR_BIN_SQR + 1];                   // 1KB, zero at the beginning
    local volatile uint* s_tile_alloc_prefix_sum[CR_BIN_SQR + 1];                   // 1KB, zero at the beginning
    local volatile int* s_tile_stream_curr_ofs [CR_BIN_SQR];                       // 1KB
    local volatile uint s_first_alloc_seg;
    local volatile uint s_first_active_idx;

    // Private
    int tile_log     = CR_TILE_LOG2 + CR_SUBPIXEL_LOG2;
    int thread_local_id  = get_local_id(0) + get_local_id(1) * 32;
    int emit_shift   = CR_BIN_LOG2 * 2 + 5; // We scan ((num_emits << emit_shift) | num_allocs) over tiles.

    if (g_num_subtris > p_max_subtris || g_num_bin_segs > p_max_bin_segs)
        return;

    // Initialize sharedmem arrays.

    s_tile_emit_prefix_sum[0] = 0;
    s_tile_alloc_prefix_sum[0] = 0;
    s_scan_temp[get_local_id(1)][get_local_id(0)] = 0;

    // Sort bins in descending order of triangle count.

    for (int bin_idx = thread_local_id; bin_idx < p_num_bins; bin_idx += CR_COARSE_WARPS * 32)
    {
        int count = 0;
        for (int i = 0; i < CR_BIN_STREAMS_SIZE; i++)
        s_bin_order[bin_idx] = (~count << (CR_MAXBINS_LOG2 * 2)) | bin_idx;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    sort_shared(s_bin_order, p_num_bins);

    // Process each bin by one block.

    for (;;)
    {
        // Pick a bin for the block.

        if (thread_local_id == 0)
            s_work_counter = atomic_add(g_coarse_counter, 1);
        barrier(CLK_LOCAL_MEM_FENCE);

        int work_counter = s_work_counter;
        if (work_counter >= p_num_bins)
        {
            break;
        }

        uint bin_order = s_bin_order[work_counter];
        bool bin_empty = ((~bin_order >> (CR_MAXBINS_LOG2 * 2)) == 0);
        if (bin_empty && !p_deferred_clear)
        {
            break;
        }

        int bin_idx = bin_order & (CR_MAXBINS_SQR - 1);

        // Initialize input/output streams.

        int tri_queue_write_pos = 0;
        int tri_queue_read_pos = 0;

        if (thread_local_id < CR_BIN_STREAMS_SIZE)
        {
            int seg_idx = g_bin_first_seg[(bin_idx << CR_BIN_STREAMS_LOG2) + thread_local_id];
            s_bin_stream_curr_seg[thread_local_id] = seg_idx;
            s_bin_stream_first_tri[thread_local_id] = (seg_idx == -1) ? ~0u : g_bin_seg_data[seg_idx << CR_BIN_SEG_LOG2];
        }

        for (int tile_in_bin = CR_COARSE_WARPS * 32 - 1 - thread_local_id; tile_in_bin < CR_BIN_SQR; tile_in_bin += CR_COARSE_WARPS * 32)
            s_tile_stream_curr_ofs[tile_in_bin] = -CR_TILE_SEG_SIZE;

        // Initialize per-bin state.

        int bin_y = idiv_fast(bin_idx, p_width_bins);
        int bin_x = bin_idx - bin_y * p_width_bins;
        int origin_x = (bin_x << (CR_BIN_LOG2 + tile_log)) - (p_viewport_width << (CR_SUBPIXEL_LOG2 - 1));
        int origin_y = (bin_y << (CR_BIN_LOG2 + tile_log)) - (p_viewport_height << (CR_SUBPIXEL_LOG2 - 1));
        int max_tile_x_in_bin = min(p_width_tiles - (bin_x << CR_BIN_LOG2), CR_BIN_SIZE) - 1;
        int max_tile_y_in_bin = min(p_height_tiles - (bin_y << CR_BIN_LOG2), CR_BIN_SIZE) - 1;
        int bin_tile_idx = (bin_x + bin_y * p_width_tiles) << CR_BIN_LOG2;


        // Entire block: Merge input streams and process triangles.

        if (!bin_empty)
        do
        {
            //------------------------------------------------------------------------
            // Merge.
            //------------------------------------------------------------------------

            // Entire block: Not enough triangles => merge and queue segments.
            // NOTE: The bin exit criterion assumes that we queue more triangles than we actually need.

            while (tri_queue_write_pos - tri_queue_read_pos <= CR_COARSE_WARPS * 32)
            {
                // First warp: Choose the segment with the lowest initial triangle index.

                if (thread_local_id < CR_BIN_STREAMS_SIZE)
                {
                    // Find the stream with the lowest triangle index.

                    uint first_tri = s_bin_stream_first_tri[thread_local_id];
                    uint t = first_tri;
                    local volatile uint* p = &s_scan_temp[0][thread_local_id + 16];

                    #if (CR_BIN_STREAMS_SIZE > 1)
                        p[0] = t, t = min(t, p[-1]);
                    #endif
                    #if (CR_BIN_STREAMS_SIZE > 2)
                        p[0] = t, t = min(t, p[-2]);
                    #endif
                    #if (CR_BIN_STREAMS_SIZE > 4)
                        p[0] = t, t = min(t, p[-4]);
                    #endif
                    #if (CR_BIN_STREAMS_SIZE > 8)
                        p[0] = t, t = min(t, p[-8]);
                    #endif
                    #if (CR_BIN_STREAMS_SIZE > 16)
                        p[0] = t, t = min(t, p[-16]);
                    #endif
                    p[0] = t;

                    // Consume and broadcast.

                    if (s_scan_temp[0][CR_BIN_STREAMS_SIZE - 1 + 16] == first_tri)
                    {
                        int seg_idx = s_bin_stream_curr_seg[thread_local_id];
                        s_bin_stream_selected_ofs = seg_idx << CR_BIN_SEG_LOG2;
                        if (seg_idx != -1)
                        {
                            int seg_size = g_bin_seg_count[seg_idx];
                            int seg_next = g_bin_seg_next[seg_idx];
                            s_bin_stream_selected_size = seg_size;
                            s_tri_queue_write_pos = tri_queue_write_pos + seg_size;
                            s_bin_stream_curr_seg[thread_local_id] = seg_next;
                            s_bin_stream_first_tri[thread_local_id] = (seg_next == -1) ? ~0u : g_bin_seg_data[seg_next << CR_BIN_SEG_LOG2];
                        }
                    }
                }

                // No more segments => break.

                barrier(CLK_LOCAL_MEM_FENCE);
                tri_queue_write_pos = s_tri_queue_write_pos;
                int seg_ofs = s_bin_stream_selected_ofs;
                if (seg_ofs < 0)
                    break;

                int seg_size = s_bin_stream_selected_size;
                barrier(CLK_LOCAL_MEM_FENCE);

                // Fetch triangles into the queue.

                for (int idx_in_seg = CR_COARSE_WARPS * 32 - 1 - thread_local_id; idx_in_seg < seg_size; idx_in_seg += CR_COARSE_WARPS * 32)
                {
                    int tri_idx = g_bin_seg_data[seg_ofs + idx_in_seg];
                    s_tri_queue[(tri_queue_write_pos - seg_size + idx_in_seg) & (CR_COARSE_QUEUE_SIZE - 1)] = tri_idx;
                }

            }

            // All threads: Clear emit masks.

            for (int mask_idx = thread_local_id; mask_idx < CR_COARSE_WARPS * CR_BIN_SQR; mask_idx += CR_COARSE_WARPS * 32)
                s_warp_emit_mask[mask_idx >> (CR_BIN_LOG2 * 2)][mask_idx & (CR_BIN_SQR - 1)] = 0;

            barrier(CLK_LOCAL_MEM_FENCE);

            //------------------------------------------------------------------------
            // Raster.
            //------------------------------------------------------------------------

            // Triangle per thread: Read from the queue.

            int tri_idx = -1;
            if (tri_queue_read_pos + thread_local_id < tri_queue_write_pos)
                tri_idx = s_tri_queue[(tri_queue_read_pos + thread_local_id) & (CR_COARSE_QUEUE_SIZE - 1)];

            uint4 tri_data = (uint4){0, 0, 0, 0};
            if (tri_idx != -1)
            {
                int data_idx = tri_idx >> 3;
                int subtri_idx = tri_idx & 7;
                if (subtri_idx != 7)
                    data_idx = g_tri_header[data_idx].misc + subtri_idx;
                tri_data = read_imageui(g_t_tri_header, p_s_tri_header, data_idx);
            }

            // 32 triangles per warp: Record emits (= tile intersections).

            if (sub_group_any(tri_idx != -1))
            {
                int v0x = sub_s16lo_s16lo(tri_data.x, origin_x);
                int v0y = sub_s16hi_s16lo(tri_data.x, origin_y);
                int d01x = sub_s16lo_s16lo(tri_data.y, tri_data.x);
                int d01y = sub_s16hi_s16hi(tri_data.y, tri_data.x);
                int d02x = sub_s16lo_s16lo(tri_data.z, tri_data.x);
                int d02y = sub_s16hi_s16hi(tri_data.z, tri_data.x);

                // Compute tile-based AABB.

                int lox = add_clamp_0_x((v0x + min_min(d01x, 0, d02x)) >> tile_log, 0, max_tile_x_in_bin);
                int loy = add_clamp_0_x((v0y + min_min(d01y, 0, d02y)) >> tile_log, 0, max_tile_y_in_bin);
                int hix = add_clamp_0_x((v0x + max_max(d01x, 0, d02x)) >> tile_log, 0, max_tile_x_in_bin);
                int hiy = add_clamp_0_x((v0y + max_max(d01y, 0, d02y)) >> tile_log, 0, max_tile_y_in_bin);
                int sizex = add_sub(hix, 1, lox);
                int sizey = add_sub(hiy, 1, loy);
                int area = sizex * sizey;

                // Miscellaneous init.

                local uchar* curr_ptr = (local uchar*)&s_warp_emit_mask[get_local_id(1)][lox + (loy << CR_BIN_LOG2)];
                int ptr_y_inc = CR_BIN_SIZE * 4 - (sizex << 2);
                uint mask_bit = 1 << get_local_id(0);

                // Case A: All AABBs are small => record the full AABB using atomics.

                if (sub_group_all(sizex <= 2 && sizey <= 2))
                {

                    if (tri_idx != -1)
                    {
                        atomic_or((local uint*)curr_ptr, mask_bit);
                        if (sizex == 2) atomic_or((local uint*)(curr_ptr + 4), mask_bit);
                        if (sizey == 2) atomic_or((local uint*)(curr_ptr + CR_BIN_SIZE * 4), mask_bit);
                        if (sizex == 2 && sizey == 2) atomic_or((local uint*)(curr_ptr + 4 + CR_BIN_SIZE * 4), mask_bit);
                    }

                }
                else
                {
                    // Compute warp-AABB (scan-32).

                    uint aabb_mask = add_sub(2 << hix, 0x20000 << hiy, 1 << lox) - (0x10000 << loy);
                    if (tri_idx == -1)
                        aabb_mask = 0;

                    local volatile uint* p = &s_scan_temp[get_local_id(1)][get_local_id(0) + 16];
                    p[0] = aabb_mask, aabb_mask |= p[-1];
                    p[0] = aabb_mask, aabb_mask |= p[-2];
                    p[0] = aabb_mask, aabb_mask |= p[-4];
                    p[0] = aabb_mask, aabb_mask |= p[-8];
                    p[0] = aabb_mask, aabb_mask |= p[-16];
                    p[0] = aabb_mask, aabb_mask = s_scan_temp[get_local_id(1)][47];

                    uint mask_x = aabb_mask & 0xFFFF;
                    uint mask_y = aabb_mask >> 16;
                    int wlox = findLeadingOne(mask_x ^ (mask_x - 1));
                    int wloy = findLeadingOne(mask_y ^ (mask_y - 1));
                    int whix = findLeadingOne(mask_x);
                    int whiy = findLeadingOne(mask_y);
                    int warea = (add_sub(whix, 1, wlox)) * (add_sub(whiy, 1, wloy));

                    // Initialize edge functions.

                    int d12x = d02x - d01x;
                    int d12y = d02y - d01y;
                    v0x -= lox << tile_log;
                    v0y -= loy << tile_log;

                    int t01 = v0x * d01y - v0y * d01x;
                    int t02 = v0y * d02x - v0x * d02y;
                    int t12 = d01x * d12y - d01y * d12x - t01 - t02;
                    int b01 = add_sub(t01 >> tile_log, max(d01x, 0), min(d01y, 0));
                    int b02 = add_sub(t02 >> tile_log, max(d02y, 0), min(d02x, 0));
                    int b12 = add_sub(t12 >> tile_log, max(d12x, 0), min(d12y, 0));

                    d01x += sizex * d01y;
                    d02x += sizex * d02y;
                    d12x += sizex * d12y;

                    // Case B: Warp-AABB is not much larger than largest AABB => Check tiles in warp-AABB, record using ballots.

                    if (sub_group_any(warea * 4 <= area * 8))
                    {
                        if (tri_idx != -1)
                        {
                            for (int y = wloy; y <= hiy; y++)
                            {
                                if (y < loy) continue;
                                for (int x = wlox; x <= hix; x++)
                                {
                                    if (x < lox) continue;
                                    *(local uint*)curr_ptr = sub_group_ballot(b01 >= 0 && b02 >= 0 && b12 >= 0);
                                    curr_ptr += 4, b01 -= d01y, b02 += d02y, b12 -= d12y;
                                }
                                curr_ptr += ptr_y_inc, b01 += d01x, b02 -= d02x, b12 += d12x;
                            }
                        }
                    }

                    // Case C: General case => Check tiles in AABB, record using atomics.

                    else
                    {

                        if (tri_idx != -1)
                        {
                            local uchar* skip_ptr = curr_ptr + (sizex << 2);
                            local uchar* end_ptr  = curr_ptr + (sizey << (CR_BIN_LOG2 + 2));
                            do
                            {
                                if (b01 >= 0 && b02 >= 0 && b12 >= 0)
                                    atomic_or((local uint*)curr_ptr, mask_bit);
                                curr_ptr += 4, b01 -= d01y, b02 += d02y, b12 -= d12y;
                                if (curr_ptr == skip_ptr)
                                    curr_ptr += ptr_y_inc, b01 += d01x, b02 -= d02x, b12 += d12x, skip_ptr += CR_BIN_SIZE * 4;
                            }
                            while (curr_ptr != end_ptr);
                        }

                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            //------------------------------------------------------------------------
            // Count.
            //------------------------------------------------------------------------

            // Tile per thread: Initialize prefix sums.

            for (int tile_in_bin = thread_local_id; tile_in_bin < CR_BIN_SQR; tile_in_bin += CR_COARSE_WARPS * 32)
            {
                // Compute prefix sum of emits over warps.

                local uchar* src_ptr = (local uchar*)&s_warp_emit_mask[0][tile_in_bin];
                local uchar* dst_ptr = (local uchar*)&s_warp_emit_prefix_sum[0][tile_in_bin];
                int tile_emits = 0;
                for (int i = 0; i < CR_COARSE_WARPS; i++)
                {
                    tile_emits += popcount(*(uint*)src_ptr);
                    *(uint*)dst_ptr = tile_emits;
                    src_ptr += (CR_BIN_SQR + 1) * 4;
                    dst_ptr += (CR_BIN_SQR + 1) * 4;
                }

                // Determine the number of segments to allocate.

                int space_left = -s_tile_stream_curr_ofs[tile_in_bin] & (CR_TILE_SEG_SIZE - 1);
                int tile_allocs = (tile_emits - space_left + CR_TILE_SEG_SIZE - 1) >> CR_TILE_SEG_LOG2;
                local volatile uint* p = &s_tile_emit_prefix_sum[tile_in_bin + 1];

                // All counters within the warp are small => compute prefix sum using ballot.

                if (!sub_group_any(tile_emits >= 2))
                {
                    uint m = getLaneMaskLe();
                    *p = (popcount(sub_group_ballot(tile_emits & 1) & m) << emit_shift) | popcount(sub_group_ballot(tile_allocs & 1) & m);
                }

                // Otherwise => scan-32 within the warp.

                else
                {
                    uint sum = (tile_emits << emit_shift) | tile_allocs;
                    *p = sum; if (get_local_id(0) >= 1)  sum += p[-1];
                    *p = sum; if (get_local_id(0) >= 2)  sum += p[-2];
                    *p = sum; if (get_local_id(0) >= 4)  sum += p[-4];
                    *p = sum; if (get_local_id(0) >= 8)  sum += p[-8];
                    *p = sum; if (get_local_id(0) >= 16) sum += p[-16];
                    *p = sum;
                }
            }

            // First warp: Scan-8.

            barrier(CLK_LOCAL_MEM_FENCE);

            if (thread_local_id < CR_BIN_SQR / 32)
            {
                int sum = s_tile_emit_prefix_sum[(thread_local_id << 5) + 32];
                local volatile uint* p = &s_scan_temp[0][thread_local_id + 16];
                p[0] = sum;
                #if (CR_BIN_SQR > 1 * 32)
                    sum += p[-1], p[0] = sum;
                #endif
                #if (CR_BIN_SQR > 2 * 32)
                    sum += p[-2], p[0] = sum;
                #endif
                #if (CR_BIN_SQR > 4 * 32)
                    sum += p[-4], p[0] = sum;
                #endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Tile per thread: Finalize prefix sums.
            // Single thread: Allocate segments.

            for (int tile_in_bin = thread_local_id; tile_in_bin < CR_BIN_SQR; tile_in_bin += CR_COARSE_WARPS * 32)
            {
                int sum = s_tile_emit_prefix_sum[tile_in_bin + 1] + s_scan_temp[0][(tile_in_bin >> 5) + 15];
                int num_emits = sum >> emit_shift;
                int num_allocs = sum & ((1 << emit_shift) - 1);
                s_tile_emit_prefix_sum[tile_in_bin + 1] = num_emits;
                s_tile_alloc_prefix_sum[tile_in_bin + 1] = num_allocs;

                if (tile_in_bin == CR_BIN_SQR - 1 && num_allocs != 0)
                {
                    int t = atomic_add(&g_num_tile_segs, num_allocs);
                    s_first_alloc_seg = (t + num_allocs <= p_max_tile_segs) ? t : 0;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            int first_alloc_seg   = s_first_alloc_seg;
            int total_emits      = s_tile_emit_prefix_sum[CR_BIN_SQR];
            int total_allocs     = s_tile_alloc_prefix_sum[CR_BIN_SQR];

            //------------------------------------------------------------------------
            // Emit.
            //------------------------------------------------------------------------

            // Emit per thread: Write triangle index to globalmem.

            for (int emit_in_bin = thread_local_id; emit_in_bin < total_emits; emit_in_bin += CR_COARSE_WARPS * 32)
            {
                // Find tile in bin.

                local uchar* tile_base = (local uchar*)&s_tile_emit_prefix_sum[0];
                local uchar* tile_ptr = tile_base;
                local uchar* ptr;

                #if (CR_BIN_SQR > 128)
                    ptr = tile_ptr + 0x80 * 4; if (emit_in_bin >= *(local uint*)ptr) tile_ptr = ptr;
                #endif
                #if (CR_BIN_SQR > 64)
                    ptr = tile_ptr + 0x40 * 4; if (emit_in_bin >= *(local uint*)ptr) tile_ptr = ptr;
                #endif
                #if (CR_BIN_SQR > 32)
                    ptr = tile_ptr + 0x20 * 4; if (emit_in_bin >= *(local uint*)ptr) tile_ptr = ptr;
                #endif
                #if (CR_BIN_SQR > 16)
                    ptr = tile_ptr + 0x10 * 4; if (emit_in_bin >= *(local uint*)ptr) tile_ptr = ptr;
                #endif
                #if (CR_BIN_SQR > 8)
                    ptr = tile_ptr + 0x08 * 4; if (emit_in_bin >= *(local uint*)ptr) tile_ptr = ptr;
                #endif
                #if (CR_BIN_SQR > 4)
                    ptr = tile_ptr + 0x04 * 4; if (emit_in_bin >= *(local uint*)ptr) tile_ptr = ptr;
                #endif
                #if (CR_BIN_SQR > 2)
                    ptr = tile_ptr + 0x02 * 4; if (emit_in_bin >= *(local uint*)ptr) tile_ptr = ptr;
                #endif
                #if (CR_BIN_SQR > 1)
                    ptr = tile_ptr + 0x01 * 4; if (emit_in_bin >= *(local uint*)ptr) tile_ptr = ptr;
                #endif

                int tile_in_bin = (tile_ptr - tileBase) >> 2;
                int emit_in_tile = emit_in_bin - *(local uint*)tile_ptr;

                // Find warp in tile.

                int warp_step = (CR_BIN_SQR + 1) * 4;
                local uchar* warp_base = (local uchar*)&s_warp_emit_prefix_sum[0][tile_in_bin] - warp_step;
                local uchar* warp_ptr = warp_base;

                #if (CR_COARSE_WARPS > 8)
                    ptr = warp_ptr + 0x08 * warp_step; if (emit_in_tile >= *(local uint*)ptr) warp_ptr = ptr;
                #endif
                #if (CR_COARSE_WARPS > 4)
                    ptr = warp_ptr + 0x04 * warp_step; if (emit_in_tile >= *(local uint*)ptr) warp_ptr = ptr;
                #endif
                #if (CR_COARSE_WARPS > 2)
                    ptr = warp_ptr + 0x02 * warp_step; if (emit_in_tile >= *(local uint*)ptr) warp_ptr = ptr;
                #endif
                #if (CR_COARSE_WARPS > 1)
                    ptr = warp_ptr + 0x01 * warp_step; if (emit_in_tile >= *(local uint*)ptr) warp_ptr = ptr;
                #endif

                int warp_in_tile = (warp_ptr - warp_base) >> (CR_BIN_LOG2 * 2 + 2);
                uint emit_mask = *(local uint*)(warp_ptr + warp_step + ((U8*)s_warp_emit_mask - (U8*)s_warp_emit_prefix_sum));
                int emit_in_warp = emit_in_tile - *(local uint*)(warp_ptr + warp_step) + popcount(emit_mask);

                // Find thread in warp.

                int thread_in_warp = 0;
                int pop = popcount(emit_mask & 0xFFFF);
                bool pred = (emit_in_warp >= pop);
                if (pred) emit_in_warp -= pop;
                if (pred) emit_mask >>= 0x10;
                if (pred) thread_in_warp += 0x10;

                pop = popcount(emit_mask & 0xFF);
                pred = (emit_in_warp >= pop);
                if (pred) emit_in_warp -= pop;
                if (pred) emit_mask >>= 0x08;
                if (pred) thread_in_warp += 0x08;

                pop = popcount(emit_mask & 0xF);
                pred = (emit_in_warp >= pop);
                if (pred) emit_in_warp -= pop;
                if (pred) emit_mask >>= 0x04;
                if (pred) thread_in_warp += 0x04;

                pop = popcount(emit_mask & 0x3);
                pred = (emit_in_warp >= pop);
                if (pred) emit_in_warp -= pop;
                if (pred) emit_mask >>= 0x02;
                if (pred) thread_in_warp += 0x02;

                if (emit_in_warp >= (emit_mask & 1))
                    thread_in_warp++;

                // Figure out where to write.

                int curr_ofs = s_tile_stream_curr_ofs[tile_in_bin];
                int space_left = -curr_ofs & (CR_TILE_SEG_SIZE - 1);
                int outOfs = emit_in_tile;

                if (outOfs < space_left)
                    outOfs += curr_ofs;
                else
                {
                    int alloc_lo = first_alloc_seg + s_tile_alloc_prefix_sum[tile_in_bin];
                    outOfs += (alloc_lo << CR_TILE_SEG_LOG2) - space_left;
                }

                // Write.

                int queue_idx = warp_in_tile * 32 + thread_in_warp;
                int tri_idx = s_tri_queue[(tri_queue_read_pos + queue_idx) & (CR_COARSE_QUEUE_SIZE - 1)];

                g_tile_seg_data[outOfs] = tri_idx;
            }

            //------------------------------------------------------------------------
            // Patch.
            //------------------------------------------------------------------------

            // Allocated segment per thread: Initialize next-pointer and count.

            for (int i = CR_COARSE_WARPS * 32 - 1 - thread_local_id; i < total_allocs; i += CR_COARSE_WARPS * 32)
            {
                int seg_idx = first_alloc_seg + i;
                g_tile_seg_next[seg_idx] = seg_idx + 1;
                g_tile_seg_count[seg_idx] = CR_TILE_SEG_SIZE;
            }

            // Tile per thread: Fix previous segment's next-pointer and update s_tile_stream_curr_ofs.

            barrier(CLK_LOCAL_MEM_FENCE);
            for (int tile_in_bin = CR_COARSE_WARPS * 32 - 1 - thread_local_id; tile_in_bin < CR_BIN_SQR; tile_in_bin += CR_COARSE_WARPS * 32)
            {
                int old_ofs = s_tile_stream_curr_ofs[tile_in_bin];
                int new_ofs = old_ofs + s_warp_emit_prefix_sum[CR_COARSE_WARPS - 1][tile_in_bin];
                int alloc_lo = s_tile_alloc_prefix_sum[tile_in_bin];
                int alloc_hi = s_tile_alloc_prefix_sum[tile_in_bin + 1];

                if (alloc_lo != alloc_hi)
                {
                    local int* next_ptr = &g_tile_seg_next[(old_ofs - 1) >> CR_TILE_SEG_LOG2];
                    if (old_ofs < 0)
                        next_ptr = &g_tile_first_seg[bin_tile_idx + global_tile_idx(tile_in_bin)];
                    *next_ptr = first_alloc_seg + alloc_lo;

                    new_ofs--;
                    new_ofs &= CR_TILE_SEG_SIZE - 1;
                    new_ofs |= (first_alloc_seg + alloc_hi - 1) << CR_TILE_SEG_LOG2;
                    new_ofs++;
                }
                s_tile_stream_curr_ofs[tile_in_bin] = new_ofs;
            }

            // Advance queue read pointer.
            // Queue became empty => bin done.

            tri_queue_read_pos += CR_COARSE_WARPS * 32;
        }
        while (tri_queue_read_pos < tri_queue_write_pos);

        // Tile per thread: Fix next-pointer and count of the last segment.
        // 32 tiles per warp: Count active tiles.

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int tile_in_bin = thread_local_id; tile_in_bin < CR_BIN_SQR; tile_in_bin += CR_COARSE_WARPS * 32)
        {
            int tile_x = tile_in_bin & (CR_BIN_SIZE - 1);
            int tile_y = tile_in_bin >> CR_BIN_LOG2;
            bool force = (p_deferred_clear & tile_x <= max_tile_x_in_bin & tile_y <= max_tile_y_in_bin);

            int ofs = s_tile_stream_curr_ofs[tile_in_bin];
            int seg_idx = (ofs - 1) >> CR_TILE_SEG_LOG2;
            int seg_count = ofs & (CR_TILE_SEG_SIZE - 1);

            if (ofs >= 0)
                g_tile_seg_next[seg_idx] = -1;
            else if (force)
            {
                s_tile_stream_curr_ofs[tile_in_bin] = 0;
                g_tile_first_seg[bin_tile_idx + tile_x + tile_y * p_width_tiles] = -1;
            }

            if (seg_count != 0)
                g_tile_seg_count[seg_idx] = seg_count;

            s_scan_temp[0][(tile_in_bin >> 5) + 16] = popcount(sub_group_ballot(ofs >= 0 | force));
        }

        // First warp: Scan-8.
        // One thread: Allocate space for active tiles.

        barrier(CLK_LOCAL_MEM_FENCE);
        if (thread_local_id < CR_BIN_SQR / 32)
        {
            local volatile uint* p = &s_scan_temp[0][thread_local_id + 16];
            uint sum = p[0];
            #if (CR_BIN_SQR > 1 * 32)
                sum += p[-1], p[0] = sum;
            #endif
            #if (CR_BIN_SQR > 2 * 32)
                sum += p[-2], p[0] = sum;
            #endif
            #if (CR_BIN_SQR > 4 * 32)
                sum += p[-4], p[0] = sum;
            #endif

            if (thread_local_id == CR_BIN_SQR / 32 - 1)
                s_first_active_idx = atomic_add(g_num_active_tiles, sum);
        }

        // Tile per thread: Output active tiles.

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int tile_in_bin = thread_local_id; tile_in_bin < CR_BIN_SQR; tile_in_bin += CR_COARSE_WARPS * 32)
        {
            if (s_tile_stream_curr_ofs[tile_in_bin] < 0)
                continue;

            int active_idx = s_first_active_idx;
            active_idx += s_scan_temp[0][(tile_in_bin >> 5) + 15];
            active_idx += popcount(sub_group_ballot(true) & getLaneMaskLt());
            g_active_tiles[active_idx] = bin_tile_idx + global_tile_idx(tile_in_bin);
        }

    }

}