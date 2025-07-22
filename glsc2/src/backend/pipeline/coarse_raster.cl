#ifdef __COMPILER_RELATIVE_PATH__
#include <backend/types.cl>
#include <backend/utils/sub_group_mask.cl>
#include <backend/utils/sync.cl>
#else
#include "glsc2/src/backend/types.cl"
#include "glsc2/src/backend/utils/sub_group_mask.cl"
#include "glsc2/src/backend/utils/sync.cl"
#endif

inline void sort_shared(local volatile uint* ptr, int num_items)
{
    int thread_local_id = get_local_linear_id(); // get_local_id(0) + get_local_id(1) * get_local_size(0);
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

inline int global_tile_idx(int tile_in_bin, int c_width_tiles)
{
    int tile_x = tile_in_bin & (CR_BIN_SIZE - 1);
    int tile_y = tile_in_bin >> CR_BIN_LOG2;
    return tile_x + tile_y * c_width_tiles;
}

//----------------------------------------------------------------------------------------

kernel 
__attribute__((reqd_work_group_size(DEVICE_SUB_GROUP_THREADS, DEVICE_COARSE_SUB_GROUPS, 1)))
void coarse_raster(
    global int* a_coarse_counter,
    global int* a_num_active_tiles,
    global const int* a_num_bin_segs,
    global int* a_num_tile_segs,
    global const int* a_num_subtris,

    global int* g_active_tiles,
    global const int* g_bin_first_seg,    
    global const int* g_bin_seg_count,
    global const int* g_bin_seg_data,
    global const int* g_bin_seg_next,
    global const int* g_bin_total,
    global int* g_tile_first_seg,
    global int* g_tile_seg_count,
    global int* g_tile_seg_data,
    global int* g_tile_seg_next,
    global const triangle_header_t*  g_tri_header,

    #ifdef DEVICE_IMAGE_ENABLED
    read_only image1d_buffer_t t_tri_header,
    #endif

    const int c_deferred_clear,
    const int c_height_tiles,
    const int c_max_bin_segs,
    const int c_max_subtris,
    const int c_max_tile_segs,
    const int c_num_bins,
    const int c_viewport_height,
    const int c_viewport_width,
    const int c_width_bins,
    const int c_width_tiles
) {
    
    local volatile uint s_work_counter;
    local volatile uint s_scan_temp          [DEVICE_COARSE_SUB_GROUPS][48];              // 3KB

    local volatile uint s_bin_order           [CR_MAXBINS_SQR];                   // 1KB
    local volatile int s_bin_stream_curr_seg  [CR_BIN_STREAMS_SIZE];              // 0KB
    local volatile int s_bin_stream_first_tri [CR_BIN_STREAMS_SIZE];              // 0KB
    local volatile int s_tri_queue            [CR_COARSE_QUEUE_SIZE];             // 4KB
    local volatile int s_tri_queue_write_pos;
    local volatile uint s_bin_stream_selected_ofs;
    local volatile uint s_bin_stream_selected_size;

    local volatile uint s_warp_emit_mask      [DEVICE_COARSE_SUB_GROUPS][CR_BIN_SQR + 1];  // 16KB, +1 to avoid bank collisions
    local volatile uint s_warp_emit_prefix_sum [DEVICE_COARSE_SUB_GROUPS][CR_BIN_SQR + 1];  // 16KB, +1 to avoid bank collisions
    local volatile uint s_tile_emit_prefix_sum [CR_BIN_SQR + 1];                   // 1KB, zero at the beginning
    local volatile uint s_tile_alloc_prefix_sum [CR_BIN_SQR + 1];                   // 1KB, zero at the beginning
    local volatile int s_tile_stream_curr_ofs [CR_BIN_SQR];                       // 1KB
    local volatile uint s_first_alloc_seg;
    local volatile uint s_first_active_idx;

    local volatile uint l_temp [DEVICE_COARSE_SUB_GROUPS*DEVICE_SUB_GROUP_THREADS];

    // Private
    int tile_log         = CR_TILE_LOG2 + CR_SUBPIXEL_LOG2;
    int thread_local_id  = get_local_linear_id();
    int emit_shift       = CR_BIN_LOG2 * 2 + 5; // We scan ((num_emits << emit_shift) | num_allocs) over tiles.

    if (*a_num_subtris > c_max_subtris || *a_num_bin_segs > c_max_bin_segs)
        return;

    // Initialize sharedmem arrays.

    s_tile_emit_prefix_sum[0] = 0;
    s_tile_alloc_prefix_sum[0] = 0;
    s_scan_temp[get_local_id(1)][get_local_id(0)] = 0;

    // Sort bins in descending order of triangle count.

    for (int bin_idx = thread_local_id; bin_idx < c_num_bins; bin_idx += get_local_linear_size())
    {
        int count = 0;
        for (int i = 0; i < CR_BIN_STREAMS_SIZE; i++)
            count += g_bin_total[(bin_idx << CR_BIN_STREAMS_LOG2) + i];
        s_bin_order[bin_idx] = (~count << (CR_MAXBINS_LOG2 * 2)) | bin_idx;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    sort_shared(s_bin_order, c_num_bins);

    // Process each bin by one block.

    for (;;)
    {
        // Pick a bin for the block.

        if (thread_local_id == 0)
            s_work_counter = atomic_add(a_coarse_counter, 1);
        barrier(CLK_LOCAL_MEM_FENCE);

        int work_counter = s_work_counter;
        if (work_counter >= c_num_bins)
        {
            break;
        }

        uint bin_order = s_bin_order[work_counter];
        bool bin_empty = ((~bin_order >> (CR_MAXBINS_LOG2 * 2)) == 0);
        if (bin_empty && !c_deferred_clear)
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

        for (int tile_in_bin = thread_local_id; tile_in_bin < CR_BIN_SQR; tile_in_bin += get_local_linear_size())
            s_tile_stream_curr_ofs[tile_in_bin] = -CR_TILE_SEG_SIZE;

        // Initialize per-bin state.

        int bin_y = bin_idx / c_width_bins; // idiv_fast(bin_idx, c_width_bins);
        int bin_x = bin_idx - bin_y * c_width_bins;
        int origin_x = (bin_x << (CR_BIN_LOG2 + tile_log)) - (c_viewport_width << (CR_SUBPIXEL_LOG2 - 1));
        int origin_y = (bin_y << (CR_BIN_LOG2 + tile_log)) - (c_viewport_height << (CR_SUBPIXEL_LOG2 - 1));
        int max_tile_x_in_bin = min(c_width_tiles - (bin_x << CR_BIN_LOG2), CR_BIN_SIZE) - 1;
        int max_tile_y_in_bin = min(c_height_tiles - (bin_y << CR_BIN_LOG2), CR_BIN_SIZE) - 1;
        int bin_tile_idx = (bin_x + bin_y * c_width_tiles) << CR_BIN_LOG2;


        // Entire block: Merge input streams and process triangles.

        if (!bin_empty)
        do
        {
            //------------------------------------------------------------------------
            // Merge.
            //------------------------------------------------------------------------

            // Entire block: Not enough triangles => merge and queue segments.
            // NOTE: The bin exit criterion assumes that we queue more triangles than we actually need.

            while (tri_queue_write_pos - tri_queue_read_pos <= get_local_linear_size())
            {
                // First warp: Choose the segment with the lowest initial triangle index.

                // TODO: Optimize for sub groups when bin streams > sub group
                // Find the stream with the lowest triangle index.
                #if (CR_BIN_STREAMS_SIZE <= DEVICE_SUB_GROUP_THREADS && (DEVICE_SUB_GROUP_INTRINSICTS_SUPPORT || DEVICE_SUB_GROUP_RAW))
                {
                    bool thread_condition;
                    #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
                        thread_condition = get_sub_group_id() == 0;
                    #else
                        thread_condition = thread_local_id < CR_BIN_STREAMS_SIZE;
                    #endif

                    if (thread_condition)
                    {
                        
                        uint id, first_tri, min_first_tri;
                        #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
                        {
                            id = get_sub_group_local_id();
                            first_tri = id < CR_BIN_STREAMS_SIZE ? s_bin_stream_first_tri[id] : UINT_MAX;
                            min_first_tri = sub_group_reduce_min(first_tri);
                        }
                        #else
                        {
                            id = thread_local_id;
                            first_tri = s_bin_stream_first_tri[id];
                            sub_group_scan_inclusive_min(first_tri, l_temp);
                            min_first_tri = l_temp[CR_BIN_STREAMS_SIZE-1];
                        }
                        #endif
                        
                        if (min_first_tri == first_tri)
                        {
                            int seg_idx = s_bin_stream_curr_seg[id];
                            s_bin_stream_selected_ofs = seg_idx << CR_BIN_SEG_LOG2;
                            if (seg_idx != -1)
                            {
                                int seg_size = g_bin_seg_count[seg_idx];
                                int seg_next = g_bin_seg_next[seg_idx];
                                s_bin_stream_selected_size = seg_size;
                                s_tri_queue_write_pos = tri_queue_write_pos + seg_size;
                                s_bin_stream_curr_seg[id] = seg_next;
                                s_bin_stream_first_tri[id] = (seg_next == -1) ? ~0u : g_bin_seg_data[seg_next << CR_BIN_SEG_LOG2];
                            }
                        }
                        
                    }
                }
                #else
                {
                    uint first_tri = UINT_MAX; 
                    if (thread_local_id < CR_BIN_STREAMS_SIZE)
                        first_tri = s_bin_stream_first_tri[thread_local_id];

                    // TODO: reduce the scope of the reduce depending on bin streams
                    uint min_first_tri = local_reduce_min(first_tri, l_temp);

                    if (min_first_tri == first_tri && thread_local_id < CR_BIN_STREAMS_SIZE)
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
                #endif
                // No more segments => break.

                barrier(CLK_LOCAL_MEM_FENCE);
                tri_queue_write_pos = s_tri_queue_write_pos;
                int seg_ofs = s_bin_stream_selected_ofs;
                if (seg_ofs < 0)
                    break;

                int seg_size = s_bin_stream_selected_size;
                barrier(CLK_LOCAL_MEM_FENCE);

                // Fetch triangles into the queue.
                for (int idx_in_seg = thread_local_id; idx_in_seg < seg_size; idx_in_seg += get_local_linear_size())
                {
                    int tri_idx = g_bin_seg_data[seg_ofs + idx_in_seg];
                    s_tri_queue[(tri_queue_write_pos - seg_size + idx_in_seg) & (CR_COARSE_QUEUE_SIZE - 1)] = tri_idx;
                }

            }

            // All threads: Clear emit masks.

            for (int mask_idx = thread_local_id; mask_idx < DEVICE_COARSE_SUB_GROUPS * CR_BIN_SQR; mask_idx += get_local_linear_size())
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
                int subtri_idx = tri_idx & 0x7;
                if (subtri_idx != 7)
                    data_idx = g_tri_header[data_idx].misc + subtri_idx;
                #ifdef DEVICE_IMAGE_ENABLED
                tri_data = read_imageui(t_tri_header, data_idx);
                #else
                tri_data = *(((global uint4*) g_tri_header) + data_idx); 
                #endif
            }

            // TODO: Refactor this part to be optimal for disable sub groups kernels
            // 32 triangles per warp: Record emits (= tile intersections).
            #ifdef CONF_SUB_GROUP_ENABLED
            if (sub_group_any(tri_idx != -1))
            #endif
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
                #ifdef CONF_SUB_GROUP_ENABLED
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
                #endif
                {
                    // Compute warp-AABB (scan-32).

                    uint aabb_mask = add_sub(2 << hix, 0x20000 << hiy, 1 << lox) - (0x10000 << loy);
                    if (tri_idx == -1)
                        aabb_mask = 0;

                    #ifdef CONF_SUB_GROUP_ENABLED
                    aabb_mask = sub_group_reduce_or_ui(aabb_mask);
                    #else
                    aabb_mask = local_1dim_reduce_or(aabb_mask, l_temp);
                    #endif

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
                    #ifdef CONF_SUB_GROUP_ENABLED
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
                                    *(local uint*)curr_ptr = sub_group_masked_ballot(b01 >= 0 && b02 >= 0 && b12 >= 0, sub_group_activemask());
                                    curr_ptr += 4, b01 -= d01y, b02 += d02y, b12 -= d12y;
                                }
                                curr_ptr += ptr_y_inc, b01 += d01x, b02 -= d02x, b12 += d12x;
                            }
                        }
                    }

                    // Case C: General case => Check tiles in AABB, record using atomics.

                    else
                    #endif
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
            // TODO: this only works if CR_BIN_SQR is divisible by get_local_size(0)
            #ifdef CONF_SUB_GROUP_ENABLED
            for (int tile_in_bin = thread_local_id; tile_in_bin < CR_BIN_SQR; tile_in_bin += get_local_linear_size())
            #else
            for (int tile_in_bin_chunk = 0; tile_in_bin_chunk < CR_BIN_SQR; tile_in_bin_chunk += get_local_linear_size())
            #endif
            {
                // Compute prefix sum of emits over warps.
                #ifndef CONF_SUB_GROUP_ENABLED
                int tile_in_bin = tile_in_bin_chunk + thread_local_id;
                #endif

                local uchar *src_ptr, *dst_ptr;
                local volatile uint* p;
                int tile_emits, tile_allocs;
                #ifndef CONF_SUB_GROUP_ENABLED
                if (tile_in_bin < CR_BIN_SQR)
                #endif
                {
                    src_ptr = (local uchar*)&s_warp_emit_mask[0][tile_in_bin];
                    dst_ptr = (local uchar*)&s_warp_emit_prefix_sum[0][tile_in_bin];
                    tile_emits = 0;
                    for (int i = 0; i < DEVICE_COARSE_SUB_GROUPS; i++)
                    {
                        tile_emits += popcount(*(uint*)src_ptr);
                        *(uint*)dst_ptr = tile_emits;
                        src_ptr += (CR_BIN_SQR + 1) * 4;
                        dst_ptr += (CR_BIN_SQR + 1) * 4;
                    }

                    // Determine the number of segments to allocate.

                    int space_left = -s_tile_stream_curr_ofs[tile_in_bin] & (CR_TILE_SEG_SIZE - 1);
                    tile_allocs = (tile_emits - space_left + CR_TILE_SEG_SIZE - 1) >> CR_TILE_SEG_LOG2;
                    p = &s_tile_emit_prefix_sum[tile_in_bin + 1];
                }

                #ifdef CONF_SUB_GROUP_ENABLED
                // All counters within the warp are small => compute prefix sum using ballot.
                if (!sub_group_any(tile_emits >= 2))
                {
                    uint m = getLaneMaskLe();
                    *p = (popcount(sub_group_ballot(tile_emits & 1) & m) << emit_shift) | popcount(sub_group_ballot(tile_allocs & 1) & m);
                }
                // Otherwise => scan-32 within the warp.
                else
                #endif
                {
                    uint sum = (tile_emits << emit_shift) | tile_allocs;
                    uint scan_sum;
                    #ifdef CONF_SUB_GROUP_ENABLED
                    scan_sum = sub_group_scan_inclusive_add_ui(sum);
                    #else
                    scan_sum = local_1dim_scan_inclusive_add(sum, l_temp);
                    #endif

                    #ifndef CONF_SUB_GROUP_ENABLED
                    if (tile_in_bin < CR_BIN_SQR)
                    #endif
                    {
                        *p = scan_sum;
                    }
                }
            }

            // First warp: Scan-8.

            barrier(CLK_LOCAL_MEM_FENCE);

            // TODO: this is only valid if CR_BIN_SQR / 32 < sub_group_size()
            #if (CR_BIN_SQR / DEVICE_SUB_GROUP_THREADS <= DEVICE_SUB_GROUP_THREADS)
            {
                bool thread_condition = true;
                #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
                    thread_condition = thread_local_id < get_sub_group_size();
                #endif
                #ifdef DEVICE_SUB_GROUP_RAW_ENABLED
                    thread_condition = thread_local_id < CR_BIN_SQR / 32;
                #endif

                if (thread_condition){
                    int sum = 0;
                    if (thread_local_id < CR_BIN_SQR / 32)
                        sum = s_tile_emit_prefix_sum[(thread_local_id << 5) + 32];
                    
                    int scan_sum;
                    #if (DEVICE_SUB_GROUP_INTRINSICTS_SUPPORT && DEVICE_SUB_GROUP_RAW == 0)
                        scan_sum = sub_group_scan_inclusive_add(sum);
                    #else
                        scan_sum = local_1dim_scan_inclusive_add(sum, l_temp);
                    #endif

                    if (thread_local_id < CR_BIN_SQR / 32)
                        s_scan_temp[0][thread_local_id + 16] = scan_sum;

                }
            }
            #else 
                #error CR_BIN_SQR is to large.
            #endif

            barrier(CLK_LOCAL_MEM_FENCE);

            // Tile per thread: Finalize prefix sums.
            // Single thread: Allocate segments.

            for (int tile_in_bin = thread_local_id; tile_in_bin < CR_BIN_SQR; tile_in_bin += get_local_linear_size())
            {
                int sum = s_tile_emit_prefix_sum[tile_in_bin + 1] + s_scan_temp[0][(tile_in_bin >> 5) + 15];
                int num_emits = sum >> emit_shift;
                int num_allocs = sum & ((1 << emit_shift) - 1);
                s_tile_emit_prefix_sum[tile_in_bin + 1] = num_emits;
                s_tile_alloc_prefix_sum[tile_in_bin + 1] = num_allocs;

                if (tile_in_bin == CR_BIN_SQR - 1 && num_allocs != 0)
                {
                    int t = atomic_add(a_num_tile_segs, num_allocs);
                    s_first_alloc_seg = (t + num_allocs <= c_max_tile_segs) ? t : 0;
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

            for (int emit_in_bin = thread_local_id; emit_in_bin < total_emits; emit_in_bin += get_local_linear_size())
            {
                // int emit_in_bin = emit_in_bin_chunk + thread_local_id;
                
                // Find tile in bin.

                local uint* tile_base = (local uint*) &s_tile_emit_prefix_sum[0];
                local uint* tile_ptr = tile_base;
                local uint* ptr;

                #pragma unroll
                for (int i=CR_BIN_SQR; i>0; i /=2) {
                    ptr = tile_ptr + i/2;
                    if (emit_in_bin >= *ptr) tile_ptr = ptr;
                }

                int tile_in_bin = (tile_ptr - tile_base);
                int emit_in_tile = emit_in_bin - *tile_ptr;

                // Find warp in tile.

                int warp_step = (CR_BIN_SQR + 1);
                local uint* warp_base = (local uint*)&s_warp_emit_prefix_sum[0][tile_in_bin] - warp_step;
                local uint* warp_ptr = warp_base;

                #pragma unroll
                for (int i=DEVICE_COARSE_SUB_GROUPS; i>0; i /=2) {
                    ptr = warp_ptr + i/2 * warp_step;
                    if (emit_in_tile >= *ptr) warp_ptr = ptr;
                }

                int warp_in_tile = (warp_ptr - warp_base) >> (CR_BIN_LOG2 * 2);
                uint emit_mask = *(warp_ptr + warp_step + ((uint*)s_warp_emit_mask - (uint*)s_warp_emit_prefix_sum));
                int emit_in_warp = emit_in_tile - *(warp_ptr + warp_step) + popcount(emit_mask);

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

            for (int i = thread_local_id; i < total_allocs; i += get_local_linear_size())
            {
                int seg_idx = first_alloc_seg + i;
                g_tile_seg_next[seg_idx] = seg_idx + 1;
                g_tile_seg_count[seg_idx] = CR_TILE_SEG_SIZE;
            }

            // Tile per thread: Fix previous segment's next-pointer and update s_tile_stream_curr_ofs.
            // TODO: rm this sync, maybe unecessary ??
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int tile_in_bin = thread_local_id; tile_in_bin < CR_BIN_SQR; tile_in_bin += get_local_linear_size())
            {
                int old_ofs = s_tile_stream_curr_ofs[tile_in_bin];
                int new_ofs = old_ofs + s_warp_emit_prefix_sum[DEVICE_COARSE_SUB_GROUPS - 1][tile_in_bin];
                int alloc_lo = s_tile_alloc_prefix_sum[tile_in_bin];
                int alloc_hi = s_tile_alloc_prefix_sum[tile_in_bin + 1];

                if (alloc_lo != alloc_hi)
                {
                    global int* next_ptr = &g_tile_seg_next[(old_ofs - 1) >> CR_TILE_SEG_LOG2];
                    if (old_ofs < 0)
                        next_ptr = &g_tile_first_seg[bin_tile_idx + global_tile_idx(tile_in_bin, c_width_tiles)];
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

            tri_queue_read_pos += get_local_linear_size();
        }
        while (tri_queue_read_pos < tri_queue_write_pos);

        // Tile per thread: Fix next-pointer and count of the last segment.
        // 32 tiles per warp: Count active tiles.

        barrier(CLK_LOCAL_MEM_FENCE);

        #ifdef CONF_SUB_GROUP_ENABLED
        for (int tile_in_bin = thread_local_id; tile_in_bin < CR_BIN_SQR; tile_in_bin += get_local_linear_size())
        #else
        for (int tile_in_bin_chunk = 0; tile_in_bin_chunk < CR_BIN_SQR; tile_in_bin_chunk += get_local_linear_size())
        #endif
        {
            #ifndef CONF_SUB_GROUP_ENABLED
            int tile_in_bin = tile_in_bin_chunk + thread_local_id;
            #endif
            bool is_tile_in_bin = tile_in_bin < CR_BIN_SQR;
            int ofs;
            bool force;
            if (is_tile_in_bin) {

                int tile_x = tile_in_bin & (CR_BIN_SIZE - 1);
                int tile_y = tile_in_bin >> CR_BIN_LOG2;
                bool force = (c_deferred_clear && tile_x <= max_tile_x_in_bin && tile_y <= max_tile_y_in_bin); // check this

                int ofs;

                #ifndef CONF_SUB_GROUP_ENABLED
                if (is_tile_in_bin)
                #endif
                {
                    ofs = s_tile_stream_curr_ofs[tile_in_bin];
                }
                int seg_idx = (ofs - 1) >> CR_TILE_SEG_LOG2;
                int seg_count = ofs & (CR_TILE_SEG_SIZE - 1);

                if (ofs >= 0)
                    g_tile_seg_next[seg_idx] = -1;
                else if (force)
                {
                    s_tile_stream_curr_ofs[tile_in_bin] = 0;
                    g_tile_first_seg[bin_tile_idx + tile_x + tile_y * c_width_tiles] = -1;
                }

                if (seg_count != 0)
                    g_tile_seg_count[seg_idx] = seg_count;
            
            }

            uint bitmask = local_1dim_ballot(ofs >= 0 || force || is_tile_in_bin, l_temp);
            // uint bitmask = sub_group_ballot(ofs >= 0 || force || is_tile_in_bin);

            s_scan_temp[0][(tile_in_bin >> 5) + 16] = popcount(bitmask);
        }

        // First warp: Scan-8.
        // One thread: Allocate space for active tiles.

        barrier(CLK_LOCAL_MEM_FENCE);
        // #ifdef CONF_SUB_GROUP_ENABLED
        // if (thread_local_id < get_local_size(0))
        // #endif
        {
            local volatile uint* p = &s_scan_temp[0][thread_local_id + 16];
            uint sum = 0;
            if (thread_local_id < CR_BIN_SQR / 32) {
                // s_scan_temp[0][thread_local_id + 8] = 0;
                sum = s_scan_temp[0][thread_local_id + 16];
            }
            // uint scan_sum;
            // #ifdef CONF_SUB_GROUP_ENABLED
            // #else
            //     scan_sum = local_scan_inclusive_add_1dim_ui(sum, l_temp);
            // #endif
            // sum = sub_group_scan_inclusive_add_ui(sum);
            #pragma unroll
            for (int i=1; i*DEVICE_SUB_GROUP_THREADS < CR_BIN_SQR; i*=2) {
                sum += p[-i], p[0] = sum;
                
                // last iter do not sync
                if (i*DEVICE_SUB_GROUP_THREADS*2 < CR_BIN_SQR) 
                    barrier(CLK_LOCAL_MEM_FENCE);
            }
            /*
            #if (CR_BIN_SQR > 1 * 32)
            if (thread_local_id < CR_BIN_SQR / 32) {
                sum += p[-1], p[0] = sum;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            #endif
            #if (CR_BIN_SQR > 2 * 32)
            if (thread_local_id < CR_BIN_SQR / 32) {
                sum += p[-2], p[0] = sum;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            #endif
            #if (CR_BIN_SQR > 4 * 32)
            if (thread_local_id < CR_BIN_SQR / 32) {
                sum += p[-4], p[0] = sum;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            #endif
            */
            if (thread_local_id == CR_BIN_SQR / 32 - 1)
                s_first_active_idx = atomic_add(a_num_active_tiles, sum);
        }

        // Tile per thread: Output active tiles.

        barrier(CLK_LOCAL_MEM_FENCE);
        #ifdef CONF_SUB_GROUP_ENABLED
        for (int tile_in_bin = thread_local_id; tile_in_bin < CR_BIN_SQR; tile_in_bin += DEVICE_COARSE_SUB_GROUPS * 32)
        #else
        for (int tile_in_bin_chunk = 0; tile_in_bin_chunk < CR_BIN_SQR; tile_in_bin_chunk += get_local_linear_size())
        #endif
        {
            #ifndef CONF_SUB_GROUP_ENABLED
            int tile_in_bin = tile_in_bin_chunk + thread_local_id;
            bool enable = tile_in_bin < CR_BIN_SQR;
            #endif

            bool pass = 1;
            #ifndef CONF_SUB_GROUP_ENABLED
            if (enable)
            #endif
            {
                pass = s_tile_stream_curr_ofs[tile_in_bin] < 0;
            }

            #ifdef CONF_SUB_GROUP_ENABLED
            if (pass)
                continue;
            #endif

            int active_idx = s_first_active_idx;
            #ifndef CONF_SUB_GROUP_ENABLED
            if (!pass)
            #endif
            {
                active_idx += s_scan_temp[0][(tile_in_bin >> 5) + 15];
            }

            uint activemask;
            #ifdef CONF_SUB_GROUP_ENABLED
            activemask = sub_group_activemask();
            #else
            activemask = local_1dim_reduce_or((pass ? 0 : 1) << get_local_id(0), l_temp);
            #endif
            active_idx += popcount(activemask & getLaneMaskLt());
            #ifndef CONF_SUB_GROUP_ENABLED
            if (!pass)
            #endif
            {
                g_active_tiles[active_idx] = bin_tile_idx + global_tile_idx(tile_in_bin, c_width_tiles);
            }
        }

    }

}