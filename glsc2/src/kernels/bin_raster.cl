#ifdef __COMPILER_RELATIVE_PATH__ 
#include "common/headers.cl"
#else
#include "glsc2/src/kernels/common/headers.cl"
#endif

// Configuration
#define CONF_WARP_SIZE_LOG2 5
#define CONF_BR_WARPS_LOG2 4
// Architecture
#define CONF_WARP_SIZE (1 << CONF_WARP_SIZE_LOG2)
// Bin rasterizer 
#define CONF_BR_WARPS (1 << CONF_BR_WARPS_LOG2)

// Debug definitions
#ifdef CONF_DEBUG_KERNEL
typedef struct {
    int over_total;
} global_bin_raster_debug_t;

typedef struct {
    int over_total;
} local_bin_raster_debug_t;

typedef struct {
    uint my_idx;
    uint l_broadcast;
    uint4 tri_data;
} warp_bin_raster_debug_t;
#endif



// uint local_scan_inclusive_add_ui(uint value, local volatile uint* l_temp) {
//     uint local_id = get_local_id(0);
//     local volatile uint* ptr = &l_temp[local_id];
//     *ptr = value;
//     #pragma unroll
//     for(int i=0; i<CONF_WARP_SIZE_LOG2 + CONF_BR_WARPS_LOG2; ++i) {
//         barrier(CLK_LOCAL_MEM_FENCE);
//         if (local_id >= 1 << i) {
//             value += ptr[-(1 << i)];    
//             *ptr = value;   
//         }
//     }
//     return value;
// }


/**
    2 dim kernel 
    TODO: maybe could be done 1 dim, but would need to change a lot the kernel
 */
kernel
#ifndef CONF_BIN_RASTER_RANDOM_SIZE
//#ifndef CONF_SUB_GROUP_ENABLED
//__attribute__((reqd_work_group_size(CONF_BIN_RASTER_LOCAL_SIZE, 1, 1)))
//#else
__attribute__((reqd_work_group_size(CONF_WARP_SIZE, CONF_BR_WARPS, 1)))
//#endif
#endif
void bin_raster(
    global int* a_bin_counter,
    global int* a_num_bin_segs,
    global const int* a_num_subtris,

    global int* g_bin_first_seg,
    global int* g_bin_seg_count,
    global int* g_bin_seg_data,
    global int* g_bin_seg_next,
    global int* g_bin_total,
    global const CRTriangleHeader* g_tri_header,
    global const uchar* g_tri_subtris,
    
    #ifdef __IMAGE_SUPPORT__
    read_only image1d_buffer_t t_tri_header,
    #endif

    private const int c_bin_batch_sz,
    private const int c_height_bins,
    private const int c_max_bin_segs,
    private const int c_max_subtris,
    private const int c_num_bins,
    private const int c_num_tris,
    private const int c_viewport_height,
    private const int c_viewport_width,
    private const int c_width_bins

    #ifdef CONF_DEBUG_KERNEL
    , global warp_bin_raster_debug_t* g_w_debug // sz == 32
    //, global local_bin_raster_debug_t* g_l_debug // sz == CR_BIN_WARPS
    //, global global_bin_raster_debug_t* g_g_debug // 
    #endif
) {

    // Local space
    local volatile uint s_broadcast   [CR_BIN_WARPS + 16];
    local volatile int  s_out_ofs     [CR_MAXBINS_SQR];
    local volatile int  s_out_total   [CR_MAXBINS_SQR];
    local volatile int  s_over_index  [CR_MAXBINS_SQR];
    local volatile int  s_out_mask    [CR_BIN_WARPS][CR_MAXBINS_SQR + 1]; // +1 to avoid bank collisions
    local volatile int  s_out_count   [CR_BIN_WARPS][CR_MAXBINS_SQR + 1]; // +1 to avoid bank collisions
    local volatile int  s_tri_buf     [CR_BIN_WARPS*32*4];                // triangle ring buffer
    local volatile uint s_batch_pos;
    local volatile uint s_buf_count;
    local volatile uint s_over_total;
    local volatile uint s_alloc_base;

    #ifndef CONF_WARP_ENABLED
    local volatile uint l_temp [CONF_WARP_SIZE*CONF_BR_WARPS];
    #endif
    
    if (*a_num_subtris > c_max_subtris) 
        return;

    // Private space
    int local_id = get_local_id(0) + get_local_id(1) * get_local_size(0);
    uint local_size = get_local_size(0) * get_local_size(1);
    int batch_pos = 0;

    // first 16 elements of s_broadcast are always zero
    if (local_id < 16)
        s_broadcast[local_id] = 0;

    // initialize output linked lists and offsets
    if (local_id < c_num_bins)
    {
        g_bin_first_seg[(local_id << CR_BIN_STREAMS_LOG2) + get_group_id(0)] = -1;
        s_out_ofs[local_id] = -CR_BIN_SEG_SIZE;
        s_out_total[local_id] = 0;
    }

    // repeat until done
    for (;;) {
        // get batch
        if (local_id == 0)
            s_batch_pos = atomic_add(a_bin_counter,c_bin_batch_sz);
        barrier(CLK_LOCAL_MEM_FENCE);
        batch_pos = s_batch_pos;

        // all batches done?
        if (batch_pos >= c_num_tris)
            break;

        // Private space
        int buf_index = 0;
        int buf_count = 0;
        int batch_end = min(batch_pos + c_bin_batch_sz, c_num_tris);

        // Loop over batch
        do {
            
            while (buf_count < local_size && batch_pos < batch_end) {
                
                // Get subtriangle count
                int tri_idx = batch_pos + local_id;
                int num = 0;
                if (tri_idx < batch_end)
                    num = g_tri_subtris[tri_idx];

                // cumulative sum of subtriangles within each warp
                uint scan_exc_num;
                #ifndef CONF_SUB_GROUP_ENABLED
                {
                    uint scan_inc_num = local_scan_inclusive_add_ui(num, l_temp);
                    scan_exc_num = scan_inc_num - num;
                }
                #else
                {
                    // sub-groups implementation
                    uint scan_inc_num = sub_group_scan_inclusive_add_ui(num);
                    // broadcast sub group scan inclusive to +1 offset sub group.
                    if (get_local_id(0) == get_local_size(0)-1) {
                        uint scan_exc_ofs = ((get_local_id(1)+1) & (get_local_size(1)-1)) + get_local_size(1); 
                        s_broadcast[scan_exc_ofs] = scan_inc_num * (get_local_id(1) != get_local_size(1)-1);
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    
                    // This only works if local_size / sub_group_size <= sub_group_size.
                    // TODO: Generalize it.
                    if (local_id < get_local_size(0)) {
                        uint sub_group_scan_exc_num;
                        if (local_id < CONF_BR_WARPS) 
                            sub_group_scan_exc_num = s_broadcast[local_id+get_local_size(1)];
                        else
                            sub_group_scan_exc_num = 0;
                        uint local_scan_exc_num = sub_group_scan_inclusive_add_ui(sub_group_scan_exc_num);
                        if (local_id < CONF_BR_WARPS) 
                            s_broadcast[local_id+get_local_size(1)] = local_scan_exc_num;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    
                    scan_exc_num = s_broadcast[get_local_id(1)+get_local_size(1)] + scan_inc_num - num;
                }
                #endif
                s_batch_pos = batch_pos + local_size;
                if (local_id == local_size-1)
                    s_buf_count = buf_count + scan_exc_num + num;
                barrier(CLK_LOCAL_MEM_FENCE);

                // skip if no subtriangles
                if (num) {
                    uint pos = buf_count + scan_exc_num; // my_idx + s_broadcast[get_local_id(1) + 16 - 1];

                    // only write if entire triangle fits
                    if (pos + num <= FW_ARRAY_SIZE(s_tri_buf))
                    {
                        pos += buf_index; // adjust for current start position
                        pos &= FW_ARRAY_SIZE(s_tri_buf)-1; // does the ring operation
                        if (num == 1)
                            s_tri_buf[pos] = tri_idx * 8 + 7; // single triangle
                        else {
                            for (int i=0; i < num; i++) {
                                s_tri_buf[pos] = tri_idx * 8 + i;
                                pos++;
                                pos &= FW_ARRAY_SIZE(s_tri_buf)-1;
                            }
                        }
                    } else if (pos <= FW_ARRAY_SIZE(s_tri_buf))
                    {
                        // this triangle is the first that failed, overwrite total count and triangle count
                        s_batch_pos = batch_pos + local_id;
                        s_buf_count = pos;
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                batch_pos = s_batch_pos;
                buf_count = s_buf_count;
            }

            // make every warp clear its output buffers
            for (int i=get_local_id(0); i < c_num_bins; i += get_local_size(0))
                s_out_mask[get_local_id(1)][i] = 0;

            // choose our triangle
            uint4 tri_data = (uint4){0, 0, 0, 0};
            if (local_id < buf_count)
            {
                uint tri_pos = buf_index + local_id;
                tri_pos &= FW_ARRAY_SIZE(s_tri_buf)-1;

                // find triangle
                int tri_idx = s_tri_buf[tri_pos];
                int data_idx = tri_idx >> 3;
                int subtri_idx = tri_idx & 7;
                if (subtri_idx != 7)
                    data_idx = g_tri_header[data_idx].misc + subtri_idx;

                // read triangle
                #ifdef __IMAGE_SUPPORT__
                tri_data = read_imageui(t_tri_header, data_idx);
                #else
                tri_data = *(((global uint4*) g_tri_header) + data_idx); 
                // * ((global uint4*) &g_tri_header[data_idx]);
                #endif
            }
            
            // setup bounding box and edge functions, and rasterize
            // TODO: Sub group code in this part heavily relies on CUDA warp sync, refactorit to make it more OpenCL friendly.
            int lox, loy, hix, hiy;
            if (local_id < buf_count) {
                int v0x = add_s16lo_s16lo(tri_data.x, c_viewport_width  * (CR_SUBPIXEL_SIZE >> 1));
                int v0y = add_s16hi_s16lo(tri_data.x, c_viewport_height * (CR_SUBPIXEL_SIZE >> 1));
                int d01x = sub_s16lo_s16lo(tri_data.y, tri_data.x);
                int d01y = sub_s16hi_s16hi(tri_data.y, tri_data.x);
                int d02x = sub_s16lo_s16lo(tri_data.z, tri_data.x);
                int d02y = sub_s16hi_s16hi(tri_data.z, tri_data.x);
                int bin_log = CR_BIN_LOG2 + CR_TILE_LOG2 + CR_SUBPIXEL_LOG2;
                lox = add_clamp_0_x((v0x + min_min(d01x, 0, d02x)) >> bin_log, 0, c_width_bins  - 1);
                loy = add_clamp_0_x((v0y + min_min(d01y, 0, d02y)) >> bin_log, 0, c_height_bins - 1);
                hix = add_clamp_0_x((v0x + max_max(d01x, 0, d02x)) >> bin_log, 0, c_width_bins  - 1);
                hiy = add_clamp_0_x((v0y + max_max(d01y, 0, d02y)) >> bin_log, 0, c_height_bins - 1);
                // triangle lies between hix & hiy & lox & loy
                uint bit = 1 << get_local_id(0);
                #ifdef CONF_SUB_GROUP_ENABLED
                uint activemask = sub_group_activemask();
                bool multi = (hix != lox || hiy != loy);
                if (sub_group_masked_any(multi, activemask)) {
                    bool _complex = (hix > lox+1 || hiy > loy+1);
                    if (sub_group_masked_any(_complex, activemask))
                #endif
                    // TODO: No optimization for sub group disable kernel 
                    {
                        int d12x = d02x - d01x, d12y = d02y - d01y;
                        v0x -= lox << bin_log, v0y -= loy << bin_log;

                        int t01 = v0x * d01y - v0y * d01x;
                        int t02 = v0y * d02x - v0x * d02y;
                        int t12 = d01x * d12y - d01y * d12x - t01 - t02;
                        int b01 = add_sub(t01 >> bin_log, max(d01x, 0), min(d01y, 0));
                        int b02 = add_sub(t02 >> bin_log, max(d02y, 0), min(d02x, 0));
                        int b12 = add_sub(t12 >> bin_log, max(d12x, 0), min(d12y, 0));

                        int width = hix - lox + 1;
                        d01x += width * d01y;
                        d02x += width * d02y;
                        d12x += width * d12y;

                        local uchar* currPtr = (local uchar*)&s_out_mask[get_local_id(1)][lox + loy * c_width_bins];
                        local uchar* skipPtr = (local uchar*)&s_out_mask[get_local_id(1)][(hix + 1) + loy * c_width_bins];
                        local uchar* endPtr  = (local uchar*)&s_out_mask[get_local_id(1)][lox + (hiy + 1) * c_width_bins];
                        int stride  = c_width_bins * 4;
                        int ptrYInc = stride - width * 4;

                        do
                        {
                            if (b01 >= 0 && b02 >= 0 && b12 >= 0)
                                atomic_or((local uint*)currPtr, bit);
                            currPtr += 4, b01 -= d01y, b02 += d02y, b12 -= d12y;
                            if (currPtr == skipPtr)
                                currPtr += ptrYInc, b01 += d01x, b02 -= d02x, b12 += d12x, skipPtr += stride;
                        }
                        while (currPtr != endPtr);
                    }
                #ifdef CONF_SUB_GROUP_ENABLED
                    else {
                        int bin_idx = lox + c_width_bins * loy;
                        atomic_or((local uint*)&s_out_mask[get_local_id(1)][bin_idx], bit);
                        if (hix > lox) atomic_or((local uint*)&s_out_mask[get_local_id(1)][bin_idx + 1], bit);
                        if (hiy > loy) atomic_or((local uint*)&s_out_mask[get_local_id(1)][bin_idx + c_width_bins], bit);
                        if (hix > lox && hiy > loy) atomic_or((local uint*)&s_out_mask[get_local_id(1)][bin_idx + c_width_bins + 1], bit);
                    }
                } else {
                    int bin_idx = lox + c_width_bins * loy;
                    bool won;
                    do
                    {
                        // s_broadcast[get_local_id(1) + 16] = bin_idx;
                        // int winner = s_broadcast[get_local_id(1) + 16];
                        int winner = sub_group_masked_broadcast_ui(bin_idx, findLeadingOne(activemask), activemask);
                        won = (bin_idx == winner);
                        //uint inner_activemask = sub_group_activemask();
                        uint mask = sub_group_masked_ballot(won, activemask);
                        activemask &= ~mask;
                        s_out_mask[get_local_id(1)][winner] = mask;
                    } while (!won);
                }
                #endif
            }

            // count per-bin contributions
            s_over_total = 0; // overflow counter

            // ensure that out masks are done
            barrier(CLK_LOCAL_MEM_FENCE);

            int over_index = -1;
            // Sync for non-warp relies on local_liner_size <= c_num_bins 
            // TODO: Generalize
            if (local_id < c_num_bins)
            {
                local uchar* src_ptr = (local uchar*)&s_out_mask[0][local_id];
                local uchar* dst_ptr = (local uchar*)&s_out_count[0][local_id];
                int total = 0;
                for (int i = 0; i < CR_BIN_WARPS; i++)
                {
                    total += popcount(*(local uint*)src_ptr);
                    *(local uint*)dst_ptr = total;
                    src_ptr += (CR_MAXBINS_SQR + 1) * 4;
                    dst_ptr += (CR_MAXBINS_SQR + 1) * 4;
                }

                // overflow => request a new segment
                int ofs = s_out_ofs[local_id];
                bool overflow = ((ofs - 1) >> CR_BIN_SEG_LOG2) != (((ofs - 1) + total) >> CR_BIN_SEG_LOG2);
                #ifdef CONF_SUB_GROUP_ENABLED
                if (overflow)
                #endif
                {
                    uint mask;
                    #ifdef CONF_SUB_GROUP_ENABLED
                    {
                        // TODO: this code relies on 32 bit warps mask
                        mask = sub_group_activemask();
                    }
                    #else 
                    {
                        l_temp[get_local_id(1)] = 0;
                        barrier(CLK_LOCAL_MEM_FENCE); // TODO: Maybe this is not necesary???
                        atomic_or(&l_temp[get_local_id(1)], 1 << get_local_id(0));
                        barrier(CLK_LOCAL_MEM_FENCE);
                        mask = l_temp[get_local_id(1)];
                    }
                    #endif
                    over_index = popcount(mask & getLaneMaskLt());
                    if (over_index == 0)  
                        s_broadcast[get_local_id(1) + 16] = atomic_add((local uint*)&s_over_total, popcount(mask));
                    over_index += s_broadcast[get_local_id(1) + 16];
                    s_over_index[local_id] = over_index;
                }
            }

            // sync after over_total is ready
            barrier(CLK_LOCAL_MEM_FENCE);

            uint over_total = s_over_total;
            uint alloc_base = 0;
            if (over_total > 0)
            {
                // allocate memory
                if (local_id == 0)
                {
                    uint alloc_base = atomic_add(a_num_bin_segs, over_total);
                    s_alloc_base = (alloc_base + over_total <= c_max_bin_segs) ? alloc_base : 0;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                alloc_base = s_alloc_base;

                // did my bin overflow?
                if (over_index != -1)
                {
                    // calculate new segment index
                    int seg_idx = alloc_base + over_index;

                    // add to linked list
                    if (s_out_ofs[local_id] < 0)
                        g_bin_first_seg[(local_id << CR_BIN_STREAMS_LOG2) + get_group_id(0)] = seg_idx;
                    else
                        g_bin_seg_next[(s_out_ofs[local_id] - 1) >> CR_BIN_SEG_LOG2] = seg_idx;

                    // defaults
                    g_bin_seg_next [seg_idx] = -1;
                    g_bin_seg_count[seg_idx] = CR_BIN_SEG_SIZE;
                }
            }

            // concurrent emission -- each warp handles its own triangle
            if (local_id < buf_count)
            {
                int triPos  = (buf_index + local_id) & (FW_ARRAY_SIZE(s_tri_buf) - 1);
                int currBin = lox + loy * c_width_bins;
                int skipBin = (hix + 1) + loy * c_width_bins;
                int endBin  = lox + (hiy + 1) * c_width_bins;
                int binYInc = c_width_bins - (hix - lox + 1);

                // loop over triangle's bins
                do
                {
                    uint outMask = s_out_mask[get_local_id(1)][currBin];
                    if (outMask & (1<<get_local_id(0)))
                    {
                        int idx = popcount(outMask & getLaneMaskLt());
                        if (get_local_id(1) > 0)
                            idx += s_out_count[get_local_id(1)-1][currBin];

                        int base = s_out_ofs[currBin];
                        int free = (-base) & (CR_BIN_SEG_SIZE - 1);
                        if (idx >= free)
                            idx += ((alloc_base + s_over_index[currBin]) << CR_BIN_SEG_LOG2) - free;
                        else
                            idx += base;

                        g_bin_seg_data[idx] = s_tri_buf[triPos];
                    }

                    currBin++;
                    if (currBin == skipBin)
                        currBin += binYInc, skipBin += c_width_bins;
                }
                while (currBin != endBin);
            }

            // wait all triangles to finish, then replace overflown segment offsets
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_id < c_num_bins)
            {
                uint total  = s_out_count[CR_BIN_WARPS - 1][local_id];
                uint oldOfs = s_out_ofs[local_id];
                if (over_index == -1)
                    s_out_ofs[local_id] = oldOfs + total;
                else
                {
                    int addr = oldOfs + total;
                    addr = ((addr - 1) & (CR_BIN_SEG_SIZE - 1)) + 1;
                    addr += (alloc_base + over_index) << CR_BIN_SEG_LOG2;
                    s_out_ofs[local_id] = addr;
                }
                s_out_total[local_id] += total;
            }

            // these triangles are now done
            int count = min(buf_count, CR_BIN_WARPS * 32);
            buf_count -= count;
            buf_index += count;
            buf_index &= FW_ARRAY_SIZE(s_tri_buf)-1;

        } while(buf_count > 0 || batch_pos < batch_end);

        // flush all bins
        if (local_id < c_num_bins) {
            int ofs = s_out_ofs[local_id];
            if (ofs & (CR_BIN_SEG_SIZE-1)) {
                int seg = ofs >> CR_BIN_SEG_LOG2;
                g_bin_seg_count[seg] = ofs & (CR_BIN_SEG_SIZE-1);
                s_out_ofs[local_id] = (ofs - CR_BIN_SEG_SIZE - 1) & -CR_BIN_SEG_SIZE;
            }
        }
    }

    // output totals
    if (local_id < c_num_bins)
        g_bin_total[(local_id << CR_BIN_STREAMS_LOG2) + get_group_id(0)] = s_out_total[local_id];

}