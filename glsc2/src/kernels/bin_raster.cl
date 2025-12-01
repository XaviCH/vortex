#ifdef __COMPILER_RELATIVE_PATH__
#include "common.cl"
#else
#include "glsc2/src/kernels/common.cl"
#endif

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


/**
    Processed triangles are going to be stored in bins, depending if they fall inside.
    Each CTA has their full set of bins. Each CTA would batch a different set of triangles
    to test each bin. 
 */
kernel
__attribute__((reqd_work_group_size(DEVICE_SUB_GROUP_THREADS, DEVICE_BIN_SUB_GROUPS, 1)))
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
    
    #ifdef CONF_BIN_IMAGE_ENABLED
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
    //, global local_bin_raster_debug_t* g_l_debug // sz == DEVICE_BIN_SUB_GROUPS
    //, global global_bin_raster_debug_t* g_g_debug // 
    #endif
) {

    // Local space
    // TODO: remove limitations on maxbins
    local volatile int  s_out_ofs     [CR_MAXBINS_SQR];
    local volatile int  s_out_total   [CR_MAXBINS_SQR];
    local volatile int  s_over_index  [CR_MAXBINS_SQR];
    // TODO: s_out_mask relies on 32 sub group size, change for a more OpenCL friendly code.
    local volatile int  s_out_mask    [DEVICE_BIN_SUB_GROUPS][CR_MAXBINS_SQR + 1];        // +1 to avoid bank collisions
    local volatile int  s_out_count   [DEVICE_BIN_SUB_GROUPS][CR_MAXBINS_SQR + 1];        // +1 to avoid bank collisions
    local volatile int  s_tri_buf     [DEVICE_BIN_SUB_GROUPS*DEVICE_SUB_GROUP_THREADS*4];  // triangle ring buffer

    local volatile uint s_batch_pos;
    local volatile uint s_buf_count;
    local volatile uint s_over_total;
    local volatile uint s_alloc_base;

    #ifdef CONF_BIN_SUB_GROUP_ENABLED
    local volatile uint l_temp [DEVICE_BIN_SUB_GROUPS];
    #else 
    local volatile uint l_temp [DEVICE_SUB_GROUP_THREADS*DEVICE_BIN_SUB_GROUPS];
    #endif

    if (*a_num_subtris > c_max_subtris) 
        return;

    // Private space
    int local_id = get_local_linear_id();
    uint local_size = get_local_linear_size();
    int batch_pos = 0;

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

        // loop over batch
        do {
            
            // fill the s_tri_buf with tri_idx[31:3] + sub_tri_idx[2:0]  
            while (buf_count < local_size && batch_pos < batch_end) {
                
                // get subtriangle count
                int tri_idx = batch_pos + local_id;
                int num = 0;
                if (tri_idx < batch_end)
                    num = g_tri_subtris[tri_idx];

                // cumulative sum of subtriangles within each subgroup
                uint scan_exc_num = local_scan_inclusive_add_ui(num, l_temp) - num;

                s_batch_pos = batch_pos + local_size;
                if (local_id == local_size-1)
                    s_buf_count = buf_count + scan_exc_num + num;
                barrier(CLK_LOCAL_MEM_FENCE);

                // skip if no subtriangles
                if (num) {
                    uint pos = buf_count + scan_exc_num;

                    // only write if entire triangle fits
                    if (pos + num <= FW_ARRAY_SIZE(s_tri_buf))
                    {
                        pos += buf_index; // adjust for current start position
                        pos &= FW_ARRAY_SIZE(s_tri_buf)-1; // does the ring operation
                        if (num == 1)
                            s_tri_buf[pos] = (tri_idx << 3) | 0x7u; // single triangle
                        else {
                            for (int i=0; i < num; i++) {
                                s_tri_buf[pos] = (tri_idx << 3) | i;
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

            // TODO: maybe move it to another part, here just creates noise
            // clear its output buffers
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
                #ifdef CONF_BIN_IMAGE_ENABLED
                tri_data = read_imageui(t_tri_header, data_idx);
                #else
                tri_data = *(((global uint4*) g_tri_header) + data_idx); 
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

                #ifdef CONF_BIN_SUB_GROUP_ENABLED
                uint activemask = sub_group_activemask();
                bool multi = (hix != lox || hiy != loy);
                if (sub_group_masked_any(multi, activemask)) {
                    bool _complex = (hix > lox+1 || hiy > loy+1);
                    if (sub_group_masked_any(_complex, activemask))
                #endif
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
                #ifdef CONF_BIN_SUB_GROUP_ENABLED
                    else {
                        int bin_idx = lox + c_width_bins * loy;
                        atomic_or((local uint*)&s_out_mask[get_local_id(1)][bin_idx], bit);
                        if (hix > lox) atomic_or((local uint*)&s_out_mask[get_local_id(1)][bin_idx + 1], bit);
                        if (hiy > loy) atomic_or((local uint*)&s_out_mask[get_local_id(1)][bin_idx + c_width_bins], bit);
                        if (hix > lox && hiy > loy) atomic_or((local uint*)&s_out_mask[get_local_id(1)][bin_idx + c_width_bins + 1], bit);
                    }
                } else {
                    int bin_idx = lox + c_width_bins * loy;
                    int tmp_idx;
                    do {
                        int tmp_idx = sub_group_broadcast_first_ui(bin_idx);
                        s_out_mask[get_sub_group_local_id()][tmp_idx] = sub_group_ballot(bin_idx == tmp_idx);
                    } while(bin_idx != tmp_idx);
                    
                }
                #endif
            }

            // count per-bin contributions
            s_over_total = 0; // overflow counter

            // ensure that out masks are done
            barrier(CLK_LOCAL_MEM_FENCE);

            int over_index = -1;
            {
                int total = 0, ofs;
                bool overflow = 0;

                if (local_id < c_num_bins) {
                    for (int sub_group = 0; sub_group < get_num_sub_groups(); ++sub_group) {
                        total += popcount(s_out_mask[sub_group][local_id]);
                        s_out_count[sub_group][local_id] = total;
                    }
                
                    // overflow => request a new segment
                    ofs = s_out_ofs[local_id];
                    // TODO: Checkout this, maybe can be write to be more understandable
                    overflow = ((ofs - 1) >> CR_BIN_SEG_LOG2) != (((ofs - 1) + total) >> CR_BIN_SEG_LOG2);
                }

                // exc cumm scan of all overflows in the work group 
                uint over_total;
                uint exc_scan_over_index;
                #ifdef CONF_BIN_SUB_GROUP_ENABLED
                {
                    exc_scan_over_index = popcount(sub_group_ballot(overflow) & getLaneMaskLt());
                    if (get_sub_group_local_id() == get_sub_group_size()-1)
                        over_total = atomic_add(&s_over_total, exc_scan_over_index + overflow);
                    over_total = sub_group_broadcast_ui(over_total, get_sub_group_size()-1);
                }
                #else
                {
                    over_total = s_over_total;
                    exc_scan_over_index = local_scan_inclusive_add_ui(overflow, l_temp) - overflow;
                    if (get_local_linear_id()-1 == c_num_bins)
                        s_over_total = exc_scan_over_index + overflow;
                }
                #endif

                if (overflow) {
                    over_index = exc_scan_over_index + over_total;
                    s_over_index[local_id] = over_index;
                }
                
            }

            // sync after over_total is ready
            barrier(CLK_LOCAL_MEM_FENCE);

            uint over_total = s_over_total;
            uint alloc_base = 0;
            if (over_total > 0)
            {
                // allocate memory if fits
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
                    uint out_mask = s_out_mask[get_local_id(1)][currBin];
                    if (out_mask & (1<<get_local_id(0)))
                    {
                        int idx = popcount(out_mask & getLaneMaskLt());
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
                uint total  = s_out_count[DEVICE_BIN_SUB_GROUPS - 1][local_id];
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
            int count = min(buf_count, DEVICE_BIN_SUB_GROUPS * 32);
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