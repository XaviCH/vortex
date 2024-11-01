// Configuration Device
#define BLOCK_BIN_QUEUE_SIZE 10
#define CD_WARP_SIZE 32

// Configuration Rasterizen
#define CR_MAXVIEWPORT_LOG2     11      // ViewportSize / PixelSize.
#define CR_SUBPIXEL_LOG2        4       // PixelSize / SubpixelSize.

#define CR_MAXBINS_LOG2         4       // ViewportSize / BinSize.
#define CR_BIN_LOG2             4       // BinSize / TileSize.
#define CR_TILE_LOG2            3       // TileSize / PixelSize.

#define CR_BIN_WARPS            16
#define CR_BIN_STREAMS_LOG2     4
#define CR_BIN_SEG_LOG2         9       // 32-bit entries.

// Utils
#define CR_MAXVIEWPORT_SIZE     (1 << CR_MAXVIEWPORT_LOG2)
#define CR_SUBPIXEL_SIZE        (1 << CR_SUBPIXEL_LOG2)
#define CR_SUBPIXEL_SQR         (1 << (CR_SUBPIXEL_LOG2 * 2))

#define CR_BIN_SEG_SIZE         (1 << CR_BIN_SEG_LOG2)
#define CR_MAXBINS_SQR          (1 << (CR_MAXBINS_LOG2 * 2))

#define FW_ARRAY_SIZE(X) ((int)(sizeof(X) / sizeof((X)[0])))

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
inline int      add_s16lo_s16lo     (int a, int b)              { int v; asm("vadd.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      add_s16hi_s16lo     (int a, int b)              { int v; asm("vadd.s32.s32.s32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      sub_s16lo_s16lo     (int a, int b)              { int v; asm("vsub.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      sub_s16hi_s16hi     (int a, int b)              { int v; asm("vsub.s32.s32.s32 %0, %1.h1, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
inline int      max_max             (int a, int b, int c)       { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
inline int      min_min             (int a, int b, int c)       { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
inline uint     add_sub             (uint a, uint b, uint c)    { uint v; asm("vsub.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(c), "r"(b)); return v; }
inline int      add_clamp_0_x       (int a, int b, int c)       { int v; asm("vadd.u32.s32.s32.sat.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
inline uint     getLaneMaskLt       (void)                      { uint r; asm("mov.u32 %0, %%lanemask_lt;" : "=r"(r)); return r; }

// inline uint     get_max_sub_group_size() { return 32; }
#else
inline int      add_s16lo_s16lo     (int a, int b)              { return (a & 0xFFFF) + (b & 0xFFFF); }
inline int      add_s16hi_s16lo     (int a, int b)              { return (a >> 16) + (b & 0xFFFF); }
inline int      sub_s16lo_s16lo     (int a, int b)              { return (a & 0xFFFF) - (b & 0xFFFF); }
inline int      sub_s16hi_s16hi     (int a, int b)              { return (a >> 16) - (b >> 16); }
inline int      max_max             (int a, int b, int c)       { return max(a, max(b, c)); }
inline int      min_min             (int a, int b, int c)       { return min(a, min(b, c)); }
inline uint     add_sub             (uint a, uint b, uint c)    { return a+b-c; }
inline int      add_clamp_0_x       (int a, int b, int c)       { return clamp(a+b,0,c); }
inline uint     getLaneMaskLt       (void) {
    // TODO: this might just work with 32 warp
    uint mask;
    for(uint warp = 0; warp < get_max_sub_group_size(); ++warp) mask |= 1 << warp;
    return mask >> (get_max_sub_group_size() - get_local_id(0));
}
#endif

// Bin rasterizer 
kernel void binraster(
    // global atomics
    global   int* g_num_subtris,
    global   int* g_bin_counter, 
    global   int* g_num_bin_segs, 
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
    // input
    private  int    p_max_subtris,
    global   uchar* g_tri_subtris,
    global   CRTriangleHeader*  g_tri_header,
    global   void*  g_tri_data, // unused
    private  uint   p_tri_data_item_sz, // unused
    // output
    private  int    p_max_bin_segs,
    global   int*   g_bin_first_seg,    
    global   int*   g_bin_total,
    global   int*   g_bin_seg_data,
    global   int*   g_bin_seg_next,
    global   int*   g_bin_seg_count,
    // texture
    image1d_t g_t_tri_header,
    sampler_t p_s_tri_header
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
    
    if (*g_num_subtris > p_max_subtris) 
        return;

    // Private space
    int thread_local_id = get_local_id(0) + get_local_id(1) * get_max_sub_group_size(); 
    int batch_pos = 0;

    // first 16 elements of s_broadcast are always zero
    if (thread_local_id < 16)
        s_broadcast[thread_local_id] = 0;

    // initialize output linked lists and offsets
    if (thread_local_id < p_max_subtris)
    {
        g_bin_first_seg[(thread_local_id << CR_BIN_STREAMS_LOG2) + get_global_id(0)] = -1;
        s_out_ofs[thread_local_id] = -CR_BIN_SEG_SIZE;
        s_out_total[thread_local_id] = 0;
    }

    // repeat until done
    while (true) {

        // get batch
        if (thread_local_id == 0)
            s_batch_pos = atomic_add(g_bin_counter,p_bin_batch_sz);
        barrier(CLK_LOCAL_MEM_FENCE);
        batch_pos = s_batch_pos;

        // all batches done?
        if (batch_pos >= p_num_tris)
            break;

        // Private space
        int buf_index = 0;
        int buf_count = 0;
        int batch_end = min(batch_pos + p_bin_batch_sz, p_num_tris);

        // Loop over batch
        do {
            
            while (buf_count < CR_BIN_WARPS*get_max_sub_group_size() && batch_pos < batch_end) {
                
                // Get subtriangle count
                int tri_idx = batch_pos + thread_local_id;
                int num = 0;
                if (tri_idx < batch_end) 
                    num = g_tri_subtris[tri_idx];

                // TODO: cumulative sum of subtriangles within each warp
                uint my_idx = popcount(sub_group_ballot(num & 1) & getLaneMaskLt());
                if (sub_group_any(num > 1))
                {
                    my_idx += popcount(sub_group_ballot(num & 2) & getLaneMaskLt()) * 2;
                    my_idx += popcount(sub_group_ballot(num & 4) & getLaneMaskLt()) * 4;
                }
                s_broadcast[get_local_id(1) + 16] = my_idx + num;
                barrier(CLK_LOCAL_MEM_FENCE);

                // cumulative sum of per-warp subtriangle counts
                if (thread_local_id < CR_BIN_WARPS) {
                    volatile uint* ptr = &s_broadcast[thread_local_id + 16];
                    uint val = *ptr;
                    #if (BIN_WARPS > 1)
                        val += ptr[-1]; *ptr = val;
                    #endif
                    #if (BIN_WARPS > 2)
                        val += ptr[-2]; *ptr = val;
                    #endif
                    #if (BIN_WARPS > 4)
                        val += ptr[-4]; *ptr = val;
                    #endif
                    #if (BIN_WARPS > 8)
                        val += ptr[-8]; *ptr = val;
                    #endif
                    #if (BIN_WARPS > 16)
                        val += ptr[-16]; *ptr = val;
                    #endif

                    // initially assume that we consume everything
                    s_batch_pos = batch_pos + CR_BIN_WARPS * get_max_sub_group_size();
                    s_buf_count = buf_count + val;
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                // skip if no subtriangles
                if (num) {
                    uint pos = buf_count + my_idx + s_broadcast[get_local_id(1) + 16 - 1];

                    // only write if entire triangle fits
                    if (pos + num <= FW_ARRAY_SIZE(s_tri_buf))
                    {
                        pos += buf_index; // adjust for current start position
                        pos &= FW_ARRAY_SIZE(s_tri_buf)-1;
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
                        s_batch_pos = batch_pos + thread_local_id;
                        s_buf_count = pos;
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                batch_pos = s_batch_pos;
                buf_count = s_buf_count;
            }

            // make every warp clear its output buffers
            for (int i=get_local_id(0); i < p_num_bins; i += 32)
                s_out_mask[get_local_id(1)][i] = 0;

            // choose our triangle
            uint4 tri_data = (uint4){0, 0, 0, 0};
            if (thread_local_id < buf_count)
            {
                uint tri_pos = buf_index + thread_local_id;
                tri_pos &= FW_ARRAY_SIZE(s_tri_buf)-1;

                // find triangle
                int tri_idx = s_tri_buf[tri_pos];
                int data_idx = tri_idx >> 3;
                int subtri_idx = tri_idx & 7;
                if (subtri_idx != 7)
                    data_idx = g_tri_header[data_idx].misc + subtri_idx;

                // read triangle

                tri_data = read_imageui(g_t_tri_header, p_s_tri_header, data_idx);
            }

            int lox, loy, hix, hiy;
            if (thread_local_id < buf_count) {
                int v0x = add_s16lo_s16lo(tri_data.x, p_viewport_width  * (CR_SUBPIXEL_SIZE >> 1));
                int v0y = add_s16hi_s16lo(tri_data.x, p_viewport_height * (CR_SUBPIXEL_SIZE >> 1));
                int d01x = sub_s16lo_s16lo(tri_data.y, tri_data.x);
                int d01y = sub_s16hi_s16hi(tri_data.y, tri_data.x);
                int d02x = sub_s16lo_s16lo(tri_data.z, tri_data.x);
                int d02y = sub_s16hi_s16hi(tri_data.z, tri_data.x);
                int bin_log = CR_BIN_LOG2 + CR_TILE_LOG2 + CR_SUBPIXEL_LOG2;
                lox = add_clamp_0_x((v0x + min_min(d01x, 0, d02x)) >> bin_log, 0, p_width_bins  - 1);
                loy = add_clamp_0_x((v0y + min_min(d01y, 0, d02y)) >> bin_log, 0, p_height_bins - 1);
                hix = add_clamp_0_x((v0x + max_max(d01x, 0, d02x)) >> bin_log, 0, p_width_bins  - 1);
                hiy = add_clamp_0_x((v0y + max_max(d01y, 0, d02y)) >> bin_log, 0, p_height_bins - 1);

                uint bit = 1 << get_local_id(0);
                bool multi = (hix != lox || hiy != loy);
                if (!sub_group_any(multi))
                {
                    int bin_idx = lox + p_width_bins * loy;
                    bool won;
                    do
                    {
                        s_broadcast[get_local_id(1) + 16] = bin_idx;
                        int winner = s_broadcast[get_local_id(1) + 16];
                        won = (bin_idx == winner);
                        uint mask = sub_group_ballot(won);
                        s_out_mask[get_local_id(1)][winner] = mask;
                    } while (!won);
                } else
                {
                    bool complex = (hix > lox+1 || hiy > loy+1);
                    if (!sub_group_any(complex))
                    {
                        int bin_idx = lox + p_width_bins * loy;
                        atomic_or((local uint*)&s_out_mask[get_local_id(1)][bin_idx], bit);
                        if (hix > lox) atomic_or((local uint*)&s_out_mask[get_local_id(1)][bin_idx + 1], bit);
                        if (hiy > loy) atomic_or((local uint*)&s_out_mask[get_local_id(1)][bin_idx + p_width_bins], bit);
                        if (hix > lox && hiy > loy) atomic_or((local uint*)&s_out_mask[get_local_id(1)][bin_idx + p_width_bins + 1], bit);
                    } else
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

                        local uchar* currPtr = (local uchar*)&s_out_mask[get_local_id(1)][lox + loy * p_width_bins];
                        local uchar* skipPtr = (local uchar*)&s_out_mask[get_local_id(1)][(hix + 1) + loy * p_width_bins];
                        local uchar* endPtr  = (local uchar*)&s_out_mask[get_local_id(1)][lox + (hiy + 1) * p_width_bins];
                        int stride  = p_width_bins * 4;
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
                }
            }

            // count per-bin contributions
            s_over_total = 0; // overflow counter

            // ensure that out masks are done
            barrier(CLK_LOCAL_MEM_FENCE);

            int over_index = -1;
            if (thread_local_id < p_num_bins)
            {
                local uchar* src_ptr = (local uchar*)&s_out_mask[0][thread_local_id];
                local uchar* dst_ptr = (local uchar*)&s_out_count[0][thread_local_id];
                int total = 0;
                for (int i = 0; i < CR_BIN_WARPS; i++)
                {
                    total += popcount(*(local uint*)src_ptr);
                    *(local uint*)dst_ptr = total;
                    src_ptr += (CR_MAXBINS_SQR + 1) * 4;
                    dst_ptr += (CR_MAXBINS_SQR + 1) * 4;
                }

                // overflow => request a new segment
                int ofs = s_out_ofs[thread_local_id];
                if (((ofs - 1) >> CR_BIN_SEG_LOG2) != (((ofs - 1) + total) >> CR_BIN_SEG_LOG2))
                {
                    uint mask = sub_group_ballot(true);
                    over_index = popcount(mask & getLaneMaskLt());
                    if (over_index == 0)
                        s_broadcast[get_local_id(1) + 16] = atomic_add((local uint*)&s_over_total, popcount(mask));
                    over_index += s_broadcast[get_local_id(1) + 16];
                    s_over_index[thread_local_id] = over_index;
                }
            }

            // sync after over_total is ready
            barrier(CLK_LOCAL_MEM_FENCE);

            uint over_total = s_over_total;
            uint alloc_base = 0;
            if (over_total > 0)
            {
                // allocate memory
                if (thread_local_id == 0)
                {
                    uint alloc_base = atomic_add(g_num_bin_segs, over_total);
                    s_alloc_base = (alloc_base + over_total <= p_max_bin_segs) ? alloc_base : 0;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                alloc_base = s_alloc_base;

                // did my bin overflow?
                if (over_index != -1)
                {
                    // calculate new segment index
                    int seg_idx = alloc_base + over_index;

                    // add to linked list
                    if (s_out_ofs[thread_local_id] < 0)
                        g_bin_first_seg[(thread_local_id << CR_BIN_STREAMS_LOG2) + get_global_id(0)] = seg_idx;
                    else
                        g_bin_seg_next[(s_out_ofs[thread_local_id] - 1) >> CR_BIN_SEG_LOG2] = seg_idx;

                    // defaults
                    g_bin_seg_next [seg_idx] = -1;
                    g_bin_seg_count[seg_idx] = CR_BIN_SEG_SIZE;
                }
            }

            // concurrent emission -- each warp handles its own triangle
            if (thread_local_id < buf_count)
            {
                int triPos  = (buf_index + thread_local_id) & (FW_ARRAY_SIZE(s_tri_buf) - 1);
                int currBin = lox + loy * p_width_bins;
                int skipBin = (hix + 1) + loy * p_width_bins;
                int endBin  = lox + (hiy + 1) * p_width_bins;
                int binYInc = p_width_bins - (hix - lox + 1);

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
                        currBin += binYInc, skipBin += p_width_bins;
                }
                while (currBin != endBin);
            }

            // wait all triangles to finish, then replace overflown segment offsets
            barrier(CLK_LOCAL_MEM_FENCE);
            if (thread_local_id < p_num_bins)
            {
                uint total  = s_out_count[CR_BIN_WARPS - 1][thread_local_id];
                uint oldOfs = s_out_ofs[thread_local_id];
                if (over_index == -1)
                    s_out_ofs[thread_local_id] = oldOfs + total;
                else
                {
                    int addr = oldOfs + total;
                    addr = ((addr - 1) & (CR_BIN_SEG_SIZE - 1)) + 1;
                    addr += (alloc_base + over_index) << CR_BIN_SEG_LOG2;
                    s_out_ofs[thread_local_id] = addr;
                }
                s_out_total[thread_local_id] += total;
            }

            // these triangles are now done
            int count = min(buf_count, CR_BIN_WARPS * 32);
            buf_count -= count;
            buf_index += count;
            buf_index &= FW_ARRAY_SIZE(s_tri_buf)-1;

        } while(buf_count > 0 || batch_pos < batch_end);

        // flush all bins
        if (thread_local_id < p_num_bins) {
            int ofs = s_out_ofs[thread_local_id];
            if (ofs & (CR_BIN_SEG_SIZE-1)) {
                int seg = ofs >> CR_BIN_SEG_LOG2;
                g_bin_seg_count[seg] = ofs & (CR_BIN_SEG_SIZE-1);
                s_out_ofs[thread_local_id] = (ofs - CR_BIN_SEG_SIZE - 1) & -CR_BIN_SEG_SIZE;
            }
        }
    }

    // output totals
    if (thread_local_id < p_num_bins)
        g_bin_total[(thread_local_id << CR_BIN_STREAMS_LOG2) + get_global_id(0)] = s_out_total[thread_local_id];

}