#include "common/headers.cl"

inline void snapTriangle(
    float4 v0, float4 v1, float4 v2, int2* p0, int2* p1, int2* p2, 
    float3* rcpW, int2* lo, int2* hi,
    int c_viewport_width, int c_viewport_height 
    )
{
    float viewScaleX = (float)(c_viewport_width  << (CR_SUBPIXEL_LOG2 - 1));
    float viewScaleY = (float)(c_viewport_height << (CR_SUBPIXEL_LOG2 - 1));
    *rcpW = (float3)(1.0f / v0.w, 1.0f / v1.w, 1.0f / v2.w);
    *p0 = (int2)(f32_to_s32_sat(v0.x * rcpW->x * viewScaleX), f32_to_s32_sat(v0.y * rcpW->x * viewScaleY));
    *p1 = (int2)(f32_to_s32_sat(v1.x * rcpW->y * viewScaleX), f32_to_s32_sat(v1.y * rcpW->y * viewScaleY));
    *p2 = (int2)(f32_to_s32_sat(v2.x * rcpW->z * viewScaleX), f32_to_s32_sat(v2.y * rcpW->z * viewScaleY));
    *lo = (int2)(min_min(p0->x, p1->x, p2->x), min_min(p0->y, p1->y, p2->y));
    *hi = (int2)(max_max(p0->x, p1->x, p2->x), max_max(p0->y, p1->y, p2->y));
}

//------------------------------------------------------------------------
// 0 = Visible.
// 1 = Backfacing.
// 2 = Between pixels.

inline int prepareTriangle(
    int2 p0, int2 p1, int2 p2, int2 lo, int2 hi,
    int2* d1, int2* d2, int* area, 
    int c_viewport_width, int c_viewport_height,
    int c_samples_log2
) {
    // Backfacing or degenerate => cull.

    *d1 = (int2)(p1.x - p0.x, p1.y - p0.y);
    *d2 = (int2)(p2.x - p0.x, p2.y - p0.y);
    *area = d1->x * d2->y - d1->y * d2->x;

    if (*area <= 0)
        return 1; // Backfacing.

    // AABB falls between samples => cull.

    int sampleSize = 1 << (CR_SUBPIXEL_LOG2 - c_samples_log2);
    int biasX = (c_viewport_width  << (CR_SUBPIXEL_LOG2 - 1)) - (sampleSize >> 1);
    int biasY = (c_viewport_height << (CR_SUBPIXEL_LOG2 - 1)) - (sampleSize >> 1);
    int lox = (int)add_add(lo.x, sampleSize - 1, biasX) & -sampleSize;
    int loy = (int)add_add(lo.y, sampleSize - 1, biasY) & -sampleSize;
    int hix = (hi.x + biasX) & -sampleSize;
    int hiy = (hi.y + biasY) & -sampleSize;

    if (lox > hix || loy > hiy)
        return 2; // Between pixels.

    // AABB covers 1 or 2 samples => cull if they are not covered.

    int diff = add_sub(hix, hiy, lox) - loy;
    if (diff <= sampleSize)
    {
        int2 t0 = (int2)(add_sub(p0.x, biasX, lox), add_sub(p0.y, biasY, loy));
        int2 t1 = (int2)(add_sub(p1.x, biasX, lox), add_sub(p1.y, biasY, loy));
        int2 t2 = (int2)(add_sub(p2.x, biasX, lox), add_sub(p2.y, biasY, loy));
        int e0 = t0.x * t1.y - t0.y * t1.x;
        int e1 = t1.x * t2.y - t1.y * t2.x;
        int e2 = t2.x * t0.y - t2.y * t0.x;

        if (e0 < 0 || e1 < 0 || e2 < 0)
        {
            if (diff == 0)
                return 2; // Between pixels.

            t0 = (int2)(add_sub(p0.x, biasX, hix), add_sub(p0.y, biasY, hiy));
            t1 = (int2)(add_sub(p1.x, biasX, hix), add_sub(p1.y, biasY, hiy));
            t2 = (int2)(add_sub(p2.x, biasX, hix), add_sub(p2.y, biasY, hiy));
            e0 = t0.x * t1.y - t0.y * t1.x;
            e1 = t1.x * t2.y - t1.y * t2.x;
            e2 = t2.x * t0.y - t2.y * t0.x;

            if (e0 < 0 || e1 < 0 || e2 < 0)
                return 2; // Between pixels.
        }
    }

    // Otherwise => proceed to output the triangle.

    return 0; // Visible.
}

//------------------------------------------------------------------------

inline void setupTriangle(
    CRTriangleHeader* th, CRTriangleData* td, int3 vidx,
    float4 v0, float4 v1, float4 v2,
    float2 b0, float2 b1, float2 b2,
    int2 p0, int2 p1, int2 p2, float3 rcpW,
    int2 d1, int2 d2, int area,

    int c_viewport_width, int c_viewport_height,
    int c_samples_log2, uint c_render_mode_flags
    )
{
    uint dep = 0;
    float areaRcp;
    int2 wv0;

    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0 ||
        (c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_LERP) != 0)
    {
        areaRcp = 1.0f / (float)area;
        wv0.x = p0.x + (c_viewport_width  << (CR_SUBPIXEL_LOG2 - 1));
        wv0.y = p0.y + (c_viewport_height << (CR_SUBPIXEL_LOG2 - 1));
    }

    // Setup depth plane equation.

    uint3 zpleq;
    uint zmin = 0, zslope = 0;
    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0)
    {
        float zcoef = (float)(CR_DEPTH_MAX - CR_DEPTH_MIN) * 0.5f;
        float zbias = (float)(CR_DEPTH_MAX + CR_DEPTH_MIN) * 0.5f;
        float3 zvert;
        zvert.x = (v0.z * zcoef) * rcpW.x + zbias;
        zvert.y = (v1.z * zcoef) * rcpW.y + zbias;
        zvert.z = (v2.z * zcoef) * rcpW.z + zbias;

        int2 zv0;
        zv0.x = wv0.x - (1 << (CR_SUBPIXEL_LOG2 - c_samples_log2 - 1));
        zv0.y = wv0.y - (1 << (CR_SUBPIXEL_LOG2 - c_samples_log2 - 1));
        zpleq = setupPleq(zvert, zv0, d1, d2, areaRcp, c_samples_log2);

        zmin = f32_to_u32_sat(min(min(zvert.x, zvert.y), zvert.z) - (float)CR_LERP_ERROR(c_samples_log2));
        if (c_samples_log2 != 0)
        {
            uint tmp = abs((int)zpleq.x) + abs(max((int)zpleq.y, -FW_S32_MAX));
            zslope = tmp << max(c_samples_log2 - 1, 0);
            if ((zslope >> max(c_samples_log2 - 1, 0)) != tmp)
                zslope = FW_U32_MAX;
        }

        dep += zpleq.x + zpleq.y + zpleq.z + zmin + zslope;
    }
    
    // Setup lerp plane equations.
    
    uint3 wpleq, upleq, vpleq;
    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_LERP) != 0)
    {
        float wcoef = min(min(v0.w, v1.w), v2.w) * (float)CR_BARY_MAX;
        float3 wvert = (float3)(wcoef * rcpW.x, wcoef * rcpW.y, wcoef * rcpW.z);
        float3 uvert = (float3)(b0.x * wvert.x, b1.x * wvert.y, b2.x * wvert.z);
        float3 vvert = (float3)(b0.y * wvert.x, b1.y * wvert.y, b2.y * wvert.z);

        wpleq = setupPleq(wvert, wv0, d1, d2, areaRcp, c_samples_log2 + 1);
        upleq = setupPleq(uvert, wv0, d1, d2, areaRcp, c_samples_log2 + 1);
        vpleq = setupPleq(vvert, wv0, d1, d2, areaRcp, c_samples_log2 + 1);
        dep += wpleq.x + wpleq.y + wpleq.z + upleq.x + upleq.y + upleq.z;
    }
    
    // Write CRTriangleData.

    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0) {
        // td->zx = zpleq.x;
        // td->zy = zpleq.y;
        // td->zb = zpleq.z;
        // td->zslope = zslope;
        // TODO: enable vector unit
        *(uint4*)&td->zx = (uint4)(zpleq.x, zpleq.y, zpleq.z, zslope);
    }
    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_LERP) == 0) {
        // td->vb = 0;
        // td->vi0 = vidx.x;
        // td->vi1 = vidx.y;
        // td->vi2 = vidx.z;
        // TODO: enable vector unit
        *(uint4*)&td->vb = (uint4)(0, vidx.x, vidx.y, vidx.z);
    }
    else
    {
        //return; // TODO: OutofBound access ??? Idk
        // td->wx = wpleq.x;
        // td->wy = wpleq.y;
        // td->wb = wpleq.z;
        // td->ux = upleq.x;
        // td->uy = upleq.y;
        // td->ub = upleq.z;
        // td->vx = vpleq.x;
        // td->vy = vpleq.y;
        // td->vb = vpleq.z;
        // td->vi0 = vidx.x;
        // td->vi1 = vidx.y;
        // td->vi2 = vidx.z;
        // TODO: enable vector unit
        *(uint4*)&td->wx = (uint4)(wpleq.x, wpleq.y, wpleq.z, upleq.x);
        *(uint4*)&td->uy = (uint4)(upleq.y, upleq.z, vpleq.x, vpleq.y);
        *(uint4*)&td->vb = (uint4)(vpleq.z, vidx.x, vidx.y, vidx.z);
    }

    // Determine flipbits.

    uint f01 = cover8x8_selectFlips(d1.x, d1.y);
    uint f12 = cover8x8_selectFlips(d2.x - d1.x, d2.y - d1.y);
    uint f20 = cover8x8_selectFlips(-d2.x, -d2.y);

    // Write CRTriangleHeader.

    *(uint4*)th = (uint4)(
        prmt(p0.x, p0.y, 0x5410),
        prmt(p1.x, p1.y, 0x5410),
        prmt(p2.x, p2.y, 0x5410),
        (zmin & 0xfffff000u) | (f01 << 6) | (f12 << 2) | (f20 >> 2));

}

//------------------------------------------------------------------------

/**
    Triangle setup for index buffer
 */
//template <class VertexClass>
kernel
//__attribute__((reqd_work_group_size(32, 2, 1)))
void triangle_setup(
    global int* a_num_subtris,

    global const int* g_index_buffer, // maybe fit in constant memory
    global CRTriangleHeader* g_tri_header, 
    global CRTriangleData* g_tri_data,
    global uchar* g_tri_subtris,

    // #ifdef __IMAGE_SUPPORT__
    read_only image1d_buffer_t t_vertex_buffer,

    const int c_num_tris, 
    const int c_max_subtris,
    const int c_samples_log2, 
    const uint c_render_mode_flags,
    const int c_viewport_height,
    const int c_vertex_size,
    const int c_viewport_width
    
    // ifdef DEBUG
    //,global int* debug_return
)
{
    local float s_bary[CR_SETUP_WARPS * 32][18]; // TODO: WARP DEPENDANT
    local float* bary = s_bary[get_local_id(0) + get_local_id(1) * 32];

    int2 p0, p1, p2, lo, hi, d1, d2;
    float3 rcpW;
    int area;

    // Pick a task.

    int taskIdx = get_local_id(0) + 32 * (get_local_id(1) + CR_SETUP_WARPS * (get_group_id(0) + get_num_groups(0) * get_group_id(1)));
    // debug_return[taskIdx] = 0;
    if (taskIdx >= c_num_tris)
        return;

    // Read vertices.
    // debug_return[taskIdx] = 1;

    global const int* index_buffer = g_index_buffer + taskIdx*3;
    int3 vidx = {*index_buffer, *(index_buffer+1), *(index_buffer+2)}; // int3 is an int4 in opencl check
    int stride = c_vertex_size / sizeof(float4);
    float4 v0 = read_imagef(t_vertex_buffer, vidx.x * stride); // gl_Position must be in first position
    float4 v1 = read_imagef(t_vertex_buffer, vidx.y * stride);
    float4 v2 = read_imagef(t_vertex_buffer, vidx.z * stride);

    // Outside view frustum => cull.

    if (v0.w < fabs(v0.x) | v0.w < fabs(v0.y) | v0.w < fabs(v0.z))
    {
        if ((v0.w < +v0.x & v1.w < +v1.x & v2.w < +v2.x) |
            (v0.w < -v0.x & v1.w < -v1.x & v2.w < -v2.x) |
            (v0.w < +v0.y & v1.w < +v1.y & v2.w < +v2.y) |
            (v0.w < -v0.y & v1.w < -v1.y & v2.w < -v2.y) |
            (v0.w < +v0.z & v1.w < +v1.z & v2.w < +v2.z) |
            (v0.w < -v0.z & v1.w < -v1.z & v2.w < -v2.z))
        {
            g_tri_subtris[taskIdx] = 0; 
            return;
        }
    }

    // Inside depth range => try to snap vertices.
    // debug_return[taskIdx] = 2;

    if (v0.w >= fabs(v0.z) & v1.w >= fabs(v1.z) & v2.w >= fabs(v2.z))
    {
        // Inside S16 range and small enough => fast path.
        // Note: aabbLimit comes from the fact that cover8x8
        // does not support guardband with maximal viewport.

        snapTriangle(v0, v1, v2, &p0, &p1, &p2, &rcpW, &lo, &hi, 
            c_viewport_width, c_viewport_height);
        int loxy = min(lo.x, lo.y);
        int hixy = max(hi.x, hi.y);
        int aabbLimit = (1 << (CR_MAXVIEWPORT_LOG2 + CR_SUBPIXEL_LOG2)) - 1;

        if (loxy >= -32768 && hixy <= 32767 && hixy - loxy <= aabbLimit)
        {
            int res = prepareTriangle(p0, p1, p2, lo, hi, &d1, &d2, &area, c_viewport_width, c_viewport_height, c_samples_log2);
            g_tri_subtris[taskIdx] = (res == 0) ? 1 : 0;

            if (res == 0)
                setupTriangle(
                    &g_tri_header[taskIdx], &g_tri_data[taskIdx], vidx,
                    v0, v1, v2,
                    (float2)(0.0f, 0.0f),
                    (float2)(1.0f, 0.0f),
                    (float2)(0.0f, 1.0f),
                    p0, p1, p2, rcpW,
                    d1, d2, area,
                    c_viewport_width, c_viewport_height,
                    c_samples_log2, c_render_mode_flags);

            return;
        }
    }

    // Clip to view frustum.
    // debug_return[taskIdx] = 3;

    float4 ov0 = v0;
    float4 od1 = (float4)(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, v1.w - v0.w);
    float4 od2 = (float4)(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, v2.w - v0.w);
    int numVerts = clipTriangleWithFrustum(bary, (float*) &ov0, (float*) &v1, (float*) &v2, (float*) &od1, (float*) &od2);

    // Count non-culled subtriangles.

    v0.x = ov0.x + od1.x * bary[0] + od2.x * bary[1];
    v0.y = ov0.y + od1.y * bary[0] + od2.y * bary[1];
    v0.z = ov0.z + od1.z * bary[0] + od2.z * bary[1];
    v0.w = ov0.w + od1.w * bary[0] + od2.w * bary[1];
    v1.x = ov0.x + od1.x * bary[2] + od2.x * bary[3];
    v1.y = ov0.y + od1.y * bary[2] + od2.y * bary[3];
    v1.z = ov0.z + od1.z * bary[2] + od2.z * bary[3];
    v1.w = ov0.w + od1.w * bary[2] + od2.w * bary[3];
    float4 tv1 = v1;

    int numSubtris = 0;
    for (int i = 2; i < numVerts; i++)
    {
        v2.x = ov0.x + od1.x * bary[i * 2 + 0] + od2.x * bary[i * 2 + 1];
        v2.y = ov0.y + od1.y * bary[i * 2 + 0] + od2.y * bary[i * 2 + 1];
        v2.z = ov0.z + od1.z * bary[i * 2 + 0] + od2.z * bary[i * 2 + 1];
        v2.w = ov0.w + od1.w * bary[i * 2 + 0] + od2.w * bary[i * 2 + 1];

        snapTriangle(v0, v1, v2, &p0, &p1, &p2, &rcpW, &lo, &hi, c_viewport_width, c_viewport_height);
        if (prepareTriangle(p0, p1, p2, lo, hi, &d1, &d2, &area, c_viewport_width, c_viewport_height, c_samples_log2) == 0)
            numSubtris++;

        v1 = v2;
    }

    g_tri_subtris[taskIdx] = numSubtris;

    // Multiple subtriangles => allocate.

    int subtriBase = taskIdx;
    if (numSubtris > 1)
    {
        subtriBase = atomic_add(a_num_subtris, numSubtris);
        g_tri_header[taskIdx].misc = subtriBase;
        if (subtriBase + numSubtris > c_max_subtris)
            numVerts = 0;
    }

    // Setup subtriangles.

    v1 = tv1;
    for (int i = 2; i < numVerts; i++)
    {
        v2.x = ov0.x + od1.x * bary[i * 2 + 0] + od2.x * bary[i * 2 + 1];
        v2.y = ov0.y + od1.y * bary[i * 2 + 0] + od2.y * bary[i * 2 + 1];
        v2.z = ov0.z + od1.z * bary[i * 2 + 0] + od2.z * bary[i * 2 + 1];
        v2.w = ov0.w + od1.w * bary[i * 2 + 0] + od2.w * bary[i * 2 + 1];

        snapTriangle(v0, v1, v2, &p0, &p1, &p2, &rcpW, &lo, &hi, c_viewport_width, c_viewport_height);
        if (prepareTriangle(p0, p1, p2, lo, hi, &d1, &d2, &area, c_viewport_width, c_viewport_height, c_samples_log2) == 0)
        {

            setupTriangle(
                &g_tri_header[subtriBase], &g_tri_data[subtriBase], vidx,
                v0, v1, v2,
                (float2)(bary[0], bary[1]),
                (float2)(bary[i * 2 - 2], bary[i * 2 - 1]),
                (float2)(bary[i * 2 + 0], bary[i * 2 + 1]),
                p0, p1, p2, rcpW,
                d1, d2, area,
                c_viewport_width, c_viewport_height, c_samples_log2, c_render_mode_flags);

            subtriBase++;
        }

        v1 = v2;
    }

}

//------------------------------------------------------------------------
