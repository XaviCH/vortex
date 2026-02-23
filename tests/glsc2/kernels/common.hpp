#pragma once
#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <string.h>

#include <glm/glm.hpp>

#include <constants.device.h>

typedef struct CRTriangleHeader
{
    short v0x;    // Subpixels relative to viewport center. Valid if triSubtris = 1.
    short v0y;
    short v1x;
    short v1y;
    short v2x;
    short v2y;

    uint misc;   // triSubtris=1: (zmin:20, f01:4, f12:4, f20:4), triSubtris>=2: (subtriBase)

    bool operator!=(CRTriangleHeader& h) {
        return memcmp(this, &h, sizeof(CRTriangleHeader));
    }

};

typedef struct CRTriangleData
{
    uint zx;     // zx * sampleX + zy * sampleY + zb = lerp(CR_DEPTH_MIN, CR_DEPTH_MAX, (clipZ / clipW + 1) / 2)
    uint zy;
    uint zb;
    uint zslope; // (abs(zx) + abs(zy)) * (samplesPerPixel / 2)

    int wx;     // wx * (sampleX * 2 + 1) + wy * (sampleY * 2 + 1) + wb = minClipW / clipW * CR_BARY_MAX
    int wy;
    int wb;

    int ux;     // ux * (sampleX * 2 + 1) + uy * (sampleY * 2 + 1) + ub = baryU * minClipW / clipW * CR_BARY_MAX
    int uy;
    int ub;

    int vx;     // vx * (sampleX * 2 + 1) + vy * (sampleY * 2 + 1) + vb = baryV * minClipW / clipW * CR_BARY_MAX
    int vy;
    int vb;

    uint vi0;    // Vertex indices.
    uint vi1;
    uint vi2;

    bool operator!=(CRTriangleData& d) {
        return memcmp(this, &d, sizeof(CRTriangleData));
    }
}; 

template <class A, class B> 
A lerp(const A& a, const A& b, const B& t) { return (A)(a * ((B)1 - t) + b * t); }

glm::ivec3 setupPleq(const glm::vec3& values, const glm::ivec2& v0, const glm::ivec2& d1, const glm::ivec2& d2, int32_t area, int samplesLog2)
{
    double t0 = (double)values.x;
    double t1 = (double)values.y - t0;
    double t2 = (double)values.z - t0;
    double xc = (t1 * (double)d2.y - t2 * (double)d1.y) / (double)area;
    double yc = (t2 * (double)d1.x - t1 * (double)d2.x) / (double)area;

    glm::ivec2 center = (v0 * 2 + std::min(d1.x, std::min(d2.x, 0)) + std::max(d1.x, std::max(d2.x, 0))) >> (CR_SUBPIXEL_LOG2 - samplesLog2 + 1);
    glm::ivec2 vc = v0 - (center << (CR_SUBPIXEL_LOG2 - samplesLog2));

    glm::ivec3 pleq;
    pleq.x = (uint32_t)(int64_t)std::floor(xc * std::exp2(CR_SUBPIXEL_LOG2 - samplesLog2) + 0.5);
    pleq.y = (uint32_t)(int64_t)std::floor(yc * std::exp2(CR_SUBPIXEL_LOG2 - samplesLog2) + 0.5);
    pleq.z = (uint32_t)(int64_t)std::floor(t0 - xc * (double)vc.x - yc * (double)vc.y + 0.5);
    pleq.z -= pleq.x * center.x + pleq.y * center.y;
    return pleq;
}

// C functions

uint32_t cover8x8_selectFlips(int32_t dx, int32_t dy) // 10 instr
{
    uint32_t flips = 0;
    if (dy > 0 || (dy == 0 && dx <= 0))
        flips ^= (1 << CR_FLIPBIT_FLIP_X) ^ (1 << CR_FLIPBIT_FLIP_Y) ^ (1 << CR_FLIPBIT_COMPL);
    if (dx > 0)
        flips ^= (1 << CR_FLIPBIT_FLIP_X) ^ (1 << CR_FLIPBIT_FLIP_Y);
    if (::abs(dx) < ::abs(dy))
        flips ^= (1 << CR_FLIPBIT_SWAP_XY) ^ (1 << CR_FLIPBIT_FLIP_Y);
    return flips;
}

int clipPolygonWithPlane(float* baryOut, const float* baryIn, int numIn, float v0, float v1, float v2)
{
    int numOut = 0;
    if (numIn >= 3)
    {
        int ai = (numIn - 1) * 2;
        float av = v0 + v1 * baryIn[ai + 0] + v2 * baryIn[ai + 1];
        for (int bi = 0; bi < numIn * 2; bi += 2)
        {
            float bv = v0 + v1 * baryIn[bi + 0] + v2 * baryIn[bi + 1];
            if (av * bv < 0.0f)
            {
                float bc = av / (av - bv);
                float ac = 1.0f - bc;
                baryOut[numOut + 0] = baryIn[ai + 0] * ac + baryIn[bi + 0] * bc;
                baryOut[numOut + 1] = baryIn[ai + 1] * ac + baryIn[bi + 1] * bc;
                numOut += 2;
            }
            if (bv >= 0.0f)
            {
                baryOut[numOut + 0] = baryIn[bi + 0];
                baryOut[numOut + 1] = baryIn[bi + 1];
                numOut += 2;
            }
            ai = bi;
            av = bv;
        }
    }
    return (numOut >> 1);
}

int clipTriangleWithFrustum(float* bary, const float* v0, const float* v1, const float* v2, const float* d1, const float* d2)
{
    int num = 3;
    bary[0] = 0.0f, bary[1] = 0.0f;
    bary[2] = 1.0f, bary[3] = 0.0f;
    bary[4] = 0.0f, bary[5] = 1.0f;

    if ((v0[3] < fabsf(v0[0])) | (v1[3] < fabsf(v1[0])) | (v2[3] < fabsf(v2[0])))
    {
        float temp[18];
        num = clipPolygonWithPlane(temp, bary, num, v0[3] + v0[0], d1[3] + d1[0], d2[3] + d2[0]);
        num = clipPolygonWithPlane(bary, temp, num, v0[3] - v0[0], d1[3] - d1[0], d2[3] - d2[0]);
    }
    if ((v0[3] < fabsf(v0[1])) | (v1[3] < fabsf(v1[1])) | (v2[3] < fabsf(v2[1])))
    {
        float temp[18];
        num = clipPolygonWithPlane(temp, bary, num, v0[3] + v0[1], d1[3] + d1[1], d2[3] + d2[1]);
        num = clipPolygonWithPlane(bary, temp, num, v0[3] - v0[1], d1[3] - d1[1], d2[3] - d2[1]);
    }
    if ((v0[3] < fabsf(v0[2])) | (v1[3] < fabsf(v1[2])) | (v2[3] < fabsf(v2[2])))
    {
        float temp[18];
        num = clipPolygonWithPlane(temp, bary, num, v0[3] + v0[2], d1[3] + d1[2], d2[3] + d2[2]);
        num = clipPolygonWithPlane(bary, temp, num, v0[3] - v0[2], d1[3] - d1[2], d2[3] - d2[2]);
    }
    return num;
}


#define ASSERT_EQ(_A, _B) \
  { \
    if (_A != _B) { \
        printf("ASSERTION: "#_A" != "#_B"\n");\
        exit(1); \
    } \
  }
  
#define ASSERT_EQ_I(_A, _B) \
  { \
    if (_A != _B) { \
        printf("ASSERTION: "#_A" != "#_B", %d != %d\n", _A, _B);\
        exit(1); \
    } \
  }

#include <vector>


void emulateBinRaster(
    const cl_int& a_num_subtris,
    cl_int& a_num_bin_segs,
    // cl_int& a_num_active_tiles,
    const std::vector<CRTriangleHeader>& g_tri_header,
    const std::vector<cl_uchar>& g_tris_subtris,
    std::vector<cl_int>& g_bin_first_seg,
    std::vector<cl_int>& g_bin_seg_count,
    std::vector<cl_int>& g_bin_seg_data,
    std::vector<cl_int>& g_bin_seg_next,
    std::vector<cl_int>& g_bin_total,
    const cl_int c_bin_batch_sz,
    const cl_int c_max_bin_segs,
    const cl_int c_max_subtris,
    const cl_int c_num_bins,
    const cl_int c_num_tris,

    const glm::ivec2 viewport_size,
    const glm::ivec2 size_bins
)
{

    // emulate_result.g_bin_first_seg   = (cl_int*) malloc(CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * sizeof(cl_int));
    // emulate_result.g_bin_seg_data    = (cl_int*) malloc(c_max_bin_segs * CR_BIN_SEG_SIZE * sizeof(cl_int));
    // emulate_result.g_bin_seg_next    = (cl_int*) malloc(c_max_bin_segs * sizeof(cl_int));
    // emulate_result.g_bin_seg_count   = (cl_int*) malloc(c_max_bin_segs * sizeof(cl_int));
    // emulate_result.g_bin_total       = (cl_int*) malloc(CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * sizeof(cl_int));

    if (a_num_subtris > c_max_subtris)
        return;

    std::vector<int> batchTris;
    std::vector<int> currSeg(c_num_bins * CR_BIN_STREAMS_SIZE, 0);
    std::vector<int> idxInSeg(c_num_bins * CR_BIN_STREAMS_SIZE, 0);

    for (int i = 0; i < c_num_bins * CR_BIN_STREAMS_SIZE; i++)
    {
        g_bin_first_seg[i] = -1;
        g_bin_total[i] = 0;
        currSeg[i] = -1;
        idxInSeg[i] = CR_BIN_SEG_SIZE;
    }

    // Loop over batches.

    for (int batchIdx = 0; batchIdx * c_bin_batch_sz < c_num_tris; batchIdx++)
    {
        // Collect triangles.

        batchTris.clear();
        int batchStart = batchIdx * c_bin_batch_sz;
        int batchEnd = std::min(batchStart + c_bin_batch_sz, c_num_tris);
        for (int triIdx = batchStart; triIdx < batchEnd; triIdx++)
        {
			int numSubtris = g_tris_subtris[triIdx];
            for (int subtriIdx = 0; subtriIdx < numSubtris; subtriIdx++)
                batchTris.push_back((triIdx << 3) | ((numSubtris == 1) ? 7 : subtriIdx));
        }

        // Rasterize each triangle to bins.

        for (int idxInBatch = 0; idxInBatch < batchTris.size(); idxInBatch++)
        {
            int triIdx = batchTris[idxInBatch];

            int dataIdx = triIdx >> 3;
            int subtriIdx = triIdx & 7;
            if (subtriIdx != 7)
                dataIdx = g_tri_header[dataIdx].misc + subtriIdx;
            
            printf("dataIdx=%d, subtriIdx=%d\n", dataIdx, subtriIdx);

            // Read vertices and compute AABB.
            const CRTriangleHeader& tri = g_tri_header[dataIdx];
            glm::ivec2 v0 = glm::ivec2(tri.v0x, tri.v0y);
            glm::ivec2 d01 = glm::ivec2(tri.v1x, tri.v1y) - v0;
            glm::ivec2 d02 = glm::ivec2(tri.v2x, tri.v2y) - v0;
            v0 += viewport_size * CR_SUBPIXEL_SIZE / 2;
            glm::ivec2 lo = v0 + glm::min(glm::ivec2(0), glm::min(d01, d02));
            glm::ivec2 hi = v0 + glm::max(glm::ivec2(0), glm::max(d01, d02));

            // Check against each bin.
            for (int binIdx = 0; binIdx < c_num_bins; binIdx++)
            {
                int binX = binIdx % size_bins.x;
                int binY = binIdx / size_bins.x;
                int half = CR_BIN_SIZE * CR_TILE_SIZE * CR_SUBPIXEL_SIZE / 2;
                glm::ivec2 center = (glm::ivec2(binX, binY) * 2 + 1) * half;

                // Outside AABB => skip.
                if (lo.x >= center.x + half || lo.y >= center.y + half || hi.x <= center.x - half || hi.y <= center.y - half)
                    continue;

                // No intersection => skip.
                glm::ivec2 p0 = center - v0;
                glm::ivec2 p1 = p0 - d01;
                glm::ivec2 d12 = d02 - d01;
                if ((int64_t)p0.x * d01.y - (int64_t)p0.y * d01.x >= (abs(d01.x) + abs(d01.y)) * half) continue;
                if ((int64_t)p0.y * d02.x - (int64_t)p0.x * d02.y >= (abs(d02.x) + abs(d02.y)) * half) continue;
                if ((int64_t)p1.x * d12.y - (int64_t)p1.y * d12.x >= (abs(d12.x) + abs(d12.y)) * half) continue;

                // Segment full => allocate a new one.

                int si = binIdx * CR_BIN_STREAMS_SIZE + batchIdx % CR_BIN_STREAMS_SIZE;
                if (idxInSeg[si] == CR_BIN_SEG_SIZE)
                {
                    int segIdx = std::min(a_num_bin_segs++, c_max_bin_segs - 1);
                    if (currSeg[si] == -1)
                        g_bin_first_seg[si] = segIdx;
                    else
                        g_bin_seg_next[currSeg[si]] = segIdx;

                    g_bin_seg_next[segIdx] = -1;
					g_bin_seg_count[segIdx] = CR_BIN_SEG_SIZE;
                    currSeg[si] = segIdx;
                    idxInSeg[si] = 0;
                }

                // Append to the current segment.
                g_bin_seg_data[currSeg[si] * CR_BIN_SEG_SIZE + idxInSeg[si]] = triIdx;
                idxInSeg[si]++;
                g_bin_total[si]++;
            }
        }

        // Flush between batches.

        for (int i = 0; i < c_num_bins * CR_BIN_STREAMS_SIZE; i++)
		{
			if (idxInSeg[i] != CR_BIN_SEG_SIZE)
				g_bin_seg_count[currSeg[i]] = idxInSeg[i];
            idxInSeg[i] = CR_BIN_SEG_SIZE;
		}
    }
}

void emulateCoarseRaster(
    const cl_int& a_num_subtris,
    const cl_int& a_num_bin_segs,
    cl_int& a_num_active_tiles,
    const std::vector<CRTriangleHeader>& g_tri_header,
    const std::vector<cl_int>& g_bin_first_seg,
    const std::vector<cl_int>& g_bin_seg_data,
    const std::vector<cl_int>& g_bin_seg_next,
    const std::vector<cl_int>& g_bin_seg_count,
    std::vector<cl_int>& g_active_tiles,
    std::vector<cl_int>& g_tile_first_seg,
    std::vector<cl_int>& g_tile_seg_data,
    std::vector<cl_int>& g_tile_seg_count,
    std::vector<cl_int>& g_tile_seg_next,
    const cl_int c_deferred_clear,
    const cl_int c_max_tile_segs,
    const cl_int c_max_bin_segs,
    const cl_int c_max_subtris,
    size_t num_bins,
    size_t num_tiles,
    glm::ivec2 size_bins,
    glm::ivec2 size_tiles,
    glm::ivec2 viewport_size
)
{
    // Initialize.

    // const CRTriangleHeader* triHeader       = TRI_HEADER;

    // g_active_tiles.clear(); g_active_tiles.resize(CR_MAXTILES_SQR);
    // g_tile_first_seg.clear(); g_tile_first_seg.resize(CR_MAXTILES_SQR);
    // g_tile_seg_data.clear(); g_tile_seg_data.resize(c_max_tile_segs*CR_TILE_SEG_SIZE);
    // g_tile_seg_count.clear(); g_tile_seg_count.resize(c_max_tile_segs);
    // g_tile_seg_next.clear(); g_tile_seg_next.resize(c_max_tile_segs);

    // emulate_result.g_active_tiles    = (cl_int*) malloc(sizeof(cl_int[CR_MAXTILES_SQR]));
    // emulate_result.g_tile_first_seg  = (cl_int*) malloc(sizeof(cl_int[CR_MAXTILES_SQR]));
    // emulate_result.g_tile_seg_data   = (cl_int*) malloc(sizeof(cl_int[c_max_tile_segs][CR_TILE_SEG_SIZE]));
    // emulate_result.g_tile_seg_count  = (cl_int*) malloc(sizeof(cl_int[c_max_tile_segs]));
    // emulate_result.g_tile_seg_next   = (cl_int*) malloc(sizeof(cl_int[c_max_tile_segs]));

    std::vector<int> mergedTris;
    std::vector<int> currSeg(num_tiles, NULL);
    std::vector<int> idxInSeg(num_tiles, NULL);

    if (a_num_subtris > c_max_subtris || a_num_bin_segs > c_max_bin_segs)
        return;

    printf("Hey\n");

    for (int i = 0; i < num_tiles; i++)
    {
        g_tile_first_seg[i] = -1;
        currSeg[i] = -1;
        idxInSeg[i] = CR_TILE_SEG_SIZE;
    }

    // Process each bin.

    for (int binIdx = 0; binIdx < num_bins; binIdx++)
    {
        int binTileX = (binIdx % size_bins.x) * CR_BIN_SIZE;
        int binTileY = (binIdx / size_bins.x) * CR_BIN_SIZE;

        // Merge streams.

        mergedTris.clear();
        int streamSeg[CR_BIN_STREAMS_SIZE];
        for (int i = 0; i < CR_BIN_STREAMS_SIZE; i++)
            streamSeg[i] = g_bin_first_seg[binIdx * CR_BIN_STREAMS_SIZE + i];

        for (;;)
        {
            // Pick the stream with the lowest triangle index.

            int64_t smin = FW_S64_MAX;
            for (int i = 0; i < CR_BIN_STREAMS_SIZE; i++)
                if (streamSeg[i] != -1)
                    smin = std::min(smin, ((int64_t)g_bin_seg_data[streamSeg[i] * CR_BIN_SEG_SIZE] << 32) | i);
            if (smin == FW_S64_MAX)
                break;

            // Consume one segment from the stream.

            int segIdx = streamSeg[(int32_t)smin];
            streamSeg[(int32_t)smin] = g_bin_seg_next[segIdx];
            for (int i = 0; i < g_tile_seg_count[segIdx]; i++)
				mergedTris.push_back(g_bin_seg_data[segIdx * CR_BIN_SEG_SIZE + i]);
        }

        // Rasterize each triangle into tiles.

        if (mergedTris.size())
            printf("mergedtris=%d", mergedTris.size());
        
        for (int mergedIdx = 0; mergedIdx < mergedTris.size(); mergedIdx++)
        {
            int triIdx = mergedTris[mergedIdx];
            int dataIdx = triIdx >> 3;
            int subtriIdx = triIdx & 7;
            if (subtriIdx != 7)
                dataIdx = g_tri_header[dataIdx].misc + subtriIdx;

            // Read vertices and compute AABB.

            const CRTriangleHeader& tri = g_tri_header[dataIdx];
            glm::ivec2 v0 = glm::ivec2(tri.v0x, tri.v0y);
            glm::ivec2 d01 = glm::ivec2(tri.v1x, tri.v1y) - v0;
            glm::ivec2 d02 = glm::ivec2(tri.v2x, tri.v2y) - v0;
            v0 += viewport_size * CR_SUBPIXEL_SIZE / 2;
            glm::ivec2 lo = v0 + glm::min(glm::ivec2(0), glm::min(d01, d02));
            glm::ivec2 hi = v0 + glm::max(glm::ivec2(0), glm::max(d01, d02));

            // Check against each tile.

            for (int tileInBin = 0; tileInBin < CR_BIN_SQR; tileInBin++)
            {
                int tileX = tileInBin % CR_BIN_SIZE + binTileX;
                int tileY = tileInBin / CR_BIN_SIZE + binTileY;
                int half = CR_TILE_SIZE * CR_SUBPIXEL_SIZE / 2;
                glm::ivec2 center = (glm::ivec2(tileX, tileY) * 2 + 1) * half;

                // Outside viewport => skip.

                if (tileX >= size_tiles.x || tileY >= size_tiles.y)
                    continue;

                // No intersection => skip.

                if (lo.x >= center.x + half || lo.y >= center.y + half || hi.x <= center.x - half || hi.y <= center.y - half)
                    continue;

                glm::ivec2 p0 = center - v0;
                glm::ivec2 p1 = p0 - d01;
                glm::ivec2 d12 = d02 - d01;
                if ((int64_t)p0.x * d01.y - (int64_t)p0.y * d01.x >= (abs(d01.x) + abs(d01.y)) * half) continue;
                if ((int64_t)p0.y * d02.x - (int64_t)p0.x * d02.y >= (abs(d02.x) + abs(d02.y)) * half) continue;
                if ((int64_t)p1.x * d12.y - (int64_t)p1.y * d12.x >= (abs(d12.x) + abs(d12.y)) * half) continue;

                // Segment full => allocate a new one.

                int si = tileX + tileY * size_tiles.x;
                if (idxInSeg[si] == CR_TILE_SEG_SIZE)
                {
                    int segIdx = std::min(a_num_active_tiles++, c_max_tile_segs - 1);
                    if (currSeg[si] == -1)
                        g_tile_first_seg[si] = segIdx;
                    else
                        g_tile_seg_next[currSeg[si]] = segIdx;

                    g_tile_seg_next[segIdx] = -1;
                    g_tile_seg_count[segIdx] = CR_TILE_SEG_SIZE;
                    currSeg[si] = segIdx;
                    idxInSeg[si] = 0;
                }

                // Append to the current segment.

                g_tile_seg_data[currSeg[si] * CR_TILE_SEG_SIZE + idxInSeg[si]] = triIdx;
                idxInSeg[si]++;
            }
        }
    }

    // Flush.

    for (int i = 0; i < num_tiles; i++)
    {
        if (currSeg[i] == -1 && c_deferred_clear)
            g_tile_first_seg[i] = -1;
        if (currSeg[i] != -1 || c_deferred_clear) {
            g_active_tiles[a_num_active_tiles++] = i;
            printf("Hey !\n");
        }
		if (idxInSeg[i] != CR_TILE_SEG_SIZE) {
            g_tile_seg_count[currSeg[i]] = idxInSeg[i];
            printf("append seg");
        }
    }
}

struct GouraudVertex
{
    glm::vec4   clipPos;
    glm::vec4   color;          // Varying 0.
};

void emulateFineRaster(
    const cl_int& a_num_subtris,
    const cl_int& a_num_bin_segs,
    const cl_int& a_num_tile_segs,
    cl_int& a_num_active_tiles,
    const std::vector<CRTriangleHeader>& g_tri_header,
    const std::vector<CRTriangleData>& g_tri_data,
    const std::vector<cl_uchar>& g_vertex_buffer,
    const std::vector<cl_int>& g_bin_first_seg,
    const std::vector<cl_int>& g_bin_seg_data,
    const std::vector<cl_int>& g_bin_seg_next,
    const std::vector<cl_int>& g_bin_seg_count,
    std::vector<cl_int>& g_active_tiles,
    std::vector<cl_int>& g_tile_first_seg,
    std::vector<cl_int>& g_tile_seg_data,
    std::vector<cl_int>& g_tile_seg_count,
    std::vector<cl_int>& g_tile_seg_next,

    std::vector<cl_uint>& g_color_buffer,
    std::vector<cl_uint>& g_depth_buffer,

    const cl_uint   c_clear_color,
    const cl_uint   c_clear_depth,
    const cl_uint   c_color_buffer_mode,
    const cl_int c_deferred_clear,
    const cl_int c_max_bin_segs,
    const cl_int c_max_subtris,
    const cl_int c_max_tile_segs,
    const cl_int c_render_mode_flags,
    const cl_int    c_viewport_height,
    const cl_int    c_viewport_width,
    const cl_int    c_width_tiles,

    glm::ivec2 size_pixels,
    glm::ivec2 size_tiles,
    glm::ivec2 viewport_size
)
{
    // Initialize.
    int num_samples = 1;
    int samples_log2 = 0;

    // const U8*               g_vertex_buffer    = (const U8*)m_g_vertex_buffer->getPtr(m_vertexOfs);
    // const CRTriangleHeader* g_tri_header       = (const CRTriangleHeader*)m_g_tri_header.getPtr();
    // const CRTriangleData*   g_tri_data         = (const CRTriangleData*)m_g_tri_data.getPtr();
    // CRAtomics&              atomics         = *(CRAtomics*)m_module->getGlobal("g_crAtomics").getMutablePtr();
    // const S32*              activeTiles     = (S32*)m_activeTiles.getPtr();
    // const S32*              tileFirstSeg    = (S32*)m_tileFirstSeg.getPtr();
    // const S32*              tileSegData     = (S32*)m_tileSegData.getPtr();
    // const S32*              tileSegNext     = (S32*)m_tileSegNext.getPtr();
    // const S32*              tileSegCount    = (S32*)m_tileSegCount.getPtr();

    bool                    enableBlend     = (c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_BLENDER) != 0; // (String(m_pipeSpec.blendShaderName) == "BlendSrcOver");

    std::vector<int32_t> mergedTris;
    // std::vector<uint32_t> colorBuffer(NULL, size_pixels.y * size_pixels.x * num_samples);
    // std::vector<uint32_t> depthBuffer(NULL, size_pixels.y * size_pixels.x * num_samples);
    if (a_num_subtris > c_max_subtris || a_num_bin_segs > c_max_bin_segs || a_num_tile_segs > c_max_tile_segs)
    return;
    
    // Deferred clear => clear framebuffer.

    if (c_deferred_clear)
    {
        for (int i = 0; i < g_color_buffer.size(); i++)
        {
            g_color_buffer[i] = c_clear_color;
            g_depth_buffer[i] = c_clear_depth;
        }
    }

    // Otherwise => download framebuffer.

    /*
    else
    {
        CUDA_MEMCPY2D copy;
        copy.srcXInBytes    = 0;
        copy.srcY           = 0;
        copy.srcMemoryType  = CU_MEMORYTYPE_ARRAY;
        copy.srcArray       = m_colorBuffer->getCudaArray();
        copy.dstXInBytes    = 0;
        copy.dstY           = 0;
        copy.dstMemoryType  = CU_MEMORYTYPE_HOST;
        copy.dstHost        = colorBuffer.getPtr();
        copy.dstPitch       = m_sizePixels.x * m_numSamples * sizeof(U32);
        copy.WidthInBytes   = m_sizePixels.x * m_numSamples * sizeof(U32);
        copy.Height         = m_sizePixels.y;

        CudaModule::checkError("cuMemcpy2D", cuMemcpy2D(&copy));

        copy.srcArray       = m_depthBuffer->getCudaArray();
        copy.dstHost        = depthBuffer.getPtr();

        CudaModule::checkError("cuMemcpy2D", cuMemcpy2D(&copy));
    }
    */
    // Process each tile-triangle intersection.

    for (int activeIdx = 0; activeIdx < a_num_active_tiles; activeIdx++)
    {
        int tileIdx = g_active_tiles[activeIdx];
        glm::ivec2 tilePixelPos = glm::ivec2(tileIdx % size_tiles.x, tileIdx / size_tiles.x) * CR_TILE_SIZE;

        // Collect triangles.

        mergedTris.clear();
        for (int segIdx = g_tile_first_seg[tileIdx]; segIdx != -1; segIdx = g_tile_seg_next[segIdx])
            for (int i = 0; i < g_tile_seg_count[segIdx]; i++)
                mergedTris.push_back(g_tile_seg_data[segIdx * CR_TILE_SEG_SIZE + i]);

        // Rasterize each triangle into framebuffer.
        // printf("merged tris %d\n", mergedTris.size());
        for (int mergedIdx = 0; mergedIdx < mergedTris.size(); mergedIdx++)
        {
            int triIdx = mergedTris[mergedIdx];
            int dataIdx = triIdx >> 3;
            int subtriIdx = triIdx & 7;
            if (subtriIdx != 7)
                dataIdx = g_tri_header[dataIdx].misc + subtriIdx;

            // Read vertices.

		    const CRTriangleHeader& th  = g_tri_header[dataIdx];
		    const CRTriangleData&   td  = g_tri_data[dataIdx];
            const GouraudVertex&    vd0 = *(const GouraudVertex*)(g_vertex_buffer.data() + td.vi0 * sizeof(GouraudVertex));
            const GouraudVertex&    vd1 = *(const GouraudVertex*)(g_vertex_buffer.data() + td.vi1 * sizeof(GouraudVertex));
            const GouraudVertex&    vd2 = *(const GouraudVertex*)(g_vertex_buffer.data() + td.vi2 * sizeof(GouraudVertex));

            glm::ivec2 v0 = glm::ivec2(th.v0x, th.v0y) + viewport_size * (CR_SUBPIXEL_SIZE / 2);
            glm::ivec2 v1 = glm::ivec2(th.v1x, th.v1y) + viewport_size * (CR_SUBPIXEL_SIZE / 2);
            glm::ivec2 v2 = glm::ivec2(th.v2x, th.v2y) + viewport_size * (CR_SUBPIXEL_SIZE / 2);

            // Setup edge functions.

            glm::ivec2 d0 = v1 - v0;
            glm::ivec2 d1 = v2 - v1;
            glm::ivec2 d2 = v0 - v2;
            int64_t b0 = (int64_t)v0.x * d0.y - (int64_t)v0.y * d0.x;
            int64_t b1 = (int64_t)v1.x * d1.y - (int64_t)v1.y * d1.x;
            int64_t b2 = (int64_t)v2.x * d2.y - (int64_t)v2.y * d2.x;
            int64_t c0 = b0 + (abs(d0.x) + abs(d0.y)) * (CR_SUBPIXEL_SIZE / 2);
            int64_t c1 = b1 + (abs(d1.x) + abs(d1.y)) * (CR_SUBPIXEL_SIZE / 2);
            int64_t c2 = b2 + (abs(d2.x) + abs(d2.y)) * (CR_SUBPIXEL_SIZE / 2);
            if (d0.y > 0 || (d0.y == 0 && d0.x <= 0)) b0--;
            if (d1.y > 0 || (d1.y == 0 && d1.x <= 0)) b1--;
            if (d2.y > 0 || (d2.y == 0 && d2.x <= 0)) b2--;

            // Check against each pixel.

            for (int pixelIdx = 0; pixelIdx < CR_TILE_SQR; pixelIdx++)
            {
                glm::ivec2 pixelPos = tilePixelPos + glm::ivec2(pixelIdx % CR_TILE_SIZE, pixelIdx / CR_TILE_SIZE);
                int pixelOfs = (tilePixelPos.x + pixelPos.y * size_pixels.x) * num_samples + (pixelIdx % CR_TILE_SIZE);

                // Test pixel coverage (conservative).

                int64_t xx = (int64_t)(pixelPos.x * CR_SUBPIXEL_SIZE + CR_SUBPIXEL_SIZE / 2);
                int64_t yy = (int64_t)(pixelPos.y * CR_SUBPIXEL_SIZE + CR_SUBPIXEL_SIZE / 2);
                if (xx * d0.y - yy * d0.x > c0) continue;
                if (xx * d1.y - yy * d1.x > c1) continue;
                if (xx * d2.y - yy * d2.x > c2) continue;

                // Test sample coverage (exact).
                // Test and update depth.

                uint32_t coverMask = 0;
                uint32_t writeMask = 0;
                for (int i = 0; i < num_samples; i++)
                {
                    uint32_t sampleX = pixelPos.x * num_samples + 0; //c_msaaPatterns[samples_log2][i];
                    uint32_t sampleY = pixelPos.y * num_samples + i;

                    int64_t xx = (int64_t)((sampleX * 2 + 1) << (CR_SUBPIXEL_LOG2 - samples_log2 - 1));
                    int64_t yy = (int64_t)((sampleY * 2 + 1) << (CR_SUBPIXEL_LOG2 - samples_log2 - 1));
                    if (xx * d0.y - yy * d0.x > b0) continue;
                    if (xx * d1.y - yy * d1.x > b1) continue;
                    if (xx * d2.y - yy * d2.x > b2) continue;

                    coverMask |= 1 << i;

                    if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0)
                    {
                        uint32_t depth = td.zx * sampleX + td.zy * sampleY + td.zb;
                        if (depth >= g_depth_buffer[pixelOfs + i * CR_TILE_SIZE])
                            continue;
                        g_depth_buffer[pixelOfs + i * CR_TILE_SIZE] = depth;
                    }
                    writeMask |= 1 << i;
                }

                // No samples to write => skip shader & ROP.

                if (writeMask == 0)
                    continue;

                // Interpolate color.

                glm::vec4 color; // = glm::vec4(1,1,1,1);
                
                if ((c_render_mode_flags & RENDER_MODE_FLAG_ENABLE_LERP) == 0)
                    color = vd2.color;
                else
                {
                    int ctr = -1; //selectMSAACentroid(m_samplesLog2, coverMask);
                    int sampleX = pixelPos.x * num_samples * 2 + ((ctr == -1) ? num_samples : 1 ); // c_msaaPatterns[m_samplesLog2][ctr] * 2 + 1);
                    int sampleY = pixelPos.y * num_samples * 2 + ((ctr == -1) ? num_samples : ctr * 2 + 1);
                    float w = 1.0f / (float)(td.wx * sampleX + td.wy * sampleY + td.wb);
                    float u = w * (float)(td.ux * sampleX + td.uy * sampleY + td.ub);
                    float v = w * (float)(td.vx * sampleX + td.vy * sampleY + td.vb);
                    color = vd0.color + (vd1.color - vd0.color) * u + (vd2.color - vd0.color) * v;
                }

                // Blend.

                uint32_t src = 
                    ((uint32_t) (color.x * 255)) & 0xFF <<  0 | 
                    ((uint32_t) (color.y * 255)) & 0xFF <<  8 | 
                    ((uint32_t) (color.z * 255)) & 0xFF << 16 | 
                    ((uint32_t) (color.w * 255)) & 0xFF << 24 ;

                uint32_t srcFactor = src >> 24;
                uint32_t dstFactor = 255 - srcFactor;

                for (int i = 0; i < num_samples; i++)
                {
                    if ((writeMask & (1 << i)) != 0)
                    {
                        uint32_t& dst = g_color_buffer[pixelOfs + i * CR_TILE_SIZE];
                        if (!enableBlend)
                            dst = src;
                        else
                            dst =
                                ((((((src >> 0)  & 0xFF) * srcFactor + ((dst >> 0)  & 0xFF) * dstFactor) * 0x010101 + 0x800000) >> 24) << 0)  |
                                ((((((src >> 8)  & 0xFF) * srcFactor + ((dst >> 8)  & 0xFF) * dstFactor) * 0x010101 + 0x800000) >> 24) << 8)  |
                                ((((((src >> 16) & 0xFF) * srcFactor + ((dst >> 16) & 0xFF) * dstFactor) * 0x010101 + 0x800000) >> 24) << 16) |
                                ((((((src >> 24) & 0xFF) * srcFactor + ((dst >> 24) & 0xFF) * dstFactor) * 0x010101 + 0x800000) >> 24) << 24);
                    }
                }
            }
        }
    }

    // Upload framebuffer.
    /*
    {
        CUDA_MEMCPY2D copy;
        copy.srcXInBytes    = 0;
        copy.srcY           = 0;
        copy.srcMemoryType  = CU_MEMORYTYPE_HOST;
        copy.srcHost        = colorBuffer.getPtr();
        copy.srcPitch       = m_sizePixels.x * m_numSamples * sizeof(U32);
        copy.dstXInBytes    = 0;
        copy.dstY           = 0;
        copy.dstMemoryType  = CU_MEMORYTYPE_ARRAY;
        copy.dstArray       = m_colorBuffer->getCudaArray();
        copy.WidthInBytes   = m_sizePixels.x * m_numSamples * sizeof(U32);
        copy.Height         = m_sizePixels.y;

        CudaModule::checkError("cuMemcpy2D", cuMemcpy2D(&copy));

        copy.srcHost        = depthBuffer.getPtr();
        copy.dstArray       = m_depthBuffer->getCudaArray();

        CudaModule::checkError("cuMemcpy2D", cuMemcpy2D(&copy));
    }
    */
}
