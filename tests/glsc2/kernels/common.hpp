#pragma once
#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <string.h>

#include <glm/glm.hpp>

#include <constants.h>

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
  