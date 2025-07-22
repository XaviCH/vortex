#ifndef TYPES_DEVICE_H
#define TYPES_DEVICE_H

#ifdef __COMPILER_RELATIVE_PATH__
#include <constants.device.h>
#else
#include "constants.device.h"
#endif


// defining primitives for c and cl context
#ifdef __OPENCL_VERSION__
typedef uint cl_uint;
typedef ushort cl_ushort;
typedef short cl_short;
typedef bool cl_bool;
#else
#include <CL/opencl.h>
#endif

typedef struct {
    cl_uint flags;
} render_mode_t;

inline cl_bool is_render_mode_flag_triangle_fan(const render_mode_t render_mode) {
    return (render_mode.flags & RENDER_MODE_FLAG_TRIANGLE_FAN) != 0; 
}

inline cl_bool is_render_mode_flag_triangle_strip(const render_mode_t render_mode) {
    return (render_mode.flags & RENDER_MODE_FLAG_TRIANGLE_STRIP) != 0; 
}

inline cl_bool is_render_mode_flag_enable_depth(const render_mode_t render_mode) {
    return (render_mode.flags & RENDER_MODE_FLAG_ENABLE_DEPTH) != 0; 
}

inline cl_bool is_render_mode_flag_enable_lerp(const render_mode_t render_mode) {
    return (render_mode.flags & RENDER_MODE_FLAG_ENABLE_LERP) != 0; 
}

inline cl_bool is_render_mode_flag_enable_cull_front(const render_mode_t render_mode) {
    return (render_mode.flags & RENDER_MODE_FLAG_ENABLE_CULL_FRONT) != 0; 
}

inline cl_bool is_render_mode_flag_enable_cull_back(const render_mode_t render_mode) {
    return (render_mode.flags & RENDER_MODE_FLAG_ENABLE_CULL_BACK) != 0; 
}

typedef enum {
    FRONT = 0,
    BACK = 1
} face_t;

typedef struct {
    cl_uint misc;
} triangle_header_misc_t;

inline face_t           get_th_misc_face(const triangle_header_misc_t th_misc) { return (face_t)(th_misc.misc>>31); }
inline cl_ushort   get_th_misc_zmax(const triangle_header_misc_t th_misc) { return ((th_misc.misc >>  4) | 0xFFu) & 0xFFFFu; }
inline cl_ushort   get_th_misc_zmin(const triangle_header_misc_t th_misc) { return ((th_misc.misc >> 12) | 0x00u) & 0xFF00u; }

inline void set_th_misc_face(triangle_header_misc_t* th_misc, face_t face) { 
    th_misc->misc &= 0x7FFFFFFFu;
    th_misc->misc |= ((unsigned int)face << 31);
};
inline void set_th_misc_zmax(triangle_header_misc_t* th_misc, cl_ushort zmax) {
    th_misc->misc &= 0xFFF00FFFu;
    th_misc->misc |= ((unsigned int)zmax << 4) & 0xFF000u; // just write the 8 most significant bits
};
inline void set_th_misc_zmin(triangle_header_misc_t* th_misc, cl_ushort zmin) {
    th_misc->misc &= 0xF00FFFFFu;
    th_misc->misc |= ((unsigned int)zmin << 12) & 0xFF00000u; // just write the 8 most significant bits
};

typedef struct
{
    cl_short v0x;    // Subpixels relative to viewport center. Valid if triSubtris = 1.
    cl_short v0y;
    cl_short v1x;
    cl_short v1y;
    cl_short v2x;
    cl_short v2y;

    cl_uint misc;   // triSubtris=1: (zmin:20, f01:4, f12:4, f20:4), triSubtris>=2: (subtriBase)
} triangle_header_t;

typedef struct
{
    cl_uint zx;     // zx * sampleX + zy * sampleY + zb = lerp(CR_DEPTH_MIN, CR_DEPTH_MAX, (clipZ / clipW + 1) / 2)
    cl_uint zy;
    cl_uint zb;
    cl_uint zslope; // (abs(zx) + abs(zy)) * (samplesPerPixel / 2)

    int wx;     // wx * (sampleX * 2 + 1) + wy * (sampleY * 2 + 1) + wb = minClipW / clipW * CR_BARY_MAX
    int wy;
    int wb;

    int ux;     // ux * (sampleX * 2 + 1) + uy * (sampleY * 2 + 1) + ub = baryU * minClipW / clipW * CR_BARY_MAX
    int uy;
    int ub;

    int vx;     // vx * (sampleX * 2 + 1) + vy * (sampleY * 2 + 1) + vb = baryV * minClipW / clipW * CR_BARY_MAX
    int vy;
    int vb;

    unsigned int vi0;    // Vertex indices.
    unsigned int vi1;
    unsigned int vi2;
} triangle_data_t;

typedef struct {
    unsigned int offset;
    unsigned int stride;
    unsigned int misc; // size[3], type[3], normalized[1], vertex_attrib_pointer_active[1]
} vertex_attribute_data_t;

/*
typedef struct
{
    unsigned int blending : 1;
    unsigned int cull_back : 1;
    unsigned int cull_front : 1;
    unsigned int depth_test : 1;
    unsigned int dithering : 1;
    unsigned int lerp  : 1;
    unsigned int scissor_test : 1;
    unsigned int stencil_test : 1;
    unsigned int polygon_offset_fill : 1;
} render_mode_t;
*/

typedef struct {
    unsigned short width, height, misc;
    /*
    unsigned int internalformat : 4;
    unsigned int wrap_s : 2;
    unsigned int wrap_t : 2;
    unsigned int min_filter : 3;
    unsigned int mag_filter : 1;
    */
} sampler2D_t;


unsigned int get_sampler2D_internalformat(sampler2D_t sampler2D)  { return (sampler2D.misc >>  0) & 0xFu; }
unsigned int get_sampler2D_wrap_s(sampler2D_t sampler2D)          { return (sampler2D.misc >>  4) & 0x3u; }
unsigned int get_sampler2D_wrap_t(sampler2D_t sampler2D)          { return (sampler2D.misc >>  6) & 0x3u; }
unsigned int get_sampler2D_min_filter(sampler2D_t sampler2D)      { return (sampler2D.misc >>  8) & 0x7u; }
unsigned int get_sampler2D_mag_filter(sampler2D_t sampler2D)      { return (sampler2D.misc >> 11) & 0x1u; }

void set_sampler2D_internalformat(sampler2D_t* sampler2D, unsigned short internalformat) 
{ 
    sampler2D->misc = (sampler2D->misc & ~0xFu) | (internalformat & 0xFu); 
}

void set_sampler2D_wrap_s(sampler2D_t* sampler2D, unsigned short wrap_s) 
{ 
    sampler2D->misc &= ~(0x3u << 4);
    sampler2D->misc |= wrap_s << 4;
}

void set_sampler2D_wrap_t(sampler2D_t* sampler2D, unsigned short wrap_t) 
{ 
    sampler2D->misc &= ~(0x3u << 6);
    sampler2D->misc |= wrap_t << 6;
}

#endif