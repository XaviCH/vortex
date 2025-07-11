#ifndef TYPES_DEVICE_H
#define TYPES_DEVICE_H

typedef struct
{
    short v0x;    // Subpixels relative to viewport center. Valid if triSubtris = 1.
    short v0y;
    short v1x;
    short v1y;
    short v2x;
    short v2y;

    unsigned int misc;   // triSubtris=1: (zmin:20, f01:4, f12:4, f20:4), triSubtris>=2: (subtriBase)
} triangle_header_t;

typedef struct 
{
    unsigned int zx;     // zx * sampleX + zy * sampleY + zb = lerp(CR_DEPTH_MIN, CR_DEPTH_MAX, (clipZ / clipW + 1) / 2)
    unsigned int zy;
    unsigned int zb;
    unsigned int zslope; // (abs(zx) + abs(zy)) * (samplesPerPixel / 2)

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

void set_sampler2D_internalformat(sampler2D_t sampler2D, unsigned short internalformat) { sampler2D.misc = (sampler2D.misc & 0xF0u) | (internalformat & 0xFu); }

#endif