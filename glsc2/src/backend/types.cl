#ifndef BACKEND_TYPES_CL
#define BACKEND_TYPES_CL

#ifdef __COMPILER_RELATIVE_PATH__
#include <types.device.h>
#else
#include "glsc2/src/types.device.h"
#endif

#ifdef DEVICE_IMAGE_ENABLED
    typedef read_only image1d_buffer_t ro_vertex_buffer_t;
    typedef write_only image1d_buffer_t wo_vertex_buffer_t;
    typedef read_only image2d_t ro_texture2d_t;
    typedef read_write image2d_t rw_texture2d_t;

    typedef read_write image2d_t colorbuffer_t;
    typedef read_write image2d_t depthbuffer_t;
    typedef read_write image2d_t stencilbuffer_t;
#else
    typedef global const float4* restrict ro_vertex_buffer_t;
    typedef global float4* restrict wo_vertex_buffer_t;
    typedef global const void* restrict ro_texture2d_t;
    typedef global void* restrict rw_texture2d_t;
    
    typedef global void* restrict colorbuffer_t;
    typedef global ushort* restrict depthbuffer_t;
    typedef global uchar* restrict stencilbuffer_t;
#endif

inline float4 read_vertex_buffer(ro_vertex_buffer_t vertex_buffer, uint index) { 
    float4 value;

    #ifdef DEVICE_IMAGE_ENABLED
        value = read_imagef(vertex_buffer, index);
    #else
        value = vertex_buffer[index]; 
    #endif

    return value;
}

static inline uint read_colorbuffer(colorbuffer_t colorbuffer, uint2 pos, uint2 size, uint mode)
{

    #ifdef DEVICE_IMAGE_ENABLED
    {
        return uint4_to_uint(read_imageui(colorbuffer, pos));
    }
    #else
    {
        uint offset = pos.y * size.x + pos.x;

        switch(mode) 
        {
            default:
            case TEX_R8:
                return *((global const uchar*) colorbuffer + offset);
            case TEX_RG8:
                return *((global const ushort*) colorbuffer + offset);
            case TEX_RGB8:
            {
                global const uchar* ptr = (global const uchar*) colorbuffer + offset*3;
                return 
                    ((uint) ptr[0] <<  0) |
                    ((uint) ptr[1] <<  8) |
                    ((uint) ptr[2] << 16) |
                    0xFF000000            ;
            }
            case TEX_RGBA8:
                return *((global const uint*) colorbuffer + offset);
            case TEX_RGBA4:
            {
                uint tex = *((global const ushort*) colorbuffer + offset);
                return 
                    (tex & 0x000F) << 0 |
                    (tex & 0x00F0) << 4 |
                    (tex & 0x0F00) << 8 |
                    (tex & 0xF000) << 12;
            }
            case TEX_RGB5_A1:
            {
                uint tex = *((global const ushort*) colorbuffer + offset);
                return 
                    (tex & 0x001F) << 0 |
                    (tex & 0x03E0) << 5 |
                    (tex & 0x7C00) << 10|
                    (tex & 0x8000) << 15;
            }
            case TEX_RGB565:
            {
                uint tex = *((global const ushort*) colorbuffer + offset);
                return 
                    (tex & 0x001F) << 0 |
                    (tex & 0x07E0) << 5 |
                    (tex & 0xF800) << 11;
            }
        }
    }
    #endif
}

static inline void write_colorbuffer(colorbuffer_t colorbuffer, uint2 pos, uint2 size, uint mode, uint color) 
{
    #ifdef DEVICE_IMAGE_ENABLED
    {
        write_imageui(colorbuffer, pos, uint_to_uint4(color));
    }
    #else
    {
        uint offset = pos.y * size.x + pos.x;

        switch(mode) {
            case TEX_R8:
                *((global uchar*) colorbuffer + offset) = color;
                break;
            case TEX_RG8:
                *((global ushort*) colorbuffer + offset) = color;
                break;
            case TEX_RGB8:
                {
                    global uchar* buffer = colorbuffer + offset*3;
                    buffer[0] = color >>  0;
                    buffer[1] = color >>  8;
                    buffer[2] = color >> 16;
                }
                break;
            case TEX_RGBA8:
                *(((global uint*) colorbuffer) + offset) = color;
                break;
            case TEX_RGBA4:
                *((global ushort*) colorbuffer + offset) = 
                    (color & 0x0000000F) >> 0 |
                    (color & 0x00000F00) >> 4 |
                    (color & 0x000F0000) >> 8 |
                    (color & 0x0F000000) >> 12;
                break;
            case TEX_RGB5_A1:
                *((global ushort*) colorbuffer + offset) = 
                    (color & 0x0000001F) >> 0 |
                    (color & 0x00001F00) >> 3 |
                    (color & 0x001F0000) >> 6 |
                    (color & 0x01000000) >> 9 ;
                break;
            case TEX_RGB565:
                *((global ushort*) colorbuffer + offset) = 
                    (color & 0x0000001F) >> 0 |
                    (color & 0x00003F00) >> 3 |
                    (color & 0x001F0000) >> 5 ;
                break;
        }
    }
    #endif
}

static inline ushort read_depthbuffer(depthbuffer_t depthbuffer, uint2 pos, uint2 size)
{
    #ifdef DEVICE_IMAGE_ENABLED
    {
        return read_imageui(depthbuffer, pos).x;
    }
    #else
    {
        uint offset = pos.y * size.x + pos.x;
        return depthbuffer[offset];
    }
    #endif
}

static inline void write_depthbuffer(depthbuffer_t depthbuffer, uint2 pos, uint2 size, ushort depth) 
{
    #ifdef DEVICE_IMAGE_ENABLED
    {
        write_imageui(colorbuffer, pos, depth);
    }
    #else
    {
        uint offset = pos.y * size.x + pos.x;
        depthbuffer[offset] = depth;
    }
    #endif
}

static inline ushort read_stencilbuffer(stencilbuffer_t stencilbuffer, uint2 pos, uint2 size)
{
    #ifdef DEVICE_IMAGE_ENABLED
    {
        return read_imageui(stencilbuffer, pos).x;
    }
    #else
    {
        uint offset = pos.y * size.x + pos.x;
        return stencilbuffer[offset];
    }
    #endif
}

static inline void write_stencilbuffer(stencilbuffer_t stencilbuffer, uint2 pos, uint2 size, ushort stencil) 
{
    #ifdef DEVICE_IMAGE_ENABLED
    {
        write_imageui(colorbuffer, pos, stencil);
    }
    #else
    {
        uint offset = pos.y * size.x + pos.x;
        stencilbuffer[offset] = stencil;
    }
    #endif
}


#endif