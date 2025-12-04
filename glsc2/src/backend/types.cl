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
#else
    typedef global const float4* restrict ro_vertex_buffer_t;
    typedef global float4* restrict wo_vertex_buffer_t;
    typedef global const void* restrict ro_texture2d_t;
    typedef global void* restrict rw_texture2d_t;
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

#endif