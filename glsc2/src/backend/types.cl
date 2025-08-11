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
#else
typedef global const float4* ro_vertex_buffer_t;
typedef global float4* wo_vertex_buffer_t;
#endif

#ifdef DEVICE_SUB_GROUP_ENABLED
typedef struct {
    #if     (DEVICE_SUB_GROUP_THREADS <= 8)
        uchar mask;
    #elif   (DEVICE_SUB_GROUP_THREADS <= 16)
        ushort mask;
    #elif   (DEVICE_SUB_GROUP_THREADS <= 32)
        uint mask;
    #elif   (DEVICE_SUB_GROUP_THREADS <= 64)
        ulong mask;
    #elif   (DEVICE_SUB_GROUP_THREADS <= 128)
        uint4 mask;
    #else
        #error DEVICE_SUB_GROUP_THREADS too large to be supported. 
    #endif
} sub_group_mask_t;
#endif

#endif