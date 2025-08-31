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
typedef global const float4* restrict ro_vertex_buffer_t;
typedef global float4* restrict wo_vertex_buffer_t;
#endif

#endif