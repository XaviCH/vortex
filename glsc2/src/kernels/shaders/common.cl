/**
 * TODO: 
 *
 *
 */

#ifndef KERNELS_SHADERS_COMMON_CL
#define KERNELS_SHADERS_COMMON_CL

#ifdef __COMPILER_RELATIVE_PATH__
#include "common.h"
#include "../../types.device.h"
#else
#include "glsc2/src/kernels/shaders/common.h"
#include "glsc2/src/types.device.h"
#endif

// Utils

typedef float2 vec2;
typedef float3 vec3;
typedef float4 vec4;

#ifdef CONF_FINE_IMAGE_ENABLED
#define KERNEL_ARG_VERTEX_BUFFER write_only image1d_buffer_t vertex_buffer
typedef write_only image1d_buffer_t vertex_buffer_t;
#else
#define KERNEL_ARG_VERTEX_BUFFER global float4* vertex_buffer
typedef global float4* vertex_buffer_t;
#endif

#ifdef CONF_FINE_IMAGE_ENABLED
typedef read_only image2d_t image2D_t;
#else
typedef const global uchar* image2D_t;
#endif

float4 mul(float16 mat, float4 vec) {
  float4 result = 0;

  for(int i=0; i<16; ++i) {
    result[i%4] += mat[i]*vec[i/4]; 
  }

  return result;
}

float4 texture2D(sampler2D_t sampler, image2D_t image, float2 coord) {
    int width, height;

    coord.x = (coord.x - floor(coord.x));
    coord.y = (coord.y - floor(coord.y));

    width   = sampler.width * coord.x;
    height  = sampler.height * coord.y;

    global uchar* color = image + (height*sampler.width + width)*4;
    return (float4) ((float)*color / 255, (float)*(color+1) / 255, (float)*(color+2) / 255, (float)*(color+3) / 255);
    
}

#define TEXTURE2D(sampler, coord) texture2D(sampler, gl_image_##sampler, coord);

// Varying definitions

#ifdef VARYING_VEC2
#define PARAM_VARYING_VEC2 COMMA_CHAIN(float2, VARYING_VEC2)
#define DEFINE_VARYING_VEC2 STRUCT_CHAIN(float2, VARYING_VEC2)
#define OUTPUT_VARYING_VEC2 STRUCT_CHAIN(float4, VARYING_VEC2)
#define SET_STRUCT_VARYING_VEC2 SET_STRUCT_F4_F2_CHAIN(output, 0, 1, VARYING_VEC2)
#define SET_VARYING_VEC2 SET_VARYING_F2_CHAIN(VARYING_VEC2)
#else
#define PARAM_VARYING_VEC2
#define DEFINE_VARYING_VEC2
#define OUTPUT_VARYING_VEC2
#define SET_STRUCT_VARYING_VEC2
#define SET_VARYING_VEC2
#endif

#ifdef VARYING_VEC3
#define PARAM_VARYING_VEC3 COMMA_CHAIN(float3, VARYING_VEC3)
#define DEFINE_VARYING_VEC3 STRUCT_CHAIN(float3, VARYING_VEC3)
#define OUTPUT_VARYING_VEC3 STRUCT_CHAIN(float4, VARYING_VEC3)
#define SET_STRUCT_VARYING_VEC3 SET_STRUCT_F4_F3_CHAIN(output, 1, VARYING_VEC3)
#define SET_VARYING_VEC3 SET_VARYING_F3_CHAIN(VARYING_VEC3)
#else
#define PARAM_VARYING_VEC3
#define DEFINE_VARYING_VEC3
#define OUTPUT_VARYING_VEC3
#define SET_STRUCT_VARYING_VEC3
#define SET_VARYING_VEC3
#endif

#ifdef VARYING_VEC4
#define PARAM_VARYING_VEC4 COMMA_CHAIN(float3, VARYING_VEC4)
#define DEFINE_VARYING_VEC4 STRUCT_CHAIN(float4, VARYING_VEC4)
#define OUTPUT_VARYING_VEC4 STRUCT_CHAIN(float4, VARYING_VEC4)
#define SET_STRUCT_VARYING_VEC4 SET_STRUCT_CHAIN(output, VARYING_VEC4)
#define SET_VARYING_VEC4 SET_VARYING_CHAIN(VARYING_VEC4)
#else
#define PARAM_VARYING_VEC4
#define DEFINE_VARYING_VEC4
#define OUTPUT_VARYING_VEC4
#define SET_STRUCT_VARYING_VEC4
#define SET_VARYING_VEC4
#endif

#define DEFINE_VARYINGS \
    DEFINE_VARYING_VEC2 \
    DEFINE_VARYING_VEC3 \
    DEFINE_VARYING_VEC4

#define SET_VARYINGS \
    { \
    uint varying = 0; \
    SET_VARYING_VEC2 \
    SET_VARYING_VEC3 \
    SET_VARYING_VEC4 \
    }

typedef struct {
    DEFINE_VARYING_VEC2
    DEFINE_VARYING_VEC3
    DEFINE_VARYING_VEC4
} varying_attributes_t;

// VS definitions

void __attribute__((overloadable)) set_attribute_from_kernel(global const float* in, float2* out) {
    *out = ((global const float2*) in)[get_global_linear_id()];
}
void __attribute__((overloadable)) set_attribute_from_kernel(global const float* in, float3* out) {
    out->x = in[get_global_linear_id() * 3 + 0]; 
    out->y = in[get_global_linear_id() * 3 + 1]; 
    out->z = in[get_global_linear_id() * 3 + 2]; 
}
void __attribute__((overloadable)) set_attribute_from_kernel(global const float* in, float4* out) {
    *out = ((global const float4*) in)[get_global_linear_id()];
}

#ifdef ATTRIBUTE_VEC2
#define KERNEL_ARG_ATTRIBUTE_VEC2 COMMA_CHAIN(global const float*, JOIN_CHAIN(_,ATTRIBUTE_VEC2))
#define DEFINE_ATTRIBUTE_VEC2 STRUCT_CHAIN(float2, ATTRIBUTE_VEC2)
#define SET_ATTRIBUTE_VEC2 SET_ATTRIBUTE_CHAIN(ATTRIBUTE_VEC2)
#else
#define KERNEL_ARG_ATTRIBUTE_VEC2
#define DEFINE_ATTRIBUTE_VEC2
#define SET_ATTRIBUTE_VEC2
#endif

#ifdef ATTRIBUTE_VEC3
#define KERNEL_ARG_ATTRIBUTE_VEC3 COMMA_CHAIN(global const float*, JOIN_CHAIN(_,ATTRIBUTE_VEC3))
#define DEFINE_ATTRIBUTE_VEC3 STRUCT_CHAIN(float3, ATTRIBUTE_VEC3)
#define SET_ATTRIBUTE_VEC3 SET_ATTRIBUTE_CHAIN(ATTRIBUTE_VEC3)
#else
#define KERNEL_ARG_ATTRIBUTE_VEC3
#define DEFINE_ATTRIBUTE_VEC3
#endif

#ifdef VS_UNIFORM_MAT4
#define KERNEL_ARG_VS_UNIFORM_MAT4 COMMA_CHAIN(constant float16*, JOIN_CHAIN(_,VS_UNIFORM_MAT4))
#define DEFINE_VS_UNIFORM_MAT4 STRUCT_CHAIN(float16, VS_UNIFORM_MAT4)
#define SET_VS_UNIFORM_MAT4 SET_UNIFORM_CHAIN(VS_UNIFORM_MAT4)
#else
#define KERNEL_ARG_VS_UNIFORM_MAT4
#define DEFINE_VS_UNIFORM_MAT4
#define SET_VS_UNIFORM_MAT4
#endif

typedef struct {
    float4 gl_Position;
    OUTPUT_VARYING_VEC2
    OUTPUT_VARYING_VEC3
    OUTPUT_VARYING_VEC4
} vertex_shader_output_t;


inline void fill_vertex_buffer(
    vertex_shader_output_t* output, 
    KERNEL_ARG_VERTEX_BUFFER
) {
    size_t id = get_global_linear_id();
    uint output_size = sizeof(vertex_shader_output_t)/sizeof(float4);
    float4* f4_output = (float4*)output;

    #pragma unroll
    for(uint attrib = 0; attrib < output_size; ++attrib) {
        uint offset = id*output_size + attrib;
        float4 value = *(f4_output + attrib);

        #ifdef CONF_FINE_IMAGE_ENABLED
            write_imagef(vertex_buffer, offset, value);
        #else
            vertex_buffer[offset] = value;
        #endif
    }
}

#define VS_MAIN(...) \
    kernel void gl_vertex_shader( \
        KERNEL_ARG_ATTRIBUTE_VEC2 \
        KERNEL_ARG_ATTRIBUTE_VEC3 \
        KERNEL_ARG_VS_UNIFORM_MAT4 \
        KERNEL_ARG_VERTEX_BUFFER \
    ) { \
        /* Define accessible objects from vertex shader */ \
        float4 gl_Position; \
        DEFINE_VARYING_VEC2 \
        DEFINE_VARYING_VEC3 \
        DEFINE_ATTRIBUTE_VEC2 \
        DEFINE_ATTRIBUTE_VEC3 \
        DEFINE_VS_UNIFORM_MAT4 \
        /* Set values from buffers */ \
        SET_ATTRIBUTE_VEC2 \
        SET_ATTRIBUTE_VEC3 \
        SET_VS_UNIFORM_MAT4 \
        /* Run vertex shader */ \
        __VA_ARGS__ \
        /* Fill vertex buffer */ \
        vertex_shader_output_t output; \
        output.gl_Position = gl_Position; \
        SET_STRUCT_VARYING_VEC2 \
        SET_STRUCT_VARYING_VEC3 \
        SET_STRUCT_VARYING_VEC4 \
        fill_vertex_buffer(&output, vertex_buffer); \
    }

// FS definitions

#ifdef FS_UNIFORM_FLOAT
#define FS_KERNEL_PARAM_UNIFORM_FLOAT COMMA_CHAIN(constant float*, JOIN_CHAIN(_,FS_UNIFORM_FLOAT))
#define FS_KERNEL_ARG_UNIFORM_FLOAT COMMA_CHAIN(, JOIN_CHAIN(_,FS_UNIFORM_FLOAT))
#define FS_DEFINE_UNIFORM_FLOAT STRUCT_CHAIN(float, FS_UNIFORM_FLOAT)
#define FS_SET_UNIFORM_FLOAT SET_UNIFORM_CHAIN(FS_UNIFORM_FLOAT)
#else
#define FS_KERNEL_PARAM_UNIFORM_FLOAT
#define FS_KERNEL_ARG_UNIFORM_FLOAT
#define FS_DEFINE_UNIFORM_FLOAT
#define FS_SET_UNIFORM_FLOAT
#endif

#ifdef FS_UNIFORM_SAMPLER2D
#define FS_KERNEL_PARAM_UNIFORM_SAMPLER2D PARAM_SAMPLER_CHAIN(sampler2D_t, image2D_t, FS_UNIFORM_SAMPLER2D)
#define FS_KERNEL_ARG_UNIFORM_SAMPLER2D PARAM_SAMPLER_CHAIN( , ,FS_UNIFORM_SAMPLER2D)
#else
#define FS_KERNEL_PARAM_UNIFORM_SAMPLER2D
#define FS_KERNEL_ARG_UNIFORM_SAMPLER2D
#endif

#define FS_DEFINES \
    float4 gl_FragColor; \
    DEFINE_VARYINGS \
    FS_DEFINE_UNIFORM_FLOAT \

#define FS_SETS \
    SET_VARYINGS \
    FS_SET_UNIFORM_FLOAT

#define FS_KERNEL_PARAMS \
    FS_KERNEL_PARAM_UNIFORM_FLOAT \
    FS_KERNEL_PARAM_UNIFORM_SAMPLER2D

#define FS_KERNEL_ARGS \
    FS_KERNEL_ARG_UNIFORM_FLOAT \
    FS_KERNEL_ARG_UNIFORM_SAMPLER2D

typedef struct {
    float4 gl_FragColor;
    float gl_FragDepth;
} fragment_shader_output_t;

#define FS_MAIN(...) \
    inline bool gl_fragment_shader( \
        FS_KERNEL_PARAMS \
        fragment_shader_output_t* output, \
        KERNEL_ARG_VERTEX_BUFFER, \
        uint3 vert_idx, float3 bary \
    ) { \
        /* Define accessible objects from fragment shader */ \
        FS_DEFINES \
        /* Set values from vertex_buffer */ \
        FS_SETS \
        /* Run vertex shader */ \
        __VA_ARGS__ \
        /* Fill fragment shader output */ \
        output->gl_FragColor = gl_FragColor; \
        return true; \
    }


inline float4 get_varying_at_vertex(
    int varying_idx, int vert_idx,
    #ifdef CONF_FINE_IMAGE_ENABLED
    image1d_buffer_t t_vertex_buffer
    #else
    global const float4* g_vertex_buffer
    #endif
) {
    size_t idx =  vert_idx * (sizeof(vertex_shader_output_t) / sizeof(float4)) + varying_idx + 1;
    #ifdef CONF_FINE_IMAGE_ENABLED
    return read_imagef(t_vertex_buffer, idx);
    #else
    return g_vertex_buffer[idx];
    #endif
}

inline float4 interpolate_varying(
    int varying_idx, const uint3 vert_idx, const float3 bary, 
    #ifdef CONF_FINE_IMAGE_ENABLED
    image1d_buffer_t vertex_buffer
    #else
    global const float4* vertex_buffer
    #endif
) {
    float4 v0 = get_varying_at_vertex(varying_idx, vert_idx.x, vertex_buffer);
    float4 v1 = get_varying_at_vertex(varying_idx, vert_idx.y, vertex_buffer);
    float4 v2 = get_varying_at_vertex(varying_idx, vert_idx.z, vertex_buffer);
    return v0 * bary.x + v1 * bary.y + v2 * bary.z; 
}


// Data reflection for OpenGL frontend

kernel void gl_tmp_fragment_shader(
    FS_KERNEL_PARAMS
    private int test
) {
    return;
}

kernel void gl_varying_data(
    PARAM_VARYING_VEC2
    PARAM_VARYING_VEC3
    PARAM_VARYING_VEC4
    private int test
) {
    return;
}

#endif
