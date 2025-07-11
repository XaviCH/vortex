/**
 * Wrapper to simplify creating a shader using OpenCL
 * 
 * TODO: Add GLSL typing and built-in functions
 */

#ifndef KERNELS_SHADERS_WRAPPER_CL
#define KERNELS_SHADERS_WRAPPER_CL

#ifdef __COMPILER_RELATIVE_PATH__
#include "macros.h"
#include "built_in.cl"
#include "../../constants.h"
#else
#include "glsc2/src/kernels/shaders/macros.h"
#include "glsc2/src/kernels/shaders/built_in.cl"
#include "glsc2/src/constants.h"
#endif

// Utils

#ifdef CONF_FINE_IMAGE_ENABLED
#define KERNEL_ARG_VERTEX_BUFFER write_only image1d_buffer_t vertex_buffer
typedef write_only image1d_buffer_t vertex_buffer_t;
#else
#define KERNEL_ARG_VERTEX_BUFFER global float4* vertex_buffer
typedef global float4* vertex_buffer_t;
#endif

#define TEXTURE2D(sampler, coord) texture2D(sampler, gl_image_##sampler, coord);

// Varying definitions

#ifdef VARYING_FLOAT
#define PARAM_VARYING_FLOAT COMMA_CHAIN(float, VARYING_FLOAT)
#define DEFINE_VARYING_FLOAT STRUCT_CHAIN(float, VARYING_FLOAT)
#define OUTPUT_VARYING_FLOAT STRUCT_CHAIN(float4, VARYING_FLOAT)
#define SET_STRUCT_VARYING_FLOAT SET_STRUCT_F4_F1_CHAIN(output, 0, 0, 1, VARYING_FLOAT)
#define SET_VARYING_FLOAT SET_VARYING_F1_CHAIN(VARYING_FLOAT)
#else
#define PARAM_VARYING_FLOAT
#define DEFINE_VARYING_FLOAT
#define OUTPUT_VARYING_FLOAT
#define SET_STRUCT_VARYING_FLOAT
#define SET_VARYING_FLOAT
#endif

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
    DEFINE_VARYING_FLOAT \
    DEFINE_VARYING_VEC2 \
    DEFINE_VARYING_VEC3 \
    DEFINE_VARYING_VEC4

#define OUTPUT_VARYINGS \
    OUTPUT_VARYING_FLOAT \
    OUTPUT_VARYING_VEC2 \
    OUTPUT_VARYING_VEC3 \
    OUTPUT_VARYING_VEC4

#define PARAM_VARYINGS \
    PARAM_VARYING_FLOAT \
    PARAM_VARYING_VEC2 \
    PARAM_VARYING_VEC3 \
    PARAM_VARYING_VEC4

#define SET_STRUCT_VARYINGS \
    SET_STRUCT_VARYING_FLOAT \
    SET_STRUCT_VARYING_VEC2 \
    SET_STRUCT_VARYING_VEC3 \
    SET_STRUCT_VARYING_VEC4

#define SET_VARYINGS \
    { \
    uint varying = 0; \
    SET_VARYING_FLOAT \
    SET_VARYING_VEC2 \
    SET_VARYING_VEC3 \
    SET_VARYING_VEC4 \
    }


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

void __attribute__((overloadable)) set_attribute_from_float4(float4 in, float2* out) {
    *out = in.xy;
}
void __attribute__((overloadable)) set_attribute_from_float4(float4 in, float3* out) {
    *out = in.xyz;
}
void __attribute__((overloadable)) set_attribute_from_float4(float4 in, float4* out) {
    *out = in;
}

//-------------

#ifdef ATTRIBUTE_VEC2
#define COUNT_ATTRIBUTE_VEC2 COUNT(ATTRIBUTE_VEC2)
#define KERNEL_ARG_ATTRIBUTE_VEC2 COMMA_CHAIN(global const float*, JOIN_CHAIN(_,ATTRIBUTE_VEC2))
#define DEFINE_ATTRIBUTE_VEC2 STRUCT_CHAIN(float2, ATTRIBUTE_VEC2)
#define SET_ATTRIBUTE_VEC2 SET_ATTRIBUTE_CHAIN(ATTRIBUTE_VEC2)
#else
#define COUNT_ATTRIBUTE_VEC2 0
#define KERNEL_ARG_ATTRIBUTE_VEC2
#define DEFINE_ATTRIBUTE_VEC2
#define SET_ATTRIBUTE_VEC2
#endif

#ifdef ATTRIBUTE_VEC3
#define COUNT_ATTRIBUTE_VEC3 COUNT(ATTRIBUTE_VEC3)
#define KERNEL_ARG_ATTRIBUTE_VEC3 COMMA_CHAIN(global const float*, JOIN_CHAIN(_,ATTRIBUTE_VEC3))
#define DEFINE_ATTRIBUTE_VEC3 STRUCT_CHAIN(float3, ATTRIBUTE_VEC3)
#define SET_ATTRIBUTE_VEC3 SET_ATTRIBUTE_CHAIN(ATTRIBUTE_VEC3)
#else
#define COUNT_ATTRIBUTE_VEC3 0
#define KERNEL_ARG_ATTRIBUTE_VEC3
#define DEFINE_ATTRIBUTE_VEC3
#define SET_ATTRIBUTE_VEC3
#endif

#ifdef ATTRIBUTE_VEC4
#define COUNT_ATTRIBUTE_VEC4 COUNT(ATTRIBUTE_VEC4)
#define KERNEL_ARG_ATTRIBUTE_VEC4 COMMA_CHAIN(global const float*, JOIN_CHAIN(_,ATTRIBUTE_VEC4))
#define DEFINE_ATTRIBUTE_VEC4 STRUCT_CHAIN(float4, ATTRIBUTE_VEC4)
#define SET_ATTRIBUTE_VEC4 SET_ATTRIBUTE_CHAIN(ATTRIBUTE_VEC4)
#else
#define COUNT_ATTRIBUTE_VEC4 0
#define KERNEL_ARG_ATTRIBUTE_VEC4
#define DEFINE_ATTRIBUTE_VEC4
#define SET_ATTRIBUTE_VEC4
#endif

#define COUNT_ATTRIBUTES ( \
    COUNT_ATTRIBUTE_VEC2 + \
    COUNT_ATTRIBUTE_VEC3 + \
    COUNT_ATTRIBUTE_VEC4 )

#define DEFINE_ATTRIBUTES \
    DEFINE_ATTRIBUTE_VEC2 \
    DEFINE_ATTRIBUTE_VEC3 \
    DEFINE_ATTRIBUTE_VEC4

#define KERNEL_ARG_ATTRIBUTES \
    KERNEL_ARG_ATTRIBUTE_VEC2 \
    KERNEL_ARG_ATTRIBUTE_VEC3 \
    KERNEL_ARG_ATTRIBUTE_VEC4

#define SET_ATTRIBUTES \
    { \
    uint gl_attribute_location = 0; \
    SET_ATTRIBUTE_VEC2 \
    SET_ATTRIBUTE_VEC3 \
    SET_ATTRIBUTE_VEC4 \
    }

//-------------------

#ifdef VS_UNIFORM_FLOAT
#define VS_KERNEL_ARG_UNIFORM_FLOAT COMMA_CHAIN(constant float*, JOIN_CHAIN(_,VS_UNIFORM_FLOAT))
#define VS_DEFINE_UNIFORM_FLOAT STRUCT_CHAIN(float, VS_UNIFORM_FLOAT)
#define VS_SET_UNIFORM_FLOAT SET_UNIFORM_CHAIN(VS_UNIFORM_FLOAT)
#else
#define VS_KERNEL_ARG_UNIFORM_FLOAT
#define VS_DEFINE_UNIFORM_FLOAT
#define VS_SET_UNIFORM_FLOAT
#endif

#ifdef VS_UNIFORM_INT2
#define VS_KERNEL_ARG_UNIFORM_INT2 COMMA_CHAIN(constant int2*, JOIN_CHAIN(_,VS_UNIFORM_INT2))
#define VS_DEFINE_UNIFORM_INT2 STRUCT_CHAIN(int2, VS_UNIFORM_INT2)
#define VS_SET_UNIFORM_INT2 SET_UNIFORM_CHAIN(VS_UNIFORM_INT2)
#else
#define VS_KERNEL_ARG_UNIFORM_INT2
#define VS_DEFINE_UNIFORM_INT2
#define VS_SET_UNIFORM_INT2
#endif

#ifdef VS_UNIFORM_VEC4
#define VS_KERNEL_ARG_UNIFORM_VEC4 COMMA_CHAIN(constant float4*, JOIN_CHAIN(_,VS_UNIFORM_VEC4))
#define VS_DEFINE_UNIFORM_VEC4 STRUCT_CHAIN(float4, VS_UNIFORM_VEC4)
#define VS_SET_UNIFORM_VEC4 SET_UNIFORM_CHAIN(VS_UNIFORM_VEC4)
#else
#define VS_KERNEL_ARG_UNIFORM_VEC4
#define VS_DEFINE_UNIFORM_VEC4
#define VS_SET_UNIFORM_VEC4
#endif

#ifdef VS_UNIFORM_MAT4
#define VS_KERNEL_ARG_UNIFORM_MAT4 COMMA_CHAIN(constant float16*, JOIN_CHAIN(_,VS_UNIFORM_MAT4))
#define VS_DEFINE_UNIFORM_MAT4 STRUCT_CHAIN(float16, VS_UNIFORM_MAT4)
#define VS_SET_UNIFORM_MAT4 SET_UNIFORM_CHAIN(VS_UNIFORM_MAT4)
#else
#define VS_KERNEL_ARG_UNIFORM_MAT4
#define VS_DEFINE_UNIFORM_MAT4
#define VS_SET_UNIFORM_MAT4
#endif

#define VS_DEFINE_UNIFORMS \
    VS_DEFINE_UNIFORM_FLOAT \
    VS_DEFINE_UNIFORM_INT2 \
    VS_DEFINE_UNIFORM_VEC4 \
    VS_DEFINE_UNIFORM_MAT4

#define VS_KERNEL_ARG_UNIFORMS \
    VS_KERNEL_ARG_UNIFORM_FLOAT \
    VS_KERNEL_ARG_UNIFORM_INT2 \
    VS_KERNEL_ARG_UNIFORM_VEC4 \
    VS_KERNEL_ARG_UNIFORM_MAT4

#define VS_SET_UNIFORMS \
    VS_SET_UNIFORM_FLOAT \
    VS_SET_UNIFORM_INT2 \
    VS_SET_UNIFORM_VEC4 \
    VS_SET_UNIFORM_MAT4

//------------------------

typedef struct {
    float4 gl_Position;
    OUTPUT_VARYINGS
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

// TODO: Mem acces on wrong padding cause error
inline float4 gl_get_vertex_attribute_from_pointer(global void* data, vertex_attribute_data_t vertex_attribute_data) {
    data += vertex_attribute_data.offset;

    switch (vertex_attribute_data.misc & (VERTEX_ATTRIBUTE_SIZE_MASK | VERTEX_ATTRIBUTE_TYPE_MASK)) {
        default:
        case VERTEX_ATTRIBUTE_SIZE_1 | VERTEX_ATTRIBUTE_TYPE_FLOAT:
            data += (sizeof(float) + vertex_attribute_data.stride) * get_global_linear_id();
            return (float4)(
                *(global float*)data, 
                0, 
                0, 
                1
                );
        case VERTEX_ATTRIBUTE_SIZE_2 | VERTEX_ATTRIBUTE_TYPE_FLOAT:
            data += (sizeof(float2) + vertex_attribute_data.stride) * get_global_linear_id();
            return (float4)(
                ((global float*)data)[0], 
                ((global float*)data)[1], 
                0, 
                1
                );
        case VERTEX_ATTRIBUTE_SIZE_3 | VERTEX_ATTRIBUTE_TYPE_FLOAT:
            data += (sizeof(float[3]) + vertex_attribute_data.stride) * get_global_linear_id();
            return (float4)(
                ((global float*)data)[0],
                ((global float*)data)[1],
                ((global float*)data)[2],
                1
                );
        case VERTEX_ATTRIBUTE_SIZE_4 | VERTEX_ATTRIBUTE_TYPE_FLOAT:
            data += (sizeof(float4) + vertex_attribute_data.stride) * get_global_linear_id();
            return (float4)(
                ((global float*)data)[0],
                ((global float*)data)[1],
                ((global float*)data)[2],
                ((global float*)data)[3]
                );
    };
}

#define VS_MAIN(...) \
    kernel void gl_vertex_shader( \
        global float4* vertex_attributes, \
        global vertex_attribute_data_t* vertex_attribute_datas, \
        KERNEL_ARG_ATTRIBUTES \
        VS_KERNEL_ARG_UNIFORMS \
        KERNEL_ARG_VERTEX_BUFFER \
    ) { \
        /* Define accessible objects from vertex shader */ \
        float4 gl_Position; \
        DEFINE_VARYINGS \
        DEFINE_ATTRIBUTES \
        VS_DEFINE_UNIFORMS \
        /* Set values from buffers */ \
        SET_ATTRIBUTES \
        VS_SET_UNIFORMS \
        /* Run vertex shader */ \
        __VA_ARGS__ \
        /* Fill vertex buffer */ \
        vertex_shader_output_t output; \
        output.gl_Position = gl_Position; \
        SET_STRUCT_VARYINGS \
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

#ifdef FS_UNIFORM_INT
#define FS_KERNEL_PARAM_UNIFORM_INT COMMA_CHAIN(constant int*, JOIN_CHAIN(_,FS_UNIFORM_INT))
#define FS_KERNEL_ARG_UNIFORM_INT COMMA_CHAIN(, JOIN_CHAIN(_,FS_UNIFORM_INT))
#define FS_DEFINE_UNIFORM_INT STRUCT_CHAIN(int, FS_UNIFORM_INT)
#define FS_SET_UNIFORM_INT SET_UNIFORM_CHAIN(FS_UNIFORM_INT)
#else
#define FS_KERNEL_PARAM_UNIFORM_INT
#define FS_KERNEL_ARG_UNIFORM_INT
#define FS_DEFINE_UNIFORM_INT
#define FS_SET_UNIFORM_INT
#endif

#ifdef FS_UNIFORM_VEC4
#define FS_KERNEL_PARAM_UNIFORM_VEC4 COMMA_CHAIN(constant float4*, JOIN_CHAIN(_,FS_UNIFORM_VEC4))
#define FS_KERNEL_ARG_UNIFORM_VEC4 COMMA_CHAIN(, JOIN_CHAIN(_,FS_UNIFORM_VEC4))
#define FS_DEFINE_UNIFORM_VEC4 STRUCT_CHAIN(float4, FS_UNIFORM_VEC4)
#define FS_SET_UNIFORM_VEC4 SET_UNIFORM_CHAIN(FS_UNIFORM_VEC4)
#else
#define FS_KERNEL_PARAM_UNIFORM_VEC4
#define FS_KERNEL_ARG_UNIFORM_VEC4
#define FS_DEFINE_UNIFORM_VEC4
#define FS_SET_UNIFORM_VEC4
#endif

#ifdef FS_UNIFORM_SAMPLER2D
#define FS_KERNEL_PARAM_UNIFORM_SAMPLER2D PARAM_SAMPLER_CHAIN(sampler2D_t, image2D_t, FS_UNIFORM_SAMPLER2D)
#define FS_KERNEL_ARG_UNIFORM_SAMPLER2D PARAM_SAMPLER_CHAIN( , ,FS_UNIFORM_SAMPLER2D)
#else
#define FS_KERNEL_PARAM_UNIFORM_SAMPLER2D
#define FS_KERNEL_ARG_UNIFORM_SAMPLER2D
#endif

//------------

#define FS_DEFINES \
    float4 gl_FragColor; \
    DEFINE_VARYINGS \
    FS_DEFINE_UNIFORM_FLOAT \
    FS_DEFINE_UNIFORM_INT \
    FS_DEFINE_UNIFORM_VEC4

#define FS_SETS \
    SET_VARYINGS \
    FS_SET_UNIFORM_FLOAT \
    FS_SET_UNIFORM_INT \
    FS_SET_UNIFORM_VEC4

#define FS_KERNEL_PARAMS \
    FS_KERNEL_PARAM_UNIFORM_FLOAT \
    FS_KERNEL_PARAM_UNIFORM_INT \
    FS_KERNEL_PARAM_UNIFORM_VEC4 \
    FS_KERNEL_PARAM_UNIFORM_SAMPLER2D

#define FS_KERNEL_ARGS \
    FS_KERNEL_ARG_UNIFORM_FLOAT \
    FS_KERNEL_ARG_UNIFORM_INT \
    FS_KERNEL_ARG_UNIFORM_VEC4 \
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
    PARAM_VARYINGS
    private int test
) {
    return;
}

#endif
