#ifndef BACKEND_SHADERS_VERTEX_CL
#define BACKEND_SHADERS_VERTEX_CL

#ifdef __COMPILER_RELATIVE_PATH__
    #include <backend/shaders/wrapper.cl>
    #include <backend/types.cl>
    #include <constants.h>
#else
    #include "glsc2/src/backend/shaders/wrapper.cl"
    #include "glsc2/src/backend/types.cl"
    #include "glsc2/src/constants.h"
#endif


inline float4 gl_get_vertex_attribute_from_pointer(global void* data, vertex_attribute_data_t vertex_attribute_data) {
    data += vertex_attribute_data.offset;

    switch (vertex_attribute_data.misc & (VERTEX_ATTRIBUTE_SIZE_MASK | VERTEX_ATTRIBUTE_TYPE_MASK)) {
        default:
        case (VERTEX_ATTRIBUTE_SIZE_1 << 3) | VERTEX_ATTRIBUTE_TYPE_FLOAT:
            data += (vertex_attribute_data.stride) * get_global_linear_id();
            return (float4)(
                *(global float*)data, 
                0, 
                0, 
                1
                );
        case (VERTEX_ATTRIBUTE_SIZE_2 << 3) | VERTEX_ATTRIBUTE_TYPE_FLOAT:
            data += (vertex_attribute_data.stride) * get_global_linear_id();
            return (float4)(
                ((global float*)data)[0],
                ((global float*)data)[1], 
                0, 
                1
                );
        case (VERTEX_ATTRIBUTE_SIZE_3 << 3) | VERTEX_ATTRIBUTE_TYPE_FLOAT:
            data += (vertex_attribute_data.stride) * get_global_linear_id();
            return (float4)(
                ((global float*)data)[0],
                ((global float*)data)[1],
                ((global float*)data)[2],
                1
                );
        case (VERTEX_ATTRIBUTE_SIZE_4 << 3) | VERTEX_ATTRIBUTE_TYPE_FLOAT:
            data += (vertex_attribute_data.stride) * get_global_linear_id();
            return (float4)(
                ((global float*)data)[0],
                ((global float*)data)[1],
                ((global float*)data)[2],
                ((global float*)data)[3]
                );
    };
}

inline float4 gl_get_vertex_attribute(
    global const float4* vertex_attributes, 
    global const vertex_attribute_data_t* vertex_attribute_datas,
    int attribute_location
) {
    if ((vertex_attribute_datas[attribute_location].misc & VERTEX_ATTRIBUTE_ACTIVE_POINTER) != 0)
        return gl_get_vertex_attribute_from_pointer((global void*)vertex_attributes, vertex_attribute_datas[attribute_location]);
    
    return vertex_attributes[attribute_location];
}

#define GL_SET_VERTEX_ATTRIBUTE_ARGS \
    global void* src, \
    global const float4* vertex_attributes, \
    global const vertex_attribute_data_t* vertex_attribute_datas, \
    int attribute_location

inline void __attribute__((overloadable)) gl_set_vertex_attribute(float2* dst, GL_SET_VERTEX_ATTRIBUTE_ARGS)
{
    *dst = gl_get_vertex_attribute(vertex_attributes, vertex_attribute_datas, attribute_location).xy;
}
inline void __attribute__((overloadable)) gl_set_vertex_attribute(float3* dst, GL_SET_VERTEX_ATTRIBUTE_ARGS)
{
    *dst = gl_get_vertex_attribute(vertex_attributes, vertex_attribute_datas, attribute_location).xyz;
}
inline void __attribute__((overloadable)) gl_set_vertex_attribute(float4* dst, GL_SET_VERTEX_ATTRIBUTE_ARGS)
{
    *dst = gl_get_vertex_attribute(vertex_attributes, vertex_attribute_datas, attribute_location);
}

#undef GL_SET_VERTEX_ATTRIBUTE_ARGS

inline void gl_fill_vertex_buffer(
    vertex_shader_output_t* output, 
    wo_vertex_buffer_t vertex_buffer
) {
    size_t id = get_global_id(0); // Enabled for multiple kernels synchronously writing to the same buffer.
    uint output_size = sizeof(vertex_shader_output_t)/sizeof(float4);
    float4* f4_output = (float4*)output;

    #pragma unroll
    for(uint attrib = 0; attrib < output_size; ++attrib) {
        uint offset = id*output_size + attrib;
        float4 value = *(f4_output + attrib);

        #ifdef DEVICE_IMAGE_ENABLED
            write_imagef(vertex_buffer, offset, value);
        #else
            vertex_buffer[offset] = value;
        #endif
    }
}

/**
 * TODO: 
 *     Implement constant memory.
 *     KERNEL_ARG_ATTRIBUTES could be one dimensional array,
 *     and vertex_attribute_datas could contain the offsets.
 */
#define VS_MAIN(...) \
    kernel void gl_vertex_shader( \
        global const float4* vertex_attributes, \
        global const vertex_attribute_data_t* vertex_attribute_datas, \
        global const void* gl_uniforms, \
        VS_KERNEL_PARAMS \
        wo_vertex_buffer_t vertex_buffer \
    ) { \
        /* Define accessible objects from vertex shader */ \
        VS_DEFINES \
        /* Set values from buffers */ \
        VS_SETS \
        /* Run vertex shader */ \
        __VA_ARGS__ \
        /* Fill vertex buffer */ \
        vertex_shader_output_t output; \
        output.gl_Position = gl_Position; \
        SET_STRUCT_VARYINGS \
        gl_fill_vertex_buffer(&output, vertex_buffer); \
    }

#endif // BACKEND_SHADERS_VERTEX_CL