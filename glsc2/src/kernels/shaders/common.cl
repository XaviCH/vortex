#ifndef KERNELS_SHADERS_COMMON_CL
#define KERNELS_SHADERS_COMMON_CL

#ifndef VERTEX_INPUT
#define VERTEX_INPUT int error;
#error VERTEX_INPUT not defined
#endif

#ifndef KERNEL_VERTEX_INPUT
#define KERNEL_VERTEX_INPUT int error,
#error KERNEL_VERTEX_INPUT not defined
#endif

#ifndef VERTEX_UNIFORMS
#define VERTEX_UNIFORMS
#endif

#ifndef VARYING_ATTRIBUTES
#define VARYING_ATTRIBUTES
#endif

// Scope to parse vertex kernel input to vertex attributes input
#ifndef VERTEX_KERNEL_TO_INPUT
#define VERTEX_KERNEL_TO_INPUT {}
#error VERTEX_KERNEL_TO_INPUT not defined
#endif


typedef struct {
    VERTEX_INPUT
} vertex_shader_input_t;

typedef struct {
    float4 gl_Position; // gl_position
    VARYING_ATTRIBUTES
} vertex_shader_output_t;

inline void vertex_shader(vertex_shader_input_t*, vertex_shader_output_t*);

kernel void gl_vertex_shader(
    KERNEL_VERTEX_INPUT

    #ifdef CONF_FINE_IMAGE_ENABLED
    write_only image1d_buffer_t t_vertex_buffer
    #else
    global float4* g_vertex_buffer
    #endif
) {
    vertex_shader_input_t input;
    vertex_shader_output_t output;

    size_t id = get_global_linear_id();
    
    VERTEX_KERNEL_TO_INPUT;

    vertex_shader(&input, &output);

    uint output_size = sizeof(vertex_shader_output_t)/sizeof(float4);
    float4* f4_output = (float4*)&output;

    #pragma unroll
    for(uint attrib = 0; attrib < output_size; ++attrib) {
        uint offset = id*output_size + attrib;
        float4 value = *(f4_output + attrib);

        #ifdef CONF_FINE_IMAGE_ENABLED
        {
            write_imagef(t_vertex_buffer, offset, value);
        }
        #else
        {
            g_vertex_buffer[offset] = value;
        }
        #endif
    }

}


// Fragment Shader

#ifndef FRAGMENT_INPUT
#define FRAGMENT_INPUT
#endif

#ifndef KERNEL_FRAGMENT_INPUT
#define KERNEL_FRAGMENT_INPUT
#endif

#ifndef KERNEL_FRAGMENT_INPUT_NAMES
#define KERNEL_FRAGMENT_INPUT_NAMES
#endif

// Scope to parse fragment kernel input to fragment uniforms input
#ifndef FRAGMENT_KERNEL_TO_INPUT
#define FRAGMENT_KERNEL_TO_INPUT {}
#endif

typedef struct {
    ushort s, t, min_filter, mag_filter;
} texture_wraps_t;

typedef struct __attribute__((packed)) {
  uint width, height;
  uint internalformat;
  texture_wraps_t wraps;
} sampler2D_t;

typedef struct {
    VARYING_ATTRIBUTES
} varying_attributes_t;

typedef struct {
    VARYING_ATTRIBUTES
    FRAGMENT_INPUT
} fragment_shader_input_t;

typedef struct {
    float4 gl_FragColor;
    float gl_FragDepth;
} fragment_shader_output_t;

inline bool fragment_shader(fragment_shader_input_t*, fragment_shader_output_t*);

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

/**
    @pre
 */
inline bool gl_fragment_shader(
    KERNEL_FRAGMENT_INPUT

    fragment_shader_output_t* output,
    #ifdef CONF_FINE_IMAGE_ENABLED
    image1d_buffer_t vertex_buffer,
    #else
    global const float4* vertex_buffer,
    #endif
    uint3 vert_idx,
    float3 bary
) {
    fragment_shader_input_t input;
    
    float4* f4_input = (float4*) &input;
    #pragma unroll
    for(uint attrib = 0; attrib < sizeof(varying_attributes_t)/sizeof(float4); ++attrib) {
        f4_input[attrib] = interpolate_varying(attrib, vert_idx, bary, vertex_buffer);
    }

    FRAGMENT_KERNEL_TO_INPUT;

    return fragment_shader(&input, output);
}

#endif
