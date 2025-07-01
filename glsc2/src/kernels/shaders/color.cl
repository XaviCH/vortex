// 
#define VERTEX_INPUT \
    float3 position; \
    float3 color; \
    float16 perspective; \
    float16 view; \
    float16 model;

// Kernel vertex inputs
#define KERNEL_VERTEX_INPUT \
    global const float* position, \
    global const float* color, \
    constant float16 *perspective, \
    constant float16 *view, \
    constant float16 *model,

#define VARYING_ATTRIBUTES \
    float4 color;

#define VERTEX_KERNEL_TO_INPUT \
    {   \
        global const float* pos_off = position + id*3; \
        input.position = (float3){pos_off[0], pos_off[1], pos_off[2]}; \
        global const float* col_off = color + id*3; \
        input.color = (float3){col_off[0], col_off[1], col_off[2]}; \
        input.perspective = *perspective; \
        input.view = *view; \
        input.model = *model; \
    }

#ifdef __COMPILER_RELATIVE_PATH__
#include "common.cl"
#else
#include "glsc2/src/kernels/shaders/common.cl"
#endif


inline void vertex_shader(
    vertex_shader_input_t* input,
    vertex_shader_output_t* output
) {
    output->gl_Position = (float4){input->position, 1};
    output->color       = (float4){input->color, 1};
}

// return true if fragment is not discarded, otherwise false
inline bool fragment_shader(
    fragment_shader_input_t* input, 
    fragment_shader_output_t* output
) {
    output->gl_FragColor = (float4){input->color.xyz, 0.5f};
    return true;
}
