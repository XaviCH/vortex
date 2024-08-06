#ifndef __types_h__
#define __types_h__ 1

#include <CL/opencl.h>
#include <GLSC2/glsc2.h>
#include "config.h"

typedef struct {
    GLint x, y;
    GLsizei width, height;
} box_t;

typedef struct {
    float values[4]; // default: {0.0, 0.0, 0.0, 1.0} 
} vertex_attrib_t;

typedef struct {
    GLint size;
    GLenum type;
    GLboolean normalized;
    GLsizei stride;
    void *pointer;
    GLuint binding;
} vertex_attrib_pointer_t;

typedef union {
    vertex_attrib_t vec4;
    vertex_attrib_pointer_t pointer;
} vertex_attrib_u;

typedef struct {
    unsigned int vertex_location, fragment_location, size, type;
    unsigned char name[ARG_NAME_SIZE]; 
} arg_data_t;

typedef struct { 
    GLboolean last_load_attempt, last_validation_attempt;
    unsigned char log[PROGRAM_LOG_SIZE];
} program_data_t;

typedef struct {
    // Program data
    GLboolean               last_load_attempt; 
    GLboolean               last_validation_attempt;
    unsigned char           log[PROGRAM_LOG_SIZE];
    unsigned int            log_length;
    cl_program              program;
    // Kernel data
    cl_kernel               vertex_kernel;
    cl_kernel               fragment_kernel;
    // Uniform data
    unsigned int            active_uniforms;
    arg_data_t              uniforms_data[MAX_UNIFORM_SIZE];
    unsigned int            uniforms_array_size[MAX_UNIFORM_SIZE];
    cl_mem                  uniforms_mem[MAX_UNIFORM_SIZE];
    // Vertex data
    unsigned int            active_vertex_attribs;
    arg_data_t              vertex_attribs_data[MAX_VERTEX_ATTRIBS];
    unsigned int            gl_position_location;
    // Varying data
    unsigned int            varying_size;
    arg_data_t              varying_data[16];
    // Texture data
    unsigned int            texture_unit_size;
    uint32_t                sampler_value[MAX_COMBINED_TEXTURE_IMAGE_UNITS]; // points to an active texture unit
    arg_data_t              sampler_data[MAX_COMBINED_TEXTURE_IMAGE_UNITS];
    arg_data_t              image_data[MAX_COMBINED_TEXTURE_IMAGE_UNITS];
    // Fragment data
    unsigned int            gl_fragcolor_location;
    int                     gl_fragcoord_location;
    int                     gl_discard_location;
} program_t;

typedef struct {
    cl_kernel points, line_stip, line_loop, lines, triangle_strip, triangle_fan, triangles;
} rasterization_kernel_container_t;

typedef struct {
    cl_kernel rgba4, rgba8;
} readnpixels_kernel_container_t;

typedef struct {
    rasterization_kernel_container_t rasterization;
    readnpixels_kernel_container_t readnpixels;
    cl_kernel viewport_division, perspective_division, strided_write, depth_test, stencil_test, scissor_test, blending, clear, dithering;
} kernel_container_t;

typedef struct {
    GLboolean depth_test, stencil_test, scissor_test, pixel_ownership, dither, blend, cull_face;
} enabled_container_t;

typedef struct {
    GLboolean r,g,b,a;
} color_mask_t;

typedef struct {
    GLuint front, back;
} stencil_mask_t;

typedef struct {
    color_mask_t color;
    GLboolean depth;
    stencil_mask_t stencil;
} mask_container_t;

typedef struct
{
    uint8_t unpack_aligment;
} pixel_store_t;

typedef struct {
    uint32_t s, t, min_filter, mag_filter;
} texture_wraps_t;

typedef struct
{
    texture_wraps_t wraps;
    uint32_t width, height;
    GLenum internalformat;
    cl_mem mem;
    GLboolean used;
} texture_t;

typedef struct
{
    uint32_t binding;
} active_texture_t;

typedef struct {
    GLboolean used;
    GLenum target;
    void* mem;
} buffer_t;

typedef struct {
    GLenum target;
    uint32_t position;
} attachment_t;

typedef struct {
    attachment_t color_attachment0, depth_attachment, stencil_attachment;
    GLboolean used;
} framebuffer_t;

typedef struct {
    cl_mem mem;
    GLenum internalformat;
    GLsizei width, height;
    GLboolean used;
} renderbuffer_t;

typedef struct { GLfloat n, f; } depth_range_t;

typedef struct { 
    GLenum func; 
    GLint ref; 
    GLuint mask; 
} stencil_function_t;

typedef struct { 
    GLenum sfail,  dpfail,  dppasss; 
} stencil_operation_t;

typedef struct { 
    stencil_function_t function; 
    stencil_operation_t operation;
} stencil_face_data_t;

typedef struct { 
    stencil_face_data_t front, back; 
} stencil_data_t;

typedef struct {
    GLint left, bottom;
    GLsizei width, height; 
} scissor_data_t;

typedef struct {
    GLenum modeRGB, modeAlpha;
} blend_equation_t;

typedef struct {
    GLenum srcRGB, dstRGB, srcAlpha, dstAlpha;
} blend_func_t;

typedef struct {
    GLfloat red, green, blue, alpha;
} blend_color_t;

typedef struct {
    blend_equation_t equation;
    blend_func_t func;
    blend_color_t color;
} blend_data_t;

typedef struct {
    GLfloat red, green, blue, alpha;
} clear_color_t;

typedef struct {
    clear_color_t color;
    GLfloat depth; 
    GLint stencil;
} clear_data_t;

#endif