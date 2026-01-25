#ifndef FRONTEND_DEVICE_H
#define FRONTEND_DEVICE_H

#include <CL/opencl.h>
#include <string.h>
#include <stdio.h>

#include <backend/pipeline/triangle_setup.o.c>
#include <backend/pipeline/bin_raster.o.c>
#include <backend/pipeline/coarse_raster.o.c>
#include <backend/pipeline/force_clear.o.c>
#include <constants.device.h>
#include <frontend/types.h>


#define CL_UNSUPORTED_MAPING(...) \
{                                                                       \
    printf("OpenCL unsupported mapping at %s:%d. " #__VA_ARGS__ "\n", __FILE__, __LINE__);    \
    exit(-1);                                                      \
}

#define CL_CHECK(...)       \
{                                                                       \
    cl_int __error = __VA_ARGS__;                                             \
    if (__error != CL_SUCCESS) {                                   \
    printf("OpenCL error at %s:%d. " #__VA_ARGS__ " returned %d.\n", __FILE__, __LINE__, (int)__error);    \
    exit(__error);                                                      \
    } \
}

#define CL_ASSIGN_CHECK(_LEFT, ...)                                     \
{                                                                       \
    cl_int error;                                                       \
    _LEFT = __VA_ARGS__;                                                      \
    if (error != CL_SUCCESS) {                                     \
    printf("OpenCL error at %s:%d. " #__VA_ARGS__ " returned %d.\n", __FILE__, __LINE__, (int)error);      \
    exit(error);   \
    }                                                     \
}


typedef struct {
    cl_kernel kernel;
    uint32_t  number_attributes;
    uint32_t  vertex_size;
} vertex_data_t;

typedef struct {
    vertex_data_t vertex;
    cl_kernel fragment, uniform_data, attribute_data, varying_data;
} device_shader_kernels_t;

typedef struct {
    cl_command_queue queues[DEVICE_BIN_QUEUE_SIZE][DEVICE_BIN_QUEUE_SIZE]; // TODO: multiple queues
} bin_queue_t;

typedef struct {
    cl_platform_id          platform_id;
    cl_device_id            device_id;
    cl_context              context;

    cl_program              triangle_setup_program;
    cl_program              bin_raster_program;
    cl_program              coarse_raster_program;
    cl_program              clear_program;

    cl_mem                  mem_vertex_attrib_pointer;  // dummy attrib pointers
    cl_mem                  mem_texture;                // dummy texture
    cl_mem                  mem_depthbuffer;            // dummy depthbuffer
    cl_mem                  mem_stencilbuffer;          // dummy stencilbuffer
    
    size_t                program_size;
    cl_program                      programs        [HOST_PROGRAMS_SIZE];
    device_shader_kernels_t    shaders      [HOST_PROGRAMS_SIZE];

    size_t               buffers_size;
    cl_mem                  buffers         [HOST_BUFFERS_SIZE];
    size_t                renderbuffers_size;
    cl_mem                  renderbuffers   [HOST_RENDERBUFFERS_SIZE];
    size_t                textures_size;
    cl_mem                  textures        [HOST_TEXTURES_SIZE];

    cl_kernel           bin_raster_kernel;
    cl_kernel           clear_kernel;
    cl_kernel           coarse_raster_kernel;
    cl_kernel           read_pixels_kernel;
    cl_kernel           triangle_setup_arrays_kernel;
    cl_kernel           triangle_setup_range_kernel;

    cl_uchar            enabled_colorbuffer, enabled_depthbuffer, enabled_stencilbuffer;
    cl_mem              t_colorbuffer, t_depthbuffer, t_stencilbuffer;
    cl_command_queue    queue;
    cl_event            framebuffer_wait_event;

    size_t bin_queues_size;
    bin_queue_t         bin_queues [HOST_FRAMEBUFFER_SIZE];
} device_t;

typedef union {
    cl_mem device;
    void* host;
} vertex_attrib_pointer_mem_u;

typedef enum {
    DEVICE_BINDING_VERTEX_UNIFORM = 0,
    DEVICE_BINDING_VERTEX_FRAGMENT_UNIFORM = 1,
} uniform_type_t;

typedef enum {
    DEVICE_VERTEX_ATTRIBUTE_POINTER = 0,
    HOST_VERTEX_ATTRIBUTE_POINTER = 1,
} vertex_attrib_pointer_type_t;

typedef struct {
    
    device_t *device;

    // Buffers
    cl_mem g_tri_header;
    cl_mem g_tri_data;
    cl_mem g_tri_subtris;
    cl_mem g_vertex_buffer;

    cl_mem g_bin_first_seg; 
    cl_mem g_bin_seg_data; 
    cl_mem g_bin_seg_next;
    cl_mem g_bin_seg_count;
    cl_mem g_bin_total;

    cl_mem g_active_tiles;
    cl_mem g_tile_first_seg;
    cl_mem g_tile_seg_data;
    cl_mem g_tile_seg_next;
    cl_mem g_tile_seg_count;

    // Atomics
    cl_mem a_bin_counter;
    cl_mem a_coarse_counter;
    cl_mem a_fine_counter;
    cl_mem a_num_active_tiles;
    cl_mem a_num_bin_segs;
    cl_mem a_num_subtris;
    cl_mem a_num_tile_segs;

    #ifdef DEVICE_IMAGE_ENABLED
    // Images
    cl_mem t_tri_data;
    cl_mem t_tri_header;
    cl_mem t_vertex_buffer;
    #endif
    size_t                  vertex_attrib_pointers_size; // shader data

    // Binded state
    uniform_type_t                  vertex_uniform_type;
    vertex_attrib_pointer_type_t    vertex_attrib_pointer_types     [DEVICE_VERTEX_ATTRIBUTE_SIZE]; //
    size_t                          vertex_attrib_pointer_strides   [DEVICE_VERTEX_ATTRIBUTE_SIZE]; // in case of host pointer size of each vertex 
    vertex_attrib_pointer_mem_u     vertex_attrib_pointer_mems      [DEVICE_VERTEX_ATTRIBUTE_SIZE];

    cl_mem                          texture_units_mems[DEVICE_TEXTURE_UNITS];    // binded texture ids
    
    // Vertex async state
    size_t                  vertex_command_queue_index; // TODO: do not rely
    cl_command_queue        vertex_command_queues       [DEVICE_VERTEX_COMMAND_QUEUE_SIZE];
    cl_event                assembly_wait_events        [DEVICE_VERTEX_COMMAND_QUEUE_SIZE];
    cl_mem                  vertex_attributes_mem       [DEVICE_VERTEX_COMMAND_QUEUE_SIZE];     // sizeof(float[DEVICE_VERTEX_ATTRIBUTE_SIZE][4])
    cl_mem                  vertex_attribute_data_mem   [DEVICE_VERTEX_COMMAND_QUEUE_SIZE];     // sizeof(vertex_attribute_data_t[DEVICE_VERTEX_ATTRIBUTE_SIZE])
    cl_mem                  vertex_uniform_mem          [DEVICE_VERTEX_COMMAND_QUEUE_SIZE];     // sizeof(cl_uchar[DEVICE_VERTEX_COMMAND_QUEUE_SIZE][DEVICE_UNIFORM_CAPACITY])            

    // TODO: this is not used, but could be implemented, only use frament_texture_datas_mem for now
    // cl_mem                  vertex_texture_datas_mem    [DEVICE_VERTEX_COMMAND_QUEUE_SIZE];     // sizeof(texture_data_t[DEVICE_VERTEX_COMMAND_QUEUE_SIZE][DEVICE_TEXTURE_UNITS])
    
    
    // Primitive Configuration
    // Raster async state
    cl_command_queue        raster_command_queue;
    cl_event                bin_wait_event                  [DEVICE_BIN_QUEUE_SIZE];
    cl_mem                  fragment_texture_datas_mem;                                          // sizeof(texture_data_t[DEVICE_TEXTURE_UNITS])
    cl_mem                  fragment_uniform_mem;                                           // sizeof(cl_uchar[TRIANGLE_PRIMITIVE_CONFIGS][DEVICE_UNIFORM_CAPACITY])
    cl_mem                  fragment_uniform_subbuffer_mems [TRIANGLE_PRIMITIVE_CONFIGS];    // subbuffers for vertex shader, TODO: do not use sub-buffers in OpenCL
    cl_mem                  rop_configs_mem;                                                // sizeof(rop_config_t[TRIANGLE_PRIMITIVE_CONFIGS])

} device_context_t;

// ---------------------------------------------------------------
// TODO: rm control from device.h to orchestrator.h
static size_t __device_get_vertex_command_index(device_context_t *context) 
{
    return context->vertex_command_queue_index % DEVICE_VERTEX_COMMAND_QUEUE_SIZE;
}

static void __device_advance_vertex_command_index(device_context_t *context) 
{
    context->vertex_command_queue_index += 1;
}

static void __device_reset_vertex_command_index(device_context_t *context) 
{
    context->vertex_command_queue_index = 0;
}
// ---------------------------------------------------------------

// ---------------------------------------------------------------
// Utils
static size_t __device_get_max_number_subtriangles() 
{
    return  DEVICE_MAX_NUMBER_TRIANGLES + DEVICE_MAX_NUMBER_SUBTRIANGLES;
}

static size_t __device_get_max_number_triangles() 
{
    return  DEVICE_MAX_NUMBER_TRIANGLES;
}

static size_t __device_get_max_number_bin_segments() 
{
    return CR_MAXBINS_SQR*CR_BIN_STREAMS_SIZE; // At least one segment x bin
}

static size_t __device_get_max_number_tile_segments() 
{
    return CR_MAXTILES_SQR; // At least one segment x tile
}

static size_t __device_get_bin_batch_size() 
{
    return DEVICE_BIN_SUB_GROUPS*DEVICE_SUB_GROUP_THREADS; // at least use all threads available, maybe can be fine tuned for larger executions
}

static size_t __device_get_num_bins_from_viewport(size_t viewport_size) 
{
    return ((viewport_size-1) / (CR_TILE_SIZE * CR_BIN_SIZE)) + 1;
}

static size_t __device_get_num_tiles_from_viewport(size_t viewport_size) 
{
    return ((viewport_size-1) / CR_TILE_SIZE) + 1;
}

static cl_mem __device_get_texture_vertex_buffer(device_context_t* context) 
{
    #ifdef DEVICE_IMAGE_ENABLED
        return context->t_vertex_buffer;
    #else
        return context->g_vertex_buffer;
    #endif
}

static cl_mem __device_get_texture_triangle_data(device_context_t* context) 
{
    #ifdef DEVICE_IMAGE_ENABLED
        return context->t_tri_data;
    #else
        return context->g_tri_data;
    #endif
}

static bin_queue_t* __device_get_bin_queue(
    device_t* device, 
    size_t bin_queue_id
) {
    return device->bin_queues + bin_queue_id;
}


static uint32_t size_from_name_type(const char* name_type) {
    #define RETURN_IF_SIZE_FROM(_TYPE)                              \
        if (strncmp(name_type, _TYPE, sizeof(_TYPE) - 1) == 0) {    \
            substr_size = name_type + sizeof(_TYPE) - 1;            \
            if (*substr_size == '*' || *substr_size == '\0') return 1;                     \
            return atoi(substr_size);                               \
        }

    const char* substr_size;
    RETURN_IF_SIZE_FROM("float");
    RETURN_IF_SIZE_FROM("int");
    RETURN_IF_SIZE_FROM("uint");
    RETURN_IF_SIZE_FROM("short");
    RETURN_IF_SIZE_FROM("char");
    RETURN_IF_SIZE_FROM("bool");

    #undef RETURN_IF_SIZE_FROM

    // OpenGL - OpenCL special types
    if (strcmp(name_type, "sampler2D_t") == 0) return 1;
    #ifdef HOSTGPU
    if (strcmp(name_type, "image_t") == 0) return 1;
    #else
    if (strncmp(name_type, "uchar", sizeof("uchar") -1) == 0) return 1;
    #endif

    printf("ERROR not found size for type_name=%s\n", name_type);
}

static uint32_t type_from_name_type(const char* name_type) {
    if (strncmp(name_type, "float",  sizeof("float") -1)  == 0) return GL_FLOAT;
    if (strncmp(name_type, "int",    sizeof("int")   -1)  == 0) return GL_INT;
    if (strncmp(name_type, "uint",    sizeof("uint")   -1)  == 0) return GL_INT;
    if (strncmp(name_type, "short",  sizeof("short") -1)  == 0) return GL_SHORT;
    if (strncmp(name_type, "char",   sizeof("char")  -1)  == 0) return GL_BYTE;
    if (strncmp(name_type, "bool",   sizeof("bool")  -1)  == 0) return GL_BYTE;
    
    printf("ERROR: not found type for type name=%s\n", name_type);

    #ifdef DEBUG
    printf("%s\n", name_type);
    CL_UNSUPORTED_MAPING();
    #else
    return 0;
    #endif
}

inline uint32_t get_num_tris(GLenum mode, GLsizei count) 
{
    switch (mode)
    {
        default:
        case GL_TRIANGLES: return count / 3;
        case GL_TRIANGLE_FAN: return count - 2;
        case GL_TRIANGLE_STRIP: return count - 2;
    }
}

inline uint32_t get_num_primitives(GLenum mode, uint32_t count) {
    switch (mode)
    {
    case GL_POINTS: return count;
    case GL_LINES: return count / 2;
    case GL_LINE_LOOP: return count;
    case GL_LINE_STRIP: return count - 1;
    case GL_TRIANGLES: return count / 3;
    case GL_TRIANGLE_STRIP: return count <= 2 ? 0 : count - 2;
    case GL_TRIANGLE_FAN: return count <= 2 ? 0 : count - 2;
    default: return 0; 
    }
}

inline uint32_t get_num_primitives_assembled(GLenum mode, size_t offset, size_t size) {
    switch (mode)
    {
    case GL_POINTS: return size - offset;
    case GL_LINES: return (size - offset) / 2;
    case GL_LINE_LOOP: return size - offset;
    case GL_LINE_STRIP: return size - offset - 1;
    case GL_TRIANGLES: return (size - offset) / 3;
    case GL_TRIANGLE_STRIP: return size - offset - 2;
    case GL_TRIANGLE_FAN: return size - offset - 2;
    default: return 0;
    }
};

inline int32_t get_primitive_offset(GLenum mode) {
    switch (mode)
    {
    case GL_POINTS:
    case GL_LINES:
    case GL_TRIANGLES: 
        return 0;
    case GL_LINE_LOOP: 
    case GL_LINE_STRIP:
    case GL_TRIANGLE_FAN:
        return -1;
    case GL_TRIANGLE_STRIP:
        return -2;
    default: return 0; 
    }
}
// ---------------------------------------------------------------

// ---------------------------------------------------------------
// Device initializers
static void device_init(device_t* shared) 
{
    // OpenCL setup
    CL_CHECK(clGetPlatformIDs(1, &shared->platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(shared->platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &shared->device_id, NULL));
    CL_ASSIGN_CHECK(shared->context, clCreateContext(NULL, 1, &shared->device_id, NULL, NULL,  &error));

    // Load programs
    size_t triangle_setup_size = sizeof(triangle_setup_o);
    const unsigned char *triangle_setup_bin = triangle_setup_o;
    CL_ASSIGN_CHECK(shared->triangle_setup_program, clCreateProgramWithBinary(shared->context, 1, &shared->device_id, &triangle_setup_size, &triangle_setup_bin, NULL, &error));
    CL_CHECK(clBuildProgram(shared->triangle_setup_program, 1, &shared->device_id, NULL, NULL, NULL));

    size_t bin_raster_size = sizeof(bin_raster_o);
    const unsigned char *bin_raster_bin = bin_raster_o;
    CL_ASSIGN_CHECK(shared->bin_raster_program, clCreateProgramWithBinary(shared->context, 1, &shared->device_id, &bin_raster_size, &bin_raster_bin, NULL, &error));
    CL_CHECK(clBuildProgram(shared->bin_raster_program, 1, &shared->device_id, NULL, NULL, NULL));

    size_t coarse_raster_size = sizeof(coarse_raster_o);
    const unsigned char *coarse_raster_bin = coarse_raster_o;
    CL_ASSIGN_CHECK(shared->coarse_raster_program, clCreateProgramWithBinary(shared->context, 1, &shared->device_id, &coarse_raster_size, &coarse_raster_bin, NULL, &error));
    CL_CHECK(clBuildProgram(shared->coarse_raster_program, 1, &shared->device_id, NULL, NULL, NULL));

    size_t force_clear_size = sizeof(force_clear_o);
    const unsigned char *force_clear_bin = force_clear_o;
    CL_ASSIGN_CHECK(shared->clear_program, clCreateProgramWithBinary(shared->context, 1, &shared->device_id, &force_clear_size, &force_clear_bin, NULL, &error));
    CL_CHECK(clBuildProgram(shared->clear_program, 1, &shared->device_id, NULL, NULL, NULL));

    // Create kernels
    CL_ASSIGN_CHECK(shared->triangle_setup_arrays_kernel,   clCreateKernel(shared->triangle_setup_program,  "triangle_setup_arrays",    &error));
    CL_ASSIGN_CHECK(shared->triangle_setup_range_kernel,    clCreateKernel(shared->triangle_setup_program,  "triangle_setup_range",     &error));
    CL_ASSIGN_CHECK(shared->bin_raster_kernel,              clCreateKernel(shared->bin_raster_program,      "bin_raster",               &error));
    CL_ASSIGN_CHECK(shared->coarse_raster_kernel,           clCreateKernel(shared->coarse_raster_program,   "coarse_raster",            &error));
    CL_ASSIGN_CHECK(shared->clear_kernel,                   clCreateKernel(shared->clear_program,           "force_clear",              &error));

    #ifdef DEVICE_IMAGE_ENABLE
    {
        cl_image_format image_format; 
        
        image_format = (cl_image_format) {
            .image_channel_data_type = CL_UNSIGNED_INT16,
            .image_channel_order = CL_DEPTH,
        };
        CL_ASSIGN_CHECK(shared->mem_depthbuffer, clCreateImage2D(shared->context, CL_MEM_READ_WRITE, image_format, 1, 1, 0, NULL, &error));
        image_format = (cl_image_format) {
            .image_channel_data_type = CL_UNSIGNED_INT8,
            .image_channel_order = CL_A,
        };
        CL_ASSIGN_CHECK(shared->mem_stencilbuffer, clCreateImage2D(shared->context, CL_MEM_READ_WRITE, image_format, 1, 1, 0, NULL, &error));
        image_format = (cl_image_format) {
            .image_channel_data_type = CL_UNSIGNED_INT8,
            .image_channel_order = CL_RGBA,
        };
        CL_ASSIGN_CHECK(shared->mem_texture, clCreateImage2D(shared->context, CL_MEM_READ_WRITE, image_format, 1, 1, 0, NULL, &error));
    }
    #else
    {
        CL_ASSIGN_CHECK(shared->mem_depthbuffer, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &error));
        CL_ASSIGN_CHECK(shared->mem_stencilbuffer, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &error));
        CL_ASSIGN_CHECK(shared->mem_texture, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &error));
    }
    #endif

    CL_ASSIGN_CHECK(shared->mem_vertex_attrib_pointer, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &error));

    CL_ASSIGN_CHECK(shared->queue, clCreateCommandQueue(shared->context, shared->device_id, 0, &error));
}

void device_init_context(
    device_context_t *context, 
    device_t* device
) {
    context->device = device;

    // Buffers
    const size_t max_number_subtriangles    = __device_get_max_number_subtriangles();
    const size_t triangle_header_size       = sizeof(triangle_header_t[max_number_subtriangles]);
    const size_t triangle_data_size         = sizeof(triangle_data_t[max_number_subtriangles]);
    const size_t vertex_buffer_size         = sizeof(cl_float4[DEVICE_VERTICES_SIZE][DEVICE_VARYING_SIZE + 1]);

    CL_ASSIGN_CHECK(context->g_tri_subtris,     clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_uchar[max_number_subtriangles]),   NULL, &error));
    CL_ASSIGN_CHECK(context->g_tri_header,      clCreateBuffer(device->context, CL_MEM_READ_WRITE, triangle_header_size,                        NULL, &error));
    CL_ASSIGN_CHECK(context->g_tri_data,        clCreateBuffer(device->context, CL_MEM_READ_WRITE, triangle_data_size,                          NULL, &error));     
    CL_ASSIGN_CHECK(context->g_vertex_buffer,   clCreateBuffer(device->context, CL_MEM_READ_WRITE, vertex_buffer_size,                          NULL, &error));

    #ifdef DEVICE_IMAGE_ENABLED
    {
        cl_image_desc image_desc;
        cl_image_format image_format;
        
        image_format = {
            .image_channel_order = CL_RGBA,
            .image_channel_data_type = CL_UNSIGNED_INT32,
        };

        image_desc = {
            .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
            .image_row_pitch = 0,
            .image_width = triangle_header_size / sizeof(cl_uint4),
            .mem_object = context->g_tri_header, 
        };
        
        CL_ASSIGN_CHECK(context->t_tri_header, clCreateImage(device->context, CL_MEM_READ_ONLY, &image_format, &image_desc, NULL, &error));

        image_desc = {
            .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
            .image_width = triangle_data_size / sizeof(cl_uint4),
            .image_row_pitch = 0,
            .mem_object = context->g_tri_data, 
        };

        CL_ASSIGN_CHECK(context->t_tri_data, clCreateImage(device->context, CL_MEM_READ_ONLY, &image_format, &image_desc, NULL, &error));

        image_format = {
            .image_channel_order = CL_RGBA,
            .image_channel_data_type = CL_FLOAT,
        };
        image_desc = {
            .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
            .image_width = vertex_buffer_size / sizeof(cl_float4),
            .image_row_pitch = 0,
            .mem_object = context->g_vertex_buffer,
        };

        CL_ASSIGN_CHECK(context->t_vertex_buffer, clCreateImage(device->context, CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &error));
    }
    #endif

    size_t max_number_bin_segments  = CR_MAXBINS_SQR*CR_BIN_STREAMS_SIZE; // At least one segment x bin
    CL_ASSIGN_CHECK(context->g_bin_first_seg, clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]), NULL, &error)); 
    CL_ASSIGN_CHECK(context->g_bin_seg_data, clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_bin_segments * CR_BIN_SEG_SIZE]), NULL, &error)); 
    CL_ASSIGN_CHECK(context->g_bin_seg_next, clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_bin_segments]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_bin_seg_count, clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_bin_segments]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_bin_total, clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]), NULL, &error));

    size_t max_number_tile_segments = CR_MAXTILES_SQR; // At least one segment x tile
    CL_ASSIGN_CHECK(context->g_active_tiles, clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXTILES_SQR]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_tile_first_seg, clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXTILES_SQR]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_tile_seg_data, clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_tile_segments * CR_TILE_SEG_SIZE]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_tile_seg_next, clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_tile_segments]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_tile_seg_count, clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_tile_segments]), NULL, &error));

    // Atomics
    cl_uint ZERO = 0;
    size_t max_number_triangles = DEVICE_MAX_NUMBER_TRIANGLES;
    CL_ASSIGN_CHECK(context->a_bin_counter, clCreateBuffer(device->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO, &error));
    CL_ASSIGN_CHECK(context->a_coarse_counter, clCreateBuffer(device->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO, &error));
    CL_ASSIGN_CHECK(context->a_fine_counter, clCreateBuffer(device->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO, &error));
    CL_ASSIGN_CHECK(context->a_num_active_tiles, clCreateBuffer(device->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO, &error));
    CL_ASSIGN_CHECK(context->a_num_bin_segs, clCreateBuffer(device->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO, &error));
    CL_ASSIGN_CHECK(context->a_num_subtris, clCreateBuffer(device->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &max_number_triangles, &error));
    CL_ASSIGN_CHECK(context->a_num_tile_segs, clCreateBuffer(device->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO, &error));

    // Vertex context objects
    context->vertex_command_queue_index = 0;

    for (size_t i = 0; i < DEVICE_VERTEX_COMMAND_QUEUE_SIZE; ++i) {
        context->assembly_wait_events[i] = NULL;

        CL_ASSIGN_CHECK(context->vertex_command_queues[i],      clCreateCommandQueue(device->context, device->device_id, 0, &error)); 
        CL_ASSIGN_CHECK(context->vertex_attributes_mem[i],      clCreateBuffer(device->context, CL_MEM_READ_ONLY, sizeof(cl_float4[DEVICE_VERTEX_ATTRIBUTE_SIZE]), NULL, &error));
        CL_ASSIGN_CHECK(context->vertex_attribute_data_mem[i],  clCreateBuffer(device->context, CL_MEM_READ_ONLY, sizeof(vertex_attribute_data_t[DEVICE_VERTEX_ATTRIBUTE_SIZE]), NULL, &error));
        CL_ASSIGN_CHECK(context->vertex_uniform_mem[i],         clCreateBuffer(device->context, CL_MEM_READ_ONLY, sizeof(cl_uchar[DEVICE_UNIFORM_CAPACITY]), NULL, &error));
    }

    // Fragment context objects
    CL_ASSIGN_CHECK(context->raster_command_queue, clCreateCommandQueue(device->context, device->device_id, 0, &error));

    CL_ASSIGN_CHECK(context->fragment_texture_datas_mem, clCreateBuffer(device->context, CL_MEM_READ_ONLY, sizeof(gl_texture_data_t[DEVICE_TEXTURE_UNITS]), NULL, &error));

    CL_ASSIGN_CHECK(context->fragment_uniform_mem, clCreateBuffer(device->context, CL_MEM_READ_ONLY, sizeof(cl_uchar[TRIANGLE_PRIMITIVE_CONFIGS][DEVICE_UNIFORM_CAPACITY]), NULL, &error));
    // TODO: rm this sub-buffer usage and use size_t offset in the kernel 
    for (int i = 0; i < TRIANGLE_PRIMITIVE_CONFIGS; ++i) {
        cl_buffer_region buffer_region = {
                .origin = DEVICE_UNIFORM_CAPACITY * i,
                .size = DEVICE_UNIFORM_CAPACITY
        };
        CL_ASSIGN_CHECK(context->fragment_uniform_subbuffer_mems[i], clCreateSubBuffer(
            context->fragment_uniform_mem, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
            &buffer_region, &error
        ));
    }
    
    CL_ASSIGN_CHECK(context->rop_configs_mem, clCreateBuffer(device->context, CL_MEM_READ_ONLY, sizeof(rop_config_t[TRIANGLE_PRIMITIVE_CONFIGS]), NULL, &error));

    for (size_t unit; unit < DEVICE_TEXTURE_UNITS; ++unit)
    {
        context->texture_units_mems[unit] = device->mem_texture;
    }
}
// ---------------------------------------------------------------

// ---------------------------------------------------------------
// Kernel argument setters
static void __device_set_vertex_shader_kernel_args(
    cl_kernel kernel,
    device_context_t* context, 
    size_t num_attributes,
    cl_mem g_vertex_attribute,
    cl_mem g_vertex_attribute_data,
    cl_mem g_uniforms,
    cl_mem *attribute_pointers
) {
    cl_mem t_vertex_buffer = __device_get_texture_vertex_buffer(context);

    cl_uint arg_idx = 0;

    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &g_vertex_attribute));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &g_vertex_attribute_data));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &g_uniforms));

    for (cl_uint attribute = 0; attribute < num_attributes; ++attribute)
    {
        CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &attribute_pointers[attribute]));
    }

    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &t_vertex_buffer)); 
}

static void __device_set_triangle_setup_arrays_kernel_args(
    cl_kernel kernel,
    device_context_t* context, 
    cl_uint c_vertex_offset,
    render_mode_t c_mode,
    cl_uint c_vertex_size,
    cl_uint c_fragment_config_id,
    cl_uint c_viewport_height,
    cl_uint c_viewport_width
) {
    cl_kernel kernel = context->device->triangle_setup_arrays_kernel;

    cl_uint c_max_subtris = __device_get_max_number_subtriangles();
    cl_uint c_samples_log2 = 0; // TODO: not impl
    cl_mem  t_vertex_buffer = __device_get_texture_vertex_buffer(context);

    cl_uint arg_idx = 0;

    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),                 &context->a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),                 &context->g_tri_header));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),                 &context->g_tri_data));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),                 &context->g_tri_subtris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),                 &t_vertex_buffer)); 
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_vertex_offset),        &c_vertex_offset));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_max_subtris),          &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_mode),                 &c_mode));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_samples_log2),         &c_samples_log2));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_vertex_size),          &c_vertex_size));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_viewport_height),      &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_viewport_width),       &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_fragment_config_id),   &c_fragment_config_id));
}

static void __device_set_triangle_setup_range_kernel_args(
    cl_kernel kernel,
    device_context_t* context, 
    cl_mem g_index_buffer,
    cl_uint c_vertex_offset,
    render_mode_t c_mode,
    cl_uint c_vertex_size,
    cl_uint c_fragment_config_id,
    cl_uint c_viewport_height,
    cl_uint c_viewport_width
) {
    cl_uint c_max_subtris = __device_get_max_number_subtriangles();
    cl_uint c_samples_log2 = 0; // TODO: not impl
    cl_mem  t_vertex_buffer = __device_get_texture_vertex_buffer(context);

    cl_uint arg_idx = 0;

    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),                 &context->a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),                 &g_index_buffer));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),                 &context->g_tri_header));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),                 &context->g_tri_data));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),                 &context->g_tri_subtris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),                 &t_vertex_buffer)); 
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_vertex_offset),        &c_vertex_offset));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_max_subtris),          &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_mode),                 &c_mode));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_samples_log2),         &c_samples_log2));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_vertex_size),          &c_vertex_size));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_viewport_height),      &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_viewport_width),       &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_fragment_config_id),   &c_fragment_config_id));
}

static void __device_set_bin_raster_kernel_args(
    cl_kernel kernel,
    device_context_t* context,
    cl_uint c_num_tris,
    cl_uint c_viewport_height,
    cl_uint c_viewport_width
) {
    cl_uint c_max_subtris = __device_get_max_number_subtriangles();
    cl_uint c_max_bin_segs  = __device_get_max_number_bin_segments();
    cl_uint c_samples_log2 = 0; // TODO: not impl
    cl_uint c_bin_batch_sz = __device_get_bin_batch_size();
    cl_uint c_height_bins = __device_get_num_bins_from_viewport(c_viewport_height);
    cl_uint c_width_bins = __device_get_num_bins_from_viewport(c_viewport_width);
    cl_uint c_num_bins = c_height_bins * c_width_bins;

    cl_uint arg_idx = 0;

    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->a_bin_counter));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->a_num_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_bin_first_seg));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_bin_seg_count));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_bin_seg_data));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_bin_seg_next));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_bin_total));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_tri_header));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_tri_subtris));
    #ifdef DEVICE_IMAGE_ENABLED
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->t_tri_header));
    #endif
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_bin_batch_sz),      &c_bin_batch_sz));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_bin_batch_sz),      &c_bin_batch_sz));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_height_bins),       &c_height_bins));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_max_bin_segs),      &c_max_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_max_subtris),       &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_num_bins),          &c_num_bins));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_num_tris),          &c_num_tris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_width_bins),        &c_width_bins));
}

static void __device_set_coarse_raster_kernel_args(
    cl_kernel kernel,
    device_context_t* context,
    cl_uint c_deferred_clear,
    cl_uint c_viewport_height,
    cl_uint c_viewport_width
) {
    cl_uint c_max_subtris = __device_get_max_number_subtriangles();
    cl_uint c_max_bin_segs  = __device_get_max_number_bin_segments();
    cl_uint c_max_tile_segs = __device_get_max_number_tile_segments(); // At least one segment x tile
    cl_uint c_height_tiles = __device_get_num_tiles_from_viewport(c_viewport_height);
    cl_uint c_width_tiles = __device_get_num_tiles_from_viewport(c_viewport_width);

    cl_uint c_height_bins = __device_get_num_bins_from_viewport(c_viewport_height);
    cl_uint c_width_bins = __device_get_num_bins_from_viewport(c_viewport_width);
    cl_uint c_num_bins = c_height_bins * c_width_bins;

    cl_uint arg_idx = 0;

    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->a_coarse_counter));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->a_num_active_tiles));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->a_num_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->a_num_tile_segs));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->g_active_tiles));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->g_bin_first_seg));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->g_bin_seg_count));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->g_bin_seg_data));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->g_bin_seg_next));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->g_bin_total));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->g_tile_first_seg));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->g_tile_seg_count));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->g_tile_seg_data));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->g_tile_seg_next));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->g_tri_header));
    #ifdef DEVICE_IMAGE_ENABLED
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->t_tri_header));
    #endif
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_deferred_clear),    &c_deferred_clear));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_height_tiles),      &c_height_tiles));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_max_bin_segs),      &c_max_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_max_subtris),       &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_max_tile_segs),     &c_max_tile_segs));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_num_bins),          &c_num_bins));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_width_bins),        &c_width_bins));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_width_tiles),       &c_width_tiles));
}

static void __device_set_fragment_shader_kernel_args(
    cl_kernel kernel,
    device_context_t* context, 
    cl_mem t_colorbuffer,
    cl_mem t_depthbuffer,
    cl_mem t_stencilbuffer,
    clear_data_t c_clear_data,
    enabled_data_t c_enabled_data,
    cl_uint c_colorbuffer_mode,
    cl_uint c_viewport_height,
    cl_uint c_viewport_width,
    gl_framebuffer_data_t c_framebuffer_data
) {
    cl_mem t_tri_data = __device_get_texture_triangle_data(context);
    cl_mem t_vertex_buffer = __device_get_texture_vertex_buffer(context);
    const cl_uint c_max_bin_segs  = __device_get_max_number_bin_segments();
    const cl_uint c_max_tile_segs = __device_get_max_number_tile_segments();
    const cl_uint c_max_subtris = __device_get_max_number_subtriangles();
    cl_uint c_width_tiles = __device_get_num_tiles_from_viewport(c_viewport_width);

    cl_uint arg_idx = 0;
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->fragment_uniform_mem));

    for(int i=0; i<DEVICE_TEXTURE_UNITS; ++i)
    {
        CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->texture_units_mems[i]));
    }

    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->fragment_texture_datas_mem));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->a_fine_counter));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->a_num_active_tiles));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->a_num_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->a_num_tile_segs));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_active_tiles));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_tile_first_seg));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_tile_seg_count));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_tile_seg_data));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_tile_seg_next));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->g_tri_header));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &t_colorbuffer));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &t_depthbuffer));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &t_stencilbuffer));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &t_tri_data));
    #ifdef DEVICE_IMAGE_ENABLED
    {
        CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &context->t_tri_header));
    }
    #endif
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &t_vertex_buffer));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),              &context->rop_configs_mem));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_clear_data),        &c_clear_data));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_enabled_data),      &c_enabled_data));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_colorbuffer_mode),  &c_colorbuffer_mode));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_max_bin_segs),      &c_max_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_max_subtris),       &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_max_tile_segs),     &c_max_tile_segs));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_width_tiles),       &c_width_tiles));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_framebuffer_data),  &c_framebuffer_data));
}

static void __device_set_clear_kernel_args(
    device_context_t* context,
    cl_kernel kernel,
    cl_mem t_color_buffer,
    cl_mem t_depth_buffer,
    cl_mem t_stencil_buffer,
    cl_uint c_colorbuffer_mode,
    cl_uint c_viewport_width,
    clear_data_t c_data,
    enabled_data_t c_enabled
) {
    cl_uint arg_idx = 0;

    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &t_color_buffer));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &t_depth_buffer));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &t_stencil_buffer));
    #ifndef DEVICE_IMAGE_ENABLED
    {
        CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_colorbuffer_mode), &c_colorbuffer_mode));
        CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_viewport_width), &c_viewport_width));
    }
    #endif
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_data),      &c_data));
    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(c_enabled),   &c_enabled));
}

static void __device_set_read_pixels_kernel_args(
    device_context_t* context,
    cl_kernel kernel,
    cl_mem g_pixel_buffer,
    cl_uint c_viewport_height,
    cl_uint c_viewport_width,
    cl_uint c_read_format,
    cl_uint c_read_type
) {
    cl_uint arg_idx = 0;

    // TODO
}
// ---------------------------------------------------------------


void device_finish(device_context_t* context) 
{
    CL_CHECK(clFinish(context->raster_command_queue));
}

// ---------------------------------------------------------------
// Device create functions

static size_t device_create_bin_queue(
    device_t* device
) {
    size_t bin_queue_id = device->bin_queues_size;

    if (bin_queue_id >= HOST_BIN_QUEUES_SIZE) {
        fprintf(stderr, "Error: Exceeded maximum number of bin queues.\n");
        exit(1);
    }

    bin_queue_t* bin_queue = &device->bin_queues[bin_queue_id];

    for(int i=0; i<DEVICE_BIN_QUEUE_SIZE; ++i) 
    {
        for(int j=0; j<DEVICE_BIN_QUEUE_SIZE; ++j)
        {
            CL_ASSIGN_CHECK(bin_queue->queues[i][j], clCreateCommandQueue(
                device->context,
                device->device_id,
                0,
                &error
            ));
        }
    }

    device->bin_queues_size += 1;

    return bin_queue_id;
}

static size_t device_create_program_from_binary(device_t* device, size_t size, const unsigned char* binary) 
{
    size_t program_id = device->program_size;

    // create program object
    CL_ASSIGN_CHECK(device->programs[program_id], clCreateProgramWithBinary(device->context, 1, &device->device_id, &size, &binary, NULL, &error));
    CL_CHECK(clBuildProgram(device->programs[program_id], 1, &device->device_id, NULL, NULL, NULL));

    // create program kernels
    cl_program program = device->programs[program_id];
    device_shader_kernels_t* kernels = &device->shaders[program_id];

    CL_ASSIGN_CHECK(kernels->vertex.kernel,         clCreateKernel(program, "gl_vertex_shader",             &error));
    CL_ASSIGN_CHECK(kernels->fragment,              clCreateKernel(program, "fine_raster_single_sample",    &error));
    CL_ASSIGN_CHECK(kernels->uniform_data,          clCreateKernel(program, "gl_uniform_data",              &error));
    CL_ASSIGN_CHECK(kernels->attribute_data,        clCreateKernel(program, "gl_attribute_data",            &error));
    CL_ASSIGN_CHECK(kernels->varying_data,          clCreateKernel(program, "gl_varying_data",              &error));

    cl_uint kernel_varying_num_args;
    clGetKernelInfo(kernels->varying_data, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &kernel_varying_num_args, NULL);
    kernels->vertex.vertex_size = sizeof(cl_float4[kernel_varying_num_args]);

    cl_uint kernel_attributes_num_args;
    clGetKernelInfo(kernels->attribute_data, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &kernel_attributes_num_args, NULL);
    kernels->vertex.number_attributes = kernel_attributes_num_args - 1;

    device->program_size += 1;

    return program_id;
}
// ---------------------------------------------------------------

// ---------------------------------------------------------------
// Device getter functions
static void device_get_program_uniform_arg_data(device_t* shared, size_t program_id, size_t location, arg_data_t* arg_data) 
{
    cl_kernel kernel = shared->shaders[program_id].uniform_data;

    char name[128];
    char type_name[32];

    CL_CHECK(clGetKernelArgInfo(kernel, location, CL_KERNEL_ARG_NAME,        sizeof(name),       &name,      NULL));
    CL_CHECK(clGetKernelArgInfo(kernel, location, CL_KERNEL_ARG_TYPE_NAME,   sizeof(type_name),  &type_name, NULL));

    arg_data->size = size_from_name_type(type_name);
    arg_data->type = type_from_name_type(type_name);
    strcpy(arg_data->name, name);
}

static size_t device_get_program_uniform_size(device_t* device, size_t program_id) 
{
    cl_kernel kernel = device->shaders[program_id].uniform_data;

    cl_uint uniform_size;
    CL_CHECK(clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &uniform_size, NULL));

    return uniform_size - 1;
}

static void device_get_program_vertex_attrib_arg_data(device_t* device, size_t program_id, size_t location, arg_data_t* arg_data) 
{
    cl_kernel kernel = device->shaders[program_id].attribute_data;

    char name[128];
    char type_name[32];

    CL_CHECK(clGetKernelArgInfo(kernel, location, CL_KERNEL_ARG_NAME,        sizeof(name),       &name,      NULL));
    CL_CHECK(clGetKernelArgInfo(kernel, location, CL_KERNEL_ARG_TYPE_NAME,   sizeof(type_name),  &type_name, NULL));

    arg_data->size = size_from_name_type(type_name);
    arg_data->type = type_from_name_type(type_name);
    strcpy(arg_data->name, &name[1]);
}

static size_t device_get_program_vertex_attrib_size(device_t* device, size_t program_id)
{
    return device->shaders[program_id].vertex.number_attributes;
}
// ---------------------------------------------------------------
/*
static void device_load_dummy_to_colorbuffer(device_context_t *context) 
{
    context->t_color_buffer = context->shared_objects->mem_depthbuffer; // chage to colorbuffer
    context->enabled_stencilbuffer = 0;
}

static void device_load_dummy_to_depthbuffer(device_context_t *context) 
{
    context->t_depth_buffer = context->shared_objects->mem_depthbuffer;
    context->enabled_depthbuffer = 0;
}

static void device_load_dummy_to_stencilbuffer(device_context_t *context) 
{
    context->t_stencil_buffer = context->shared_objects->mem_stencilbuffer;
    context->enabled_stencilbuffer = 0;
}

void device_load_renderbuffer_to_colorbuffer(device_context_t* context, size_t renderbuffer_id, uint32_t texture_mode) 
{
    context->t_color_buffer = context->shared_objects->renderbuffers[renderbuffer_id];
    context->c_color_buffer_mode = texture_mode;
    context->enabled_colorbuffer = 1;
}

void device_load_renderbuffer_to_depthbuffer(device_context_t* context, size_t renderbuffer_id, uint32_t texture_mode) 
{
    context->t_depth_buffer = context->shared_objects->renderbuffers[renderbuffer_id];
    // context->c_depth_buffer_mode = texture_mode;
    context->enabled_depthbuffer = 1;
}

void device_load_renderbuffer_to_stencilbuffer(device_context_t* context, size_t renderbuffer_id, uint32_t texture_mode) 
{
    context->t_stencil_buffer = context->shared_objects->renderbuffers[renderbuffer_id];
    // context->c_stencil_buffer_mode = texture_mode;
    context->enabled_stencilbuffer = 1;
}

void device_load_texture_to_colorbuffer(device_context_t* context, size_t texture_id, uint32_t texture_mode)
{
    context->t_color_buffer = context->shared_objects->textures[texture_id];
    context->c_color_buffer_mode = texture_mode;
    context->enabled_colorbuffer = 1;
}

void device_load_texture_to_depthbuffer(device_context_t* context, size_t texture_id, uint32_t texture_mode)
{
    context->t_depth_buffer = context->shared_objects->textures[texture_id];
    //context->c_depth_buffer_mode = texture_mode;
    context->enabled_depthbuffer = 1;
}

void device_load_texture_to_stencilbuffer(device_context_t* context, size_t texture_id, uint32_t texture_mode)
{
    context->t_stencil_buffer = context->shared_objects->textures[texture_id];
    //context->c_stencil_buffer_mode = texture_mode;
    context->enabled_stencilbuffer = 1;
}

uint32_t device_has_triangles_enqueued(device_context_t* context)
{
    return context->assembled_triangles != 0;
}
*/

static size_t __device_get_bytes_from_texture_mode(cl_uint texture_mode) 
{
    switch (texture_mode) {
        case TEX_R8:
        case TEX_STENCIL_INDEX8:
            return 1;
        case TEX_RG8:
        case TEX_RGBA4:
        case TEX_RGB5_A1:
        case TEX_RGB565:
        case TEX_DEPTH_COMPONENT16:
            return 2;
        case TEX_RGB8:
            return 3;
        default:
        case TEX_RGBA8:
            return 4;
    }
}

static cl_image_format __device_get_image_format_from_texture_mode(cl_uint texture_mode) 
{
    cl_image_format image_format;

    switch (texture_mode) {
        case TEX_R8:
            image_format = (cl_image_format) {
                .image_channel_order = CL_R,
                .image_channel_data_type = CL_UNSIGNED_INT8,
            };
            break;
        case TEX_STENCIL_INDEX8:
            image_format = (cl_image_format) {
                .image_channel_order = CL_A,
                .image_channel_data_type = CL_UNSIGNED_INT8,
            };
        case TEX_RG8:
            image_format = (cl_image_format) {
                .image_channel_order = CL_RG,
                .image_channel_data_type = CL_UNSIGNED_INT8,
            };
            break;
        case TEX_RGB8:
            image_format = (cl_image_format) {
                .image_channel_order = CL_RGB,
                .image_channel_data_type = CL_UNSIGNED_INT8,
            };
            break;
        case TEX_RGB565:
            image_format = (cl_image_format) {
                .image_channel_order = CL_RGB,
                .image_channel_data_type = CL_UNORM_SHORT_565,
            };
            break;
        case TEX_RGBA4:
            CL_UNSUPORTED_MAPING("TEX_RGBA4");
            break;
        case TEX_RGB5_A1:
            image_format = (cl_image_format) {
                .image_channel_order = CL_RGBA,
                .image_channel_data_type = CL_UNORM_SHORT_555,
            };
            break;
        case TEX_DEPTH_COMPONENT16:
            image_format = (cl_image_format) {
                .image_channel_order = CL_DEPTH, // this may be not supported for OpenCL 1.2 check cl_khr_depth_images extension
                .image_channel_data_type = CL_UNSIGNED_INT16,
            };
        default:
        case TEX_RGBA8:
            image_format = (cl_image_format) {
                .image_channel_order = CL_RGBA,
                .image_channel_data_type = CL_UNSIGNED_INT8,
            };
            break;
    }

    return image_format;
}

static size_t device_create_renderbuffer(device_t* device, size_t width, size_t height, cl_uint texture_mode)
{
    size_t renderbuffer_id = device->renderbuffers_size;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        cl_image_format image_format = __device_get_image_format_from_texture_mode(texture_mode);
        CL_ASSIGN_CHECK(device->renderbuffers[renderbuffer_id], 
            clCreateImage2D(device->context, CL_MEM_READ_WRITE, &image_format, width, height, 0, NULL, &error)
        );
    }
    #else
    {
        size_t buffer_size = width * height * __device_get_bytes_from_texture_mode(texture_mode);
        CL_ASSIGN_CHECK(device->renderbuffers[renderbuffer_id], 
            clCreateBuffer(device->context, CL_MEM_READ_WRITE, buffer_size, NULL, &error)
        );
    }
    #endif

    device->renderbuffers_size += 1;

    return renderbuffer_id;
}

/*
static size_t device_create_colorbuffer(device_t* device, size_t width, size_t height, cl_uint colorbuffer_mode) 
{
    device_shared_objects_t* shared = context->shared_objects;
    size_t renderbuffer_id = shared->renderbuffers_size;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        cl_image_format image_format = get_image_format_from_colorbuffer_mode(colorbuffer_mode);
        CL_ASSIGN_CHECK(shared->renderbuffers[renderbuffer_id], clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, width, height, 0, NULL, &error));
    }
    #else
    {
        size_t buffer_size = width * height * get_bytes_from_colorbuffer_mode(colorbuffer_mode);
        CL_ASSIGN_CHECK(shared->renderbuffers[renderbuffer_id], clCreateBuffer(shared->context, CL_MEM_READ_WRITE, buffer_size, NULL, &error));
    }
    #endif

    shared->renderbuffers_size += 1;
    return renderbuffer_id;
}

int device_create_depthbuffer(device_context_t* context, size_t width, size_t height) 
{
    device_shared_objects_t* shared = context->shared_objects;
    size_t renderbuffer_id = shared->renderbuffers_size;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        cl_image_format image_format = {
            .image_channel_order = CL_DEPTH, // this may be not supported for OpenCL 1.2 check cl_khr_depth_images extension
            .image_channel_data_type = CL_UNSIGNED_INT16,
        };
        CL_ASSIGN_CHECK(shared->renderbuffers[renderbuffer_id], clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, width, height, 0, NULL, &error));
    }
    #else
    {
        size_t buffer_size = width * height * sizeof(cl_ushort);
        CL_ASSIGN_CHECK(shared->renderbuffers[renderbuffer_id], clCreateBuffer(shared->context, CL_MEM_READ_WRITE, buffer_size, NULL, &error));
    }
    #endif
    
    shared->renderbuffers_size += 1;
    return renderbuffer_id;
}

int device_create_stencilbuffer(device_context_t* context, size_t width, size_t height) 
{
    device_shared_objects_t* shared = context->shared_objects;
    size_t renderbuffer_id = shared->renderbuffers_size;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        cl_image_format image_format = {
            .image_channel_order = CL_A,
            .image_channel_data_type = CL_UNSIGNED_INT8,
        };
        CL_ASSIGN_CHECK(shared->renderbuffers[renderbuffer_id], clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, width, height, 0, NULL, &error));
    }
    #else
    {
        size_t buffer_size = width * height * sizeof(cl_uchar);
        CL_ASSIGN_CHECK(shared->renderbuffers[renderbuffer_id], clCreateBuffer(shared->context, CL_MEM_READ_WRITE, buffer_size, NULL, &error));
    }
    #endif
    
    shared->renderbuffers_size += 1;
    return renderbuffer_id;
}
*/
size_t device_create_ro_buffer(device_t* device, size_t size) 
{
    size_t buffer_id = device->buffers_size;

    CL_ASSIGN_CHECK(device->buffers[buffer_id], clCreateBuffer(device->context, CL_MEM_READ_ONLY, size, NULL, &error));

    device->buffers_size += 1;

    return buffer_id;
}

size_t device_create_2d_texture(device_t* device, size_t width, size_t height, cl_uint texture_mode) 
{
    size_t texture_id = device->textures_size;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        cl_image_format image_format = __device_get_image_format_from_texture_mode(texture_mode);
        CL_ASSIGN_CHECK(device->textures[texture_id], 
            clCreateImage2D(device->context, CL_MEM_READ_WRITE, &image_format, width, height, 0, NULL, &error)
        );
    }
    #else
    {
        size_t pixel_size = __device_get_bytes_from_texture_mode(texture_mode);
        size_t buffer_size = width * height * pixel_size;
        CL_ASSIGN_CHECK(device->textures[texture_id], 
            clCreateBuffer(device->context, CL_MEM_READ_WRITE, buffer_size, NULL, &error)
        );
    }
    #endif

    device->textures_size += 1;

    return texture_id;
}

void device_write_2d_texture(
    device_t* device, 
    size_t texture_id, 
    size_t x, size_t y, 
    size_t width, size_t height, 
    uint32_t device_texture_mode,
    uint32_t host_texture_mode,
    const void* data
) 
{
    // TODO: Do it with a kernel
    #ifdef DEVICE_IMAGE_ENABLED
    {
        size_t origin[3] = {x, y, 0};
        size_t region[3] = {width, height, 1};
        CL_CHECK(clEnqueueWriteImage(
            device->queue,
            device->textures[texture_id],
            CL_TRUE,
            origin,
            region,
            0,
            0,
            data,
            0,
            NULL,
            NULL
        ));
    }
    #else
    {
        size_t pixel_size = __device_get_bytes_from_texture_mode(device_texture_mode);
        size_t buffer_size = width * height * pixel_size;
        CL_CHECK(clEnqueueWriteBuffer(
            device->queue,
            device->textures[texture_id],
            CL_TRUE,
            0,
            buffer_size,
            data,
            0,
            NULL,
            NULL
        ));
    }
    #endif
}

static void device_write_buffer(device_t* device, size_t buffer_id, size_t offset, size_t size, const void* data) 
{
    CL_CHECK(clEnqueueWriteBuffer(
        device->queue,
        device->buffers[buffer_id],
        CL_TRUE,
        offset,
        size,
        data,
        0,
        NULL,
        NULL
    ));
}

/*
void device_bind_dummy_texture_unit(device_context_t* context, size_t unit_index)
{
    context->texture_units[unit_index] = context->shared_objects->mem_texture;
}
*/

static void device_bind_texture_unit(device_context_t* context, size_t unit_index, size_t texture_id) 
{
    context->texture_units_mems[unit_index] = context->device->textures[texture_id];
}

// TODO: textures are not implemented for vertex
static void device_load_texture_datas(device_context_t* context, const gl_texture_data_t* texture_datas) 
{
    CL_CHECK(clEnqueueWriteBuffer(
        context->raster_command_queue,
        context->fragment_texture_datas_mem,
        CL_TRUE,
        0,
        sizeof(gl_texture_data_t[DEVICE_TEXTURE_UNITS]),
        texture_datas,
        0,
        NULL,
        NULL
    ));
}

/*
void device_bind_dummy_to_vertex_attribute_pointer(device_context_t* context, size_t attrib_index)
{
    cl_mem mem_attribute = context->shared_objects->mem_vertex_attrib_pointer;

    context->mem_vertex_attrib_pointers[attrib_index] = mem_attribute;
}
*/

void device_reset_context(device_context_t *context) 
{
    context->vertex_command_queue_index = 0;
    // clean atomics
    // TODO: use a single async command queue
    cl_int ZERO = 0;
    cl_uint max_number_triangles = __device_get_max_number_triangles();
    CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_bin_counter,        CL_FALSE, 0, sizeof(cl_int), &ZERO, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_coarse_counter,     CL_FALSE, 0, sizeof(cl_int), &ZERO, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_fine_counter,       CL_FALSE, 0, sizeof(cl_int), &ZERO, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_num_active_tiles,   CL_FALSE, 0, sizeof(cl_int), &ZERO, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_num_subtris,        CL_FALSE, 0, sizeof(cl_int), &max_number_triangles, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_num_bin_segs,       CL_FALSE, 0, sizeof(cl_int), &ZERO, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_num_tile_segs,      CL_FALSE, 0, sizeof(cl_int), &ZERO, 0, NULL, NULL));
    CL_CHECK(clFinish(context->raster_command_queue));

    printf("device_reset_context\n");
}

cl_uint get_kernel_fine_raster_offset() 
{
    cl_uint offset = DEVICE_TEXTURE_UNITS;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        offset += 1;
    }
    #endif

    return offset;
}

/*
void device_load_program(device_context_t *context, int program_id) 
{
    context->vertex_shader_kernel   = context->shared_objects->kernels[program_id][0];
    context->fine_raster_kernel     = context->shared_objects->kernels[program_id][1];

    cl_kernel kernel_varying, kernel_attributes;
    CL_ASSIGN_CHECK(kernel_varying, clCreateKernel(context->shared_objects->programs[program_id], "gl_varying_data", &error));
    CL_ASSIGN_CHECK(kernel_attributes, clCreateKernel(context->shared_objects->programs[program_id], "gl_attribute_data", &error));

    cl_uint kernel_varying_num_args;
    clGetKernelInfo(kernel_varying, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &kernel_varying_num_args, NULL);
    cl_uint c_vertex_size = sizeof(cl_float4[kernel_varying_num_args]);

    cl_uint kernel_attributes_num_args;
    clGetKernelInfo(kernel_attributes, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &kernel_attributes_num_args, NULL);
    context->vertex_attrib_pointers_size = kernel_attributes_num_args - 1;

    CL_CHECK(clSetKernelArg(context->triangle_setup_arrays_kernel, 9, sizeof(c_vertex_size),       &c_vertex_size));
    CL_CHECK(clSetKernelArg(context->triangle_setup_range_kernel, 10, sizeof(c_vertex_size),       &c_vertex_size));

    cl_kernel kernel = context->fine_raster_kernel;
    cl_uint offset = get_kernel_fine_raster_offset();

    CL_CHECK(clSetKernelArg(kernel, 21 + offset, sizeof(context->c_color_buffer_mode), &context->c_color_buffer_mode));
    CL_CHECK(clSetKernelArg(kernel, 25 + offset, sizeof(context->c_viewport_height),   &context->c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 26 + offset, sizeof(context->c_viewport_width),    &context->c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, 27 + offset, sizeof(context->c_width_tiles),       &context->c_width_tiles));
}
*/

/*
void device_set_framebuffer_state(device_context_t *context, size_t framebuffer_width, size_t framebuffer_height, cl_uint color_buffer_mode) 
{
    // printf("device_set_framebuffer_state\n");

    context->c_viewport_height  = framebuffer_height;
    context->c_viewport_width   = framebuffer_width;
    context->c_height_tiles      = ((framebuffer_height-1) / CR_TILE_SIZE) + 1;
    context->c_width_tiles      = ((framebuffer_width-1) / CR_TILE_SIZE) + 1;
    context->c_height_bins       = ((framebuffer_height-1) / (CR_TILE_SIZE * CR_BIN_SIZE)) + 1;
    context->c_width_bins       = ((framebuffer_width-1) / (CR_TILE_SIZE * CR_BIN_SIZE)) + 1;
    context->c_color_buffer_mode = color_buffer_mode;
    
    
    cl_uint c_viewport_height = framebuffer_height;
    cl_uint c_viewport_width  = framebuffer_width;
    cl_uint c_height_bins      = context->c_height_bins;
    cl_uint c_width_bins      = context->c_width_bins;
    cl_uint c_height_tiles    = context->c_height_tiles;
    cl_uint c_width_tiles      = context->c_width_tiles;
    cl_uint c_num_bins      = c_height_bins * c_width_bins;

    cl_uint c_color_buffer_mode = context->c_color_buffer_mode;

    cl_kernel kernel;

    kernel = context->triangle_setup_arrays_kernel;
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(c_viewport_width),    &c_viewport_width));

    kernel = context->triangle_setup_range_kernel;
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(c_viewport_width),    &c_viewport_width));

    kernel = context->bin_raster_kernel;
    cl_uint extra = 0;
    #ifdef DEVICE_IMAGE_ENABLED
        extra = 1;
    #endif
    CL_CHECK(clSetKernelArg(kernel, 11 + extra, sizeof(c_height_bins),        &c_height_bins));
    CL_CHECK(clSetKernelArg(kernel, 14 + extra, sizeof(c_num_bins),   &c_num_bins));
    CL_CHECK(clSetKernelArg(kernel, 16 + extra, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 17 + extra, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, 18 + extra, sizeof(c_width_bins),        &c_width_bins));

    kernel = context->coarse_raster_kernel;
    CL_CHECK(clSetKernelArg(kernel, 17 + extra, sizeof(c_height_tiles),      &c_height_tiles));
    CL_CHECK(clSetKernelArg(kernel, 21 + extra, sizeof(c_num_bins),          &c_num_bins));
    CL_CHECK(clSetKernelArg(kernel, 22 + extra, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 23 + extra, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, 24 + extra, sizeof(c_width_bins),        &c_width_bins));
    CL_CHECK(clSetKernelArg(kernel, 25 + extra, sizeof(c_width_tiles),       &c_width_tiles));

    // if there is a fine raster kernel loaded, update its args too
    if (context->fine_raster_kernel == NULL) return;

    kernel = context->fine_raster_kernel;
    cl_uint offset = get_kernel_fine_raster_offset();

    CL_CHECK(clSetKernelArg(kernel, 21 + offset, sizeof(c_color_buffer_mode), &c_color_buffer_mode));
    CL_CHECK(clSetKernelArg(kernel, 25 + offset, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 26 + offset, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, 27 + offset, sizeof(c_width_tiles),       &c_width_tiles));
}
*/

/*
void device_load_framebuffer(device_context_t *context, size_t colorbuffer_id, size_t depthbuffer_id, size_t stencilbuffer_id, size_t framebuffer_width, size_t framebuffer_height, cl_uint color_buffer_mode) 
{
    // printf("device_load_framebuffer\n");

    context->t_color_buffer     = context->shared_objects->renderbuffers[colorbuffer_id];
    context->t_depth_buffer     = context->shared_objects->renderbuffers[depthbuffer_id];
    context->t_stencil_buffer   = context->shared_objects->renderbuffers[stencilbuffer_id];
    context->c_viewport_height  = framebuffer_height;
    context->c_viewport_width   = framebuffer_width;
    context->c_height_tiles      = ((framebuffer_height-1) / CR_TILE_SIZE) + 1;
    context->c_width_tiles      = ((framebuffer_width-1) / CR_TILE_SIZE) + 1;
    context->c_height_bins       = ((framebuffer_height-1) / (CR_TILE_SIZE * CR_BIN_SIZE)) + 1;
    context->c_width_bins       = ((framebuffer_width-1) / (CR_TILE_SIZE * CR_BIN_SIZE)) + 1;
    context->c_color_buffer_mode = color_buffer_mode;
    
    
    cl_uint c_viewport_height = framebuffer_height;
    cl_uint c_viewport_width  = framebuffer_width;
    cl_uint c_height_bins      = context->c_height_bins;
    cl_uint c_width_bins      = context->c_width_bins;
    cl_uint c_height_tiles    = context->c_height_tiles;
    cl_uint c_width_tiles      = context->c_width_tiles;
    cl_uint c_num_bins      = c_height_bins * c_width_bins;

    cl_uint c_color_buffer_mode = context->c_color_buffer_mode;

    cl_kernel kernel;

    kernel = context->triangle_setup_arrays_kernel;
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(c_viewport_width),    &c_viewport_width));

    kernel = context->triangle_setup_range_kernel;
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(c_viewport_width),    &c_viewport_width));

    kernel = context->bin_raster_kernel;
    cl_uint extra = 0;
    #ifdef DEVICE_IMAGE_ENABLED
        extra = 1;
    #endif
    CL_CHECK(clSetKernelArg(kernel, 11 + extra, sizeof(c_height_bins),        &c_height_bins));
    CL_CHECK(clSetKernelArg(kernel, 14 + extra, sizeof(c_num_bins),         &c_num_bins));
    CL_CHECK(clSetKernelArg(kernel, 16 + extra, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 17 + extra, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, 18 + extra, sizeof(c_width_bins),        &c_width_bins));

    kernel = context->coarse_raster_kernel;
    CL_CHECK(clSetKernelArg(kernel, 17 + extra, sizeof(c_height_tiles),      &c_height_tiles));
    CL_CHECK(clSetKernelArg(kernel, 21 + extra, sizeof(c_num_bins),          &c_num_bins));
    CL_CHECK(clSetKernelArg(kernel, 22 + extra, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 23 + extra, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, 24 + extra, sizeof(c_width_bins),        &c_width_bins));
    CL_CHECK(clSetKernelArg(kernel, 25 + extra, sizeof(c_width_tiles),       &c_width_tiles));

    // if there is a fine raster kernel loaded, update its args too
    if (context->fine_raster_kernel == NULL) return;

    kernel = context->fine_raster_kernel;
    cl_uint offset = get_kernel_fine_raster_offset();
    context->enabled_colorbuffer = 1;
    context->enabled_depthbuffer = 1;
    context->enabled_stencilbuffer = 1;

    CL_CHECK(clSetKernelArg(kernel, 21 + offset, sizeof(c_color_buffer_mode), &c_color_buffer_mode));
    CL_CHECK(clSetKernelArg(kernel, 25 + offset, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 26 + offset, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, 27 + offset, sizeof(c_width_tiles),       &c_width_tiles));
}
*/

/*
void device_clear_framebuffer(device_context_t *context, cl_ulong clear_write_values, cl_ushort clear_enabled_data, cl_uint deferred_clear) 
{
    // printf("device_clear_framebuffer: c_write_values=%lx, c_enabled_data=%x\n", clear_write_values, clear_enabled_data);
    context->c_clear_write_values = clear_write_values;
    context->c_clear_enabled_data = clear_enabled_data;
    context->c_deferred_clear = deferred_clear;

    cl_uint extra = 0;
    #ifdef DEVICE_IMAGE_ENABLED
        extra = 1;
    #endif
    CL_CHECK(clSetKernelArg(context->coarse_raster_kernel, 16 + extra, sizeof(context->c_deferred_clear), &context->c_deferred_clear));
}
*/

/*
void device_load_vertex_attributes(device_context_t *context, const float attribs[DEVICE_VERTEX_ATTRIBUTE_SIZE][4], const vertex_attribute_data_t* attrib_datas) 
{
    // printf("device_load_vertex_attributes: context=%p\n", context);

    cl_command_queue queue = context->vertex_command_queues[__device_get_vertex_command_index(context)];
    cl_event write_wait_event[2];

    CL_CHECK(clEnqueueWriteBuffer(
        queue,
        context->mem_vertex_attribs,
        CL_FALSE,
        0,
        sizeof(cl_float[DEVICE_VERTEX_ATTRIBUTE_SIZE][4]),
        attribs,
        0,
        NULL,
        &write_wait_event[0]
    ));

    CL_CHECK(clEnqueueWriteBuffer(
        queue,
        context->mem_vertex_attrib_datas,
        CL_FALSE,
        0,
        sizeof(vertex_attribute_data_t[DEVICE_VERTEX_ATTRIBUTE_SIZE]),
        attrib_datas,
        0,
        NULL,
        &write_wait_event[1]
    ));

    CL_CHECK(clWaitForEvents(2, write_wait_event));
    CL_CHECK(clReleaseEvent(write_wait_event[0]));
    CL_CHECK(clReleaseEvent(write_wait_event[1]));
}
*/

// TODO: separete rop config and uniform data loading
void device_load_config(device_context_t* context, size_t config_id, rop_config_t* rop_config, cl_uchar* uniform_data) 
{
    // TODO: make it async, for now use a single vertex command queue
    cl_command_queue queue = context->vertex_command_queues[__device_get_vertex_command_index(context)];

    CL_CHECK(clEnqueueWriteBuffer(
        queue,
        context->rop_configs_mem,
        CL_FALSE,
        sizeof(rop_config_t) * config_id,
        sizeof(rop_config_t),
        rop_config,
        0,
        NULL,
        NULL
    ));

    CL_CHECK(clEnqueueWriteBuffer(
        queue,
        context->fragment_uniform_subbuffer_mems[config_id],
        CL_FALSE,
        0,
        DEVICE_UNIFORM_CAPACITY,
        uniform_data,
        0,
        NULL,
        NULL
    ));
}

void device_destroy(device_t* device) 
{
    // TODO: release all mem objects and programs
}

void device_destroy_context(device_context_t* context) 
{
    // TODO: release all mem objects and kernels
}

static void __device_print_vertex_shader_output(
    device_context_t* context,
    cl_command_queue queue,
    size_t num_vertices,
    size_t num_varying
) {
    // printf("enqueue of=%ld, sz=%ld\n",gw_offset, gw_size);
    size_t vertices_sizeof = sizeof(float[num_vertices][num_varying+1][4]);
    float *vertices = (float*) malloc(vertices_sizeof);
    clEnqueueReadBuffer(queue, context->g_vertex_buffer, CL_TRUE, 0, vertices_sizeof, vertices, 0, NULL, NULL);
    
    for(int i= 0; i<num_vertices; ++i) 
    {
        printf("vertex %d:", i);
        for(int j=0; j<num_varying+1; ++j) {
            size_t offset = i*(num_varying+1)*4 + j*4;
            printf(" (%.2f,%.2f,%.2f,%.2f)", 
                vertices[offset+0],
                vertices[offset+1],
                vertices[offset+2],
                vertices[offset+3]
            );
        }
        printf("\n");
    }

    free(vertices);
}

static cl_mem __device_get_binded_vertex_uniform(device_context_t* context, size_t config_id)
{
    switch (context->vertex_uniform_type)
    {
        default:
        case DEVICE_BINDING_VERTEX_FRAGMENT_UNIFORM:
            return context->fragment_uniform_subbuffer_mems[config_id];
        case DEVICE_BINDING_VERTEX_UNIFORM:
            return context->vertex_uniform_mem[__device_get_vertex_command_index(context)];
        break;
    }
}

static void __device_get_device_and_host_pointers(device_context_t* context, cl_mem* mem, uint8_t* host, size_t size, size_t num_vertices)
{
    vertex_attrib_pointer_type_t*   ptrs_type = context->vertex_attrib_pointer_types;
    vertex_attrib_pointer_mem_u*    ptrs_mem = context->vertex_attrib_pointer_mems;

    for (size_t attribute = 0; attribute < size; ++attribute)
    {
        cl_mem pointer;
        if (ptrs_type[attribute] == HOST_VERTEX_ATTRIBUTE_POINTER) 
        {
            CL_ASSIGN_CHECK(pointer, clCreateBuffer(
                context->device->context,
                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                num_vertices * context->vertex_attrib_pointer_strides[attribute], // TODO: add right size
                ptrs_mem[attribute].host,
                &error
            ));

            host[attribute] = 1;
        } 
        else 
        {
            pointer = ptrs_mem[attribute].device;

            host[attribute] = 0;
        }
        mem[attribute] = pointer;
    }
}

static void __device_destroy_host_pointers(device_context_t* context, cl_mem* mem, uint8_t* host, size_t size)
{
    for (size_t attribute = 0; attribute < size; ++attribute)
    {
        if (host[attribute])
        {
            CL_CHECK(clReleaseMemObject(mem[attribute]));
        }
    }
}

static void device_launch_vertex_shader(
    device_context_t* context,
    size_t shader_id,
    size_t init, size_t end, size_t offset,
    size_t config_id
)
{
    if (init != 0) CL_UNSUPORTED_MAPING();

    size_t num_vertices = end - init;
    size_t queue_index = __device_get_vertex_command_index(context);
    
    cl_command_queue queue = context->vertex_command_queues[queue_index];
    cl_kernel kernel = context->device->shaders[shader_id].vertex.kernel;


    size_t n_attributes = context->device->shaders[shader_id].vertex.number_attributes;

    uint8_t host_pointers       [DEVICE_VERTEX_ATTRIBUTE_SIZE];
    cl_mem  attribute_pointers  [DEVICE_VERTEX_ATTRIBUTE_SIZE];

    __device_get_device_and_host_pointers(
        context,
        attribute_pointers,
        host_pointers,
        n_attributes,
        num_vertices
    );

    __device_set_vertex_shader_kernel_args(
        kernel,
        context,
        n_attributes,
        context->vertex_attributes_mem[queue_index],
        context->vertex_attribute_data_mem[queue_index],
        __device_get_binded_vertex_uniform(context, config_id),
        attribute_pointers
    );

    size_t gw_offset = offset;
    size_t gw_size = num_vertices;
    
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, &gw_offset, &gw_size, NULL, 0, NULL, NULL));
    
    __device_destroy_host_pointers(
        context,
        attribute_pointers,
        host_pointers,
        n_attributes
    );
    
    // __device_print_vertex_shader_output(context, queue, num_vertices, /*num_varying=*/1);
}

static void __device_print_range_triangle_assembly_output(
    device_context_t* context,
    cl_command_queue queue,
    size_t size
) {
    size_t sizeof_primitives = sizeof(cl_uchar[size]);
    cl_uchar *primitives = (cl_uchar*) malloc(sizeof_primitives);
    CL_CHECK(clEnqueueReadBuffer(queue, context->g_tri_subtris, CL_TRUE, 0, sizeof_primitives, primitives, 0, NULL, NULL));

    printf("subtri=[%d", primitives[0]);
    for(int i= 1; i<size; ++i) {
        printf(",%d", primitives[i]);
    }
    printf("]\n");

    free(primitives);

    size_t sizeof_header = sizeof(triangle_header_t[size]);
    triangle_header_t *triangle_header = (triangle_header_t*) malloc(sizeof_header);
    CL_CHECK(clEnqueueReadBuffer(queue, context->g_tri_header, CL_TRUE, 0, sizeof_header, triangle_header, 0, NULL, NULL));

    for(int i=0; i<size; ++i) {
        triangle_header_t *header = &triangle_header[i];
        printf("header[%d]{v0x=%d,v0y=%d,v1x=%d,v1y=%d,v2x=%d,v2y=%d}\n",i, 
            header->v0x >> (CR_SUBPIXEL_LOG2 - 1), 
            header->v0y >> (CR_SUBPIXEL_LOG2 - 1),
            header->v1x >> (CR_SUBPIXEL_LOG2 - 1),
            header->v1y >> (CR_SUBPIXEL_LOG2 - 1),
            header->v2x >> (CR_SUBPIXEL_LOG2 - 1), 
            header->v2y >> (CR_SUBPIXEL_LOG2 - 1)
        );
    }

    free(triangle_header);
}

void device_launch_range_triangle_assembly(
    device_context_t* context, 
    size_t shader_id,
    render_mode_t mode, 
    size_t frag_config_id, 
    size_t vertex_offset, size_t triangle_offset, size_t num_triangles,
    size_t width, size_t height, size_t size, uint16_t* ptr
)
{
    cl_kernel kernel = context->device->triangle_setup_range_kernel;

    cl_mem g_index_buffer;
    CL_ASSIGN_CHECK(g_index_buffer, clCreateBuffer(
            context->device->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_ushort[size]), (void*) ptr, &error
    ));

    __device_set_triangle_setup_range_kernel_args(
        kernel,
        context,
        g_index_buffer,
        vertex_offset,
        mode,
        context->device->shaders[shader_id].vertex.vertex_size,
        frag_config_id,
        height, 
        width
    );

    size_t queue_index = __device_get_vertex_command_index(context);

    // free event if previously used
    if (context->assembly_wait_events[queue_index] != NULL) 
    {
        clReleaseEvent(context->assembly_wait_events[queue_index]);
    }

    cl_command_queue queue = context->vertex_command_queues[queue_index];
    size_t gwo = triangle_offset;
    size_t gws = num_triangles;

    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, &gwo, &gws, NULL, 0, NULL, &context->assembly_wait_events[queue_index]));

    CL_CHECK(clReleaseMemObject(g_index_buffer));

    __device_advance_vertex_command_index(context);

    // __device_print_range_triangle_assembly_output(context, queue, size);
}

static void device_launch_arrays_triangle_assembly(
    device_context_t* context, 
    size_t shader_id,
    render_mode_t mode, 
    size_t frag_config_id, 
    size_t vertex_offset, 
    size_t triangle_offset, 
    size_t num_triangles,
    size_t width, 
    size_t height, 
    size_t size,
    cl_ushort* ptr
) {
    cl_kernel kernel = context->device->triangle_setup_range_kernel;

    __device_set_triangle_setup_arrays_kernel_args(
        kernel,
        context,
        vertex_offset,
        mode,
        context->device->shaders[shader_id].vertex.vertex_size,
        frag_config_id,
        height, 
        width
    );

    size_t queue_index = __device_get_vertex_command_index(context);

    // free event if previously used
    if (context->assembly_wait_events[queue_index] != NULL) 
    {
        clReleaseEvent(context->assembly_wait_events[queue_index]);
    }

    cl_command_queue queue = context->vertex_command_queues[queue_index];
    size_t gwo = triangle_offset;
    size_t gws = num_triangles;

    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, &gwo, &gws, NULL, 0, NULL, &context->assembly_wait_events[queue_index]));

    __device_advance_vertex_command_index(context);
}

static cl_uint get_num_tris_arg_index() 
{
    cl_uint index = 15;

    #ifdef CONF_BIN_IMAGE_ENABLED
    ++index;
    #endif

    return index;
}

static void device_launch_bin_dispatch(
    device_context_t* context, 
    size_t num_triangles, 
    size_t width, 
    size_t height
) {
    cl_kernel kernel = context->device->bin_raster_kernel;

    __device_set_bin_raster_kernel_args(
        kernel,
        context,
        num_triangles,
        height, 
        width
    );

    size_t await_events = MIN(context->vertex_command_queue_index, DEVICE_VERTEX_COMMAND_QUEUE_SIZE);

    size_t local_work_size[2] = {DEVICE_SUB_GROUP_THREADS, DEVICE_BIN_SUB_GROUPS};
    size_t global_work_size[2] = {local_work_size[0] * CR_BIN_STREAMS_SIZE, local_work_size[1]};

    CL_CHECK(clEnqueueNDRangeKernel(
        context->raster_command_queue, 
        kernel, 
        2, 
        NULL, 
        global_work_size, 
        local_work_size, 
        await_events, 
        context->assembly_wait_events, 
        NULL)
    );

    for(size_t event=0; event<await_events; ++event) 
    {
        clReleaseEvent(context->assembly_wait_events[event]);
        context->assembly_wait_events[event] = NULL;
    }

    __device_reset_vertex_command_index(context);
}

static void device_launch_tile_dispatch(
    device_context_t* context,
    uint32_t deferred_clear,
    size_t width, 
    size_t height
) {
    cl_kernel kernel = context->device->coarse_raster_kernel;

    __device_set_coarse_raster_kernel_args(
        kernel,
        context,
        deferred_clear,
        height, 
        width
    );

    size_t local_work_size[2] = {DEVICE_SUB_GROUP_THREADS, DEVICE_COARSE_SUB_GROUPS};
    size_t global_work_size[2] = {local_work_size[0] * DEVICE_NUM_CORES, local_work_size[1]};

    CL_CHECK(clEnqueueNDRangeKernel(context->raster_command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &context->bin_wait_event[0]));
}

static void __device_set_framebuffer_data(
    gl_framebuffer_data_t* fb_data,
    size_t colorbuffer_id, 
    size_t depthbuffer_id, 
    size_t stencilbuffer_id
) {
    *fb_data = (gl_framebuffer_data_t){0};

    if (colorbuffer_id) 
    {
        set_framebuffer_data_colorbuffer_enabled(fb_data);
    } 
    else 
    {
        set_framebuffer_data_colorbuffer_disabled(fb_data);
    }
    if (depthbuffer_id) 
    {
        set_framebuffer_data_depthbuffer_enabled(fb_data);
    } 
    else 
    {
        set_framebuffer_data_depthbuffer_disabled(fb_data);
    }
    if (stencilbuffer_id) 
    {
        set_framebuffer_data_stencilbuffer_enabled(fb_data);
    } 
    else 
    {
        set_framebuffer_data_stencilbuffer_disabled(fb_data);
    }
}

static void device_launch_fragment_shader(
    device_context_t* context,
    size_t shader_id,
    clear_data_t c_data,
    enabled_data_t c_enabled,
    size_t colorbuffer_id, size_t depthbuffer_id, size_t stencilbuffer_id,
    uint32_t colorbuffer_mode,
    size_t width, size_t height, 
    size_t bin_queue_id
) {
    cl_kernel kernel = context->device->shaders[shader_id].fragment;

    gl_framebuffer_data_t framebuffer_data = {0};

    __device_set_framebuffer_data(
        &framebuffer_data,
        colorbuffer_id,
        depthbuffer_id,
        stencilbuffer_id
    );

    __device_set_fragment_shader_kernel_args(
        kernel,
        context,
        context->device->textures[colorbuffer_id],
        context->device->textures[depthbuffer_id],
        context->device->textures[stencilbuffer_id],
        c_data,
        c_enabled,
        colorbuffer_mode,
        height,
        width,
        framebuffer_data
    );

    size_t local_work_size[2] = {DEVICE_SUB_GROUP_THREADS, DEVICE_FINE_SUB_GROUPS};
    size_t global_work_size[2] = {local_work_size[0] * DEVICE_NUM_CORES, local_work_size[1]};

    bin_queue_t *bin_queue = __device_get_bin_queue(context->device, bin_queue_id);

    CL_CHECK(clEnqueueNDRangeKernel(bin_queue->queues[0][0], kernel, 2, NULL, global_work_size, local_work_size, 1, context->bin_wait_event, NULL));
}



static void device_launch_clear_framebuffer(
    device_t* device,
    clear_data_t c_data,
    enabled_data_t c_enabled,
    size_t colorbuffer_id, size_t depthbuffer_id, size_t stencilbuffer_id,
    uint32_t colorbuffer_mode,
    size_t width, size_t height, 
    size_t bin_queue_id
) {
    cl_kernel kernel = device->clear_kernel;
    bin_queue_t* bin_queue = __device_get_bin_queue(device, bin_queue_id);

    cl_uint c_colorbuffer_mode = colorbuffer_mode;
    cl_uint c_viewport_width = width;

    cl_uint count = 0;

    CL_CHECK(clSetKernelArg(kernel, count++, sizeof(cl_mem), &device->renderbuffers[colorbuffer_id]));
    CL_CHECK(clSetKernelArg(kernel, count++, sizeof(cl_mem), &device->renderbuffers[depthbuffer_id]));
    CL_CHECK(clSetKernelArg(kernel, count++, sizeof(cl_mem), &device->renderbuffers[stencilbuffer_id]));
    #ifndef DEVICE_IMAGE_ENABLED
    {
        CL_CHECK(clSetKernelArg(kernel, count++, sizeof(c_colorbuffer_mode), &c_colorbuffer_mode));
        CL_CHECK(clSetKernelArg(kernel, count++, sizeof(c_viewport_width), &c_viewport_width));
    }
    #endif
    CL_CHECK(clSetKernelArg(kernel, count++, sizeof(c_data), &c_data));
    CL_CHECK(clSetKernelArg(kernel, count++, sizeof(c_enabled), &c_enabled));

    size_t global_work_offset[2] = {0, 0};
    size_t global_work_size[2] = {width, height};
    
    CL_CHECK(clEnqueueNDRangeKernel(bin_queue->queues[0][0], kernel, 2, global_work_offset, global_work_size, NULL, 0, NULL, &device->framebuffer_wait_event));
}

static void device_launch_read_pixels(
    device_t* device,
    size_t colorbuffer_id,
    uint32_t colorbuffer_mode,
    size_t c_width, size_t c_height,
    size_t x, size_t y,
    size_t width, size_t height,
    size_t bin_queue_id,
    uint32_t ptr_format,
    void* ptr
) {
    bin_queue_t* bin_queues = __device_get_bin_queue(device, bin_queue_id);

    cl_command_queue queue = bin_queues->queues[0][0];

    cl_mem colorbuffer = device->renderbuffers[colorbuffer_id];

    #ifdef DEVICE_IMAGE_ENABLED
    {
        size_t origin[3] = {x, y, 0};
        size_t region[3] = {width, height, 1};
        CL_CHECK(clEnqueueReadImage(
            queue,
            colorbuffer,
            CL_TRUE,
            origin,
            region,
            0,
            0,
            ptr,
            0,
            NULL,
            NULL
        ));
    }
    #else
    {
        size_t pixel_size = 4; // assuming RGBA8 for shared objects
        size_t buffer_size = width * height * pixel_size;
        size_t buffer_offset = (y * width + x) * pixel_size;

        CL_CHECK(clEnqueueReadBuffer(
            queue,
            colorbuffer,
            CL_TRUE,
            buffer_offset,
            buffer_size,
            ptr,
            0,
            NULL,
            NULL
        ));
    }
    #endif
}

void device_wait_bin_queue(
    device_t* device,
    size_t bin_queue_id
) {
    bin_queue_t* bin_queues = __device_get_bin_queue(device, bin_queue_id);

    for(int i=0; i<DEVICE_BIN_QUEUE_SIZE; ++i) 
    {
        for(int j=0; j<DEVICE_BIN_QUEUE_SIZE; ++j)
        {
            cl_command_queue queue = bin_queues->queues[i][j];

            CL_CHECK(clFinish(queue));
        }
    }
}

void device_bind_vertex_fragment_uniform(
    device_context_t* context
) {
    context->vertex_uniform_type = DEVICE_BINDING_VERTEX_FRAGMENT_UNIFORM;
}


void device_load_vertex_attribute_data(
    device_context_t* context,
    vertex_attribute_data_t* data
) {
    size_t idx = __device_get_vertex_command_index(context);
    cl_command_queue queue = context->vertex_command_queues[idx];

    CL_CHECK(clEnqueueWriteBuffer(
        queue,
        context->vertex_attribute_data_mem[idx],
        CL_FALSE,
        0,
        sizeof(vertex_attribute_data_t[DEVICE_VERTEX_ATTRIBUTE_SIZE]),
        data,
        0,
        NULL,
        NULL
    ));
}

void device_load_vertex_attributes(
    device_context_t* context, 
    float* vertex_attribs
) {
    size_t idx = __device_get_vertex_command_index(context);

    cl_command_queue queue = context->vertex_command_queues[idx];

    CL_CHECK(clEnqueueWriteBuffer(
        queue,
        context->vertex_attributes_mem[idx],
        CL_FALSE,
        0,
        sizeof(cl_float[DEVICE_VERTEX_ATTRIBUTE_SIZE][4]),
        vertex_attribs,
        0,
        NULL,
        NULL
    ));
}

void device_bind_buffer_to_vertex_attribute_pointer(
    device_context_t* context, 
    size_t attribute, 
    size_t buffer_id
) {
    cl_mem buffer_mem = context->device->buffers[buffer_id];

    context->vertex_attrib_pointer_types    [attribute]         = DEVICE_VERTEX_ATTRIBUTE_POINTER;
    context->vertex_attrib_pointer_mems     [attribute].device  = buffer_mem;
}

void device_bind_host_pointer_to_vertex_attribute_pointer(
    device_context_t* context, 
    size_t attribute,
    size_t stride,
    void* ptr
) {
    context->vertex_attrib_pointer_types    [attribute]         = HOST_VERTEX_ATTRIBUTE_POINTER;
    context->vertex_attrib_pointer_strides  [attribute]         = stride;
    context->vertex_attrib_pointer_mems     [attribute].host    = ptr;
}

void device_load_vertex_uniform(
    device_context_t* context, 
    void* uniform_data
) {
    size_t idx = __device_get_vertex_command_index(context);

    cl_command_queue queue = context->vertex_command_queues[idx];

    CL_CHECK(clEnqueueWriteBuffer(
        queue,
        context->vertex_uniform_mem[idx],
        CL_FALSE,
        0,
        sizeof(cl_char[DEVICE_UNIFORM_CAPACITY]),
        uniform_data,
        0,
        NULL,
        NULL
    ));
}

#endif // FRONTEND_DEVICE_H
