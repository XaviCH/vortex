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
} __device_vertex_shader_data_t;

typedef struct {
    __device_vertex_shader_data_t vertex;
    cl_kernel fragment, uniform_data, attribute_data, varying_data;
} __device_shader_kernels_t;

typedef struct {
    cl_command_queue queues[DEVICE_BIN_QUEUE_SIZE][DEVICE_BIN_QUEUE_SIZE]; // TODO: multiple queues
} __device_bin_queue_t;

typedef union {
    size_t  device_id;
    void*   host;
} __device_vertex_attribute_pointer_mem_u;

typedef struct {
    union {
        size_t is_host;
        size_t stride;
    };
    __device_vertex_attribute_pointer_mem_u mem;
} __device_vertex_attribute_pointer_t;

typedef struct {
    cl_mem              mem;
    cl_command_queue    queue;
    cl_event            write_event;
} __device_mem_t;

typedef struct {
    cl_platform_id              platform_id;
    cl_device_id                device_id;
    cl_context                  context;

    cl_program                  triangle_setup_program;
    cl_program                  bin_raster_program;
    cl_program                  coarse_raster_program;
    cl_program                  clear_program;

    cl_kernel                   bin_raster_kernel;
    cl_kernel                   clear_kernel;
    cl_kernel                   coarse_raster_kernel;
    cl_kernel                   read_pixels_kernel;
    cl_kernel                   triangle_setup_arrays_kernel;
    cl_kernel                   triangle_setup_range_kernel;

    size_t                      programs_size;
    cl_program                  programs            [HOST_PROGRAMS_SIZE];
    __device_shader_kernels_t     shaders             [HOST_PROGRAMS_SIZE];

    size_t                      bin_queues_size;
    __device_bin_queue_t                 bin_queues          [HOST_FRAMEBUFFER_SIZE];

    // host writes are waited, so there is no need to sync 
    size_t                      buffers_size;
    __device_mem_t              buffers             [HOST_BUFFERS_SIZE];

    size_t                      textures_size;
    __device_mem_t              textures            [HOST_TEXTURES_SIZE];
} device_t;

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
    __device_mem_t a_bin_counter;
    __device_mem_t a_coarse_counter;
    __device_mem_t a_fine_counter;
    __device_mem_t a_num_active_tiles;
    __device_mem_t a_num_bin_segs;
    __device_mem_t a_num_subtris;
    __device_mem_t a_num_tile_segs;

    #ifdef DEVICE_IMAGE_ENABLED
    // Images
    cl_mem t_tri_data;
    cl_mem t_tri_header;
    cl_mem t_vertex_buffer;
    #endif

    // Binded objects from device
    __device_vertex_attribute_pointer_t     vertex_attribute_pointers  [DEVICE_VERTEX_ATTRIBUTE_SIZE];
    size_t                                  texture_units_ids          [DEVICE_TEXTURE_UNITS];

    // Vertex async state
    // TODO: maybe unify into a OoO queue
    size_t                          vertex_command_queue_index;
    cl_command_queue                vertex_command_queues       [DEVICE_VERTEX_COMMAND_QUEUE_SIZE];

    size_t                          vertex_attributes_index; 
    __device_mem_t                  vertex_attributes_mem       [DEVICE_VERTEX_COMMAND_QUEUE_SIZE];     // sizeof(float[DEVICE_VERTEX_ATTRIBUTE_SIZE][4])

    size_t                          vertex_attribute_data_index; 
    __device_mem_t                  vertex_attribute_data_mem   [DEVICE_VERTEX_COMMAND_QUEUE_SIZE];     // sizeof(vertex_attribute_data_t[DEVICE_VERTEX_ATTRIBUTE_SIZE])

    size_t                          vertex_uniform_index;
    __device_mem_t                  vertex_uniform_mem          [DEVICE_VERTEX_COMMAND_QUEUE_SIZE];     // sizeof(cl_uchar[DEVICE_VERTEX_COMMAND_QUEUE_SIZE][DEVICE_UNIFORM_CAPACITY])            

    // TODO: this is not used, but could be implemented, only use frament_texture_datas_mem for now
    // cl_mem                  vertex_texture_datas_mem    [DEVICE_VERTEX_COMMAND_QUEUE_SIZE];     // sizeof(texture_data_t[DEVICE_VERTEX_COMMAND_QUEUE_SIZE][DEVICE_TEXTURE_UNITS])
    
    // Primitive Configuration
    // Raster async state
    cl_command_queue        raster_command_queue;
    cl_event                bin_wait_event                  [DEVICE_BIN_QUEUE_SIZE][DEVICE_BIN_QUEUE_SIZE];
    __device_mem_t          fragment_texture_datas_mem;                                          // sizeof(texture_data_t[DEVICE_TEXTURE_UNITS])
    __device_mem_t          fragment_uniform_mem;                                           // sizeof(cl_uchar[TRIANGLE_PRIMITIVE_CONFIGS][DEVICE_UNIFORM_CAPACITY])
    // cl_mem                  fragment_uniform_subbuffer_mems [TRIANGLE_PRIMITIVE_CONFIGS];    // subbuffers for vertex shader, TODO: do not use sub-buffers in OpenCL
    __device_mem_t          rop_configs_mem;                                                // sizeof(rop_config_t[TRIANGLE_PRIMITIVE_CONFIGS])

} device_context_t;

//-----------------------------------------------------------------------------------------------

static size_t __device_get_sizeof_image(cl_image_format* image_format, cl_image_desc* image_desc)
{
    size_t size = 0;
    switch (image_format->image_channel_order) {
        case CL_R:
        case CL_A:
        case CL_INTENSITY:
            size = 1;
            break;
        case CL_RG:
        case CL_RA:
            size = 2;
            break;
        case CL_RGB:
            size = 3;
            break;
        case CL_RGBA:
        case CL_BGRA:
            size = 4;
            break;
    }

    switch (image_format->image_channel_data_type) {
        case CL_SNORM_INT8:
        case CL_UNORM_INT8:
        case CL_SIGNED_INT8:
        case CL_UNSIGNED_INT8:
            size *= 1;
            break;
        case CL_SNORM_INT16:
        case CL_UNORM_INT16:
        case CL_SIGNED_INT16:  
        case CL_UNSIGNED_INT16: 
            size *= 2;
            break;
        case CL_SIGNED_INT32:  
        case CL_UNSIGNED_INT32: 
            size *= 4;
            break; 
    }

    return image_desc->image_width * image_desc->image_height * size; // Assuming no row pitch for simplicity
}

static void __device_init_mem(
    device_t* device,
    __device_mem_t* mem,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr
) {
    CL_ASSIGN_CHECK(mem->queue, clCreateCommandQueue(device->context, device->device_id, 0, &error)); 
    CL_ASSIGN_CHECK(mem->mem,   clCreateBuffer(device->context, flags, size, host_ptr, &error));
    mem->write_event = NULL;
}

static void __device_init_2d_image(
    device_t* device,
    __device_mem_t* mem,
    cl_mem_flags flags,
    cl_image_format* image_format,
    cl_image_desc* image_desc,
    void* host_ptr
) {
    CL_ASSIGN_CHECK(mem->queue, clCreateCommandQueue(device->context, device->device_id, 0, &error));

    #ifdef DEVICE_IMAGE_ENABLED
    {
        CL_ASSIGN_CHECK(mem->mem,   clCreateImage(device->context, flags, image_format, image_desc, host_ptr, &error));
    }
    #else
    {
        size_t size = __device_get_sizeof_image(image_format, image_desc);

        size_t last_level = size;
        for (cl_uint level = 1; level < image_desc->num_mip_levels; ++level)
        {
            size += last_level / 4;
        }
        
        CL_ASSIGN_CHECK(mem->mem,   clCreateBuffer(device->context, flags, size, host_ptr, &error));
    }
    #endif

    mem->write_event = NULL;
}

static cl_mem __device_acquire_mem(
    __device_mem_t* mem,
    cl_command_queue command_queue 
) {
    if (mem->write_event) 
    {
        CL_CHECK(clEnqueueBarrierWithWaitList(command_queue, 1, &mem->write_event, NULL));
    }

    return mem->mem;
}

static void __device_copy_mem(
    __device_mem_t* src,
    __device_mem_t* dst,
    size_t src_offset,
    size_t dst_offset,
    size_t size
) {
    __device_acquire_mem(src, dst->queue);

    if (dst->write_event)
    {
        CL_CHECK(clReleaseEvent(dst->write_event));
    }
    
    CL_CHECK(clEnqueueCopyBuffer(dst->queue, src->mem, dst->mem, src_offset, dst_offset, size, 0, NULL, &dst->write_event));
}

static void __device_write_mem(
    __device_mem_t* mem,
    cl_bool blocking_write,
    size_t offset,
    size_t size,
    const void* data
) {
    if (mem->write_event) 
    {
        CL_CHECK(clReleaseEvent(mem->write_event));
        mem->write_event = NULL;
    }

    cl_event* wait_event = blocking_write ? NULL : &mem->write_event;
    
    CL_CHECK(clEnqueueWriteBuffer(
        mem->queue,
        mem->mem, 
        blocking_write, 
        offset, 
        size, 
        data, 
        0, 
        NULL, 
        wait_event
    ));
}

static void __device_write_2d_texture(
    __device_mem_t* mem,
    cl_bool blocking_write,
    size_t origin[3],
    size_t region[3],
    size_t input_row_pitch,
    size_t input_slice_pitch,
    const void* data
) {
    if (mem->write_event) 
    {
        CL_CHECK(clReleaseEvent(mem->write_event));
    }

    cl_event* wait_event = blocking_write ? NULL : &mem->write_event;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        CL_CHECK(clEnqueueWriteImage(
            mem->queue,
            mem->mem,
            blocking_write,
            origin,
            region,
            input_row_pitch,
            input_slice_pitch,
            data,
            0,
            NULL,
            wait_event
        ));
    }
    #else
    {
        size_t host_origin[3] = {0, 0, 0};
        size_t host_row_pitch = 0;
        size_t host_slice_pitch = 0;
        /*
        CL_CHECK(clEnqueueWriteBufferRect(
            mem->queue,
            mem->mem,
            blocking_write,
            origin,
            host_origin,
            region,
            input_row_pitch,
            input_slice_pitch,
            host_row_pitch,
            host_slice_pitch,
            data,
            0,
            NULL,
            wait_event
        ));
        */
        CL_CHECK(clEnqueueWriteBuffer(
            mem->queue,
            mem->mem,
            blocking_write,
            0,
            region[0]*region[1]*sizeof(cl_uint),
            data,
            0,
            NULL,
            wait_event
        ));
    }
    #endif
}

static void __device_barrier_mem(
    __device_mem_t* mem,
    cl_event event
) {
    CL_CHECK(clEnqueueBarrierWithWaitList(mem->queue, 1, &event, NULL));
}



// ---------------------------------------------------------------
// TODO: rm control from device.h to orchestrator.h
static size_t __device_get_vertex_command_index(device_context_t *context) 
{
    return context->vertex_command_queue_index % DEVICE_VERTEX_COMMAND_QUEUE_SIZE;
}

static cl_command_queue __device_get_vertex_command_queue(device_context_t *context) 
{
    return context->vertex_command_queues[__device_get_vertex_command_index(context)];
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
    return DEVICE_MAX_NUMBER_TRIANGLES + DEVICE_MAX_NUMBER_SUBTRIANGLES;
}

static size_t __device_get_max_number_triangles() 
{
    return DEVICE_MAX_NUMBER_TRIANGLES;
}

static size_t __device_get_max_number_bin_segments() 
{
    return CR_MAXBINS_SQR*CR_BIN_STREAMS_SIZE*(DEVICE_MAX_NUMBER_TRIANGLES/CR_BIN_SEG_SIZE); // At least one segment x bin
}

static size_t __device_get_max_number_tile_segments() 
{
    return CR_MAXTILES_SQR*(DEVICE_MAX_NUMBER_TRIANGLES/CR_TILE_SEG_SIZE); // At least one segment x tile
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

static __device_bin_queue_t* __device_get_bin_queue(
    device_t* device, 
    size_t bin_queue_id
) {
    return &device->bin_queues[bin_queue_id];
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

// ---------------------------------------------------------------

static const cl_uint  max_number_triangles = 1;// __device_get_max_number_triangles();
static cl_int   ZERO = 0;

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

    // Mem objects
    shared->textures_size = 1;
    shared->buffers_size = 1;

    cl_image_format image_format = {
        .image_channel_order = CL_RGBA,
        .image_channel_data_type = CL_UNSIGNED_INT32,
    };
    cl_image_desc image_desc = {
        .image_type = CL_MEM_OBJECT_IMAGE2D,
        .image_width = 1,
        .image_height = 1,
        .image_depth = 0,
        .image_array_size = 0,
        .image_row_pitch = 0,
        .image_slice_pitch = 0,
        .num_mip_levels = 0,
        .num_samples = 0,
        .buffer = NULL
    };
    __device_init_2d_image(shared, &shared->textures[0], CL_MEM_READ_WRITE, &image_format, &image_desc, NULL);

    __device_init_mem(shared, &shared->buffers[0], CL_MEM_READ_WRITE, sizeof(cl_uint), NULL);

    shared->bin_queues_size = 0;
    shared->programs_size = 0;
}

void device_init_context(
    device_context_t *context, 
    device_t* device
) {
    context->device = device;

    cl_uint max_number_bin_segments    = __device_get_max_number_bin_segments();
    cl_uint max_number_subtriangles    = __device_get_max_number_subtriangles();
    cl_uint max_number_tile_segments   = __device_get_max_number_tile_segments();
    cl_uint max_number_triangles       = __device_get_max_number_triangles();

    // Buffers
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

    CL_ASSIGN_CHECK(context->g_bin_first_seg,   clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]), NULL, &error)); 
    CL_ASSIGN_CHECK(context->g_bin_seg_data,    clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_bin_segments * CR_BIN_SEG_SIZE]), NULL, &error)); 
    CL_ASSIGN_CHECK(context->g_bin_seg_next,    clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_bin_segments]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_bin_seg_count,   clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_bin_segments]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_bin_total,       clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]), NULL, &error));

    CL_ASSIGN_CHECK(context->g_active_tiles,    clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXTILES_SQR]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_tile_first_seg,  clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXTILES_SQR]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_tile_seg_data,   clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_tile_segments * CR_TILE_SEG_SIZE]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_tile_seg_next,   clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_tile_segments]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_tile_seg_count,  clCreateBuffer(device->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_tile_segments]), NULL, &error));

    // Atomics
    cl_uint ZERO = 0;
    __device_init_mem(device, &context->a_bin_counter,      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO);
    __device_init_mem(device, &context->a_coarse_counter,   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO);
    __device_init_mem(device, &context->a_fine_counter,     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO);
    __device_init_mem(device, &context->a_num_active_tiles, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO);
    __device_init_mem(device, &context->a_num_bin_segs,     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO);
    __device_init_mem(device, &context->a_num_subtris,      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &max_number_triangles);
    __device_init_mem(device, &context->a_num_tile_segs,    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO);

    // Vertex context objects
    context->vertex_command_queue_index = 0;
    context->vertex_attributes_index = 0;
    context->vertex_attribute_data_index = 0;
    context->vertex_uniform_index = 0;

    for (size_t i = 0; i < DEVICE_VERTEX_COMMAND_QUEUE_SIZE; ++i) {
        CL_ASSIGN_CHECK(context->vertex_command_queues[i],      clCreateCommandQueue(device->context, device->device_id, 0, &error)); 
        
        __device_init_mem(device, &context->vertex_attributes_mem[i],       CL_MEM_READ_ONLY, sizeof(cl_float4[DEVICE_VERTEX_ATTRIBUTE_SIZE]), NULL);
        __device_init_mem(device, &context->vertex_attribute_data_mem[i],   CL_MEM_READ_ONLY, sizeof(vertex_attribute_data_t[DEVICE_VERTEX_ATTRIBUTE_SIZE]), NULL);
        __device_init_mem(device, &context->vertex_uniform_mem[i],          CL_MEM_READ_ONLY, DEVICE_UNIFORM_CAPACITY, NULL);
    }

    // Fragment context objects
    CL_ASSIGN_CHECK(context->raster_command_queue, clCreateCommandQueue(device->context, device->device_id, 0, &error));

    __device_init_mem(context->device, &context->fragment_texture_datas_mem, CL_MEM_READ_ONLY, sizeof(gl_texture_data_t[DEVICE_TEXTURE_UNITS]), NULL);

    __device_init_mem(context->device, &context->fragment_uniform_mem, CL_MEM_READ_ONLY, sizeof(cl_uchar[TRIANGLE_PRIMITIVE_CONFIGS][DEVICE_UNIFORM_CAPACITY]), NULL);

    __device_init_mem(context->device, &context->rop_configs_mem, CL_MEM_READ_ONLY, sizeof(rop_config_t[TRIANGLE_PRIMITIVE_CONFIGS]), NULL);    

    // Default bind state
    for (size_t attribute = 0; attribute < DEVICE_VERTEX_ATTRIBUTE_SIZE; ++attribute)
    {
        context->vertex_attribute_pointers[attribute].is_host = 0;
        context->vertex_attribute_pointers[attribute].mem.device_id = 0;
    }

    for (size_t texture_unit = 0; texture_unit < DEVICE_TEXTURE_UNITS; ++texture_unit)
    {
        context->texture_units_ids[texture_unit] = 0;
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
    cl_mem attribute_pointers[DEVICE_VERTEX_ATTRIBUTE_SIZE]
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
        __device_mem_t *mem = &context->device->textures[context->texture_units_ids[i]];
        CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &mem->mem));
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

    __device_bin_queue_t* bin_queue = &device->bin_queues[bin_queue_id];

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
    size_t program_id = device->programs_size;

    // create program object
    CL_ASSIGN_CHECK(device->programs[program_id], clCreateProgramWithBinary(device->context, 1, &device->device_id, &size, &binary, NULL, &error));
    CL_CHECK(clBuildProgram(device->programs[program_id], 1, &device->device_id, NULL, NULL, NULL));

    // create program kernels
    cl_program program = device->programs[program_id];
    __device_shader_kernels_t* kernels = &device->shaders[program_id];

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

    device->programs_size += 1;

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

/*
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
*/

static size_t device_create_ro_buffer(device_t* device, size_t size) 
{
    size_t buffer_id = device->buffers_size;

    __device_init_mem(device, &device->buffers[buffer_id], CL_MEM_READ_ONLY, size, NULL);
    
    device->buffers_size += 1;

    return buffer_id;
}

static size_t device_create_2d_texture(device_t* device, size_t width, size_t height, cl_uint texture_mode) 
{
    size_t texture_id = device->textures_size;

    cl_image_format image_format = __device_get_image_format_from_texture_mode(texture_mode);
    cl_image_desc image_desc = {
        .image_type = CL_MEM_OBJECT_IMAGE2D,
        .image_width = width,
        .image_height = height,
        .image_depth = 0,
        .image_array_size = 0,
        .image_row_pitch = 0,
        .image_slice_pitch = 0,
        .num_mip_levels = 0,
        .num_samples = 0,
        .buffer = NULL
    };

    __device_init_2d_image(device, &device->textures[texture_id], CL_MEM_READ_WRITE, &image_format, &image_desc, NULL);

    device->textures_size += 1;

    return texture_id;
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
    
    for(size_t i= 0; i<num_vertices; ++i) 
    {
        printf("vertex %ld:", i);
        for(size_t j=0; j<num_varying+1; ++j) {
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

static void __device_print_triangle_assembly_output(
    device_context_t* context,
    cl_command_queue queue,
    size_t triangles
) {
    size_t sizeof_primitives = sizeof(cl_uchar[triangles]);
    cl_uchar *primitives = (cl_uchar*) malloc(sizeof_primitives);
    CL_CHECK(clEnqueueReadBuffer(queue, context->g_tri_subtris, CL_TRUE, 0, sizeof_primitives, primitives, 0, NULL, NULL));

    printf("subtri=[%d", primitives[0]);
    for(size_t i= 1; i<triangles; ++i) {
        printf(",%d", primitives[i]);
    }
    printf("]\n");

    free(primitives);

    size_t sizeof_header = sizeof(triangle_header_t[triangles]);
    triangle_header_t *triangle_header = (triangle_header_t*) malloc(sizeof_header);
    CL_CHECK(clEnqueueReadBuffer(queue, context->g_tri_header, CL_TRUE, 0, sizeof_header, triangle_header, 0, NULL, NULL));

    for(size_t i=0; i<triangles; ++i) {
        triangle_header_t *header = &triangle_header[i];
        printf("header[%ld]={v0x=%d,v0y=%d,v1x=%d,v1y=%d,v2x=%d,v2y=%d}\n",i, 
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

//-------------------------------------------------------------------------------------

static void device_launch_vertex_shader(
    device_context_t* context,
    size_t shader_id,
    size_t init,
    size_t end,
    size_t offset,
    size_t primitive_id,
    int use_fragment_uniform
) {
    if (init != 0) CL_UNSUPORTED_MAPING();

    cl_command_queue queue = __device_get_vertex_command_queue(context);
    cl_kernel kernel = context->device->shaders[shader_id].vertex.kernel;
    
    size_t num_vertices = end - init;
    size_t vertex_attribute_size = context->device->shaders[shader_id].vertex.number_attributes;
    cl_mem  attribute_pointer_mems [DEVICE_VERTEX_ATTRIBUTE_SIZE];
    
    // Set kernel arguments and acquire pointers

    for (size_t attribute = 0; attribute < vertex_attribute_size; ++attribute)
    {
        __device_vertex_attribute_pointer_t *attribute_pointer = &context->vertex_attribute_pointers[attribute];
        cl_mem *pointer = &attribute_pointer_mems[attribute];
        
        if (attribute_pointer->is_host) {
            CL_ASSIGN_CHECK(*pointer, clCreateBuffer(
                context->device->context,
                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                num_vertices * attribute_pointer->stride,
                attribute_pointer->mem.host,
                &error
            ));
        }
        else
        {
            __device_mem_t* mem = &context->device->buffers[attribute_pointer->mem.device_id];
            *pointer = __device_acquire_mem(mem, queue);
        }
    }

    __device_mem_t* vertex_attributes         = &context->vertex_attributes_mem[context->vertex_attributes_index];
    __device_mem_t* vertex_attribute_data     = &context->vertex_attribute_data_mem[context->vertex_attribute_data_index]; 
    __device_mem_t* vertex_uniform            = &context->vertex_uniform_mem[context->vertex_uniform_index];

    cl_mem vertex_uniform_mem;
    // TODO: check savety of that operation
    if (use_fragment_uniform)
    {
        __device_acquire_mem(&context->fragment_uniform_mem, queue);

        cl_buffer_region buffer_region = {
                .origin = DEVICE_UNIFORM_CAPACITY * primitive_id,
                .size = DEVICE_UNIFORM_CAPACITY
        };

        printf("primitive_id=%ld\n", primitive_id);

        CL_ASSIGN_CHECK(vertex_uniform_mem, clCreateSubBuffer(
            context->fragment_uniform_mem.mem, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
            &buffer_region, &error
        ));
    }
    else
    {
        vertex_uniform_mem = __device_acquire_mem(vertex_uniform, queue);
    }

    __device_set_vertex_shader_kernel_args(
        kernel,
        context,
        vertex_attribute_size,
        __device_acquire_mem(vertex_attributes,     queue),
        __device_acquire_mem(vertex_attribute_data, queue),
        vertex_uniform_mem,
        attribute_pointer_mems
    );

    // Launch kernel

    size_t gw_offset = offset;
    size_t gw_size = num_vertices;

    cl_event wait_event;
    
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, &gw_offset, &gw_size, NULL, 0, NULL, &wait_event));
    
    // Release host pointers and synchronize device pointers

    for (size_t attribute = 0; attribute < vertex_attribute_size; ++attribute)
    {
        __device_vertex_attribute_pointer_t *attribute_pointer = &context->vertex_attribute_pointers[attribute];

        if (attribute_pointer->is_host) 
        {
            CL_CHECK(clReleaseMemObject(attribute_pointer_mems[attribute]));
        }
        else
        {
            __device_mem_t* mem = &context->device->buffers[attribute_pointer->mem.device_id];
            __device_barrier_mem(mem, wait_event);
        }
    }

    __device_barrier_mem(vertex_attributes,      wait_event);
    __device_barrier_mem(vertex_attribute_data,  wait_event);

    if (use_fragment_uniform)
    {
        CL_CHECK(clReleaseMemObject(vertex_uniform_mem));
    }
    else
    {
        __device_barrier_mem(vertex_uniform, wait_event);
    }

    CL_CHECK(clReleaseEvent(wait_event));

    __device_print_vertex_shader_output(context, queue, num_vertices, /*num_varying=*/1);
}

void device_launch_range_triangle_assembly(
    device_context_t* context, 
    size_t shader_id,
    render_mode_t mode, 
    size_t frag_config_id, 
    size_t vertex_offset, size_t triangle_offset, size_t num_triangles,
    size_t width, size_t height, size_t size, uint16_t* ptr
) {
    cl_kernel kernel = context->device->triangle_setup_range_kernel;

    cl_command_queue queue = __device_get_vertex_command_queue(context);

    cl_mem g_index_buffer;
    CL_ASSIGN_CHECK(g_index_buffer, clCreateBuffer(
            context->device->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_ushort[size]), (void*) ptr, &error
    ));

    __device_acquire_mem(&context->a_num_subtris,  queue);

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

    size_t gwo = triangle_offset;
    size_t gws = num_triangles;

    cl_event wait_event;

    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, &gwo, &gws, NULL, 0, NULL, &wait_event));

    CL_CHECK(clReleaseMemObject(g_index_buffer));

    CL_CHECK(clEnqueueBarrierWithWaitList(context->raster_command_queue, 1, &wait_event, NULL));
    CL_CHECK(clReleaseEvent(wait_event));

    __device_advance_vertex_command_index(context);

    __device_print_triangle_assembly_output(context, queue, num_triangles);
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
    size_t height 
) {
    cl_kernel kernel = context->device->triangle_setup_arrays_kernel;

    cl_command_queue queue = __device_get_vertex_command_queue(context);

    __device_acquire_mem(&context->a_num_subtris,   queue);

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

    size_t gwo = triangle_offset;
    size_t gws = num_triangles;

    cl_event wait_event;

    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, &gwo, &gws, NULL, 0, NULL, &wait_event));

    CL_CHECK(clEnqueueBarrierWithWaitList(context->raster_command_queue, 1, &wait_event, NULL));
    CL_CHECK(clReleaseEvent(wait_event));

    __device_advance_vertex_command_index(context);
 
    // __device_print_triangle_assembly_output(context, queue, num_triangles);
}

static void device_launch_bin_dispatch(
    device_context_t* context, 
    size_t num_triangles, 
    size_t width, 
    size_t height
) {
    cl_kernel kernel = context->device->bin_raster_kernel;
    cl_command_queue queue = context->raster_command_queue;

    __device_acquire_mem(&context->a_bin_counter, queue),
    __device_acquire_mem(&context->a_num_bin_segs, queue),

    __device_set_bin_raster_kernel_args(
        kernel,
        context,
        num_triangles,
        height, 
        width
    );

    size_t lws[2] = {DEVICE_SUB_GROUP_THREADS, DEVICE_BIN_SUB_GROUPS};
    size_t gws[2] = {lws[0] * CR_BIN_STREAMS_SIZE, lws[1]};

    cl_event wait_event;

    CL_CHECK(clEnqueueNDRangeKernel(queue,kernel, 2, NULL, gws, lws, 0, NULL, &wait_event));

    __device_barrier_mem(&context->a_bin_counter,   wait_event);

    CL_CHECK(clReleaseEvent(wait_event));

    __device_reset_vertex_command_index(context);

    cl_int ZERO = 0;
    __device_write_mem(&context->a_bin_counter,    CL_FALSE, 0, sizeof(cl_int), &ZERO);
}

static void device_launch_tile_dispatch(
    device_context_t* context,
    uint32_t deferred_clear,
    size_t width, 
    size_t height
) {
    cl_kernel kernel = context->device->coarse_raster_kernel;
    cl_command_queue queue = context->raster_command_queue;

    __device_acquire_mem(&context->a_coarse_counter,    queue),
    __device_acquire_mem(&context->a_num_active_tiles,  queue),
    __device_acquire_mem(&context->a_num_tile_segs,     queue),

    __device_set_coarse_raster_kernel_args(
        kernel,
        context,
        deferred_clear,
        height, 
        width
    );

    size_t lws[2] = {DEVICE_SUB_GROUP_THREADS, DEVICE_COARSE_SUB_GROUPS};
    size_t gws[2] = {lws[0] * DEVICE_NUM_CORES, lws[1]};

    #if DEVICE_BIN_QUEUE_SIZE > 1
    #error "DEVICE_BIN_QUEUE_SIZE must be 1 for now, because of the way we handle events"
    #endif

    for (size_t w=0; w<DEVICE_BIN_QUEUE_SIZE; ++w) 
    {
        for (size_t h=0; h<DEVICE_BIN_QUEUE_SIZE; ++h) 
        {
            cl_event wait_event;

            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, lws, 0, NULL, &wait_event));

            __device_barrier_mem(&context->a_coarse_counter, wait_event);
            context->bin_wait_event[w][h] = wait_event;
        }
    }

    cl_int ZERO = 0;
    __device_write_mem(&context->a_coarse_counter,    CL_FALSE, 0, sizeof(cl_int), &ZERO);
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
        context->device->textures[colorbuffer_id].mem,
        context->device->textures[depthbuffer_id].mem,
        context->device->textures[stencilbuffer_id].mem,
        c_data,
        c_enabled,
        colorbuffer_mode,
        height,
        width,
        framebuffer_data
    );

    size_t lws[2] = {DEVICE_SUB_GROUP_THREADS, DEVICE_FINE_SUB_GROUPS};
    size_t gws[2] = {lws[0] * DEVICE_NUM_CORES, lws[1]};

    __device_bin_queue_t *bin_queue = __device_get_bin_queue(context->device, bin_queue_id);

    for (size_t w=0; w<DEVICE_BIN_QUEUE_SIZE; ++w) 
    {
        for (size_t h=0; h<DEVICE_BIN_QUEUE_SIZE; ++h)
        {
            cl_command_queue queue = bin_queue->queues[w][h];

            __device_acquire_mem(&context->a_fine_counter, queue);
            __device_acquire_mem(&context->fragment_texture_datas_mem, queue);
            __device_acquire_mem(&context->fragment_uniform_mem, queue);
            __device_acquire_mem(&context->rop_configs_mem, queue);

            cl_event bin_wait_event = context->bin_wait_event[w][h];
            cl_event wait_event;

            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, lws, 1, &bin_wait_event, &wait_event));

            CL_CHECK(clReleaseEvent(bin_wait_event));

            __device_barrier_mem(&context->a_fine_counter,              wait_event);
            __device_barrier_mem(&context->fragment_texture_datas_mem,  wait_event);
            __device_barrier_mem(&context->fragment_uniform_mem,        wait_event);
            __device_barrier_mem(&context->rop_configs_mem,             wait_event);

            __device_barrier_mem(&context->a_num_active_tiles,  wait_event);
            __device_barrier_mem(&context->a_num_bin_segs,      wait_event);
            __device_barrier_mem(&context->a_num_subtris,       wait_event);
            __device_barrier_mem(&context->a_num_tile_segs,     wait_event);

            for (size_t vertex_queue = 0; vertex_queue < DEVICE_VERTEX_COMMAND_QUEUE_SIZE; ++vertex_queue) 
            {
                cl_command_queue vertex_command_queue = context->vertex_command_queues[vertex_queue];
                CL_CHECK(clEnqueueBarrierWithWaitList(vertex_command_queue, 1, &wait_event, NULL));
            }
            
            CL_CHECK(clReleaseEvent(wait_event));
        }
    }

    // reset state
    cl_uint max_number_triangles = __device_get_max_number_triangles();
    cl_int ZERO = 0;
    
    __device_write_mem(&context->a_fine_counter,        CL_FALSE, 0, sizeof(cl_int), &ZERO);
    __device_write_mem(&context->a_num_active_tiles,    CL_FALSE, 0, sizeof(cl_int), &ZERO);
    __device_write_mem(&context->a_num_bin_segs,        CL_FALSE, 0, sizeof(cl_int), &ZERO);
    __device_write_mem(&context->a_num_subtris,         CL_FALSE, 0, sizeof(cl_int), &max_number_triangles);
    __device_write_mem(&context->a_num_tile_segs,       CL_FALSE, 0, sizeof(cl_int), &ZERO);
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
    __device_bin_queue_t* bin_queue = __device_get_bin_queue(device, bin_queue_id);

    cl_uint c_colorbuffer_mode = colorbuffer_mode;
    cl_uint c_viewport_width = width;

    cl_uint count = 0;

    CL_CHECK(clSetKernelArg(kernel, count++, sizeof(cl_mem), &device->textures[colorbuffer_id]));
    CL_CHECK(clSetKernelArg(kernel, count++, sizeof(cl_mem), &device->textures[depthbuffer_id]));
    CL_CHECK(clSetKernelArg(kernel, count++, sizeof(cl_mem), &device->textures[stencilbuffer_id]));
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
    
    CL_CHECK(clEnqueueNDRangeKernel(bin_queue->queues[0][0], kernel, 2, global_work_offset, global_work_size, NULL, 0, NULL, NULL));
}

static void device_launch_read_pixels(
    device_t* device,
    size_t bin_queue_id,
    size_t colorbuffer_id,
    uint32_t colorbuffer_mode,
    size_t c_width, size_t c_height,
    size_t x, size_t y,
    size_t width, size_t height,
    uint32_t ptr_format,
    void* ptr
) {
    __device_bin_queue_t* bin_queues = __device_get_bin_queue(device, bin_queue_id);

    cl_command_queue queue = bin_queues->queues[0][0];

    cl_mem colorbuffer = device->textures[colorbuffer_id].mem;

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
    __device_bin_queue_t* bin_queues = __device_get_bin_queue(device, bin_queue_id);

    for(int i=0; i<DEVICE_BIN_QUEUE_SIZE; ++i) 
    {
        for(int j=0; j<DEVICE_BIN_QUEUE_SIZE; ++j)
        {
            cl_command_queue queue = bin_queues->queues[i][j];

            CL_CHECK(clFinish(queue));
        }
    }
}

//-------------------------------------------------------------------------------------

static void device_bind_buffer_to_vertex_attribute_pointer(
    device_context_t* context, 
    size_t attribute_id, 
    size_t buffer_id
) {
    context->vertex_attribute_pointers[attribute_id] = (__device_vertex_attribute_pointer_t) {
        .is_host = 0,
        .mem = {
            .device_id = buffer_id
        }
    };
}

static void device_bind_host_pointer_to_vertex_attribute_pointer(
    device_context_t* context, 
    size_t attribute_id,
    size_t stride,
    void* ptr
) {
    context->vertex_attribute_pointers[attribute_id] = (__device_vertex_attribute_pointer_t) {
        .stride = stride,
        .mem = {
            .host = ptr
        }
    };
}

static void device_bind_texture_unit(device_context_t* context, size_t unit_index, size_t texture_id) 
{
    context->texture_units_ids[unit_index] = texture_id;
}

//-------------------------------------------------------------------------------------

static void device_copy_context_last_state(device_context_t* dst, device_context_t* src, size_t primitive_id)
{
    memcpy(dst->texture_units_ids, src->texture_units_ids, sizeof(src->texture_units_ids));
    memcpy(dst->vertex_attribute_pointers, src->vertex_attribute_pointers, sizeof(src->vertex_attribute_pointers));

    __device_write_mem(&dst->a_bin_counter,         CL_FALSE, 0, sizeof(ZERO), &ZERO);
    __device_write_mem(&dst->a_coarse_counter,      CL_FALSE, 0, sizeof(ZERO), &ZERO);
    __device_write_mem(&dst->a_fine_counter,        CL_FALSE, 0, sizeof(ZERO), &ZERO);
    __device_write_mem(&dst->a_num_active_tiles,    CL_FALSE, 0, sizeof(ZERO), &ZERO);
    __device_write_mem(&dst->a_num_bin_segs,        CL_FALSE, 0, sizeof(ZERO), &ZERO);
    __device_write_mem(&dst->a_num_subtris,         CL_FALSE, 0, sizeof(max_number_triangles), &max_number_triangles);
    __device_write_mem(&dst->a_num_tile_segs,       CL_FALSE, 0, sizeof(ZERO), &ZERO);

    dst->vertex_attribute_data_index = 0;
    dst->vertex_attributes_index = 0;
    dst->vertex_uniform_index = 0;

    __device_copy_mem(
        &src->vertex_attribute_data_mem[src->vertex_attribute_data_index],
        &dst->vertex_attribute_data_mem[dst->vertex_attribute_data_index],
        0,
        0,
        sizeof(vertex_attribute_data_t[DEVICE_VERTEX_ATTRIBUTE_SIZE])
    );

    __device_copy_mem(
        &src->vertex_attributes_mem[src->vertex_attributes_index],
        &dst->vertex_attributes_mem[dst->vertex_attributes_index],
        0,
        0,
        sizeof(float[DEVICE_VERTEX_ATTRIBUTE_SIZE][4])
    );

    __device_copy_mem(
        &src->vertex_uniform_mem[src->vertex_uniform_index],
        &dst->vertex_uniform_mem[dst->vertex_uniform_index],
        0,
        0,
        sizeof(cl_uchar[DEVICE_UNIFORM_CAPACITY])
    );

    __device_copy_mem(
        &src->fragment_texture_datas_mem,
        &dst->fragment_texture_datas_mem,
        0,
        0,
        sizeof(gl_texture_data_t[DEVICE_TEXTURE_UNITS])
    );

    size_t sizeof_uniform = sizeof(cl_uchar[DEVICE_UNIFORM_CAPACITY]);

    __device_copy_mem(
        &src->fragment_uniform_mem,
        &dst->fragment_uniform_mem,
        sizeof_uniform * primitive_id,
        0,
        sizeof(cl_uchar[DEVICE_UNIFORM_CAPACITY])
    );

    size_t sizeof_rop_config = sizeof(rop_config_t);

    __device_copy_mem(
        &src->rop_configs_mem,
        &dst->rop_configs_mem,
        sizeof_rop_config * primitive_id,
        0,
        sizeof_rop_config
    );
}

//-------------------------------------------------------------------------------------

// TODO: write api return an event handler, to do async writing

static void device_write_vertex_attribute_data(
    device_context_t* context,
    vertex_attribute_data_t vertex_attribute_data[DEVICE_VERTEX_ATTRIBUTE_SIZE]
) {
    context->vertex_attribute_data_index += 1;
    context->vertex_attribute_data_index %= DEVICE_VERTEX_COMMAND_QUEUE_SIZE;

    size_t idx = context->vertex_attribute_data_index;

    __device_write_mem(
        &context->vertex_attribute_data_mem[idx],
        CL_TRUE,
        0, 
        sizeof(vertex_attribute_data_t[DEVICE_VERTEX_ATTRIBUTE_SIZE]), 
        vertex_attribute_data
    );
}

static void device_write_vertex_attributes(
    device_context_t* context, 
    float vertex_attributes[DEVICE_VERTEX_ATTRIBUTE_SIZE][4],
    int blocking_write
) {
    context->vertex_attributes_index += 1;
    context->vertex_attributes_index %= DEVICE_VERTEX_COMMAND_QUEUE_SIZE;

    size_t idx = context->vertex_attributes_index;

    __device_write_mem(
        &context->vertex_attributes_mem[idx],
        CL_TRUE,
        0, 
        sizeof(cl_float[DEVICE_VERTEX_ATTRIBUTE_SIZE][4]), 
        vertex_attributes
    );
}

static void device_write_vertex_uniform(
    device_context_t* context, 
    uint8_t uniform_data[DEVICE_UNIFORM_CAPACITY],
    int blocking_write
) {
    context->vertex_uniform_index += 1;
    context->vertex_uniform_index %= DEVICE_VERTEX_COMMAND_QUEUE_SIZE;

    size_t idx = context->vertex_uniform_index;

    __device_write_mem(
        &context->vertex_uniform_mem[idx], 
        CL_TRUE,
        0, 
        sizeof(cl_char[DEVICE_UNIFORM_CAPACITY]), 
        uniform_data
    );
}

static void device_write_rop_config(
    device_context_t* context, 
    size_t primitive_id, 
    rop_config_t* rop_config
) {
    size_t size = sizeof(rop_config_t); 
    size_t offset = primitive_id * size; 

    __device_write_mem(
        &context->rop_configs_mem, 
        CL_TRUE, 
        offset,
        size,
        rop_config
    );
}

// TODO: textures are not implemented for vertex
static void device_write_fragment_texture_datas(
    device_context_t* context, 
    const gl_texture_data_t texture_datas[DEVICE_TEXTURE_UNITS]
) {
    size_t size = sizeof(gl_texture_data_t[DEVICE_TEXTURE_UNITS]); 

    __device_write_mem(
        &context->fragment_texture_datas_mem, 
        CL_TRUE, 
        0, 
        size,
        texture_datas
    );
}

static void device_write_fragment_uniform(
    device_context_t* context, 
    size_t primitive_id, 
    uint8_t uniform_data[DEVICE_UNIFORM_CAPACITY]
) {
    size_t size = sizeof(uint8_t[DEVICE_UNIFORM_CAPACITY]); 
    size_t offset = primitive_id * size; 

    __device_write_mem(
        &context->fragment_uniform_mem, 
        CL_TRUE, 
        offset, 
        size,
        uniform_data
    );
}

static void device_write_2d_texture(
    device_t* device, 
    size_t texture_id, 
    size_t x, 
    size_t y, 
    size_t width,
    size_t height,
    uint32_t device_texture_mode,
    uint32_t host_texture_mode,
    const void* data
) 
{
    __device_mem_t* texture_mem = &device->textures[texture_id];

    size_t origin[3] = {x, y, 1};
    size_t region[3] = {width, height, 1};

    __device_write_2d_texture(&device->textures[texture_id], CL_TRUE, origin, region, 0, 0, data);
}

static void device_write_buffer(device_t* device, size_t buffer_id, size_t offset, size_t size, const void* data) 
{
    __device_write_mem(&device->buffers[buffer_id], CL_TRUE, offset, size, data);
}

#endif // FRONTEND_DEVICE_H
