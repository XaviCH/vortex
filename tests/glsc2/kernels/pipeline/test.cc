#include <algorithm>
#include <cmath>
#include <chrono>
#include "glm/gtc/matrix_transform.hpp"

//#include <backend/pipeline/triangle_setup.o.c>
//#include <backend/pipeline/bin_raster.o.c>
//#include <backend/pipeline/coarse_raster.o.c>
//#include <backend/pipeline/fine_raster.o.c>
#include "shader.o.c"
#include <types.device.h>
//#include <kernels/shadered_fine_raster.ocl.c>
#include <constants.device.h>
#include <tests/glsc2/common.h>
#include <tests/glsc2/debug.hpp>
#include <tests/glsc2/kernels/common.hpp>

#include <frontend/device.h>

#define CL_TARGET_OPENCL_VERSION 120
#define TEST_PATH "../../../"

// NVIDIA DEFS
int maxTris             = 4096;     //
int maxSubtrisSlack     = 4096;     // x 81B    = 324KB
int maxBinSegsSlack     = 256;      // x 2137B  = 534KB
int maxTileSegsSlack    = 4096;     // x 136B   = 544KB
int ATTRIBUTE_MULTIPROCESSOR_COUNT = 36; // SMs
int ATTRIBUTE_MAX_THREADS_PER_BLOCK = 32*DEVICE_FINE_SUB_GROUPS;

// TEST DEFAULT PARAMETERS
char VERTEX_SHADER_KERNEL_NAME[]    = "gl_vertex_shader";
char TRIANGLE_SETUP_KERNEL_NAME[]   = "triangle_setup_range";
char BIN_RASTER_KERNEL_NAME[]       = "bin_raster";
char COARSE_RASTER_KERNEL_NAME[]    = "coarse_raster";
char FINE_RASTER_KERNEL_NAME[]      = "fine_raster_single_sample";

cl_ushort INDEX_BUFFER[] = {
    0, 1, 3, 3, 1, 2,
    1, 5, 2, 2, 5, 6,
    5, 4, 6, 6, 4, 7,
    4, 0, 7, 7, 0, 3,
    3, 2, 7, 7, 2, 6,
    4, 5, 0, 0, 5, 1,
};

float position[] {
    -1, -1, -1,
    1, -1, -1,
    1, 1, -1,
    -1, 1, -1,
    -1, -1, 1,
    1, -1, 1,
    1, 1, 1,
    -1, 1, 1
};

float color[] {
    1,0,0,
    0,1,0,
    0,0,1,
    1,1,0,
    0,1,1,
    1,0,1,
    1,1,1,
    0,0,0
};

float tex[] {
    0,0,
    1,0,
    1,1,
    0,1,
    0,0,
    1,0,
    1,1,
    0,1,
};

// Test Configuration
size_t max_number_attributes    = 16;
size_t max_number_varying       = 16;
size_t max_number_vertices      = 1UL << 12;  // 4K vertices
size_t max_number_triangles     = 1UL << 12;  // 4K triangles
size_t max_number_extra_subtriangles  = 1UL << 10;  // 1K subtriangles
size_t max_number_vertex_queues = 2;
size_t max_number_bin_segments  = CR_MAXBINS_SQR*CR_BIN_STREAMS_SIZE; // At least one segment x bin
size_t max_number_tile_segments = CR_MAXTILES_SQR; // At least one segment x tile
size_t framebuffer_width        = 2048;
size_t framebuffer_height       = 2048;
bool emulate_vertex, emulate_bin, emulate_coarse, emulate_fine;


void parse_args(int argc, char** argv) {
    #define GET_AND_SET(DATA, ARG) \
        { \
            char str_arg[] = "--" #ARG "="; \
            size_t argument_size = sizeof("--" #ARG "="); \
            if (strncmp(str_arg, DATA, argument_size) == 0) ARG = atoi(&DATA[argument_size]); \
        }

    for(int arg = 0; arg < argc; ++arg) {
        char* argument = argv[arg];
        GET_AND_SET(argument, max_number_attributes);
        GET_AND_SET(argument, max_number_varying);
        GET_AND_SET(argument, max_number_vertices);
        GET_AND_SET(argument, max_number_triangles);
        GET_AND_SET(argument, max_number_extra_subtriangles);
        GET_AND_SET(argument, framebuffer_width);
        GET_AND_SET(argument, framebuffer_height);
    }

    #undef GET_AND_SET
}

int NUM_TRIS = sizeof(INDEX_BUFFER)/(sizeof(INDEX_BUFFER[0])*3);
int MAX_SUBTRIS = 1;
int MAX_BIN_SEGS = 1;
int MAX_TILE_SEGS = 1;
int WIDTH = 100;
int HEIGHT = 100;

typedef struct {
    float position[4];
    float color[4];
    float texcoord[4];
} VertexData;

int num_samples = 1;


// INPUT PARAMETERS
cl_int samples_log2 = 0;
cl_uint render_mode_flags = 0 | RENDER_MODE_FLAG_ENABLE_LERP | RENDER_MODE_FLAG_ENABLE_DEPTH;
// const glm::ivec2 viewport_size = {WIDTH, HEIGHT};

// Test CL Parameters
cl_int c_bin_batch_sz; // number of triangles being processed in each CTA
cl_uint c_blending_color;
cl_uint c_blending_data;
cl_uint c_clear_color;
cl_ushort c_clear_depth;
cl_uchar c_clear_stencil;
cl_uint c_color_buffer_mode;
cl_int c_deferred_clear;
cl_int c_depth_data;
cl_int c_height_bins;
cl_int c_height_tiles;
cl_int c_max_bin_segs;
cl_int c_max_subtris;
cl_int c_max_tile_segs;
cl_int c_num_bins;
cl_int c_num_tris;
cl_uint c_render_mode_flags;
cl_int c_samples_log2;
cl_uint c_stencil_data;
cl_int c_vertex_size;
cl_int c_viewport_height;
cl_int c_viewport_width;
cl_int c_width_bins;
cl_int c_width_tiles;
cl_ulong c_clear_write_values;
cl_ushort c_clear_enabled_data;
cl_ushort c_enabled_data;


void set_cl_parameters() {
    // setup parameters
    c_max_subtris = max_number_triangles + max_number_extra_subtriangles;
    c_viewport_height = framebuffer_height;
    c_viewport_width = framebuffer_width;
    printf("INFO: max_subtris=%d, viewport=(%d,%d)\n", c_max_subtris, c_viewport_width, c_viewport_height);
    // bin parameters
    c_bin_batch_sz = DEVICE_BIN_SUB_GROUPS*DEVICE_SUB_GROUP_THREADS; // at least use all threads available, maybe can be fine tuned for larger executions
    c_max_bin_segs = max_number_bin_segments; // biggest as posible
    c_height_bins = ((framebuffer_height-1) / (CR_TILE_SIZE * CR_BIN_SIZE)) + 1;
    c_width_bins = ((framebuffer_width-1) / (CR_TILE_SIZE * CR_BIN_SIZE)) + 1;
    c_num_bins = c_height_bins * c_width_bins;
    printf("INFO: bin_bacth_sz=%d, max_bin_segs=%d, bins=(%d,%d)\n", c_bin_batch_sz, c_max_bin_segs, c_width_bins, c_height_bins);
    // tile parameters
    c_max_tile_segs = max_number_tile_segments; // biggest as posible
    c_height_tiles = ((framebuffer_height-1) / CR_TILE_SIZE) + 1;
    c_width_tiles = ((framebuffer_width-1) / CR_TILE_SIZE) + 1;
    c_deferred_clear = 1;
    printf("INFO: max_tile_segs=%d, tiles=(%d,%d)\n", c_max_tile_segs, c_width_tiles, c_height_tiles);
    // fine parameters
    c_blending_data = 0x22220000;
    c_clear_color = 0xFFFF0000;
    c_clear_depth = 0xFFFFu;
    c_clear_stencil = 0xFFu;
    c_color_buffer_mode = TEX_RGBA8;
    c_depth_data = DEPTH_FUNC_LESS | (1 << 16);
    c_render_mode_flags = render_mode_flags;
    c_stencil_data = 0xFFFFFFFFu & STENCIL_FUNC_ALWAYS;
    c_samples_log2 = 0; // TODO: add multisampling
    c_vertex_size = sizeof(VertexData);
    c_clear_write_values = (cl_ulong)c_clear_color | ((cl_ulong)c_clear_depth << 32) | ((cl_ulong)c_clear_stencil << 48);
    c_clear_enabled_data = c_deferred_clear ? CLEAR_ENABLED_MASK : 0;
    c_enabled_data = 0xFFFF;
}

// Test CL Objects
cl_platform_id platform_id;
cl_device_id device_id;
cl_context context;

cl_command_queue command_queue;
static const cl_uint vertex_command_queue_size = 2; // TODO rm
cl_command_queue vertex_command_queues[vertex_command_queue_size];

cl_program triangle_setup_program, bin_raster_program, coarse_raster_program, fine_raster_program;
cl_kernel vertex_shader_kernel, triangle_setup_kernel, bin_raster_kernel, coarse_raster_kernel, fine_raster_kernel;

cl_mem g_tri_header, g_tri_data, g_tri_subtris, g_vertex_buffer;
cl_mem g_bin_first_seg, g_bin_seg_data, g_bin_seg_next, g_bin_seg_count, g_bin_total;
cl_mem g_active_tiles, g_tile_first_seg, g_tile_seg_data, g_tile_seg_next, g_tile_seg_count;

cl_mem a_bin_counter, a_coarse_counter, a_fine_counter, a_num_active_tiles, a_num_bin_segs, a_num_subtris, a_num_tile_segs;

cl_mem t_tri_data, t_tri_header;
cl_mem t_vertex_buffer;

cl_mem t_color_buffer, t_depth_buffer, t_stencil_buffer;

// Shader objects
cl_mem gl_vertex_attributes;
cl_mem gl_vertex_attribute_datas;
cl_mem gl_vertex_attribute_pointer_0;
cl_mem gl_vertex_attribute_pointer_1;
cl_mem gl_vertex_attribute_pointer_2;

cl_mem g_index_buffer;

cl_mem gl_uniform, gl_uniforms_0, gl_uniforms_1;

cl_mem gl_texture_units[DEVICE_TEXTURE_UNITS];
cl_mem gl_texture_data;
cl_mem g_rop_config;

device_shared_objects_t shared_objects;
device_context_t device_context;

void create_cl_objects() {

    device_create_shared_objects(&shared_objects);
    device_create_context(&device_context, &shared_objects);
    int program_id = device_create_program_from_source(&device_context, shader_o_len, shader_o);
    device_load_program(&device_context, program_id);

    /*
    CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
    context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

    #define CREATE_PROGRAM(PROGRAM, BINARY) \
        { \
            size_t binary_size = BINARY##_len; \
            unsigned char* binary = BINARY; \
            PROGRAM = CL_CHECK2(clCreateProgramWithBinary(context, 1, &device_id, &binary_size, (const unsigned char**) &binary, NULL, &_err)); \
            CL_CHECK(clBuildProgram(PROGRAM, 1, &device_id, NULL, NULL, NULL)); \
        }

    CREATE_PROGRAM(triangle_setup_program, triangle_setup_o);
    CREATE_PROGRAM(bin_raster_program, bin_raster_o);
    CREATE_PROGRAM(coarse_raster_program, coarse_raster_o);
    CREATE_PROGRAM(fine_raster_program, shader_o);

    #undef CREATE_PROGRAM

    triangle_setup_kernel   = CL_CHECK2(clCreateKernel(triangle_setup_program,  TRIANGLE_SETUP_KERNEL_NAME, &_err));
    bin_raster_kernel       = CL_CHECK2(clCreateKernel(bin_raster_program,      BIN_RASTER_KERNEL_NAME,     &_err));
    coarse_raster_kernel    = CL_CHECK2(clCreateKernel(coarse_raster_program,   COARSE_RASTER_KERNEL_NAME,  &_err));
    vertex_shader_kernel    = CL_CHECK2(clCreateKernel(fine_raster_program,     VERTEX_SHADER_KERNEL_NAME,  &_err)); 
    fine_raster_kernel      = CL_CHECK2(clCreateKernel(fine_raster_program,     FINE_RASTER_KERNEL_NAME,    &_err));
    
    command_queue = CL_CHECK2(clCreateCommandQueue(context, device_id, NULL, &_err));
    for (int queue = 0; queue < vertex_command_queue_size; ++queue) // TODO: malloc
        vertex_command_queues[queue] = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));


    g_index_buffer      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(INDEX_BUFFER), &INDEX_BUFFER, &_err));
    
    // vertex buffer
    {
        size_t vertex_buffer_size = sizeof(cl_float4[max_number_vertices][max_number_varying+1]);
        g_vertex_buffer     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, vertex_buffer_size, NULL, &_err));
        
        #ifdef DEVICE_IMAGE_ENABLED
        {
            cl_image_format image_format = {
                .image_channel_order = CL_RGBA,
                .image_channel_data_type = CL_FLOAT,
            };
            cl_image_desc image_desc = {
                .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
                .image_width = vertex_buffer_size / sizeof(cl_float4),
                .image_row_pitch = 0,
                .mem_object = g_vertex_buffer,
            };

            t_vertex_buffer = CL_CHECK2(clCreateImage(context, CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &_err));
        }
        #endif
    }
    
    // triangle buffers
    {
        size_t max_number_subtriangles = max_number_triangles + max_number_extra_subtriangles;
        
        g_tri_header        = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(triangle_header_t[max_number_subtriangles]), NULL, &_err));
        g_tri_data          = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(triangle_data_t[max_number_subtriangles]), NULL, &_err));
        g_tri_subtris       = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uchar[max_number_subtriangles]), NULL, &_err));
        
        #ifdef DEVICE_IMAGE_ENABLED
        {
            cl_image_desc image_desc;
            cl_image_format image_format = {
                .image_channel_order = CL_RGBA,
                .image_channel_data_type = CL_UNSIGNED_INT32,
            };

            image_desc = {
                .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
                .image_row_pitch = 0,
                .image_width = sizeof(triangle_header_t[max_number_subtriangles]) / sizeof(cl_uint4),
                .mem_object = g_tri_header, 
            };
            
            t_tri_header = CL_CHECK2(clCreateImage(context, CL_MEM_READ_ONLY, &image_format, &image_desc, NULL, &_err));

            image_desc = {
                .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
                .image_width = sizeof(triangle_data_t[max_number_subtriangles]) / sizeof(cl_uint4),
                .image_row_pitch = 0,
                .mem_object = g_tri_data, 
            };

            t_tri_data = CL_CHECK2(clCreateImage(context, CL_MEM_READ_ONLY, &image_format, &image_desc, NULL, &_err));
        }
        #endif
    }

    // workload buffers, cannot be shared between kernels
    {
        g_bin_first_seg     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]), NULL, &_err));
        g_bin_seg_data      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_bin_segments * CR_BIN_SEG_SIZE]), NULL, &_err));
        g_bin_seg_next      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_bin_segments]), NULL, &_err));
        g_bin_seg_count     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_bin_segments]), NULL, &_err));
        g_bin_total         = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]), NULL, &_err));
        g_active_tiles      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXTILES_SQR]), NULL, &_err));
        g_tile_first_seg    = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXTILES_SQR]), NULL, &_err));
        g_tile_seg_data     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_tile_segments * CR_TILE_SEG_SIZE]), NULL, &_err));
        g_tile_seg_count    = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_tile_segments]), NULL, &_err));
        g_tile_seg_next     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_tile_segments]), NULL, &_err));
    }

    // atomics and counters, require to reset before each use
    {
        cl_int ZERO = 0;
        a_bin_counter       = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err));
        a_coarse_counter    = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err));
        a_fine_counter      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err));
        a_num_active_tiles  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err));
        a_num_subtris       = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &max_number_triangles, &_err));
        a_num_bin_segs      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err));
        a_num_tile_segs     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err));
    }
    */

    // framebuffer objects
    /*
    {
        #ifdef DEVICE_IMAGE_ENABLED
        {
            cl_image_format image_format;
            
            image_format = {
                .image_channel_order = CL_RGBA,
                .image_channel_data_type = CL_UNSIGNED_INT8,
            };
            t_color_buffer = CL_CHECK2(clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, framebuffer_width, framebuffer_height, 0, NULL, &_err));
            
            image_format = {
                .image_channel_order = CL_DEPTH, // this may be not supported for OpenCL 1.2 check cl_khr_depth_images extension
                .image_channel_data_type = CL_UNSIGNED_INT16,
            };
            t_depth_buffer = CL_CHECK2(clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, framebuffer_width, framebuffer_height, 0, NULL, &_err));

            image_format = {
                .image_channel_order = CL_A,
                .image_channel_data_type = CL_UNSIGNED_INT8,
            };
            t_stencil_buffer = CL_CHECK2(clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, framebuffer_width, framebuffer_height, 0, NULL, &_err));
        }
        #else
        {
            t_color_buffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint[framebuffer_width][framebuffer_height]), NULL, &_err));
            t_depth_buffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_ushort[framebuffer_width][framebuffer_height]), NULL, &_err));
            t_stencil_buffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uchar[framebuffer_width][framebuffer_height]), NULL, &_err));
        }
        #endif
    }
    */
    cl_uint colorbuffer_mode = TEX_RGBA8;
    int colorbuffer_id = device_create_colorbuffer(&device_context, framebuffer_width, framebuffer_height, colorbuffer_mode);
    int depthbuffer_id = device_create_depthbuffer(&device_context, framebuffer_width, framebuffer_height);
    int stencilbuffer_id = device_create_stencilbuffer(&device_context, framebuffer_width, framebuffer_height);
    device_load_framebuffer(&device_context, colorbuffer_id, depthbuffer_id, stencilbuffer_id, framebuffer_width, framebuffer_height, colorbuffer_mode);
    device_clear_framebuffer(&device_context, c_clear_write_values, c_clear_enabled_data, c_deferred_clear);

    const int CONFIGS = 2;
    rop_config_t rop_config[CONFIGS];
    // 
    {

        render_mode_t render_mode[CONFIGS] = {
            {
                .flags = RENDER_MODE_FLAG_ENABLE_LERP | RENDER_MODE_FLAG_ENABLE_DEPTH
            },
            {
                .flags = RENDER_MODE_FLAG_ENABLE_LERP | RENDER_MODE_FLAG_ENABLE_DEPTH | RENDER_MODE_FLAG_ENABLE_BLENDER
            }
        };

        blending_data_t blending_data[CONFIGS];
        set_blending_data_equation(&blending_data[1], BLEND_FUNC_ADD, BLEND_FUNC_ADD);
        set_blending_data_function_src(&blending_data[1], BLEND_SRC_ALPHA, BLEND_SRC_ALPHA);
        set_blending_data_function_dst(&blending_data[1], BLEND_SRC_ALPHA, BLEND_SRC_ALPHA);

        cl_uint blending_color[CONFIGS] = {0, 0x0f000000};
        stencil_data_t stencil_data[CONFIGS];
        set_stencil_data_func(&stencil_data[0], FRONT, STENCIL_FUNC_EQUAL, 0xFF, 0x1);
        set_stencil_data_op(&stencil_data[0], FRONT, STENCIL_OP_KEEP, STENCIL_OP_KEEP, STENCIL_OP_KEEP);

        depth_data_t depth_data[CONFIGS];
        set_depth_data_func(&depth_data[0], DEPTH_FUNC_LESS);
        set_depth_data_func(&depth_data[1], DEPTH_FUNC_LESS);

        enabled_data_t enabled_data[CONFIGS];
        set_enabled_color_data(&enabled_data[0], 1,1,1,1);
        set_enabled_stencil_data(&enabled_data[0], 0xFF);
        set_enabled_depth_data(&enabled_data[0], 1);
        set_enabled_color_data(&enabled_data[1], 1,1,1,1);
        set_enabled_stencil_data(&enabled_data[1], 0xFF);
        set_enabled_depth_data(&enabled_data[1], 1);

        printf("INFO: enabled data[0]=%x, enabled data[1]=%x\n", enabled_data[0].misc, enabled_data[1].misc);
        
        for(int i=0; i<CONFIGS; ++i) {
            rop_config[i] = (rop_config_t) {
                .render_mode = render_mode[i],
                .blending_data  = blending_data[i],
                .blending_color = blending_color[i],
                .stencil_data   = stencil_data[i].front_misc,
                .depth_data     = depth_data[i].misc,
                .enabled_data   = enabled_data[i],
            };
        }
            
        // g_rop_config        = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(rop_config), rop_config, &_err));
    }
    
    cl_uchar uniform[CONFIGS][sizeof(cl_float16[3]) + sizeof(cl_uint)];

    // gl_uniform = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, DEVICE_UNIFORM_CAPACITY * TRIANGLE_PRIMITIVE_CONFIGS, NULL, &_err));
    {
        glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(-1,0,0)); 
        glm::mat4 perspective = glm::perspective(45.0f, (float)framebuffer_width/(float)framebuffer_height, 0.01f, 1000.0f);
        glm::mat4 view = glm::lookAt(glm::vec3{2,2,-5}, glm::vec3{0,0,0}, glm::vec3{0,1,0});
        cl_uint sampler = 0;
        memcpy(uniform[0],&model[0][0], sizeof(cl_float16));
        memcpy(&uniform[0][sizeof(cl_float16[1])],&perspective[0][0], sizeof(cl_float16));
        memcpy(&uniform[0][sizeof(cl_float16[2])],&view[0][0], sizeof(cl_float16));
        memcpy(&uniform[0][sizeof(cl_float16[3])],&sampler, sizeof(sampler));

        // cl_buffer_region region = {DEVICE_UNIFORM_CAPACITY*0, DEVICE_UNIFORM_CAPACITY*1};
        // gl_uniforms_0 = CL_CHECK2(clCreateSubBuffer(gl_uniform, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &_err));
        // CL_CHECK(clEnqueueWriteBuffer(command_queue, gl_uniforms_0, CL_TRUE, 0, sizeof(uniform[0]), uniform[0], 0, NULL, NULL));
        //gl_uniforms_0             = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uniform), uniform, &_err));
    }
    {
        glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(1,0,0)); 
        glm::mat4 perspective = glm::perspective(45.0f, (float)framebuffer_width/(float)framebuffer_height, 0.01f, 1000.0f);
        glm::mat4 view = glm::lookAt(glm::vec3{2,2,-5}, glm::vec3{0,0,0}, glm::vec3{0,1,0});
        cl_uint sampler = 1;
        memcpy(uniform[1],&model[0][0], sizeof(cl_float16));
        memcpy(&uniform[1][sizeof(cl_float16[1])],&perspective[0][0], sizeof(cl_float16));
        memcpy(&uniform[1][sizeof(cl_float16[2])],&view[0][0], sizeof(cl_float16));
        memcpy(&uniform[1][sizeof(cl_float16[3])],&sampler, sizeof(sampler));
        // gl_uniforms_1             = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uniform), uniform, &_err));
        // cl_buffer_region region = {DEVICE_UNIFORM_CAPACITY*1, DEVICE_UNIFORM_CAPACITY*2};
        // gl_uniforms_1 = CL_CHECK2(clCreateSubBuffer(gl_uniform, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &_err));
        // CL_CHECK(clEnqueueWriteBuffer(command_queue, gl_uniforms_1, CL_TRUE, 0, sizeof(uniform[1]), uniform[1], 0, NULL, NULL));
    }

    device_load_config(&device_context, 0, &rop_config[0], uniform[0]);
    device_load_config(&device_context, 1, &rop_config[1], uniform[1]);

    // vertex shader objects

    float vertex_attributes[DEVICE_VERTEX_ATTRIBUTE_SIZE][4];
    vertex_attribute_data_t vertex_attribute_data[DEVICE_VERTEX_ATTRIBUTE_SIZE] = {
        {
            .offset = 0,
            .stride = sizeof(float[2]),
            .misc = VERTEX_ATTRIBUTE_ACTIVE_POINTER | (VERTEX_ATTRIBUTE_SIZE_2 << 3) | VERTEX_ATTRIBUTE_TYPE_FLOAT
        },
        {
            .offset = 0,
            .stride = sizeof(float[3]),
            .misc = VERTEX_ATTRIBUTE_ACTIVE_POINTER | (VERTEX_ATTRIBUTE_SIZE_3 << 3) | VERTEX_ATTRIBUTE_TYPE_FLOAT
        },
        {
            .offset = 0,
            .stride = sizeof(float[3]),
            .misc = VERTEX_ATTRIBUTE_ACTIVE_POINTER | (VERTEX_ATTRIBUTE_SIZE_3 << 3) | VERTEX_ATTRIBUTE_TYPE_FLOAT
        },
    };

    // gl_vertex_attributes    = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float4[DEVICE_VERTEX_ATTRIBUTE_SIZE]), NULL, &_err));
    // gl_vertex_attribute_datas = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(vertex_attribute_data), vertex_attribute_data, &_err));

    device_load_vertex_attributes(&device_context, vertex_attributes, vertex_attribute_data);

    // gl_vertex_attribute_pointer_0 = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(tex), tex, &_err));
    // gl_vertex_attribute_pointer_1 = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(position), position, &_err));
    // gl_vertex_attribute_pointer_2 = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(color), color, &_err));

    int tex_buffer = device_create_ro_buffer(&device_context, sizeof(tex));
    int position_buffer = device_create_ro_buffer(&device_context, sizeof(position));
    int color_buffer = device_create_ro_buffer(&device_context, sizeof(color));

    device_write_on_buffer(&device_context, tex_buffer, 0, sizeof(tex), tex);
    device_write_on_buffer(&device_context, position_buffer, 0, sizeof(position), position);
    device_write_on_buffer(&device_context, color_buffer, 0, sizeof(color), color);

    device_bind_buffer_to_vertex_attribute_pointer(&device_context, 0, tex_buffer);
    device_bind_buffer_to_vertex_attribute_pointer(&device_context, 1, position_buffer);
    device_bind_buffer_to_vertex_attribute_pointer(&device_context, 2, color_buffer);

    const int LOADED_TEXTURES = 2;
    ppm_image_t *textures[LOADED_TEXTURES];
    gl_texture_data_t texture_datas[DEVICE_TEXTURE_UNITS];

    {
        textures[0] = read_ppm(TEST_PATH "/glsc2/assets/dog.ppm");
        textures[1] = read_ppm(TEST_PATH "/glsc2/assets/cat.ppm");

        for (int texture=0; texture < LOADED_TEXTURES; ++texture) {
            // create sampler object
            gl_texture_data_t *texture_data = &texture_datas[texture];
            #ifndef DEVICE_IMAGE_ENABLED
                texture_data->width = textures[texture]->x,
                texture_data->height = textures[texture]->y,
            #endif
            set_sampler2D_internalformat(&texture_data->sampler2D, TEX_RGBA8);
            set_sampler2D_wrap_s(&texture_data->sampler2D, TEXTURE_WRAP_REPEAT);
            set_sampler2D_wrap_t(&texture_data->sampler2D, TEXTURE_WRAP_REPEAT);
        }
        
        // create texture units
        /*
        for(int tex_unit=0; tex_unit<DEVICE_TEXTURE_UNITS; ++tex_unit) {
            cl_mem texture_unit;
            int tex_height, tex_width;
            if (tex_unit < LOADED_TEXTURES) {
                ppm_image_t *image = textures[tex_unit];
                tex_height = image->y;
                tex_width = image->x;
                printf("unit[%d] width=%d, height=%d.\n", tex_unit, tex_width, tex_height);
                
                uint8_t* data = (uint8_t*) malloc(image->x*image->y*sizeof(uint8_t[4]));
                for(uint32_t i=0; i<image->x*image->y; ++i) {
                    data[i*4+0] = image->data[i].red;
                    data[i*4+1] = image->data[i].green;
                    data[i*4+2] = image->data[i].blue;
                    data[i*4+3] = 0xFFu;
                }
                #ifdef DEVICE_IMAGE_ENABLED
                    cl_image_format image_format = {
                        .image_channel_order = CL_RGBA,
                        .image_channel_data_type = CL_UNSIGNED_INT8,
                    };

                    texture_unit = CL_CHECK2(clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &image_format, tex_width, tex_height, 0, data, &_err));
                #else
                    texture_unit = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int[tex_width][tex_height]), data, &_err));
                
                #endif
                free(data);
            } else {
                // create dummy texture
                int tex_height = 100, tex_width = 100;
                #ifdef DEVICE_IMAGE_ENABLED
                    cl_image_format image_format = {
                        .image_channel_order = CL_RGBA,
                        .image_channel_data_type = CL_UNSIGNED_INT8,
                    };

                    texture_unit = CL_CHECK2(clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, tex_width, tex_height, 0, NULL, &_err));
                #else
                    texture_unit = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int[tex_width][tex_height]), NULL, &_err));
                
                #endif
            }
            gl_texture_units[tex_unit] = texture_unit; 
        }

        gl_texture_data = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(texture_datas), texture_datas, &_err));
        for (int texture=0; texture < LOADED_TEXTURES; ++texture) {
            printf("INFO: loaded texture[%d]: width=%d, height=%d\n", texture, texture_datas[texture].width, texture_datas[texture].height);
        }
        */
    }

    int texture_ids[LOADED_TEXTURES];
    for(int id = 0; id < LOADED_TEXTURES; ++id) {
        ppm_image_t *image = textures[id];
        texture_ids[id] = device_create_2d_texture(&device_context, image->x, image->y, TEX_RGBA8);
        uint8_t* data = (uint8_t*) malloc(image->x*image->y*sizeof(uint8_t[4]));
        for(uint32_t i=0; i<image->x*image->y; ++i) {
            data[i*4+0] = image->data[i].red;
            data[i*4+1] = image->data[i].green;
            data[i*4+2] = image->data[i].blue;
            data[i*4+3] = 0xFFu;
        }
        device_write_2d_texture(&device_context, texture_ids[id], image->x, image->y, TEX_RGBA8, data);
        free(data);
    }

    device_bind_texture_unit(&device_context, 0, texture_ids[0]);
    device_bind_texture_unit(&device_context, 1, texture_ids[1]);

    device_load_texture_datas(&device_context, texture_datas);

}


void setWorkSizeFromSize(glm::ivec2& block_size, glm::ivec2& size_threads, size_t* global_work_size, size_t* local_work_size);
void setWorkSizeFromNum(glm::ivec2& block_size, uint32_t num_threads, size_t* global_work_size, size_t* local_work_size);


// mutable state
static cl_uint cummulative_vertices = 0;
static cl_uint num_triangles = 0;
static cl_uint vertex_command_queue_index = 0;
// unmutable state

cl_event launch_parallel_vertex_range_assembly(
    cl_kernel   vertex_kernel,
    cl_mem      uniform,
    cl_uint     num_vertices,
    cl_uint     num_attributes,
    cl_mem*     attribute_pointers,
    cl_uint     num_indices,
    cl_mem      indexes,
    cl_uint     primitive_config
) {

    cl_event wait_event;
    cl_uint queue = vertex_command_queue_index;

    // vertex shader
    static cl_uint vertex_uniform_arg = 2;
    static cl_uint attribute_pointers_arg = 3;

    {
        size_t gwo = cummulative_vertices;
        size_t gws = num_vertices;
        cl_uint ap_arg = attribute_pointers_arg;

        CL_CHECK(clSetKernelArg(vertex_kernel, vertex_uniform_arg, sizeof(uniform), &uniform));
        for (cl_uint ap = 0; ap < num_attributes; ++ap)
            CL_CHECK(clSetKernelArg(vertex_kernel, ap_arg++, sizeof(attribute_pointers[ap]), &attribute_pointers[ap]));

        CL_CHECK(clEnqueueNDRangeKernel(vertex_command_queues[queue], vertex_kernel, 1, &gwo, &gws, NULL, 0, NULL, NULL));
    }

    // triangle setup
    static cl_uint index_buffer_arg = 1;
    static cl_uint vertex_offset_arg = 6;
    static cl_uint primitive_config_arg = vertex_offset_arg+7;

    {
        size_t gwo = num_triangles;
        size_t gws = num_indices/3; // for GL_TRIANGLES mode
        cl_uint vertex_offset = cummulative_vertices;
        
        CL_CHECK(clSetKernelArg(triangle_setup_kernel, index_buffer_arg, sizeof(indexes), &indexes));
        CL_CHECK(clSetKernelArg(triangle_setup_kernel, vertex_offset_arg, sizeof(vertex_offset), &vertex_offset));
        CL_CHECK(clSetKernelArg(triangle_setup_kernel, primitive_config_arg, sizeof(primitive_config), &primitive_config));
        
        CL_CHECK(clEnqueueNDRangeKernel(vertex_command_queues[queue], triangle_setup_kernel, 1, &gwo, &gws, NULL, 0, NULL, &wait_event));
    }

    // update data

    cummulative_vertices += num_vertices;
    num_triangles += num_indices/3;
    vertex_command_queue_index += 1;

    return wait_event;
}

cl_event launch_raster(
    cl_command_queue command_queue,
    cl_uint num_triangles,
    cl_uint vertex_event_wait_size,
    cl_event* vertex_event_wait_list,
    cl_uint fragment_event_wait_size,
    cl_event* fragment_event_wait_list
)
{
    size_t global_work_size[2], local_work_size[2];
    glm::ivec2 block_size, threads_size;
    cl_event fragment_event;

    {
        cl_uint c_num_tris_arg = 15;
        #ifdef CONF_BIN_IMAGE_ENABLED
        ++c_num_tris_arg;
        #endif
        c_num_tris = num_triangles;
        CL_CHECK(clSetKernelArg(bin_raster_kernel, c_num_tris_arg, sizeof(c_num_tris), &c_num_tris));
        block_size = glm::ivec2(DEVICE_SUB_GROUP_THREADS, DEVICE_BIN_SUB_GROUPS);
        threads_size = glm::ivec2(CR_BIN_STREAMS_SIZE, 1) * block_size;
        setWorkSizeFromSize(block_size, threads_size, global_work_size, local_work_size);
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, bin_raster_kernel, 2, NULL, global_work_size, local_work_size, vertex_event_wait_size, vertex_event_wait_list, NULL));
    }

    {
        block_size = glm::ivec2(DEVICE_SUB_GROUP_THREADS, DEVICE_COARSE_SUB_GROUPS);
        threads_size = glm::ivec2(DEVICE_NUM_CORES, 1) * block_size;
        setWorkSizeFromSize(block_size, threads_size, global_work_size, local_work_size);
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, coarse_raster_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    }

    {
        block_size = glm::ivec2(DEVICE_SUB_GROUP_THREADS, DEVICE_FINE_SUB_GROUPS);
        threads_size = glm::ivec2(DEVICE_NUM_CORES, 1) * block_size;
        setWorkSizeFromSize(block_size, threads_size, global_work_size, local_work_size);
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, fine_raster_kernel, 2, NULL, global_work_size, local_work_size, fragment_event_wait_size, fragment_event_wait_list, &fragment_event));
    }

    return fragment_event;
}

int main(int argc, char** argv) {

    parse_args(argc, argv);
    set_cl_parameters();
    create_cl_objects();
    
    auto begin = std::chrono::high_resolution_clock::now();

    device_launch_range_triangle_assembly(
        &device_context,
        sizeof(position)/sizeof(float[3]),
        sizeof(INDEX_BUFFER)/sizeof(*INDEX_BUFFER),
        INDEX_BUFFER,
        0
    );

    device_launch_range_triangle_assembly(
        &device_context,
        sizeof(position)/sizeof(float[3]),
        sizeof(INDEX_BUFFER)/sizeof(*INDEX_BUFFER),
        INDEX_BUFFER,
        1
    );

    device_launch_triangle_rasterization(&device_context);

    device_finish(&device_context);

    auto end = std::chrono::high_resolution_clock::now();

    printf("PERF: exec_time=%d ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());

    
    uint32_t* result = (uint32_t*) malloc(sizeof(int[framebuffer_height][framebuffer_width]));

    device_get_pixels(&device_context, 0, 0, framebuffer_width, framebuffer_height, result);

    print_ppm("output.ppm", framebuffer_width, framebuffer_height, (uint8_t*)result);

    free(result);
}

void setWorkSizeFromSize(glm::ivec2& block_size, glm::ivec2& size_threads, size_t* global_work_size, size_t* local_work_size) {
    glm::ivec2 grid_size = (size_threads + block_size - 1) / block_size;
    local_work_size[0] = block_size.x;
    local_work_size[1] = block_size.y;
    global_work_size[0] = grid_size.x * local_work_size[0];
    global_work_size[1] = grid_size.y * local_work_size[1];
}

void setWorkSizeFromNum(glm::ivec2& block_size, uint32_t num_threads, size_t* global_work_size, size_t* local_work_size) {
    local_work_size[0] = block_size.x;
    local_work_size[1] = block_size.y;

    int threadsPerBlock = local_work_size[0] * local_work_size[1];
    int maxGridWidth = 65536;
    global_work_size[0] = (c_num_tris + threadsPerBlock - 1) / threadsPerBlock;
    global_work_size[1] = 1;

    while (global_work_size[0] > maxGridWidth)
    {
        global_work_size[0] = (global_work_size[0] + 1) >> 1;
        global_work_size[1] <<= 1;
    }
    global_work_size[0] *= local_work_size[0];
    global_work_size[1] *= local_work_size[1];
}
