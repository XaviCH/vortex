#include <algorithm>
#include <cmath>
#include <chrono>
#include "glm/gtc/matrix_transform.hpp"

#include <backend/pipeline/triangle_setup.o.c>
#include <backend/pipeline/bin_raster.o.c>
#include <backend/pipeline/coarse_raster.o.c>
//#include <backend/pipeline/fine_raster.o.c>
#include "shader.o.c"
#include <types.device.h>
//#include <kernels/shadered_fine_raster.ocl.c>
#include <constants.device.h>
#include <tests/glsc2/common.h>
#include <tests/glsc2/debug.hpp>
#include <tests/glsc2/kernels/common.hpp>

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

// TEST PARAMETERS
// Test parameters

glm::ivec2 size;
glm::ivec2 size_pixels;
glm::ivec2 size_tiles;
int32_t num_tiles;
glm::ivec2 size_bins;
int32_t num_bins;
glm::ivec2 viewport_size;
int32_t round_size;
int32_t min_batches;
int32_t max_rounds;
int32_t max_subtris;
int32_t max_tris;
int32_t num_SMs;
int32_t num_fine_warps;
int32_t num_tris;

void set_parameters() {
    size = glm::ivec2(WIDTH, HEIGHT);
    size_pixels = (size + CR_TILE_SIZE - 1) & -CR_TILE_SIZE;
    size_tiles = size_pixels >> CR_TILE_LOG2;
    num_tiles = size_tiles.x * size_tiles.y;
    size_bins = (size_tiles + CR_BIN_SIZE -1) >> CR_BIN_LOG2;
    num_bins = size_bins.x * size_bins.y;
    viewport_size = glm::ivec2(WIDTH, HEIGHT);
    round_size  = DEVICE_BIN_SUB_GROUPS * 32;
    min_batches = CR_BIN_STREAMS_SIZE * 2;
    max_rounds  = 32;
    max_tris = maxTris;
    max_subtris = std::max(MAX_SUBTRIS, maxTris + maxSubtrisSlack);
    num_SMs = ATTRIBUTE_MULTIPROCESSOR_COUNT;
    num_fine_warps = std::min(ATTRIBUTE_MAX_THREADS_PER_BLOCK / 32, DEVICE_FINE_SUB_GROUPS);
    num_tris = NUM_TRIS;
}

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


void create_cl_objects() {
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

    // framebuffer objects
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

    // 
    {
        const int CONFIGS = 2;

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
        rop_config_t rop_config[CONFIGS];
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
            
        g_rop_config        = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(rop_config), rop_config, &_err));
    }

    // vertex shader objects
    vertex_attribute_data_t vertex_attribute_data[] = {
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

    gl_vertex_attributes    = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float4[DEVICE_VERTEX_ATTRIBUTE_SIZE]), NULL, &_err));
    gl_vertex_attribute_datas = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(vertex_attribute_data), vertex_attribute_data, &_err));

    gl_uniform = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, DEVICE_UNIFORM_CAPACITY * TRIANGLE_PRIMITIVE_CONFIGS, NULL, &_err));
    {
        glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(-1,0,0)); 
        glm::mat4 perspective = glm::perspective(45.0f, (float)framebuffer_width/(float)framebuffer_height, 0.01f, 1000.0f);
        glm::mat4 view = glm::lookAt(glm::vec3{2,2,-5}, glm::vec3{0,0,0}, glm::vec3{0,1,0});
        cl_uint sampler = 0;
        cl_uchar uniform[sizeof(cl_float16[3]) + sizeof(cl_uint)];
        memcpy(uniform,&model[0][0], sizeof(cl_float16));
        memcpy(&uniform[sizeof(cl_float16[1])],&perspective[0][0], sizeof(cl_float16));
        memcpy(&uniform[sizeof(cl_float16[2])],&view[0][0], sizeof(cl_float16));
        memcpy(&uniform[sizeof(cl_float16[3])],&sampler, sizeof(sampler));

        cl_buffer_region region = {DEVICE_UNIFORM_CAPACITY*0, DEVICE_UNIFORM_CAPACITY*1};
        gl_uniforms_0 = CL_CHECK2(clCreateSubBuffer(gl_uniform, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &_err));
        CL_CHECK(clEnqueueWriteBuffer(command_queue, gl_uniforms_0, CL_TRUE, 0, sizeof(uniform), uniform, 0, NULL, NULL));
        //gl_uniforms_0             = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uniform), uniform, &_err));
    }
    {
        glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(1,0,0)); 
        glm::mat4 perspective = glm::perspective(45.0f, (float)framebuffer_width/(float)framebuffer_height, 0.01f, 1000.0f);
        glm::mat4 view = glm::lookAt(glm::vec3{2,2,-5}, glm::vec3{0,0,0}, glm::vec3{0,1,0});
        cl_uint sampler = 1;
        cl_uchar uniform[sizeof(cl_float16[3]) + sizeof(cl_uint)];
        memcpy(uniform,&model[0][0], sizeof(cl_float16));
        memcpy(&uniform[sizeof(cl_float16[1])],&perspective[0][0], sizeof(cl_float16));
        memcpy(&uniform[sizeof(cl_float16[2])],&view[0][0], sizeof(cl_float16));
        memcpy(&uniform[sizeof(cl_float16[3])],&sampler, sizeof(sampler));
        // gl_uniforms_1             = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uniform), uniform, &_err));
        cl_buffer_region region = {DEVICE_UNIFORM_CAPACITY*1, DEVICE_UNIFORM_CAPACITY*2};
        gl_uniforms_1 = CL_CHECK2(clCreateSubBuffer(gl_uniform, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &_err));
        CL_CHECK(clEnqueueWriteBuffer(command_queue, gl_uniforms_1, CL_TRUE, 0, sizeof(uniform), uniform, 0, NULL, NULL));
    }

    gl_vertex_attribute_pointer_0 = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(tex), tex, &_err));
    gl_vertex_attribute_pointer_1 = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(position), position, &_err));
    gl_vertex_attribute_pointer_2 = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(color), color, &_err));

    /*
    for(int i=0; i<DEVICE_TEXTURE_UNITS; ++i) {
        cl_mem dummy_tex;
        int tex_height = 100, tex_width = 100;
        #ifdef DEVICE_IMAGE_ENABLED
            cl_image_format image_format = {
                .image_channel_order = CL_A,
                .image_channel_data_type = CL_UNSIGNED_INT8,
            };

            dummy_tex = CL_CHECK2(clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, tex_width, tex_height, 0, NULL, &_err));
        #else
            dummy_tex = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int[tex_width][tex_height]), NULL, &_err));
        
        #endif
        gl_texture_units[i] = dummy_tex; 
    }
    */

    {
        const int LOADED_TEXTURES = 2;
        ppm_image_t *textures[LOADED_TEXTURES];
        textures[0] = read_ppm(TEST_PATH "/glsc2/assets/dog.ppm");
        textures[1] = read_ppm(TEST_PATH "/glsc2/assets/cat.ppm");

        gl_texture_data_t texture_datas[LOADED_TEXTURES];
        

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
    }
}

void set_cl_kernel_parameters() {
    cl_kernel kernel;
    cl_int counter;

    kernel = vertex_shader_kernel;
    counter = 0;
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(gl_vertex_attributes),       &gl_vertex_attributes));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(gl_vertex_attribute_datas),       &gl_vertex_attribute_datas));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(gl_uniforms_0),       &gl_uniforms_0));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(gl_vertex_attribute_pointer_0),       &gl_vertex_attribute_pointer_0));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(gl_vertex_attribute_pointer_1),       &gl_vertex_attribute_pointer_1));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(gl_vertex_attribute_pointer_2),       &gl_vertex_attribute_pointer_2));
    #ifdef CONF_SETUP_IMAGE_ENABLED
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_vertex_buffer),     &t_vertex_buffer)); 
    #else
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_vertex_buffer),     &g_vertex_buffer));
    #endif


    kernel = triangle_setup_kernel; 
    counter = 0;
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_num_subtris),       &a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_index_buffer),      &g_index_buffer));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tri_header),        &g_tri_header));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tri_data),          &g_tri_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tri_subtris),       &g_tri_subtris));
    #ifdef CONF_SETUP_IMAGE_ENABLED
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_vertex_buffer),     &t_vertex_buffer)); 
    #else
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_vertex_buffer),     &g_vertex_buffer));
    #endif
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_num_tris),          &c_num_tris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_subtris),       &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_render_mode_flags), &c_render_mode_flags));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_samples_log2),      &c_samples_log2));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_vertex_size),       &c_vertex_size));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_viewport_width),    &c_viewport_width));

    kernel = bin_raster_kernel;
    counter = 0;
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_bin_counter),       &a_bin_counter));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_num_bin_segs),      &a_num_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_num_subtris),       &a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_first_seg),     &g_bin_first_seg));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_seg_count),     &g_bin_seg_count));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_seg_data),      &g_bin_seg_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_seg_next),      &g_bin_seg_next));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_total),         &g_bin_total));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tri_header),        &g_tri_header));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tri_subtris),       &g_tri_subtris));
    #ifdef CONF_BIN_IMAGE_ENABLED
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_tri_header),        &t_tri_header));
    #endif
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_bin_batch_sz),      &c_bin_batch_sz));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_height_bins),       &c_height_bins));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_bin_segs),      &c_max_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_subtris),       &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_num_bins),          &c_num_bins));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_num_tris),          &c_num_tris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_width_bins),        &c_width_bins));

    kernel = coarse_raster_kernel; 
    counter = 0;
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_coarse_counter),    &a_coarse_counter));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_num_active_tiles),  &a_num_active_tiles));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_num_bin_segs),      &a_num_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_num_tile_segs),     &a_num_tile_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_num_subtris),       &a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_active_tiles),      &g_active_tiles));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_first_seg),     &g_bin_first_seg));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_seg_count),     &g_bin_seg_count));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_seg_data),      &g_bin_seg_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_seg_next),      &g_bin_seg_next));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_total),         &g_bin_total));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tile_first_seg),    &g_tile_first_seg));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tile_seg_count),    &g_tile_seg_count));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tile_seg_data),     &g_tile_seg_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tile_seg_next),     &g_tile_seg_next));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tri_header),        &g_tri_header));
    #ifdef CONF_COARSE_IMAGE_ENABLED
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_tri_header),        &t_tri_header));
    #endif
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_deferred_clear),    &c_deferred_clear));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_height_tiles),      &c_height_tiles));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_bin_segs),      &c_max_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_subtris),       &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_tile_segs),     &c_max_tile_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_num_bins),          &c_num_bins));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_width_bins),        &c_width_bins));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_width_tiles),       &c_width_tiles));
    // CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_local_data),       &g_local_data));

    kernel = fine_raster_kernel;
    counter = 0;
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(gl_uniforms_0),      &gl_uniforms_0));
    for(int i=0; i<DEVICE_TEXTURE_UNITS; ++i)
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(gl_texture_units[i]), &gl_texture_units[i]));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(gl_texture_data),     &gl_texture_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_fine_counter),      &a_fine_counter));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_num_active_tiles),  &a_num_active_tiles));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_num_bin_segs),      &a_num_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_num_subtris),       &a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_num_tile_segs),     &a_num_tile_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_active_tiles),      &g_active_tiles));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tile_first_seg),    &g_tile_first_seg));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tile_seg_count),    &g_tile_seg_count));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tile_seg_data),     &g_tile_seg_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tile_seg_next),     &g_tile_seg_next));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tri_header),        &g_tri_header));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_color_buffer),      &t_color_buffer));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_depth_buffer),      &t_depth_buffer));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_stencil_buffer),    &t_stencil_buffer));
    #ifdef CONF_FINE_IMAGE_ENABLED
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_tri_data),          &t_tri_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_tri_header),        &t_tri_header));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_vertex_buffer),     &t_vertex_buffer));
    #else
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tri_data),          &g_tri_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_vertex_buffer),     &g_vertex_buffer));
    #endif
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_rop_config),       &g_rop_config));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_clear_write_values),       &c_clear_write_values));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_clear_enabled_data),       &c_clear_enabled_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_color_buffer_mode), &c_color_buffer_mode));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_bin_segs),      &c_max_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_subtris),       &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_tile_segs),     &c_max_tile_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_width_tiles),       &c_width_tiles));
}



// HOST CPY PARAMETRES

std::vector<cl_ushort> h_g_index_buffer;
std::vector<CRTriangleHeader> h_g_tri_header;
std::vector<CRTriangleData> h_g_tri_data;
std::vector<cl_uchar> h_g_tri_subtris;
std::vector<cl_uchar> h_g_vertex_buffer;
std::vector<cl_int> h_g_bin_first_seg;
std::vector<cl_int> h_g_bin_seg_data;
std::vector<cl_int> h_g_bin_seg_next;
std::vector<cl_int> h_g_bin_seg_count;
std::vector<cl_int> h_g_bin_total;
std::vector<cl_int> h_g_active_tiles;
std::vector<cl_int> h_g_tile_first_seg;
std::vector<cl_int> h_g_tile_seg_data;
std::vector<cl_int> h_g_tile_seg_next;
std::vector<cl_int> h_g_tile_seg_count;
std::vector<cl_int> h_a_bin_counter;
std::vector<cl_int> h_a_coarse_counter;
std::vector<cl_int> h_a_fine_counter;
std::vector<cl_int> h_a_num_active_tiles;
std::vector<cl_int> h_a_num_bin_segs;
std::vector<cl_int> h_a_num_subtris;
std::vector<cl_int> h_a_num_tile_segs;

std::vector<cl_uint> h_t_color_buffer;
std::vector<cl_uint> h_t_depth_buffer;

void downloadDeviceData() {
    #define READ_BUFFER(_D_BUFFER, _H_VECTOR) \
        CL_CHECK(clEnqueueReadBuffer(command_queue, _D_BUFFER, CL_TRUE, 0, sizeof(*(_H_VECTOR.data()))*(_H_VECTOR.capacity()), _H_VECTOR.data(), 0, NULL, NULL))

    READ_BUFFER(g_index_buffer, h_g_index_buffer);
    READ_BUFFER(g_tri_header, h_g_tri_header);
    READ_BUFFER(g_tri_data, h_g_tri_data);
    READ_BUFFER(g_tri_subtris, h_g_tri_subtris);
    READ_BUFFER(g_vertex_buffer, h_g_vertex_buffer);
    READ_BUFFER(g_bin_first_seg, h_g_bin_first_seg);
    READ_BUFFER(g_bin_seg_data, h_g_bin_seg_data);
    READ_BUFFER(g_bin_seg_next, h_g_bin_seg_next);
    READ_BUFFER(g_bin_seg_count, h_g_bin_seg_count);
    READ_BUFFER(g_bin_total, h_g_bin_total);
    READ_BUFFER(g_active_tiles, h_g_active_tiles);
    READ_BUFFER(g_tile_first_seg, h_g_tile_first_seg);
    READ_BUFFER(g_tile_seg_data, h_g_tile_seg_data);
    READ_BUFFER(g_tile_seg_next, h_g_tile_seg_next);
    READ_BUFFER(g_tile_seg_count, h_g_tile_seg_count);
    READ_BUFFER(a_bin_counter, h_a_bin_counter);
    READ_BUFFER(a_coarse_counter, h_a_coarse_counter);
    READ_BUFFER(a_fine_counter, h_a_fine_counter);
    READ_BUFFER(a_num_active_tiles, h_a_num_active_tiles);
    READ_BUFFER(a_num_bin_segs, h_a_num_bin_segs);
    READ_BUFFER(a_num_subtris, h_a_num_subtris);
    READ_BUFFER(a_num_tile_segs, h_a_num_tile_segs);

    #undef READ_BUFFER

    size_t origin[] = {0,0,0};
    size_t region[] = {size.x,size.y,1};

    CL_CHECK(clEnqueueReadImage(command_queue, t_color_buffer, CL_TRUE, origin, region, 0, 0, h_t_color_buffer.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueReadImage(command_queue, t_depth_buffer, CL_TRUE, origin, region, 0, 0, h_t_depth_buffer.data(), 0, NULL, NULL));
}

void uploadHostData() {
    #define WRITE_BUFFER(_D_BUFFER, _H_VECTOR) \
        CL_CHECK(clEnqueueWriteBuffer(command_queue, _D_BUFFER, CL_TRUE, 0, sizeof(*(_H_VECTOR.data()))*(_H_VECTOR.capacity()), _H_VECTOR.data(), 0, NULL, NULL))

    WRITE_BUFFER(g_index_buffer, h_g_index_buffer);
    WRITE_BUFFER(g_tri_header, h_g_tri_header);
    WRITE_BUFFER(g_tri_data, h_g_tri_data);
    WRITE_BUFFER(g_tri_subtris, h_g_tri_subtris);
    WRITE_BUFFER(g_vertex_buffer, h_g_vertex_buffer);
    WRITE_BUFFER(g_bin_first_seg, h_g_bin_first_seg);
    WRITE_BUFFER(g_bin_seg_data, h_g_bin_seg_data);
    WRITE_BUFFER(g_bin_seg_next, h_g_bin_seg_next);
    WRITE_BUFFER(g_bin_seg_count, h_g_bin_seg_count);
    WRITE_BUFFER(g_bin_total, h_g_bin_total);
    WRITE_BUFFER(g_active_tiles, h_g_active_tiles);
    WRITE_BUFFER(g_tile_first_seg, h_g_tile_first_seg);
    WRITE_BUFFER(g_tile_seg_data, h_g_tile_seg_data);
    WRITE_BUFFER(g_tile_seg_next, h_g_tile_seg_next);
    WRITE_BUFFER(g_tile_seg_count, h_g_tile_seg_count);
    WRITE_BUFFER(a_bin_counter, h_a_bin_counter);
    WRITE_BUFFER(a_coarse_counter, h_a_coarse_counter);
    WRITE_BUFFER(a_fine_counter, h_a_fine_counter);
    WRITE_BUFFER(a_num_active_tiles, h_a_num_active_tiles);
    WRITE_BUFFER(a_num_bin_segs, h_a_num_bin_segs);
    WRITE_BUFFER(a_num_subtris, h_a_num_subtris);
    WRITE_BUFFER(a_num_tile_segs, h_a_num_tile_segs);

    #undef WRITE_BUFFER

    size_t origin[] = {0,0,0};
    size_t region[] = {size.x,size.y,1};

    CL_CHECK(clEnqueueWriteImage(command_queue, t_color_buffer, CL_TRUE, origin, region, 0, 0, h_t_color_buffer.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteImage(command_queue, t_depth_buffer, CL_TRUE, origin, region, 0, 0, h_t_depth_buffer.data(), 0, NULL, NULL));
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
        threads_size = glm::ivec2(num_SMs, 1) * block_size;
        setWorkSizeFromSize(block_size, threads_size, global_work_size, local_work_size);
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, coarse_raster_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    }

    {
        block_size = glm::ivec2(DEVICE_SUB_GROUP_THREADS, DEVICE_FINE_SUB_GROUPS);
        threads_size = glm::ivec2(num_SMs, 1) * block_size;
        setWorkSizeFromSize(block_size, threads_size, global_work_size, local_work_size);
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, fine_raster_kernel, 2, NULL, global_work_size, local_work_size, fragment_event_wait_size, fragment_event_wait_list, &fragment_event));
    }

    return fragment_event;
}

int main(int argc, char** argv) {

    parse_args(argc, argv);
    set_parameters();
    set_cl_parameters();
    create_cl_objects();
    set_cl_kernel_parameters();

    glm::ivec2 block_size, threads_size;
    size_t global_work_size[2], local_work_size[2], global_work_offset[2];

    cl_event event_wait_list[2];

    for (int queue = 0; queue < vertex_command_queue_size; ++queue)
        vertex_command_queues[queue] = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));
    
        
    cl_mem attributes[2] = {gl_vertex_attribute_pointer_0, gl_vertex_attribute_pointer_1};
    
    auto begin = std::chrono::high_resolution_clock::now();
        
    event_wait_list[vertex_command_queue_index] = launch_parallel_vertex_range_assembly(
        vertex_shader_kernel,
        gl_uniforms_0,
        sizeof(position)/sizeof(float[3]),
        2,
        attributes,
        sizeof(INDEX_BUFFER)/sizeof(*INDEX_BUFFER),
        g_index_buffer,
        0
    );
    
    event_wait_list[vertex_command_queue_index] = launch_parallel_vertex_range_assembly(
        vertex_shader_kernel,
        gl_uniforms_1,
        sizeof(position)/sizeof(float[3]),
        2,
        attributes,
        sizeof(INDEX_BUFFER)/sizeof(*INDEX_BUFFER),
        g_index_buffer,
        1
    );

    launch_raster(
        command_queue,
        num_triangles,
        vertex_command_queue_index,
        event_wait_list,
        0,
        NULL
    );
    
    CL_CHECK(clFinish(command_queue));

    auto end = std::chrono::high_resolution_clock::now();
    printf("INFO: cumm_vertices=%d, num_triangles=%d\n", cummulative_vertices, num_triangles);
    printf("PERF: exec_time=%d ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());

    size_t origin[] = {0,0,0};
    size_t region[] = {framebuffer_width,framebuffer_height,1};

    uint32_t* result = (uint32_t*) malloc(sizeof(int[framebuffer_height][framebuffer_width]));

    #ifdef CONF_FINE_IMAGE_ENABLED
    CL_CHECK(clEnqueueReadImage(command_queue, t_color_buffer, CL_TRUE, origin, region, 0, 0, result, 0, NULL, NULL));
    #else
    // TODO test more image formats
    CL_CHECK(clEnqueueReadBuffer(command_queue, t_color_buffer, CL_TRUE, origin[0]*origin[1], region[0]*region[1]*sizeof(uint32_t), result, 0, NULL, NULL));
    #endif
    CL_CHECK(clFinish(command_queue));

    print_ppm("output.ppm", framebuffer_width, framebuffer_height, (uint8_t*)result);
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
