#include <algorithm>
#include <cmath>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>

#include <tests/glsc2/common.h>
#include <tests/glsc2/debug.hpp>
#include <tests/glsc2/kernels/common.hpp>

#include <frontend/device.h>
#include "shader.o.c"

#define TEST_PATH "../../../"

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

typedef struct {
    float position[4];
    float color[4];
    float texcoord[4];
} VertexData;

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

device_shared_objects_t shared_objects;
device_context_t device_context;

void create_cl_objects() {

    device_create_shared_objects(&shared_objects);
    device_create_context(&device_context, &shared_objects);
    int program_id = device_create_program_from_binary(&shared_objects, shader_o_len, shader_o);
    device_load_program(&device_context, program_id);

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
        gl_stencil_data_t stencil_data[CONFIGS];
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
    }
    
    cl_uchar uniform[CONFIGS][sizeof(cl_float16[3]) + sizeof(cl_uint)];

    {
        glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(-1,0,0)); 
        glm::mat4 perspective = glm::perspective(45.0f, (float)framebuffer_width/(float)framebuffer_height, 0.01f, 1000.0f);
        glm::mat4 view = glm::lookAt(glm::vec3{2,2,-5}, glm::vec3{0,0,0}, glm::vec3{0,1,0});
        cl_uint sampler = 0;
        memcpy(uniform[0],&model[0][0], sizeof(cl_float16));
        memcpy(&uniform[0][sizeof(cl_float16[1])],&perspective[0][0], sizeof(cl_float16));
        memcpy(&uniform[0][sizeof(cl_float16[2])],&view[0][0], sizeof(cl_float16));
        memcpy(&uniform[0][sizeof(cl_float16[3])],&sampler, sizeof(sampler));
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

    device_load_vertex_attributes(&device_context, vertex_attributes, vertex_attribute_data);

    int tex_buffer = device_create_ro_buffer(&shared_objects, sizeof(tex));
    int position_buffer = device_create_ro_buffer(&shared_objects, sizeof(position));
    int color_buffer = device_create_ro_buffer(&shared_objects, sizeof(color));

    device_write_on_buffer(&shared_objects, tex_buffer, 0, sizeof(tex), tex);
    device_write_on_buffer(&shared_objects, position_buffer, 0, sizeof(position), position);
    device_write_on_buffer(&shared_objects, color_buffer, 0, sizeof(color), color);

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
    }

    int texture_ids[LOADED_TEXTURES];
    for(int id = 0; id < LOADED_TEXTURES; ++id) {
        ppm_image_t *image = textures[id];
        texture_ids[id] = device_create_2d_texture(&shared_objects, image->x, image->y, TEX_RGBA8);
        uint8_t* data = (uint8_t*) malloc(image->x*image->y*sizeof(uint8_t[4]));
        for(uint32_t i=0; i<image->x*image->y; ++i) {
            data[i*4+0] = image->data[i].red;
            data[i*4+1] = image->data[i].green;
            data[i*4+2] = image->data[i].blue;
            data[i*4+3] = 0xFFu;
        }
        device_write_2d_texture(&shared_objects, texture_ids[id], image->x, image->y, TEX_RGBA8, data);
        free(data);
    }

    device_bind_texture_unit(&device_context, 0, texture_ids[0]);
    device_bind_texture_unit(&device_context, 1, texture_ids[1]);

    device_load_texture_datas(&device_context, texture_datas);
}

// mutable state
static cl_uint cummulative_vertices = 0;
static cl_uint num_triangles = 0;
static cl_uint vertex_command_queue_index = 0;
// unmutable state

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

    printf("PERF: exec_time=%ld ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());

    uint32_t* result = (uint32_t*) malloc(sizeof(int[framebuffer_height][framebuffer_width]));

    device_launch_read_pixels(&device_context, 0, 0, framebuffer_width, framebuffer_height, result);

    print_ppm("output.ppm", framebuffer_width, framebuffer_height, (uint8_t*)result);

    free(result);
}
