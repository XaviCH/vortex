#include <algorithm>
#include <cmath>
#include <chrono>
#include "glm/gtc/matrix_transform.hpp"

#include <kernels/triangle_setup.ocl.c>
#include <kernels/bin_raster.ocl.c>
#include <kernels/coarse_raster.ocl.c>
#include <kernels/fine_raster.ocl.c>
#include <constants.device.h>
#include <tests/glsc2/common.h>
#include <tests/glsc2/debug.hpp>
#include <tests/glsc2/kernels/common.hpp>

#define CL_TARGET_OPENCL_VERSION 120

// NVIDIA DEFS
int maxSubtrisSlack     = 4096;     // x 81B    = 324KB
int maxBinSegsSlack     = 256;      // x 2137B  = 534KB
int maxTileSegsSlack    = 4096;     // x 136B   = 544KB
int ATTRIBUTE_MULTIPROCESSOR_COUNT = 15; // SMs
int ATTRIBUTE_MAX_THREADS_PER_BLOCK = 32*20;

// TEST DEFAULT PARAMETERS
char TRIANGLE_SETUP_KERNEL_NAME[]   = "triangle_setup";
char BIN_RASTER_KERNEL_NAME[]       = "bin_raster";
char COARSE_RASTER_KERNEL_NAME[]    = "coarse_raster";
char FINE_RASTER_KERNEL_NAME[]      = "fine_raster_single_sample";

int INDEX_BUFFER[] = {
    0, 1, 3, 3, 1, 2,
    1, 5, 2, 2, 5, 6,
    5, 4, 6, 6, 4, 7,
    4, 0, 7, 7, 0, 3,
    3, 2, 7, 7, 2, 6,
    4, 5, 0, 0, 5, 1,
};

float VERTEX_BUFFER[] = {
    -1, -1, -1, 1, 1, 0, 0, 0,
    1, -1, -1, 1, 0, 1, 0, 0,
    1, 1, -1, 1, 0, 0, 1, 0,
    -1, 1, -1, 1, 1, 1, 0, 0,
    -1, -1, 1, 1, 0, 1, 1, 0,
    1, -1, 1, 1, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 0,
    -1, 1, 1, 1, 0, 0, 0, 0
};
int NUM_TRIS = sizeof(INDEX_BUFFER)/(sizeof(int[3]));
int MAX_SUBTRIS = 1;
int MAX_BIN_SEGS = 1;
int MAX_TILE_SEGS = 1;
int WIDTH = 1024;
int HEIGHT = 1024;

typedef struct {
    float position[4];
    float color[4];
} VertexData;

int num_samples = 1;

// TEST PARAMETERS
bool emulate_vertex, emulate_bin, emulate_coarse, emulate_fine; 

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
int32_t num_SMs;
int32_t num_fine_warps;
int32_t num_tris;

// INPUT PARAMETERS
cl_int samples_log2 = 0;
cl_uint render_mode_flags = 0 | RENDER_MODE_FLAG_ENABLE_LERP | RENDER_MODE_FLAG_ENABLE_DEPTH;
// const glm::ivec2 viewport_size = {WIDTH, HEIGHT};

// OUTPUTS
typedef struct {
    cl_int a_num_subtris;
    cl_uchar* tri_subtris;
    CRTriangleHeader* tri_header;
    CRTriangleData* tri_data;
} result_t;

result_t kernel_result, emulate_result;

// CL GLOBALS
cl_platform_id platform_id;
cl_device_id device_id;
cl_context context;
cl_program triangle_setup_program;
cl_program bin_raster_program;
cl_program coarse_raster_program;
cl_program fine_raster_program;

cl_kernel triangle_setup_kernel;
cl_kernel bin_raster_kernel;
cl_kernel coarse_raster_kernel;
cl_kernel fine_raster_kernel;

cl_event setup_event;
cl_event bin_event;
cl_event coarse_event;
cl_event fine_event;

cl_command_queue command_queue;

// CL PARAMETERS
cl_mem g_index_buffer;
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

cl_mem a_bin_counter;
cl_mem a_coarse_counter;
cl_mem a_fine_counter;
cl_mem a_num_active_tiles;
cl_mem a_num_bin_segs;
cl_mem a_num_subtris;
cl_mem a_num_tile_segs;

cl_int c_bin_batch_sz; // number of triangles being processed in each CTA
cl_uint c_clear_color;
cl_uint c_clear_depth;
cl_uint c_color_buffer_mode;
cl_uint c_cull_face;
cl_int c_deferred_clear;
cl_uint c_framebuffer_width;
cl_int c_height_bins;
cl_int c_height_tiles;
cl_int c_max_bin_segs;
cl_int c_max_subtris;
cl_int c_max_tile_segs;
cl_int c_num_bins;
cl_int c_num_tris;
cl_uint c_render_mode_flags;
cl_int c_samples_log2;
cl_int c_vertex_size;
cl_int c_viewport_height;
cl_int c_viewport_width;
cl_int c_width_bins;
cl_int c_width_tiles;

cl_mem t_tri_data;
cl_mem t_tri_header;
cl_mem t_color_buffer;
cl_mem t_depth_buffer;
cl_mem t_vertex_buffer;

cl_mem g_result;

// HOST CPY PARAMETRES

std::vector<cl_uint> h_g_index_buffer;
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

void setParameters(int argc, char** argv);
void setCL();
void setCLParameters();
void setCLKernels();
void setWorkSizeFromSize(glm::ivec2& block_size, glm::ivec2& size_threads, size_t* global_work_size, size_t* local_work_size);
void setWorkSizeFromNum(glm::ivec2& block_size, uint32_t num_threads, size_t* global_work_size, size_t* local_work_size);

int main(int argc, char** argv) {

    setParameters(argc, argv); // TODO may read inputs
    setCL();
    setCLParameters();
    setCLKernels();

    glm::ivec2 block_size, threads_size;
    size_t global_work_size[2], local_work_size[2];

    auto begin = std::chrono::high_resolution_clock::now();

    {
        block_size = glm::ivec2(DEVICE_SUB_GROUP_THREADS, CONF_SETUP_SUB_GROUPS);
        setWorkSizeFromNum(block_size, c_num_tris, global_work_size, local_work_size);
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, triangle_setup_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    }

    {
        block_size = glm::ivec2(DEVICE_SUB_GROUP_THREADS, CONF_BIN_SUB_GROUPS);
        threads_size = glm::ivec2(CR_BIN_STREAMS_SIZE, 1) * block_size;
        setWorkSizeFromSize(block_size, threads_size, global_work_size, local_work_size);
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, bin_raster_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    }

    {
        block_size = glm::ivec2(DEVICE_SUB_GROUP_THREADS, CONF_COARSE_SUB_GROUPS);
        threads_size = glm::ivec2(num_SMs, 1) * block_size;
        setWorkSizeFromSize(block_size, threads_size, global_work_size, local_work_size);
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, coarse_raster_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    }

    {
        block_size = glm::ivec2(DEVICE_SUB_GROUP_THREADS, num_fine_warps);
        threads_size = glm::ivec2(num_SMs, 1) * block_size;
        setWorkSizeFromSize(block_size, threads_size, global_work_size, local_work_size);
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, fine_raster_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    }
    
    CL_CHECK(clFinish(command_queue));
    auto end = std::chrono::high_resolution_clock::now(); 
    printf("PERF: exec_time=%d ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());

    size_t origin[] = {0,0,0};
    size_t region[] = {WIDTH,HEIGHT,1};

    uint32_t* result = (uint32_t*) malloc(sizeof(int)*HEIGHT*WIDTH);

    #ifdef CONF_FINE_IMAGE_ENABLED
    CL_CHECK(clEnqueueReadImage(command_queue, t_color_buffer, CL_TRUE, origin, region, 0, 0, result, 0, NULL, NULL));
    #else
    // TODO test more image formats
    CL_CHECK(clEnqueueReadBuffer(command_queue, t_color_buffer, CL_TRUE, origin[0]*origin[1], region[0]*region[1]*sizeof(uint32_t), result, 0, NULL, NULL));
    #endif
    CL_CHECK(clFinish(command_queue));

    print_ppm("output.ppm", WIDTH, HEIGHT, (uint8_t*)result);
}

void setCL() {
    size_t binary_size;
    unsigned char *binary;

    CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
    context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

    binary_size = triangle_setup_ocl_len;
    binary = triangle_setup_ocl;
    triangle_setup_program = CL_CHECK2(clCreateProgramWithBinary(context, 1, &device_id, &binary_size, (const unsigned char**) &binary, NULL, &_err));
    CL_CHECK(clBuildProgram(triangle_setup_program, 1, &device_id, NULL, NULL, NULL));
    triangle_setup_kernel = CL_CHECK2(clCreateKernel(triangle_setup_program, TRIANGLE_SETUP_KERNEL_NAME, &_err));

    binary_size = bin_raster_ocl_len;
    binary = bin_raster_ocl;
    bin_raster_program = CL_CHECK2(clCreateProgramWithBinary(context, 1, &device_id, &binary_size, (const unsigned char**) &binary, NULL, &_err));
    CL_CHECK(clBuildProgram(bin_raster_program, 1, &device_id, NULL, NULL, NULL));
    bin_raster_kernel = CL_CHECK2(clCreateKernel(bin_raster_program, BIN_RASTER_KERNEL_NAME, &_err));

    binary_size = coarse_raster_ocl_len;
    binary = coarse_raster_ocl;
    coarse_raster_program = CL_CHECK2(clCreateProgramWithBinary(context, 1, &device_id, &binary_size, (const unsigned char**) &binary, NULL, &_err));
    CL_CHECK(clBuildProgram(coarse_raster_program, 1, &device_id, NULL, NULL, NULL));
    coarse_raster_kernel = CL_CHECK2(clCreateKernel(coarse_raster_program, COARSE_RASTER_KERNEL_NAME, &_err));

    binary_size = fine_raster_ocl_len;
    binary = fine_raster_ocl;
    fine_raster_program = CL_CHECK2(clCreateProgramWithBinary(context, 1, &device_id, &binary_size, (const unsigned char**) &binary, NULL, &_err));
    CL_CHECK(clBuildProgram(fine_raster_program, 1, &device_id, NULL, NULL, NULL));
    fine_raster_kernel = CL_CHECK2(clCreateKernel(fine_raster_program, FINE_RASTER_KERNEL_NAME, &_err));
    
    command_queue = CL_CHECK2(clCreateCommandQueue(context, device_id, NULL, &_err));
}

bool startswith(const char* a, const char* b) {
    return strncmp(a, b, strlen(b)) == 0;
}

void setParameters(int argc, char** argv) {

    for(int i=0; i<argc; ++i) {
        char* arg = argv[i];
        if (strcmp(arg, "--emulate-vertex") == 0)
            emulate_vertex = 1;
        if (strcmp(arg, "--emulate-bin") == 0)
            emulate_bin = 1;  
        if (strcmp(arg, "--emulate-coarse") == 0)
            emulate_coarse = 1;
        if (strcmp(arg, "--emulate-fine") == 0)
            emulate_fine = 1;
    }

    size = glm::ivec2(WIDTH, HEIGHT);
    size_pixels = (size + CR_TILE_SIZE - 1) & -CR_TILE_SIZE;
    size_tiles = size_pixels >> CR_TILE_LOG2;
    num_tiles = size_tiles.x * size_tiles.y;
    size_bins = (size_tiles + CR_BIN_SIZE -1) >> CR_BIN_LOG2;
    num_bins = size_bins.x * size_bins.y;
    viewport_size = glm::ivec2(WIDTH, HEIGHT);
    round_size  = CR_BIN_WARPS * 32;
    min_batches = CR_BIN_STREAMS_SIZE * 2;
    max_rounds  = 32;
    max_subtris = std::max(MAX_SUBTRIS, NUM_TRIS + maxSubtrisSlack);
    num_SMs = ATTRIBUTE_MULTIPROCESSOR_COUNT;
    num_fine_warps = std::min(ATTRIBUTE_MAX_THREADS_PER_BLOCK / 32, CR_FINE_MAX_WARPS);
    num_tris = NUM_TRIS;

    glm::mat4 perspective = glm::perspective(45.0f, 1.0f, 0.01f, 1000.0f);
    glm::mat4 camera = glm::lookAt(glm::vec3{2,2,-5}, glm::vec3{0,0,0}, glm::vec3{0,1,0});
    // glm::mat4 rotation; 
    // rotation = glm::rotate(rotation, 1.f, glm::vec3(0.5, 0.5, 0.5));
    for (int i=0; i<sizeof(VERTEX_BUFFER)/sizeof(float); i+=8) {
        glm::vec4 vec(VERTEX_BUFFER[i], VERTEX_BUFFER[i+1], VERTEX_BUFFER[i+2], VERTEX_BUFFER[i+3]);
        vec = perspective * camera * vec;
        VERTEX_BUFFER[i] = vec.x;
        VERTEX_BUFFER[i+1] = vec.y;
        VERTEX_BUFFER[i+2] = vec.z;
        VERTEX_BUFFER[i+3] = vec.w;
    }
    for (int i=0; i<sizeof(INDEX_BUFFER)/sizeof(int); ++i) {
        printf("vertex %d: (%f,%f,%f,%f)\n", i, VERTEX_BUFFER[INDEX_BUFFER[i]*8], VERTEX_BUFFER[INDEX_BUFFER[i]*8+1], VERTEX_BUFFER[INDEX_BUFFER[i]*8+2], VERTEX_BUFFER[INDEX_BUFFER[i]*8+3]);
    }
}

void setCLParameters() {
    c_bin_batch_sz = glm::clamp(num_tris / (round_size * min_batches), 1, max_rounds) * round_size;
    c_clear_color = 0x00000000; // TODO ??
    c_clear_depth = 0xFFFFFFFF; // TODO ??
    c_color_buffer_mode = TEX_RGBA8;
    c_cull_face = 0;
    c_deferred_clear = 1; // TODO ??
    c_framebuffer_width = WIDTH;
    c_height_bins = size_bins.y;
    c_height_tiles = size_tiles.y;
    c_max_bin_segs = std::max(MAX_BIN_SEGS, std::max(num_bins * CR_BIN_STREAMS_SIZE, (NUM_TRIS - 1) / CR_BIN_SEG_SIZE + 1) + maxBinSegsSlack);
    c_max_tile_segs = std::max(MAX_TILE_SEGS, std::max(num_tiles, (NUM_TRIS - 1) / CR_TILE_SEG_SIZE +1) + maxTileSegsSlack);
    c_max_subtris = std::max(MAX_SUBTRIS, NUM_TRIS + maxSubtrisSlack);
    c_num_bins = size_bins.x * size_bins.y;
    c_num_tris = NUM_TRIS;
    c_render_mode_flags = render_mode_flags;
    c_samples_log2 = samples_log2;
    c_vertex_size = sizeof(VertexData);
    c_viewport_height = HEIGHT;
    c_viewport_width = WIDTH;
    c_width_bins = size_bins.x;
    c_width_tiles = size_tiles.x;
    printf("c_bin_batch_sz=%d, c_num_bins=%d\n", c_bin_batch_sz, c_num_bins);

    h_a_bin_counter.resize(1);
    h_a_coarse_counter.resize(1);
    h_a_fine_counter.resize(1);
    h_a_num_active_tiles.resize(1);
    h_a_num_bin_segs.resize(1);
    h_a_num_subtris.resize(1);
    h_a_num_tile_segs.resize(1);

    g_index_buffer      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(INDEX_BUFFER), &INDEX_BUFFER, &_err));
    h_g_index_buffer.resize(sizeof(INDEX_BUFFER)/sizeof(INDEX_BUFFER[0]));
    g_vertex_buffer     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(VERTEX_BUFFER), &VERTEX_BUFFER, &_err));
    h_g_vertex_buffer.resize(sizeof(VERTEX_BUFFER)/sizeof(VERTEX_BUFFER[0]));
    g_tri_header        = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(CRTriangleHeader[c_max_subtris]), NULL, &_err));
    h_g_tri_header.resize(c_max_subtris);
    g_tri_data          = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(CRTriangleData[c_max_subtris]), NULL, &_err));
    h_g_tri_data.resize(c_max_subtris);
    g_tri_subtris       = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uchar[c_max_subtris]), NULL, &_err));
    h_g_tri_subtris.resize(c_max_subtris);
    g_bin_first_seg     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]), NULL, &_err));
    h_g_bin_first_seg.resize(CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE);
    g_bin_seg_data      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[c_max_bin_segs * CR_BIN_SEG_SIZE]), NULL, &_err));
    h_g_bin_seg_data.resize(c_max_bin_segs * CR_BIN_SEG_SIZE);
    g_bin_seg_next      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[c_max_bin_segs]), NULL, &_err));
    h_g_bin_seg_next.resize(c_max_bin_segs);
    g_bin_seg_count     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[c_max_bin_segs]), NULL, &_err));
    h_g_bin_seg_count.resize(c_max_bin_segs);
    g_bin_total         = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]), NULL, &_err));
    h_g_bin_total.resize(CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE);
    g_active_tiles      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXTILES_SQR]), NULL, &_err));
    h_g_active_tiles.resize(CR_MAXTILES_SQR);
    g_tile_first_seg    = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXTILES_SQR]), NULL, &_err));
    h_g_tile_first_seg.resize(CR_MAXTILES_SQR);
    g_tile_seg_data     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[c_max_tile_segs * CR_TILE_SEG_SIZE]), NULL, &_err));
    h_g_tile_seg_data.resize(c_max_tile_segs * CR_TILE_SEG_SIZE);
    g_tile_seg_count    = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[c_max_tile_segs]), NULL, &_err));
    h_g_tile_seg_count.resize(c_max_tile_segs);
    g_tile_seg_next     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[c_max_tile_segs]), NULL, &_err));
    h_g_tile_seg_next.resize(c_max_tile_segs);
    
    // g_result     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4[3][3]), NULL, &_err));
    // g_local_data     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint[32*CR_COARSE_WARPS]), NULL, &_err));

    {
        cl_int ZERO = 0;
        a_bin_counter       = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err));
        a_coarse_counter    = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err));
        a_fine_counter      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err)); // TODO: check this
        a_num_active_tiles  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err));
        a_num_subtris       = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err)); // atomic
        a_num_bin_segs      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err));
        a_num_tile_segs     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &ZERO, &_err));

        h_a_bin_counter[0] = 0;
        h_a_coarse_counter[0] = 0;
        h_a_fine_counter[0] = 0;
        h_a_num_active_tiles[0] = 0;
        h_a_num_subtris[0] = 0;
        h_a_num_bin_segs[0] = 0;
        h_a_num_tile_segs[0] = 0;
    }

    // #if DEVICE_IMAGE_SUPPORT == 1
    {
        cl_image_format image_format = {
            .image_channel_order = CL_RGBA,
            .image_channel_data_type = CL_FLOAT,
        };
        cl_image_desc image_desc = {
            .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
            .image_width = sizeof(VERTEX_BUFFER) / sizeof(float[4]),
            .image_row_pitch = 0,
            .mem_object = g_vertex_buffer,
        };

        t_vertex_buffer = CL_CHECK2(clCreateImage(context, CL_MEM_READ_ONLY, &image_format, &image_desc, NULL, &_err));
    }
    {
        cl_image_format image_format = {
            .image_channel_order = CL_RGBA,
            .image_channel_data_type = CL_UNSIGNED_INT32,
        };
        cl_image_desc image_desc = {
            .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
            .image_width = sizeof(CRTriangleHeader[c_max_subtris]) / sizeof(cl_uint4),
            .image_row_pitch = 0,
            .mem_object = g_tri_header, 
        };

        t_tri_header = CL_CHECK2(clCreateImage(context, CL_MEM_READ_ONLY, &image_format, &image_desc, NULL, &_err));
    }
    {
        cl_image_format image_format = {
            .image_channel_order = CL_RGBA,
            .image_channel_data_type = CL_UNSIGNED_INT32,
        };
        cl_image_desc image_desc = {
            .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
            .image_width = sizeof(CRTriangleData[c_max_subtris]) / sizeof(cl_uint4),
            .image_row_pitch = 0,
            .mem_object = g_tri_data, 
        };

        t_tri_data = CL_CHECK2(clCreateImage(context, CL_MEM_READ_ONLY, &image_format, &image_desc, NULL, &_err));
    }
    // #endif

    #if DEVICE_IMAGE_SUPPORT == 1
    {
        cl_image_format image_format;
        
        image_format = {
            .image_channel_order = CL_RGBA,
            .image_channel_data_type = CL_UNSIGNED_INT8,
        };
        t_color_buffer = CL_CHECK2(clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, WIDTH, HEIGHT, 0, NULL, &_err));
        
        image_format = {
            .image_channel_order = CL_DEPTH, // this may be not supported for OpenCL 1.2 check cl_khr_depth_images extension
            .image_channel_data_type = CL_UNSIGNED_INT32,
        };
        t_depth_buffer = CL_CHECK2(clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, WIDTH, HEIGHT, 0, NULL, &_err));
    }
    #else
    {
        t_color_buffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint[WIDTH][HEIGHT]), NULL, &_err));
        t_depth_buffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint[WIDTH][HEIGHT]), NULL, &_err));
    }
    #endif

    h_t_color_buffer.resize(size.x*size.y);
    h_t_depth_buffer.resize(size.x*size.y);

}
void setCLKernels() {
    cl_kernel kernel;
    cl_int counter;

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
    #ifdef CONF_FINE_IMAGE_ENABLED
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_tri_data),          &t_tri_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_tri_header),        &t_tri_header));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_vertex_buffer),     &t_vertex_buffer));
    #else
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_tri_data),          &g_tri_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_vertex_buffer),     &g_vertex_buffer));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_framebuffer_width), &c_framebuffer_width));
    #endif
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_clear_color),       &c_clear_color));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_clear_depth),       &c_clear_depth));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_color_buffer_mode), &c_color_buffer_mode));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_deferred_clear),    &c_deferred_clear));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_bin_segs),      &c_max_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_subtris),       &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_tile_segs),     &c_max_tile_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_render_mode_flags), &c_render_mode_flags));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_width_tiles),       &c_width_tiles));
    
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
    global_work_size[0] = (NUM_TRIS + threadsPerBlock - 1) / threadsPerBlock;
    global_work_size[1] = 1;

    while (global_work_size[0] > maxGridWidth)
    {
        global_work_size[0] = (global_work_size[0] + 1) >> 1;
        global_work_size[1] <<= 1;
    }
    global_work_size[0] *= local_work_size[0];
    global_work_size[1] *= local_work_size[1];
}
