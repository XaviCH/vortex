#include <algorithm>
#include <cmath>
#include <chrono>
#include <vector>

#include <glm/glm.hpp>

#include <kernels/bin_raster.ocl.c>
#include <tests/glsc2/common.h>
#include <tests/glsc2/debug.hpp>
#include <tests/glsc2/kernels/common.hpp>

// TEST DEFAULT PARAMETERS
char KERNEL_NAME[] = "bin_raster";
size_t BINARY_SIZE = sizeof(bin_raster_ocl);
const unsigned char* BINARY = bin_raster_ocl;

int NUM_SUBTRIS = 0; 
uint8_t TRI_SUBTRIS[] = {0, 1};
CRTriangleHeader TRI_HEADER[] = { 
    CRTriangleHeader(), {
        16384, -16384, 0, 16384, -16384, -16384, 0x7ffffe53u
    }};

int INDEX_BUFFER[] = {0,1,2,2,1,0};
float VERTEX_BUFFER[] = { 
    -1, -1, 0, 1, 1, 1, 1, 0,
    0, 1, 0, 1, 1, 1, 1, 0,
    1, -1, 0, 1, 1, 1, 1, 0,
};
int NUM_TRIS = sizeof(INDEX_BUFFER)/sizeof(int)/3;

int MAX_SUBTRIS = 1;
int MAX_BIN_SEGS = 1;
int WIDTH = 2048;
int HEIGHT = 2048;

typedef struct {
    float position[4];
    float color[4];
} VertexData;

int num_samples = 1;

// UTILS

glm::ivec2 size = glm::ivec2(WIDTH, HEIGHT);
glm::ivec2 size_pixels = (size + CR_TILE_SIZE - 1) & -CR_TILE_SIZE;
glm::ivec2 size_tiles = size_pixels >> CR_TILE_LOG2;
int32_t num_tiles = size_tiles.x * size_tiles.y;
glm::ivec2 size_bins = (size_tiles + CR_BIN_SIZE -1) >> CR_BIN_LOG2;
int32_t num_bins = size_bins.x * size_bins.y;
glm::ivec2 viewport_size = glm::ivec2(WIDTH, HEIGHT);
int roundSize  = CR_BIN_WARPS * 32;
int minBatches = CR_BIN_STREAMS_SIZE * 2;
int maxRounds  = 32;

// Nvidia defs
int maxSubtrisSlack     = 4096;     // x 81B    = 324KB
int maxBinSegsSlack     = 256;      // x 2137B  = 534KB
int maxTileSegsSlack    = 4096;     // x 136B   = 544KB


// KERNEL INPUTS
cl_mem t_tri_header;
cl_mem c_num_subtris;
cl_mem c_tri_header;
cl_mem c_tri_subtris;

cl_int c_bin_batch_sz = glm::clamp(NUM_TRIS / (roundSize * minBatches), 1, maxRounds) * roundSize;
cl_int c_height_bins = size_bins.y;
cl_int c_max_bin_segs = std::max(MAX_BIN_SEGS, std::max(num_bins * CR_BIN_STREAMS_SIZE, (NUM_TRIS - 1) / CR_BIN_SEG_SIZE + 1) + maxBinSegsSlack);
cl_int c_max_subtris = std::max(MAX_SUBTRIS, NUM_TRIS + maxSubtrisSlack);
cl_int c_num_bins = size_bins.x * size_bins.y;
cl_int c_num_tris = NUM_TRIS;
cl_int c_viewport_height = HEIGHT;
cl_int c_viewport_width = WIDTH;
cl_int c_width_bins = size_bins.x;

// OUTPUTS
typedef struct {
    cl_int a_bin_counter;
    cl_int a_num_bin_segs;
    cl_int *g_bin_first_seg;
    cl_int *g_bin_seg_data;
    cl_int *g_bin_seg_next;
    cl_int *g_bin_seg_count;
    cl_int *g_bin_total;
} result_t;

result_t kernel_result, emulate_result;

// GLOBALS
cl_int error;
cl_platform_id platform_id;
cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;

void emulateBinRaster();

#define CONF_DEBUG_KERNEL

#ifdef CONF_DEBUG_KERNEL
#define DEBUG(...) __VA_ARGS__
#else
#define DEBUG(...)
#endif

#ifdef CONF_DEBUG_KERNEL
typedef struct {
    int over_total;
} global_bin_raster_debug_t;

typedef struct {
    int over_total;
} local_bin_raster_debug_t;

typedef struct {
    cl_uint my_idx;
    cl_uint l_broadcast;
    cl_uint4 tri_data;
} warp_bin_raster_debug_t;
#endif

int main(int argc, char** argv) {

    CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
    context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));
    program = CL_CHECK2(clCreateProgramWithBinary(context, 1, &device_id, &BINARY_SIZE, &BINARY, NULL, &_err));
    CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
    kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

    // Set Up Kernel Mem
    c_tri_header            = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(TRI_HEADER), &TRI_HEADER, &_err));
    cl_image_format image_format = {
        .image_channel_order = CL_RGBA,
        .image_channel_data_type = CL_UNSIGNED_INT32,
    };
    cl_image_desc image_desc = {
        .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
        .image_width = sizeof(TRI_HEADER) / sizeof(TRI_HEADER[0]),
        .image_row_pitch = 0, // check this
        .mem_object = c_tri_header, 
    };
    t_tri_header            = CL_CHECK2(clCreateImage(context, CL_MEM_READ_ONLY, &image_format, &image_desc, NULL, &_err));
    c_num_subtris           = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(NUM_SUBTRIS), &NUM_SUBTRIS, &_err));
    c_tri_subtris           = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(TRI_SUBTRIS), &TRI_SUBTRIS, &_err));
    
    cl_int bin_counter = 0, num_bin_segs = 0;
    cl_mem a_bin_counter    = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(bin_counter), &bin_counter, &_err));
    cl_mem a_num_bin_segs   = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(num_bin_segs), &num_bin_segs, &_err));
    
    cl_mem g_bin_first_seg  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]), NULL, &_err));
    cl_mem g_bin_seg_data   = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[c_max_bin_segs * CR_BIN_SEG_SIZE]), NULL, &_err));
    cl_mem g_bin_seg_next   = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[c_max_bin_segs]), NULL, &_err));
    cl_mem g_bin_seg_count  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[c_max_bin_segs]), NULL, &_err));
    cl_mem g_bin_total      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]), NULL, &_err));

    DEBUG(cl_mem g_w_debug = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(warp_bin_raster_debug_t[32]), NULL, &_err)));

    int counter = 0;

    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_bin_counter),   &a_bin_counter));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(a_num_bin_segs),  &a_num_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_num_subtris),   &c_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_first_seg), &g_bin_first_seg));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_seg_count), &g_bin_seg_count));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_seg_data),  &g_bin_seg_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_seg_next),  &g_bin_seg_next));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_bin_total),     &g_bin_total));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_tri_header),    &c_tri_header));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_tri_subtris),   &c_tri_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(t_tri_header),    &t_tri_header));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_bin_batch_sz),     &c_bin_batch_sz));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_height_bins),      &c_height_bins));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_bin_segs),     &c_max_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_subtris),      &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_num_bins),         &c_num_bins));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_num_tris),         &c_num_tris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_viewport_height),  &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_viewport_width),   &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_width_bins),       &c_width_bins));

    DEBUG(CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(g_w_debug), &g_w_debug)));

    // Enqueue Kernel
    cl_command_queue command_queue = CL_CHECK2(clCreateCommandQueue(context, device_id, NULL, &_err));
    
    glm::ivec2 block_size(32, CR_BIN_WARPS);
    glm::ivec2 size_threads = glm::ivec2(CR_BIN_STREAMS_SIZE, 1) * block_size;
    glm::ivec2 grid_size = (size_threads + block_size - 1) / block_size;
    size_t local_work_size[] = {block_size.x, block_size.y};
    size_t global_work_size[] = {grid_size.x, grid_size.y};
    // cuda to opencl
    global_work_size[0] *= local_work_size[0];
    global_work_size[1] *= local_work_size[1];

    auto begin = std::chrono::high_resolution_clock::now();
    CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(command_queue));
    auto end = std::chrono::high_resolution_clock::now(); 
    
    // Get Kernel Resutls
    kernel_result.g_bin_first_seg   = (cl_int*) malloc(CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * sizeof(cl_int));
    kernel_result.g_bin_seg_data    = (cl_int*) malloc(c_max_bin_segs * CR_BIN_SEG_SIZE * sizeof(cl_int));
    kernel_result.g_bin_seg_next    = (cl_int*) malloc(c_max_bin_segs * sizeof(cl_int));
    kernel_result.g_bin_seg_count   = (cl_int*) malloc(c_max_bin_segs * sizeof(cl_int));
    kernel_result.g_bin_total       = (cl_int*) malloc(CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * sizeof(cl_int));
    CL_CHECK(clEnqueueReadBuffer(command_queue, a_bin_counter,      CL_TRUE, 0, sizeof(cl_int),                                         &kernel_result.a_bin_counter,   0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(command_queue, a_num_bin_segs,     CL_TRUE, 0, sizeof(cl_int),                                         &kernel_result.a_num_bin_segs,  0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(command_queue, g_bin_first_seg,    CL_TRUE, 0, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]),   kernel_result.g_bin_first_seg,  0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(command_queue, g_bin_seg_data,     CL_TRUE, 0, sizeof(cl_int[c_max_bin_segs * CR_BIN_SEG_SIZE]),       kernel_result.g_bin_seg_data,   0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(command_queue, g_bin_seg_next,     CL_TRUE, 0, sizeof(cl_int[c_max_bin_segs]),                         kernel_result.g_bin_seg_next,   0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(command_queue, g_bin_seg_count,    CL_TRUE, 0, sizeof(cl_int[c_max_bin_segs]),                         kernel_result.g_bin_seg_count,  0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(command_queue, g_bin_total,        CL_TRUE, 0, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]),   kernel_result.g_bin_total,      0, NULL, NULL));

    DEBUG(
        warp_bin_raster_debug_t *result_g_w_debug = (warp_bin_raster_debug_t*) malloc(sizeof(warp_bin_raster_debug_t[32]));
        CL_CHECK(clEnqueueReadBuffer(command_queue, g_w_debug,        CL_TRUE, 0, sizeof(warp_bin_raster_debug_t[32]),   result_g_w_debug,      0, NULL, NULL));
        for(int i=0; i<16; ++i) {
            CRTriangleHeader tri_data = *((CRTriangleHeader*) &result_g_w_debug[i].tri_data);
            printf("DEBUG: i=%d, g_w_debug.my_idx=%u, g_w_debug.l_broadcast=%x, tri_data=(%d,%d,%d,%d)\n", // ,%d,%d,%x)\n", 
                i, result_g_w_debug[i].my_idx, result_g_w_debug[i].l_broadcast,
                result_g_w_debug[i].tri_data.s[0], result_g_w_debug[i].tri_data.s[1], result_g_w_debug[i].tri_data.s[2], result_g_w_debug[i].tri_data.s[3]);
                // tri_data.v0x, tri_data.v0y, tri_data.v1x, tri_data.v1y, tri_data.v2x, tri_data.v2y, tri_data.misc);
        }
    )

    emulateBinRaster();
    
    printf("INFO: kernel time = %d ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
    
    printf("INFO: a_bin_counter = %d\n", kernel_result.a_bin_counter); 
    // ASSERT_EQ_I(emulate_result.a_bin_counter, kernel_result.a_bin_counter);
    printf("INFO: a_num_bin_segs = %d\n", kernel_result.a_num_bin_segs);
    ASSERT_EQ_I(emulate_result.a_num_bin_segs, kernel_result.a_num_bin_segs);
    printf("INFO: CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE = %d\n", CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE);
    for(int i=0; i<CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE; ++i) {
        // if (kernel_result.g_bin_first_seg[i] != -1)
        // printf("%d ", i);
        //printf("INFO: g_bin_first_seg[%d] = %d\n", i, emulate_result.g_bin_first_seg[i]);
        if (emulate_result.g_bin_first_seg[i] != kernel_result.g_bin_first_seg[i]) {
            printf("INFO: g_bin_first_seg[%d]=%d | %d\n", i, emulate_result.g_bin_first_seg[i], kernel_result.g_bin_first_seg[i]);
        }
        //ASSERT_EQ_I(emulate_result.g_bin_first_seg[i], kernel_result.g_bin_first_seg[i]);
        // if (emulate_result.g_bin_total[i] != 0)
        if (emulate_result.g_bin_total[i] != kernel_result.g_bin_total[i]) {
            printf("INFO: g_bin_total[%d]=%d | %d\n", i, emulate_result.g_bin_total[i], kernel_result.g_bin_total[i]);
            //printf("INFO: g_bin_first_seg[%d]=%d | %d\n", i, emulate_result.g_bin_first_seg[i], kernel_result.g_bin_first_seg[i]);
        }
        //ASSERT_EQ_I(emulate_result.g_bin_total[i], kernel_result.g_bin_total[i]);
    }
    for(int i=0; i<c_max_bin_segs; ++i) {
        //printf("INFO: g_bin_seg_next[%d] = %d\n", i, kernel_result.g_bin_seg_next[i]);
        //ASSERT_EQ_I(emulate_result.g_bin_seg_next[i], kernel_result.g_bin_seg_next[i]);
        //printf("INFO: g_bin_seg_count[%d] = %d\n", i, kernel_result.g_bin_seg_count[i]);
        // ASSERT_EQ_I(emulate_result.g_bin_seg_count[i], kernel_result.g_bin_seg_count[i]);
        //printf("INFO: g_bin_seg_data[%d] = %d\n", i, kernel_result.g_bin_seg_data[i]);
        
    }

    printf("TEST PASS.\n");
}

void emulateBinRaster(void)
{

    emulate_result.g_bin_first_seg   = (cl_int*) malloc(CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * sizeof(cl_int));
    emulate_result.g_bin_seg_data    = (cl_int*) malloc(c_max_bin_segs * CR_BIN_SEG_SIZE * sizeof(cl_int));
    emulate_result.g_bin_seg_next    = (cl_int*) malloc(c_max_bin_segs * sizeof(cl_int));
    emulate_result.g_bin_seg_count   = (cl_int*) malloc(c_max_bin_segs * sizeof(cl_int));
    emulate_result.g_bin_total       = (cl_int*) malloc(CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * sizeof(cl_int));

    if (NUM_SUBTRIS > c_max_subtris)
        return;

    std::vector<int> batchTris;
    std::vector<int> currSeg(c_num_bins * CR_BIN_STREAMS_SIZE, 0);
    std::vector<int> idxInSeg(c_num_bins * CR_BIN_STREAMS_SIZE, 0);

    for (int i = 0; i < c_num_bins * CR_BIN_STREAMS_SIZE; i++)
    {
        emulate_result.g_bin_first_seg[i] = -1;
        emulate_result.g_bin_total[i] = 0;
        currSeg[i] = -1;
        idxInSeg[i] = CR_BIN_SEG_SIZE;
    }

    // Loop over batches.

    for (int batchIdx = 0; batchIdx * c_bin_batch_sz < c_num_tris; batchIdx++)
    {
        // Collect triangles.

        batchTris.clear();
        int batchStart = batchIdx * c_bin_batch_sz;
        int batchEnd = std::min(batchStart + c_bin_batch_sz, c_num_tris);
        for (int triIdx = batchStart; triIdx < batchEnd; triIdx++)
        {
			int numSubtris = TRI_SUBTRIS[triIdx];
            for (int subtriIdx = 0; subtriIdx < numSubtris; subtriIdx++)
                batchTris.push_back((triIdx << 3) | ((numSubtris == 1) ? 7 : subtriIdx));
        }

        // Rasterize each triangle to bins.

        for (int idxInBatch = 0; idxInBatch < batchTris.size(); idxInBatch++)
        {
            int triIdx = batchTris[idxInBatch];

            int dataIdx = triIdx >> 3;
            int subtriIdx = triIdx & 7;
            if (subtriIdx != 7)
                dataIdx = TRI_HEADER[dataIdx].misc + subtriIdx;
            
            printf("dataIdx=%d, subtriIdx=%d\n", dataIdx, subtriIdx);

            // Read vertices and compute AABB.
            const CRTriangleHeader& tri = TRI_HEADER[dataIdx];
            glm::ivec2 v0 = glm::ivec2(tri.v0x, tri.v0y);
            glm::ivec2 d01 = glm::ivec2(tri.v1x, tri.v1y) - v0;
            glm::ivec2 d02 = glm::ivec2(tri.v2x, tri.v2y) - v0;
            v0 += viewport_size * CR_SUBPIXEL_SIZE / 2;
            glm::ivec2 lo = v0 + glm::min(glm::ivec2(0), glm::min(d01, d02));
            glm::ivec2 hi = v0 + glm::max(glm::ivec2(0), glm::max(d01, d02));

            // Check against each bin.
            for (int binIdx = 0; binIdx < c_num_bins; binIdx++)
            {
                int binX = binIdx % size_bins.x;
                int binY = binIdx / size_bins.x;
                int half = CR_BIN_SIZE * CR_TILE_SIZE * CR_SUBPIXEL_SIZE / 2;
                glm::ivec2 center = (glm::ivec2(binX, binY) * 2 + 1) * half;

                // Outside AABB => skip.
                if (lo.x >= center.x + half || lo.y >= center.y + half || hi.x <= center.x - half || hi.y <= center.y - half)
                    continue;

                // No intersection => skip.
                glm::ivec2 p0 = center - v0;
                glm::ivec2 p1 = p0 - d01;
                glm::ivec2 d12 = d02 - d01;
                if ((int64_t)p0.x * d01.y - (int64_t)p0.y * d01.x >= (abs(d01.x) + abs(d01.y)) * half) continue;
                if ((int64_t)p0.y * d02.x - (int64_t)p0.x * d02.y >= (abs(d02.x) + abs(d02.y)) * half) continue;
                if ((int64_t)p1.x * d12.y - (int64_t)p1.y * d12.x >= (abs(d12.x) + abs(d12.y)) * half) continue;

                // Segment full => allocate a new one.

                int si = binIdx * CR_BIN_STREAMS_SIZE + batchIdx % CR_BIN_STREAMS_SIZE;
                if (idxInSeg[si] == CR_BIN_SEG_SIZE)
                {
                    int segIdx = std::min(emulate_result.a_num_bin_segs++, c_max_bin_segs - 1);
                    if (currSeg[si] == -1)
                        emulate_result.g_bin_first_seg[si] = segIdx;
                    else
                        emulate_result.g_bin_seg_next[currSeg[si]] = segIdx;

                    emulate_result.g_bin_seg_next[segIdx] = -1;
					emulate_result.g_bin_seg_count[segIdx] = CR_BIN_SEG_SIZE;
                    currSeg[si] = segIdx;
                    idxInSeg[si] = 0;
                }

                // Append to the current segment.
                emulate_result.g_bin_seg_data[currSeg[si] * CR_BIN_SEG_SIZE + idxInSeg[si]] = triIdx;
                idxInSeg[si]++;
                emulate_result.g_bin_total[si]++;
            }
        }

        // Flush between batches.

        for (int i = 0; i < c_num_bins * CR_BIN_STREAMS_SIZE; i++)
		{
			if (idxInSeg[i] != CR_BIN_SEG_SIZE)
				emulate_result.g_bin_seg_count[currSeg[i]] = idxInSeg[i];
            idxInSeg[i] = CR_BIN_SEG_SIZE;
		}
    }
}