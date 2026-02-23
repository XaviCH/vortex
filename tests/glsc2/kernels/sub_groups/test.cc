#include <algorithm>
#include <cmath>
#include <chrono>

#include <kernels/test.ocl.c>
#include <tests/glsc2/common.h>
#include <tests/glsc2/debug.hpp>
#include <tests/glsc2/kernels/common.hpp>

#define CL_TARGET_OPENCL_VERSION 120

#define WARP_SIZE 32

typedef struct {
    int scan_inc_add_ui;
} result_t;

// CL GLOBALS
cl_platform_id platform_id;
cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel 
    kernel_sub_group_reduce_min_ui, 
    kernel_local_scan_inclusive_add_ui,
    kernel_local_scan_inclusive_min_ui,
    kernel_local_reduce_min_ui,
    kernel_local_reduce_or_1dim_ui,
    kernel_local_scan_inclusive_and_2dim_ui;
cl_command_queue command_queue;

cl_mem g_results;

void setCL();
void setCLParameters();
void setCLKernels();

int main(int argc, char** argv) {

    setCL();
    setCLParameters();
    setCLKernels();

    size_t global_work_size[2], local_work_size[2];

    // block_size = glm::ivec2(32, CR_SETUP_WARPS);
    // setWorkSizeFromNum(block_size, 32*CR_SETUP_WARPS, global_work_size, local_work_size);
    std::vector<cl_uint> results(WARP_SIZE*16);
    {
        auto begin = std::chrono::high_resolution_clock::now();
        global_work_size[0] = WARP_SIZE*16;
        local_work_size[0] = WARP_SIZE*16;
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel_sub_group_reduce_min_ui, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
        CL_CHECK(clFinish(command_queue));
        auto end = std::chrono::high_resolution_clock::now(); 
        printf("PERF: kernel_sub_group_reduce_min_ui time = %d ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
        
        CL_CHECK(clEnqueueReadBuffer(command_queue, g_results, CL_TRUE, 0, results.capacity() * sizeof(cl_uint), results.data(), 0, NULL, NULL));
        for (int i=0; i<WARP_SIZE*16; ++i)
            printf("results[%d]=%d\n", i, results[i]);
    }


    {
        auto begin = std::chrono::high_resolution_clock::now();
        global_work_size[0] = WARP_SIZE; global_work_size[1] = 16;
        local_work_size[0] = WARP_SIZE; local_work_size[1] = 16;
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel_local_scan_inclusive_add_ui, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
        CL_CHECK(clFinish(command_queue));
        auto end = std::chrono::high_resolution_clock::now(); 
        printf("PERF: kernel_local_scan_inclusive_add_ui time = %d ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
        
        CL_CHECK(clEnqueueReadBuffer(command_queue, g_results, CL_TRUE, 0, results.capacity() * sizeof(cl_uint), results.data(), 0, NULL, NULL));
        for (int i=0; i<WARP_SIZE*16; ++i)
           printf("results[%d]=%d\n", i, results[i]);
    }
    /*
    {
        auto begin = std::chrono::high_resolution_clock::now();
        global_work_size[0] = WARP_SIZE*16;
        local_work_size[0] = WARP_SIZE*16;
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel_local_scan_inclusive_min_ui, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
        CL_CHECK(clFinish(command_queue));
        auto end = std::chrono::high_resolution_clock::now(); 
        printf("PERF: kernel_local_scan_inclusive_min_ui time = %d ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
        
        CL_CHECK(clEnqueueReadBuffer(command_queue, g_results, CL_TRUE, 0, results.capacity() * sizeof(cl_uint), results.data(), 0, NULL, NULL));
        for (int i=0; i<WARP_SIZE*16; ++i)
           printf("results[%d]=%x\n", i, results[i]);
    }

    {
        auto begin = std::chrono::high_resolution_clock::now();
        global_work_size[0] = WARP_SIZE*16;
        local_work_size[0] = WARP_SIZE*16;
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel_local_reduce_min_ui, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
        CL_CHECK(clFinish(command_queue));
        auto end = std::chrono::high_resolution_clock::now(); 
        printf("PERF: kernel_local_reduce_min_ui time = %d ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
        
        CL_CHECK(clEnqueueReadBuffer(command_queue, g_results, CL_TRUE, 0, results.capacity() * sizeof(cl_uint), results.data(), 0, NULL, NULL));
        for (int i=0; i<WARP_SIZE*16; ++i)
           printf("results[%d]=%x\n", i, results[i]);
    }

    {
        auto begin = std::chrono::high_resolution_clock::now();
        global_work_size[0] = WARP_SIZE;
        global_work_size[1] = 16;
        local_work_size[0] = WARP_SIZE;
        local_work_size[1] = 16;
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel_local_scan_inclusive_and_2dim_ui, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
        CL_CHECK(clFinish(command_queue));
        auto end = std::chrono::high_resolution_clock::now(); 
        printf("PERF: kernel_local_scan_inclusive_and_2dim_ui time = %d ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
        
        CL_CHECK(clEnqueueReadBuffer(command_queue, g_results, CL_TRUE, 0, results.capacity() * sizeof(cl_uint), results.data(), 0, NULL, NULL));
        for (int i=0; i<WARP_SIZE*16; ++i)
           printf("results[%d]=%x\n", i, results[i]);
    }

    {
        auto begin = std::chrono::high_resolution_clock::now();
        global_work_size[0] = WARP_SIZE;
        global_work_size[1] = 16;
        local_work_size[0] = WARP_SIZE;
        local_work_size[1] = 16;
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel_local_reduce_or_1dim_ui, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
        CL_CHECK(clFinish(command_queue));
        auto end = std::chrono::high_resolution_clock::now(); 
        printf("PERF: kernel_local_reduce_or_1dim_ui time = %d ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
        
        CL_CHECK(clEnqueueReadBuffer(command_queue, g_results, CL_TRUE, 0, results.capacity() * sizeof(cl_uint), results.data(), 0, NULL, NULL));
        for (int i=0; i<WARP_SIZE*16; ++i)
           printf("results[%d]=%x\n", i, results[i]);
    }
    */

}

void setCL() {
    size_t binary_size;
    unsigned char *binary;

    CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
    context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

    binary_size = test_ocl_len;
    binary = test_ocl;
    program = CL_CHECK2(clCreateProgramWithBinary(context, 1, &device_id, &binary_size, (const unsigned char**) &binary, NULL, &_err));
    CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
    kernel_sub_group_reduce_min_ui = CL_CHECK2(clCreateKernel(program, "test_sub_group_reduce_min_ui", &_err));
    kernel_local_scan_inclusive_add_ui = CL_CHECK2(clCreateKernel(program, "test_local_scan_inclusive_add_ui", &_err));
    kernel_local_scan_inclusive_min_ui = CL_CHECK2(clCreateKernel(program, "test_local_scan_inclusive_min_ui", &_err));
    kernel_local_reduce_min_ui = CL_CHECK2(clCreateKernel(program, "test_local_reduce_min_ui", &_err));
    kernel_local_scan_inclusive_and_2dim_ui = CL_CHECK2(clCreateKernel(program, "test_local_scan_inclusive_and_2dim_ui", &_err));
    kernel_local_reduce_or_1dim_ui = CL_CHECK2(clCreateKernel(program, "test_local_reduce_or_1dim_ui", &_err));

    command_queue = CL_CHECK2(clCreateCommandQueue(context, device_id, NULL, &_err));
}

void setCLParameters() {

    g_results      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint[WARP_SIZE*16]), NULL, &_err));

}

void setCLKernels() {

    CL_CHECK(clSetKernelArg(kernel_sub_group_reduce_min_ui,     0, sizeof(g_results), &g_results));
    CL_CHECK(clSetKernelArg(kernel_local_scan_inclusive_add_ui, 0, sizeof(g_results), &g_results));
    CL_CHECK(clSetKernelArg(kernel_local_scan_inclusive_min_ui, 0, sizeof(g_results), &g_results));
    CL_CHECK(clSetKernelArg(kernel_local_reduce_min_ui,         0, sizeof(g_results), &g_results));
    CL_CHECK(clSetKernelArg(kernel_local_scan_inclusive_and_2dim_ui,         0, sizeof(g_results), &g_results));
    CL_CHECK(clSetKernelArg(kernel_local_reduce_or_1dim_ui,         0, sizeof(g_results), &g_results));

}
