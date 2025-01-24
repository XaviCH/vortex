#include <kernels/triangle_setup.ocl.c>
#include <tests/glsc2/common.h>
#include <tests/glsc2/debug.hpp>
#include <algorithm>

// CONSTANTS
char KERNEL_NAME[] = "triangleSetupImpl";
size_t BINARY_SIZE = sizeof(triangle_setup_ocl);
const unsigned char* BINARY = triangle_setup_ocl;
int INDEX_BUFFER[] = {0,1,2,2,1,0};
float VERTEX_BUFFER[] = {
    -1, -1, 0, 1,
    0, 1, 0, 1,
    1, -1, 0, 1,
};
int NUM_TRIS = 2;
int maxSubtrisSlack = 4096; // TODO: Check this
int MAX_SUBTRIS = 1;
int WIDTH = 100;
int HEIGHT = 100;


// GLOBALS
cl_int error;
cl_platform_id platform_id;
cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;

typedef struct
{
    short v0x;    // Subpixels relative to viewport center. Valid if triSubtris = 1.
    short v0y;
    short v1x;
    short v1y;
    short v2x;
    short v2y;

    uint misc;   // triSubtris=1: (zmin:20, f01:4, f12:4, f20:4), triSubtris>=2: (subtriBase)
} CRTriangleHeader;

typedef struct 
{
    uint zx;     // zx * sampleX + zy * sampleY + zb = lerp(CR_DEPTH_MIN, CR_DEPTH_MAX, (clipZ / clipW + 1) / 2)
    uint zy;
    uint zb;
    uint zslope; // (abs(zx) + abs(zy)) * (samplesPerPixel / 2)

    int wx;     // wx * (sampleX * 2 + 1) + wy * (sampleY * 2 + 1) + wb = minClipW / clipW * CR_BARY_MAX
    int wy;
    int wb;

    int ux;     // ux * (sampleX * 2 + 1) + uy * (sampleY * 2 + 1) + ub = baryU * minClipW / clipW * CR_BARY_MAX
    int uy;
    int ub;

    int vx;     // vx * (sampleX * 2 + 1) + vy * (sampleY * 2 + 1) + vb = baryV * minClipW / clipW * CR_BARY_MAX
    int vy;
    int vb;

    uint vi0;    // Vertex indices.
    uint vi1;
    uint vi2;
} CRTriangleData; 

typedef struct {
    float position[4];
    float color[4];
} VertexData;

int main(int argc, char** argv) {

    CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
    context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));
    program = CL_CHECK2(clCreateProgramWithBinary(context, 1, &device_id, &BINARY_SIZE, &BINARY, NULL, &_err));
    CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
    kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

    cl_mem c_index_buffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(INDEX_BUFFER), &INDEX_BUFFER, &_err));
    cl_int c_num_tris = NUM_TRIS;
    cl_int c_max_subtris = std::max(MAX_SUBTRIS, NUM_TRIS + maxSubtrisSlack);
    cl_mem g_tri_header = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(CRTriangleHeader[c_max_subtris]), NULL, &_err));
    cl_mem g_tri_data = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(CRTriangleData[c_max_subtris]), NULL, &_err));
    cl_int vertex_size = sizeof(VertexData);
    cl_image_format image_format = {
        .image_channel_order = CL_RGBA,
        .image_channel_data_type = CL_FLOAT,
    };
    cl_image_desc image_desc = {
        .image_type = CL_MEM_OBJECT_IMAGE1D,
        .image_width = (vertex_size / sizeof(float[4])) * NUM_TRIS * 3,
        .image_row_pitch = vertex_size * NUM_TRIS * 3,
        // .image_slice_pitch = 0,
        // .num_mip_levels = 0,
        // .num_samples = 0,
    };
    cl_mem t_vertex_buffer = CL_CHECK2(clCreateImage(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &image_format, &image_desc, &VERTEX_BUFFER, &_err));
    cl_mem g_tri_subtris = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char[c_max_subtris]), NULL, &_err));
    cl_mem g_num_subtris = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &_err)); // atomic
    cl_int c_viewport_width = WIDTH, c_viewport_height = HEIGHT;
    cl_int samples_log2 = 0;
    cl_uint render_mode_flags = 0; 

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(c_index_buffer),    &c_index_buffer));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(t_vertex_buffer),   &t_vertex_buffer));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(g_tri_header),      &g_tri_header));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(g_tri_data),        &g_tri_data));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(g_tri_subtris),     &g_tri_subtris));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(g_num_subtris),     &g_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(c_num_tris),        &c_num_tris));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(c_max_subtris),     &c_max_subtris));

    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(vertex_size),       &vertex_size));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(c_viewport_width),       &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(c_viewport_height),       &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(samples_log2),       &samples_log2));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(render_mode_flags),       &render_mode_flags));

    cl_command_queue command_queue = CL_CHECK2(clCreateCommandQueue(context, device_id, NULL, &_err));
    
    size_t local_work_size[] = {32, 2}; // WARPS & CR_SETUP_WARPS
    int threadsPerBlock = local_work_size[0] * local_work_size[1];
    int maxGridWidth = 65536; // ??
    size_t global_work_size[] = {(NUM_TRIS + threadsPerBlock - 1) / threadsPerBlock, 1};
    while (global_work_size[0] > maxGridWidth)
    {
        global_work_size[0] = (global_work_size[0] + 1) >> 1;
        global_work_size[1] <<= 1;
    }
    global_work_size[0] *= local_work_size[0];
    global_work_size[1] *= local_work_size[1];
    size_t group_size;
    CL_CHECK(clGetKernelWorkGroupInfo(kernel, device_id,CL_KERNEL_WORK_GROUP_SIZE, sizeof(group_size), &group_size, NULL));
    //printf("Kernel work group size = %d.\nThread x block = %d.\nlws = (%d,%d), gwz = (%d,%d)\n", group_size, threadsPerBlock, local_work_size[0], local_work_size[1], global_work_size[0], global_work_size[1]);
    CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(command_queue));
    printf("SUCCESS\n");
}