#include <CL/opencl.h>
#include <stdio.h>

cl_int _err;
// TODO deberia ser independiente de glsc2.c, es decir pasar por parametro de funcion todas las variables que necesitemos

cl_platform_id _getPlatformID() {    
    static cl_platform_id platform_id = NULL;

    if (!platform_id) clGetPlatformIDs(1, &platform_id, NULL);
    
    printf("_getPlatformID() return: %x\n", platform_id);
    return platform_id;
}

cl_device_id _getDeviceID() {
    static cl_device_id device_id = NULL;

    if (!device_id) clGetDeviceIDs(_getPlatformID(), CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);

    printf("_getDeviceID() return: %x\n", device_id);
    return device_id;
}

cl_context _getContext() {
    static cl_context context = NULL;

    if (!context) {
        cl_device_id device_id = _getDeviceID();
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err);
    }

    printf("_getContext() return: %x\n", context);
    return context;
} 

cl_program createProgramWithBinary(const uint8_t* binary, size_t length) {
    printf("createProgramWithBinary() binary=%x, length=%d\n", binary, length);

    cl_device_id device_id = _getDeviceID();

    cl_program program = clCreateProgramWithBinary(
        _getContext(), 1, &device_id, &length, &binary, NULL, &_err);
    
    printf("\treturn=%x, error=%d\n", program, _err);
    return program;
}
int buildProgram(void *program) {
    printf("buildProgram() program=%x\n", program);

    cl_device_id device_id = _getDeviceID();

    _err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    
    printf("\treturn=%d, error=%d\n", _err, _err);
    return _err;
}
/**** BASIC OPERATIONS
 * It works as an interface, OpenGL does not has to know that is implemented with OpenCL,
 * 
*/
#define MEM_READ_ONLY CL_MEM_READ_ONLY
#define MEM_WRITE_ONLY CL_MEM_WRITE_ONLY
#define MEM_READ_WRITE CL_MEM_READ_WRITE
#define MEM_COPY_HOST_PTR CL_MEM_COPY_HOST_PTR 

void* createBuffer(uint64_t flags, size_t size, const void* data){

    void *buffer = clCreateBuffer(_getContext(), flags, size, data, &_err);
    
    printf("createBuffer() return=%x, error=%d\n", buffer, _err);

    return buffer;
}

void* createCommandQueue(uint64_t properties) {
    return clCreateCommandQueue(_getContext(), _getDeviceID(), properties, &_err);
}

void* createKernel(void* program, const char* name) {
    printf("createKernel() program=%x, name=%s\n", program, name);
    cl_kernel kernel = clCreateKernel((cl_program) program, name, &_err);

    printf("\treturn=%x, error=%d\n", kernel, _err);
    return kernel;
}

void setKernelArg(void* kernel, unsigned int location, size_t size, const void* value) {
    printf("setKernelArg() location=%d, size=%d, value=%x\n", location, size, value);
    
    int err = clSetKernelArg((cl_kernel) kernel, location, size, value);

    printf("\terror=%d\n", err);
}

// I decide to make it simple, but maybe it will need to be extendend in future.
void enqueueNDRangeKernel(void* commandQueue, void* kernel, const size_t global_work_size) {
    printf("enqueueND() work=%d\n", global_work_size);
	int err = clEnqueueNDRangeKernel(
        (cl_command_queue) commandQueue, (cl_kernel) kernel,
        1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    
    printf("\terror=%d\n", err);
}

void enqueueReadBuffer(void* command_queue, void* buffer, size_t bufSize, void* data) {

    clEnqueueReadBuffer(command_queue, (cl_mem) buffer, CL_TRUE, 0, bufSize, data, 0, NULL, NULL);
}


void finish(void* command_queue) {
    clFinish((cl_command_queue) command_queue);
}

#include "kernel.fill.c"

cl_kernel _getKernelFill() {
    static cl_program program;
    static cl_kernel kernel;

    if (kernel == NULL) {
        program = createProgramWithBinary(GLSC2_kernel_fill_pocl, sizeof(GLSC2_kernel_fill_pocl));
        buildProgram(program);
        kernel = createKernel(program, "gl_fill");
    }

    return kernel;
}

void enqueueFillBuffer(void* command_queue, void* buffer, const void* pattern, size_t pattern_size, size_t offset, size_t size) {
    printf("enqueueFillBuffer() offset=%d, size=%d\n", offset, size);
    // just valid for pattern size == 4
    cl_kernel kernel = _getKernelFill();
    size_t global_work_size = size / 4;
    clSetKernelArg(kernel, 0, 4, pattern);
    clSetKernelArg(kernel, 1, sizeof(buffer), &buffer);
    clEnqueueNDRangeKernel((cl_command_queue) command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    // TODO: It brokes 
    // clEnqueueFillBuffer((cl_command_queue)command_queue, (cl_mem) buffer, pattern, pattern_size, offset, size, 0, NULL, NULL);
}

// formats
#define RGBA8 0x0
#define RGBA4 0x1
