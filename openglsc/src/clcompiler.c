#include <stdio.h>
#include <string.h>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/opencl.h>

#define CHECK(_ERROR) \
{ \
    if (_ERROR != CL_SUCCESS) {\
        printf("Unexpected OpenCL error (%d | %x) on %s:%d.\n", _ERROR, _ERROR, __FILE__, __LINE__);\
        exit(1);\
    }\
}

static int read_file(const char* filename, uint8_t** data, size_t* size) {
    if (NULL == filename || NULL == data || 0 == size)
        return -1;

    FILE* fp = fopen(filename, "r");
    if (NULL == fp) {
        fprintf(stderr, "Failed to load kernel.");
        return -1;
    }
    fseek(fp , 0 , SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);

    *data = (uint8_t*)malloc(fsize);
    *size = fread(*data, 1, fsize, fp);
  
    fclose(fp);
  
    return 0;
}

cl_int error;
cl_platform_id platform_id;
cl_device_id device_id;
cl_context context;

int main(int argc, char** argv) {


    error = clGetPlatformIDs(1, &platform_id, NULL);
    CHECK(error);

    error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
    CHECK(error);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL,  &error);
    CHECK(error);


    uint8_t* file;
    size_t file_size;

    read_file(argv[1], &file, &file_size);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**) &file, &file_size, &error);
    CHECK(error);

    char *options = NULL;
    if (argc > 3) {
        int flag = 3;
        size_t size = 0;
        while(flag < argc) {
            size += strlen(argv[flag++]);
        }

        options = (char*) malloc(size + argc-3);

        flag = 3;
        size_t offset = 0;
        while (flag < argc) {
            strcpy(options + offset, argv[flag]);
            offset += strlen(argv[flag]);
            options[offset] = ' ';
            ++offset;
            ++flag;
        }
        options[offset-1] = '\0';
        printf("Compiling with options: %s\n", options);
    }

    error = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
    if (error == CL_BUILD_PROGRAM_FAILURE) {
    // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }
    CHECK(error);

    size_t result;
    size_t size_result;

    error = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(result), &result, &size_result);
    CHECK(error);
    /*
    if (result != CL_BUILD_SUCCESS) {
        printf("Error at building OpenCL binary: %ld | %lx\n.", result, result);

        error = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &size_result);
        CHECK(error);

        char* log = (char*) malloc(size_result);

        error = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, size_result, log, &size_result);
        CHECK(error);

        printf("Build log:\n%s\n", log);

        exit(1);
    }*/

    uint8_t* binary;
    size_t binary_size;
    size_t returned_size;

    error = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size), &binary_size, &returned_size);
    CHECK(error);

    if (binary_size == 0) {
        printf("Driver does not support binaries.\n");
        exit(-1);
    }

    binary = (uint8_t*) malloc(binary_size);
    error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, binary_size, &binary, &returned_size);
    CHECK(error);

    FILE *out = fopen(argv[2], "wb");

    fwrite(binary,1,binary_size,out);
}