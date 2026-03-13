#include <compiler/common.h>

#define CL_TARGET_OPENCL_VERSION 120

#include <CL/opencl.h>

#define STRINGIFY(_VAL) #_VAL
#define TO_STRING(_VAL) STRINGIFY(_VAL)

#ifndef INSTALL_PATH
    #error To compile is required to set -DINSTALL_PATH="/path/to/dir" 
#endif

#define SRC_PATH        TO_STRING(INSTALL_PATH) "/glsc2/src"
#define SHADERS_PATH    TO_STRING(INSTALL_PATH) "/glsc2/src/backend/shaders"

#define FINE_RASTER_CL  TO_STRING(INSTALL_PATH) "/glsc2/src/backend/pipeline/fine_raster.cl"

#define CHECK(_ERROR) \
{ \
    if (_ERROR != CL_SUCCESS) {\
        printf("Unexpected OpenCL error (%d | %x) on %s:%d.\n", _ERROR, _ERROR, __FILE__, __LINE__);\
        exit(1);\
    }\
}

cl_int error;
cl_platform_id platform_id;
cl_device_id device_id;
cl_context context;

// args
char** input_files;
char* output_file;
char** options;

void parse_args(int argc, char** argv);

int main(int argc, char** argv) {

    // Check drivers availability
    // TODO: Change to select multiple platforms
    cl_uint num_platforms;
    cl_int platform_error;
    platform_error = clGetPlatformIDs(1, &platform_id, &num_platforms);
    switch (platform_error)
    {
    case CL_PLATFORM_NOT_FOUND_KHR:
        fprintf(stderr, "No platform found. CL_PLATFORM_NOT_FOUND_KHR %d.\n", CL_PLATFORM_NOT_FOUND_KHR);
        exit(platform_error);
    case CL_SUCCESS:
        break;
    default:
        fprintf(stderr, "Unexepected error.");
        exit(platform_error);
    }
    
    CHECK(error);

    cl_platform_id* platform_ids = NULL;
    if (num_platforms > 1)
        platform_ids = malloc(num_platforms*sizeof(cl_platform_id));
    else
        platform_ids = &platform_id;

    printf("Platforms Available.\n");
    for (cl_uint platform = 0; platform < num_platforms; ++platform) {
        size_t param_value_size_ret;
        clGetPlatformInfo(platform_ids[platform], CL_PLATFORM_NAME, 0, NULL, &param_value_size_ret);
        char* name = malloc(param_value_size_ret);
        clGetPlatformInfo(platform_ids[platform], CL_PLATFORM_NAME, param_value_size_ret, name, NULL);

        clGetPlatformInfo(platform_ids[platform], CL_PLATFORM_VENDOR, 0, NULL, &param_value_size_ret);
        char* vendor = malloc(param_value_size_ret);
        clGetPlatformInfo(platform_ids[platform], CL_PLATFORM_VENDOR, param_value_size_ret, vendor, NULL);

        clGetPlatformInfo(platform_ids[platform], CL_PLATFORM_VERSION, 0, NULL, &param_value_size_ret);
        char* version = malloc(param_value_size_ret);
        clGetPlatformInfo(platform_ids[platform], CL_PLATFORM_VERSION, param_value_size_ret, version, NULL);

        printf("\t%s, %s: %s\n", name, vendor, version);

        free(name);
        free(vendor);
        free(version);
    }

    size_t param_value_size_ret;
    clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, NULL, &param_value_size_ret);
    char* name = malloc(param_value_size_ret);
    clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, param_value_size_ret, name, NULL);

    clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 0, NULL, &param_value_size_ret);
    char* vendor = malloc(param_value_size_ret);
    clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, param_value_size_ret, vendor, NULL);

    clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, 0, NULL, &param_value_size_ret);
    char* version = malloc(param_value_size_ret);
    clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, param_value_size_ret, version, NULL);
    
    clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, 0, NULL, &param_value_size_ret);
    char* extensions = malloc(param_value_size_ret);
    clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, param_value_size_ret, extensions, NULL);

    printf("Compiling in platform %s %s with %s.\nExtensions available: %s.\n", name, vendor, version, extensions);

    free(name);
    free(vendor);
    free(version);
    free(extensions);

    error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
    CHECK(error);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL,  &error);
    CHECK(error);

    uint8_t* file[2];
    size_t file_size[2];

    read_file(argv[1], &file[0], &file_size[0]);

    const char fine_raster_cl_path[] = FINE_RASTER_CL;

    read_file(fine_raster_cl_path, &file[1], &file_size[1]);

    cl_program program = clCreateProgramWithSource(context, 2, (const char**) file, file_size, &error);
    CHECK(error);

    const char default_options[] = "-DSHADER -D__COMPILER_RELATIVE_PATH__ -cl-kernel-arg-info -cl-nv-verbose -I" SHADERS_PATH " -I" SRC_PATH;

    char *options = NULL;
    
    size_t flags_init = 3;
    size_t size = sizeof(default_options);

    for(size_t flag = flags_init; flag < argc; ++flag)
    {
        size += strlen(argv[flag]) + 1;
    }

    options = (char*) malloc(size);
    strcpy(options, default_options);

    size_t offset = sizeof(default_options) - 1;
    
    for(size_t flag = flags_init; flag < argc; ++flag)
    {
        options[offset++] = ' ';
        strcpy(options + offset, argv[flag]);
        offset += strlen(argv[flag]) + 1;
    }

    options[offset] = '\0';

    printf("Compiling with options: %s\n", options);

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

    size_t result = 0;
    size_t size_result;

    error = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(result), &result, &size_result);
    CHECK(error);

    // printf("Error at building OpenCL binary: %ld | %lx\n.", result, result);

    error = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &size_result);
    CHECK(error);

    char* log = (char*) malloc(size_result);

    error = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, size_result, log, &size_result);
    CHECK(error);

    printf("%s\n", log);
    free(log);
    if (result != CL_BUILD_SUCCESS) {
        exit(1);
    }

    size_t returned_size;
    
    size_t kernel_num;
    error = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS, sizeof(kernel_num), &kernel_num, &returned_size);
    CHECK(error);
    
    if (kernel_num == 0) {
        printf("No kernel on detected on program.\n");
        exit(-1);
    }

    uint8_t* binary;
    size_t binary_size;

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
    free(binary);
}