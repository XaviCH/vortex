#include <compiler/common.h>

#define CL_TARGET_OPENCL_VERSION 120

#include <CL/opencl.h>

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

    // choose platform
    const cl_uint num_platform_entries = DEVICE_PLATFORM_ID + 1;
    
    cl_uint num_platforms;
    cl_int platform_error;
    cl_platform_id platform_entry_ids[num_platform_entries];
    platform_error = clGetPlatformIDs(num_platform_entries, platform_entry_ids, &num_platforms);
    
    if (platform_error) 
    {
        switch (platform_error)
        {
        case CL_PLATFORM_NOT_FOUND_KHR:
            fprintf(stderr, "No platform found. CL_PLATFORM_NOT_FOUND_KHR %d.\n", platform_error);
        default:
            fprintf(stderr, "Unexepected error %d.\n", platform_error);
        }
        exit(platform_error);
    }

    platform_id = platform_entry_ids[DEVICE_PLATFORM_ID];

    cl_platform_id* platform_ids = NULL;
    if (num_platforms > 1)
        platform_ids = malloc(num_platforms*sizeof(cl_platform_id));
    else
        platform_ids = &platform_id;

    printf("Platforms Available.\n");
    for (cl_uint platform = 0; platform < num_platforms; ++platform) 
    {
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

    // Choose device
    const cl_uint num_device_entries = DEVICE_DEVICE_ID + 1;
    
    cl_device_id device_entry_ids[num_device_entries];
    error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_device_entries, device_entry_ids, NULL);
    CHECK(error);

    device_id = device_entry_ids[DEVICE_DEVICE_ID];

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
    
    /*else {
        error = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &size_result);
        CHECK(error);
        char* log = (char*) malloc(size_result);

        error = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, size_result, log, &size_result);
        CHECK(error);

        printf("Build log:\n\t%s\n", log);
        free(log);
    }*/

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