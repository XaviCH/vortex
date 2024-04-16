#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h> 
#include <string.h>
#include <chrono>
#include "GLSC2/binary.c"
#include "test/test.h"

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
   })


#define TEST(func)                                                     \
  ({                                                                   \
    char function_name[] = #func;                                      \
    printf("Running %s.\n",function_name);                             \
    uint32_t result = func();                                          \
    if (!result) printf("PASSED.\n");                                  \
    else printf("FAILED with %d errors.\n", result);                   \
    errors += result;                                                  \
  })


static bool almost_equal(float a, float b, int ulp = 4) {
  union fi_t { int i; float f; };
  fi_t fa, fb;
  fa.f = a;
  fb.f = b;
  return std::abs(fa.i - fb.i) <= ulp;
}


// TESTS
int test_color_kernel();
int test_color_kernel_discard_true();

cl_device_id device_id = NULL;
cl_context context = NULL;

int main (int argc, char **argv) {
  
  cl_platform_id platform_id;
  
  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  uint32_t errors = 0;
  TEST(test_perspective_div);
  // TEST(test_color_kernel);
  // TEST(test_color_kernel_discard_true);

  // CLEANUP
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  printf("Total errors %d.\n", errors);
  return errors;
}