#include <CL/opencl.h>

// MACROS FOR DEBUGGING

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


#define MAX_DEBUG_PRINT_BUFFER 256

#define DEBUG_REP_STR1(__X) __X
#define DEBUG_REP_STR2(__X) DEBUG_REP_STR1(__X) ", " __X
#define DEBUG_REP_STR3(__X) DEBUG_REP_STR2(__X) ", " __X  
#define DEBUG_REP_STR4(__X) DEBUG_REP_STR3(__X) ", " __X  

#define DEBUG_REP_ARRAY1(__X) __X[0]
#define DEBUG_REP_ARRAY2(__X) DEBUG_REP_ARRAY1(__X), __X[1]  
#define DEBUG_REP_ARRAY3(__X) DEBUG_REP_ARRAY2(__X), __X[2]  
#define DEBUG_REP_ARRAY4(__X) DEBUG_REP_ARRAY3(__X), __X[3]  

#define DEBUG

#ifdef DEBUG
#define PRINT_BUFFER_I(__BUFFER, __SIZE, __TYPE, __TIMES)                                                               \
  {                                                                                                                     \
    printf("Function %s at %s:%d: Print buffer %s\n", __func__, __FILE__, __LINE__, #__BUFFER);                         \
    __TYPE __buffer[__SIZE][__TIMES];                                                                                   \
    CL_CHECK(clEnqueueReadBuffer(getCommandQueue(), __BUFFER, CL_TRUE, 0, sizeof(__buffer), __buffer, 0, NULL, NULL));  \
    for(int __i=0; __i < __SIZE && __i < MAX_DEBUG_PRINT_BUFFER; ++__i) {                                               \
      printf("("DEBUG_REP_STR##__TIMES("%d")")\t" , DEBUG_REP_ARRAY##__TIMES(__buffer[__i]));                           \
    }                                                                                                                   \
    printf("\n");                                                                                                       \
  }
#else
#define PRINT_BUFFER_I(_BUFFER, _SIZE, _TYPE, __TYPE_SEQ)
#endif

#ifdef DEBUG
#define PRINT_BUFFER_F(__BUFFER, __SIZE, __TYPE, __TIMES)                                                               \
  {                                                                                                                     \
    printf("Function %s at %s:%d: Print buffer %s\n", __func__, __FILE__, __LINE__, #__BUFFER);                         \
    __TYPE __buffer[__SIZE][__TIMES];                                                                                   \
    CL_CHECK(clEnqueueReadBuffer(getCommandQueue(), __BUFFER, CL_TRUE, 0, sizeof(__buffer), __buffer, 0, NULL, NULL));  \
    for(int __i=0; __i < __SIZE && __i < MAX_DEBUG_PRINT_BUFFER; ++__i) {                                               \
      printf("("DEBUG_REP_STR##__TIMES("%f")")\t" , DEBUG_REP_ARRAY##__TIMES(__buffer[__i]));                           \
    }                                                                                                                   \
    printf("\n");                                                                                                       \
  }
#else
#define PRINT_BUFFER_I(_BUFFER, _SIZE, _TYPE, __TYPE_SEQ)
#endif