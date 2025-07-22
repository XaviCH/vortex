#ifndef BACKEND_EXTENSIONS_CL_KHR_SUBGROUPS
#define BACKEND_EXTENSIONS_CL_KHR_SUBGROUPS

#ifdef cl_khr_subgroups
    #pragma OPENCL EXTENSION cl_khr_subgroups : enable
#else
    #ifdef __opencl_c_subgroups
        #pragma OPENCL EXTENSION __opencl_c_subgroups : enable
    #else
        #ifdef __COMPILER_RELATIVE_PATH__
        #include <backend/extensions/cl_khr_subgroups/nv-sm_80+.cl>
        #else
        #include "glsc2/src/backend/extensions/cl_khr_subgroups/nv-sm_80+.cl"
        #endif
    #endif
#endif

#endif