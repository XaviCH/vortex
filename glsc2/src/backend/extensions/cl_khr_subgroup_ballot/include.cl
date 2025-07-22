#ifndef BACKEND_EXTENSIONS_CL_KHR_SUBGROUP_BALLOT
#define BACKEND_EXTENSIONS_CL_KHR_SUBGROUP_BALLOT

#ifdef cl_khr_subgroup_ballot
    #pragma OPENCL EXTENSION cl_khr_subgroup_ballot : enable
#else
    #ifdef __COMPILER_RELATIVE_PATH__
    #include <backend/extensions/cl_khr_subgroup_ballot/nv-sm_30+.cl>
    #else
    #include "glsc2/src/backend/extensions/cl_khr_subgroups/nv-sm_80+.cl"
    #endif
#endif



#endif