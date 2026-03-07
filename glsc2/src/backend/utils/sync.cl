/**
 * local_ functions may requiere all threads from a local group to enter
 * sub_group_ functions may requiere all threads from a sub group to enter
 *
 *
 */

#ifndef BACKEND_UTILS_SYNC_CL
#define BACKEND_UTILS_SYNC_CL

#ifdef __COMPILER_RELATIVE_PATH__
    #include <backend/utils/sub_group_mask.cl>
    #include <backend/utils/common.cl>

    #include <backend/extensions/cl_khr_subgroups/include.cl>
    #include <backend/extensions/cl_khr_subgroup_ballot/include.cl>
    #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
    #endif

#else
    #include "glsc2/src/backend/utils/sub_group_mask.cl"
    #include "glsc2/src/backend/utils/common.cl"
#endif

#ifdef DEVICE_SUB_GROUP_RAW_ENABLED

inline uint __attribute__((overloadable)) sub_group_scan_inclusive_add(uint value, local volatile uint* sg_temp) {
    local volatile uint* ptr = &sg_temp[get_sub_group_local_id()];
    *ptr = value;
    #pragma unroll
    for(int target=1; target < get_sub_group_size(); target *= 2) {
        if (get_sub_group_local_id() >= target) {
            value += ptr[-target];    
            *ptr = value;
        }
    }
    return value;
}

inline uint __attribute__((overloadable)) sub_group_scan_inclusive_min(uint value, local volatile uint* sg_temp) {
    local volatile uint* ptr = &sg_temp[get_sub_group_local_id()];
    *ptr = value;
    #pragma unroll
    for(int target=1; target < get_sub_group_size(); target *= 2) {
        if (get_sub_group_local_id() >= target) {
            value = min(value, ptr[-target]);    
            *ptr = value;
        }
    }
    return value;
}

inline uint __attribute__((overloadable)) sub_group_reduce_min(uint value, local volatile uint* l_temp) {
    sub_group_scan_inclusive_min(value, l_temp);
    return l_temp[get_local_linear_size()-1];
}

#endif

#ifdef DEVICE_SUB_GROUP_ENABLED
inline uint __attribute__((overloadable)) local_1dim_scan_inclusive_add(uint value, local volatile uint (*sg_temp)[DEVICE_SUB_GROUP_THREADS]) {
    uint result;

    #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
    {
        result = sub_group_scan_inclusive_add(value);
    }
    #else
    {
        uint local_id = get_sub_group_local_id();
        (*sg_temp)[local_id] = value;

        #pragma unroll
        for(int target=1; target < get_sub_group_size(); target *= 2) {
            #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
                barrier(CLK_LOCAL_MEM_FENCE);
            #endif
            if (local_id >= target) {
                value += (*sg_temp)[local_id-target];
                (*sg_temp)[local_id] = value;
            }
        }

        result = value;
    }
    #endif

    return result;
}
#endif

inline uint __attribute__((overloadable)) local_1dim_scan_inclusive_add(uint value, local volatile uint* l_temp) {
    #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
    {
        return sub_group_scan_inclusive_add(value);
    }
    #else
    {
        uint result;

        local volatile uint* ptr = &l_temp[get_local_linear_id()];
        *ptr = value;

        #pragma unroll
        for(int target=1; target < get_local_size(0); target *= 2) {
            #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
                barrier(CLK_LOCAL_MEM_FENCE);
            #endif
            if (get_local_id(0) >= target) {
                value += ptr[-target];
                *ptr = value;
            }
        }
        result = value;

        return result;
    }
    #endif
}

static inline uint __attribute__((overloadable)) local_scan_inclusive_add(uint value, local volatile uint* l_temp) {
    uint result;

    uint id = get_local_linear_id();
    local volatile uint* ptr = &l_temp[id];

    {
        *ptr = value;

        #pragma unroll
        for(uint i=1; i<get_local_size(0); i=i*2) {
            #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
                barrier(CLK_LOCAL_MEM_FENCE);
            #endif
            if (id >= i) {
                value += ptr[-i];
                *ptr = value;
            }
        }

        #pragma unroll
        for(uint i=get_local_size(0); i<get_local_linear_size(); i=i*2) {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (id >= i) {
                value += ptr[-i];
                *ptr = value;
            }
        }
    }
    // #endif

    return value;
}

/*
static inline uint __attribute__((overloadable)) local_scan_inclusive_add(uint value, local volatile uint* l_temp) {
    uint local_id = get_local_linear_id();
    local volatile uint* ptr = &l_temp[local_id];
    *ptr = value;
    #pragma unroll
    for(int i=1; i<get_local_linear_size(); i=i*2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id >= i) {
            value += ptr[-i];    
            *ptr = value;
        }
    }
    return value;
}
*/


inline uint __attribute__((overloadable)) local_scan_inclusive_min(uint value, local volatile uint* l_temp) {
    uint local_id = get_local_linear_id();
    local volatile uint* ptr = &l_temp[local_id];
    *ptr = value;
    #pragma unroll
    for(int i=1; i<get_local_linear_size(); i=i*2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id >= i) {
            value = min(value, ptr[-i]);    
            *ptr = value;
        }
    }
    return value;
}

/** 
    PRE: All threads active
    POST: Only threads with get_local_linear_id < threads has correct answer, others could have unexpected values
inline uint local_sized_reduce_min_sized_ui(uint value, local volatile uint* l_temp, uint threads) {
    uint id, num_ids, tmp;
    local volatile uint* ptr;

    #if DEVICE_SUB_GROUP_INTRINSICTS_SUPPORT == 1
    uint sub_group_min = sub_group_reduce_min_ui(get_local_linear_id() < threads ? value : UINT_MAX);
    if (threads <= get_sub_group_size()) return sub_group_min; 
    id      = get_sub_group_id();
    num_ids = get_num_sub_groups();
    #else

    #endif

    #pragma unroll
    for(uint i=1; i<num_ids; i=i*2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id >= i) {
            value += ptr[-i];
            *ptr = value;
        }
    }
}
*/

/**
    PRE: All threads are active when function is called.
 */
/*
inline uint local_reduce_min_ui(uint value, local volatile uint* l_temp) {
    uint result;

    #if DEVICE_SUB_GROUP_INTRINSICTS_SUPPORT == 1
    {
        uint sub_group_min = sub_group_reduce_min_ui(value);

        if (get_num_sub_groups() > 1) {
            l_temp[get_sub_group_local_id()] = UINT_MAX;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        #pragma unroll
        for(uint i=1; i<get_num_sub_groups(); i=i*get_sub_group_size()) {
            l_temp[get_sub_group_id()] = sub_group_min;
            barrier(CLK_LOCAL_MEM_FENCE);
            sub_group_min = sub_group_reduce_min_ui(l_temp[get_sub_group_local_id()]);
        }

        result = sub_group_min;
    }
    #else 
    {
        local_scan_inclusive_min_ui(value, l_temp);
        barrier(CLK_LOCAL_MEM_FENCE);
        result = l_temp[get_local_linear_size()-1];
    }

    #endif

    return result;
}
*/


inline uint __attribute__((overloadable)) local_1dim_broadcast(uint value, uint id, local volatile uint (*sg_temp)[DEVICE_SUB_GROUP_THREADS]) {
    uint result;

    #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
    {
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        result = sub_group_broadcast(value, id);
    }
    #else
    {
        (*sg_temp)[get_local_id(0)] = value;
        #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
            barrier(CLK_LOCAL_MEM_FENCE);
        #endif
        result = (*sg_temp)[0];
    }
    #endif

    return result;
}

inline uint __attribute__((overloadable)) local_reduce_min(uint value, local volatile uint* l_temp) {
    local_scan_inclusive_min(value, l_temp);
    barrier(CLK_LOCAL_MEM_FENCE);
    return l_temp[get_local_linear_size()-1];
}

inline uint local_scan_inclusive_and_2dim_ui(uint value, local volatile uint* l_temp) {
    uint local_id = get_local_linear_id();
    local volatile uint* ptr = &l_temp[local_id];
    *ptr = value;
    #pragma unroll
    for(int i=get_local_size(0); i<get_local_linear_size(); i=i*2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id >= i) {
            value = value & ptr[-i];    
            *ptr = value;
        }
    }
    return value;
}

inline uint local_reduce_and_2dim_ui(uint value, local volatile uint* l_temp) {
    local_scan_inclusive_and_2dim_ui(value, l_temp);
    barrier(CLK_LOCAL_MEM_FENCE);
    return l_temp[get_local_linear_size() - get_local_size(0) + get_local_id(0)];
}

#ifdef DEVICE_SUB_GROUP_ENABLED
inline uint __attribute__((overloadable)) local_1dim_scan_inclusive_or(uint value, local volatile uint (*sg_temp)[DEVICE_SUB_GROUP_THREADS]) {
    uint local_id = get_sub_group_local_id();
    local volatile uint* ptr = &(*sg_temp)[local_id];
    *ptr = value;
    #pragma unroll
    for(int i=1; i<get_sub_group_size(); i=i*2) {
        #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
            barrier(CLK_LOCAL_MEM_FENCE);
        #endif
        if (local_id >= i) {
            value = value | ptr[-i];    
            *ptr = value;
        }
    }
    return value;
}
#endif

inline uint __attribute__((overloadable)) local_1dim_scan_inclusive_or(uint value, local volatile uint* l_temp) {
    uint local_id = get_local_id(0);
    local volatile uint* ptr = &l_temp[get_local_linear_id()];
    *ptr = value;
    #pragma unroll
    for(int i=1; i<get_local_size(0); i=i*2) {
        #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
        barrier(CLK_LOCAL_MEM_FENCE);
        #endif
        if (local_id >= i) {
            value = value | ptr[-i];    
            *ptr = value;
        }
    }
    return value;
}

#ifdef DEVICE_SUB_GROUP_ENABLED
inline uint __attribute__((overloadable)) local_1dim_reduce_or(uint value, local volatile uint (*sg_temp)[DEVICE_SUB_GROUP_THREADS]) {

    local_1dim_scan_inclusive_or(value, sg_temp);
    
    #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
        barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    return (*sg_temp)[get_sub_group_size()-1];
}
#endif

inline uint __attribute__((overloadable)) local_1dim_reduce_or(uint value, local volatile uint* l_temp) {

    local_1dim_scan_inclusive_or(value, l_temp);
    
    #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
        barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    return l_temp[get_local_linear_id() - get_local_id(0) + get_local_size(0) - 1];
}

/**
 * @brief Ensure memory syncronization at least for 1st dimension threads on work group.
 */
inline void local_1dim_barrier(cl_mem_fence_flags flags) {
    #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
        #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
            sub_group_barrier(flags);
        #else
            barrier(flags);
        #endif
    #endif
}

// TODO: wrap all sync function using this generics
static inline uint __attribute__((overloadable)) local_1dim_scan_inclusive(void (func)(uint*,uint), uint value, local volatile uint* l_temp) {
    uint local_id = get_local_linear_id();
    local volatile uint* ptr = &l_temp[local_id];
    
    *ptr = value;

    #pragma unroll
    for(uint i=1; i<get_local_size(0); i=i*2) {
        local_1dim_barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id >= i) {
            func(&value, ptr[-i]);
            *ptr = value;
        }
    }

    return value;
}

static inline uint __attribute__((overloadable)) local_scan_inclusive(void (func)(uint*,uint), uint value, local volatile uint* l_temp) {
    uint local_id = get_local_linear_id();
    local volatile uint* ptr = &l_temp[local_id];
    
    local_1dim_scan_inclusive(func, value, l_temp);

    #pragma unroll
    for(uint i=get_local_size(0); i<get_local_linear_size(); i=i*2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id >= i) {
            func(&value, ptr[-i]);
            *ptr = value;
        }
    }

    return value;
}

/**
    PRE: 
        For DEVICE_SUB_GROUP_INTRINSICTS_SUPPORT == 1 is not required that all threads were active. 
        Otherwise all threads in work group must be active. 
 */
#ifdef DEVICE_SUB_GROUP_ENABLED
inline sub_group_mask_t __attribute__((overloadable)) local_1dim_ballot(bool value, local volatile sub_group_mask_t (*sg_temp)[DEVICE_SUB_GROUP_THREADS]) {
    sub_group_mask_t mask;

    #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        mask = ballot_sub_group_mask(value);
    #else
    {
        sub_group_mask_t tmp;
        clear_sub_group_mask(&tmp);
        if (value) set_bit_sub_group_mask(&tmp, get_sub_group_local_id());

        clear_sub_group_mask(&(*sg_temp)[0]);
        atomic_or_sub_group_mask(&(*sg_temp)[0], tmp);
        #ifndef DEVICE_SUB_GROUP_RAW_ENABLED
            barrier(CLK_LOCAL_MEM_FENCE);
        #endif
        mask = (*sg_temp)[0];
        // TODO: 
        // Implement it without atomics could add speed up.
        /*
        #if DEVICE_SUB_GROUP_THREADS <= 8
            local volatile uchar (*ptr)[DEVICE_SUB_GROUP_THREADS] = (local volatile uchar (*)[DEVICE_SUB_GROUP_THREADS]) sg_temp;
        #elif DEVICE_SUB_GROUP_THREADS <= 16
            local volatile ushort (*ptr)[DEVICE_SUB_GROUP_THREADS] = (local volatile ushort (*)[DEVICE_SUB_GROUP_THREADS]) sg_temp;
        #elif DEVICE_SUB_GROUP_THREADS <= 32
            local volatile uint (*ptr)[DEVICE_SUB_GROUP_THREADS] = (local volatile uint (*)[DEVICE_SUB_GROUP_THREADS]) sg_temp;
        #elif DEVICE_SUB_GROUP_THREADS <= 64
            local volatile ulong (*ptr)[DEVICE_SUB_GROUP_THREADS] = (local volatile ulong (*)[DEVICE_SUB_GROUP_THREADS]) sg_temp;
        #elif DEVICE_SUB_GROUP_THREADS <= 128
            local volatile uint4 (*ptr)[DEVICE_SUB_GROUP_THREADS] = (local volatile uint4 (*)[DEVICE_SUB_GROUP_THREADS]) sg_temp;
        #else
            #error Not supported size
        #endif

        mask.mask = local_1dim_reduce_or(tmp.mask, ptr);
        */
    }
    #endif
    
    return mask;
}

inline sub_group_mask_t __attribute__((overloadable)) local_1dim_ballot(bool value, local volatile sub_group_mask_t* l_temp) {
    sub_group_mask_t mask;

    #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        mask = ballot_sub_group_mask(value);
    #else
    {
        sub_group_mask_t tmp;
        clear_sub_group_mask(&tmp);
        if (value) set_bit_sub_group_mask(&tmp, get_local_id(0));
        mask.mask = local_1dim_reduce_or(tmp.mask, (local volatile uint*) l_temp);
    }
    #endif
    
    return mask;
}
#endif

inline uint local_scan_inclusive_max_1dim_ui(uint value, local volatile uint* l_temp) {
    uint local_id = get_local_id(0);
    local volatile uint* ptr = &l_temp[get_local_linear_id()];
    *ptr = value;
    #pragma unroll
    for(int i=1; i<get_local_size(0); i=i*2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id >= i) {
            value = max(value, ptr[-i]);    
            *ptr = value;
        }
    }
    return value;
}

inline uint local_1dim_scan_inclusive_min_ui(uint value, local volatile uint* l_temp) {
    uint local_id = get_local_id(0);
    local volatile uint* ptr = &l_temp[get_local_linear_id()];
    *ptr = value;
    #pragma unroll
    for(int i=1; i<get_local_size(0); i=i*2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id >= i) {
            value = max(value, ptr[-i]);    
            *ptr = value;
        }
    }
    return value;
}

inline uint local_reduce_max_1dim_ui(uint value, local volatile uint* l_temp) {
    local_scan_inclusive_max_1dim_ui(value, l_temp);
    barrier(CLK_LOCAL_MEM_FENCE);
    return l_temp[get_local_linear_id() - get_local_id(0) + get_local_size(0) - 1];
}

inline uint local_1dim_reduce_min_ui(uint value, local volatile uint* l_temp) {
    local_1dim_scan_inclusive_min_ui(value, l_temp);
    barrier(CLK_LOCAL_MEM_FENCE);
    return l_temp[get_local_linear_id() - get_local_id(0) + get_local_size(0) - 1];
}

inline uint local_scan_inclusive_or_ui(uint value, local volatile uint* l_temp) {
    uint local_id = get_local_linear_id();
    local volatile uint* ptr = &l_temp[local_id];
    *ptr = value;
    #pragma unroll
    for(int i=1; i<get_local_linear_size(); i=i*2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id >= i) {
            value = value | ptr[-i];    
            *ptr = value;
        }
    }
    return value;
}

inline uint local_reduce_or_ui(uint value, local volatile uint* l_temp) {
    local_scan_inclusive_or_ui(value, l_temp);
    barrier(CLK_LOCAL_MEM_FENCE);
    return l_temp[get_local_linear_size()-1];
}

inline uint local_scan_inclusive_and_ui(uint value, local volatile uint* l_temp) {
    uint local_id = get_local_linear_id();
    local volatile uint* ptr = &l_temp[local_id];
    *ptr = value;
    #pragma unroll
    for(int i=1; i<get_local_linear_size(); i=i*2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id >= i) {
            value &= ptr[-i];    
            *ptr = value;
        }
    }
    return value;
}

inline uint local_reduce_and_ui(uint value, local volatile uint* l_temp) {
    local_scan_inclusive_and_ui(value, l_temp);
    barrier(CLK_LOCAL_MEM_FENCE);
    return l_temp[get_local_linear_size()-1];
}

#endif