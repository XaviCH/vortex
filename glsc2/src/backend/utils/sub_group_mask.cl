#ifndef BACKEND_UTILS_SUB_GROUP_MASK_CL
#define BACKEND_UTILS_SUB_GROUP_MASK_CL

#ifdef __COMPILER_RELATIVE_PATH__
    #include <backend/types.cl>
    #include <backend/utils/common.cl>


    #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
    #include <backend/extensions/cl_khr_subgroup_ballot/include.cl>
    #include <backend/extensions/cl_khr_subgroups/include.cl>
    #endif

#else
    #include "glsc2/src/backend/types.cl"
    #include "glsc2/src/backend/utils/common.cl"

    #ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
    #include "glsc2/src/backend/extensions/cl_khr_subgroup_ballot/include.cl"
    #include "glsc2/src/backend/extensions/cl_khr_subgroups/include.cl"
    #endif

#endif

inline void __attribute__((overloadable)) clear_sub_group_mask(sub_group_mask_t* sub_group_mask) {
    sub_group_mask->mask = 0;
}

inline void __attribute__((overloadable)) clear_sub_group_mask(local volatile sub_group_mask_t* sub_group_mask) {
    sub_group_mask->mask = 0;
}

inline sub_group_mask_t and_sub_group_mask(sub_group_mask_t a, sub_group_mask_t b) {
    sub_group_mask_t sub_group_mask;

    sub_group_mask.mask = a.mask & b.mask;

    return sub_group_mask;
}

inline sub_group_mask_t not_sub_group_mask(sub_group_mask_t mask) {
    sub_group_mask_t sub_group_mask;

    sub_group_mask.mask = ~mask.mask;

    return sub_group_mask;
}

inline bool all_sub_group_mask(sub_group_mask_t sub_group_mask) {
    return popcount(sub_group_mask.mask) == get_sub_group_size();
}

inline bool any_sub_group_mask(sub_group_mask_t sub_group_mask) {
    return popcount(sub_group_mask.mask) != 0;
}

inline sub_group_mask_t get_lane_sub_group_mask_lt() {
    sub_group_mask_t sub_group_mask;

    #if (DEVICE_SUB_GROUP_THREADS <= 64)
        sub_group_mask.mask = (1ul << get_sub_group_local_id()) - 1; 
    #else
    {
        ulong* pointer = &sub_group_mask.mask;
        pointer[0] = get_sub_group_local_id() > 63 ? ULONG_MAX : (1ul << get_sub_group_local_id()) - 1;
        pointer[1] = get_sub_group_local_id() > 63 ? (1ul << (get_sub_group_local_id() - 64)) - 1 : 0;
    }
    #endif

    return sub_group_mask;
}

inline sub_group_mask_t get_lane_sub_group_mask_le() {
    sub_group_mask_t sub_group_mask;

    #if (DEVICE_SUB_GROUP_THREADS <= 64)
        sub_group_mask.mask = (2ul << get_sub_group_local_id()) - 1; 
    #else
    {
        ulong* pointer = &sub_group_mask.mask;
        pointer[0] = get_sub_group_local_id() > 63 ? ULONG_MAX : (2ul << get_sub_group_local_id()) - 1;
        pointer[1] = get_sub_group_local_id() > 63 ? (2ul << (get_sub_group_local_id() - 64)) - 1 : 0;
    }
    #endif

    return sub_group_mask;
}

inline uint popcount_sub_group_mask(sub_group_mask_t sub_group_mask) {
    return popcount(sub_group_mask.mask);
}

#ifdef DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
inline sub_group_mask_t ballot_sub_group_mask(bool condition) {
    sub_group_mask_t sub_group_mask;

    #if (DEVICE_SUB_GROUP_THREADS <= 32)
        sub_group_mask.mask = sub_group_ballot(condition).x;
    #elif (DEVICE_SUB_GROUP_THREADS <= 64)
        sub_group_mask.mask = *((ulong*)(&sub_group_ballot(condition).xy));
    #else
        sub_group_mask.mask = sub_group_ballot(condition);
    #endif

    return sub_group_mask;
}
#endif

inline bool get_bit_sub_group_mask(sub_group_mask_t sub_group_mask, uint position) {
    bool bit;

    #if (DEVICE_SUB_GROUP_THREADS <= 64)
        bit = (sub_group_mask.mask & (1ul << position)) != 0;
    #else
    {
        ulong* pointer = (ulong*)(&sub_group_mask) + (position > 63);
        bit = (pointer->mask & (1ul << (position % 64))) != 0;
    }
    #endif

    return bit;
}

inline void set_bit_sub_group_mask(sub_group_mask_t* sub_group_mask, uint position) {
    #if (DEVICE_SUB_GROUP_THREADS <= 64)
        sub_group_mask->mask |= 1ul << position;
    #else
    {
        ulong* pointer = (ulong*)sub_group_mask + (position > 63);
        pointer->mask |= 1ul << (position % 64);
    }
    #endif
}

inline sub_group_mask_t atomic_or_sub_group_mask(local volatile sub_group_mask_t* address, const sub_group_mask_t mask) {
    sub_group_mask_t atomic_mask;

    #if (DEVICE_SUB_GROUP_THREADS <= 8)
        uint align_offset = (uint)value & 0x3u; 
        uint aligned_mask = mask.mask << align_offset*8;
        uint* aligned_address = (uint*)(value - align_offset);
        uint result = atomic_or(aligned_address, aligned_mask);
        atomic_mask.mask = result >> align_offset*8; 
    #elif (DEVICE_SUB_GROUP_THREADS <= 16)
        uint align_offset = ((uint)value >> 1) & 0x1u; 
        uint aligned_mask = mask.mask << align_offset*16;
        uint* aligned_address = (uint*)(value - align_offset);
        uint result = atomic_or(aligned_address, aligned_mask);
        atomic_mask.mask = result >> align_offset*16;
    #elif (DEVICE_SUB_GROUP_THREADS <= 64)
        atomic_mask.mask = atomic_or(&(address->mask), mask.mask);
    #else
        #error Unsupported atomics for sub group threads that large
    #endif

    return atomic_mask;
}


#endif