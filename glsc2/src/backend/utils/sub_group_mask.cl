#ifndef BACKEND_UTILS_SUB_GROUP_MASK_CL
#define BACKEND_UTILS_SUB_GROUP_MASK_CL

#ifdef __COMPILER_RELATIVE_PATH__
#include <backend/types.cl>
#else
#include "glsc2/src/backend/types.cl"
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

inline sub_group_mask_t get_lane_sub_group_mask_lt() {
    sub_group_mask_t sub_group_mask;

    #if (DEVICE_SUB_GROUP_THREADS <= 64)
        sub_group_mask.mask = (1ul << get_local_id(0)) - 1; 
    #else
        ulong* pointer = &sub_group_mask.mask;
        pointer[0] = get_local_id(0) > 63 ? ULONG_MAX : (1ul << get_local_id(0)) - 1;
        pointer[1] = get_local_id(0) > 63 ? (1ul << (get_local_id(0) - 64)) - 1 : 0;
    #endif

    return sub_group_mask;
}

inline uint popcount_sub_group_mask(sub_group_mask_t sub_group_mask) {
    return popcount(sub_group_mask.mask);
}

inline bool get_bit_sub_group_mask(sub_group_mask_t sub_group_mask, uint position) {
    #if (DEVICE_SUB_GROUP_THREADS <= 64)
        return (sub_group_mask.mask & (1ul << position)) != 0;
    #else
        ulong* pointer = (ulong*)(&sub_group_mask) + (position > 63);
        return (pointer->mask & (1ul << (position % 64))) != 0;
    #endif
}

inline void set_bit_sub_group_mask(sub_group_mask_t* sub_group_mask, uint position) {
    #if (DEVICE_SUB_GROUP_THREADS <= 64)
        sub_group_mask->mask |= 1ul << position;
    #else
        ulong* pointer = (ulong*)sub_group_mask + (position > 63);
        pointer->mask |= 1ul << (position % 64);
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