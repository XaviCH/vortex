#ifdef __COMPILER_RELATIVE_PATH__
#include "../constants.device.h"
#else
#include "glsc2/src/constants.device.h"
#endif


inline uint get_clear_color(const ulong c_clear_write_values) {
    return (uint)(c_clear_write_values & 0xFFFFFFFFu);
}

inline ushort get_clear_depth(const ulong c_clear_write_values) {
    return (ushort)((c_clear_write_values >> 32) & 0xFFFFu);
}

inline uchar get_clear_stencil(const ulong c_clear_write_values) {
    return (uchar)((c_clear_write_values >> 48) & 0xFFu);
}

inline uchar get_clear_stencil_mask(const ushort c_clear_enabled_data) {
    return (uchar)(c_clear_enabled_data & 0xFFu);
} 

uint clear_color(uint color, const ulong c_clear_write_values, const ushort c_clear_enabled_data) {
    uint clear_color = get_clear_color(c_clear_write_values);

    uint clear_mask = 
        ((c_clear_enabled_data & CLEAR_ENABLED_COLOR_CHANNEL_RED) != 0 ? 0xFFu <<  0 : 0) |
        ((c_clear_enabled_data & CLEAR_ENABLED_COLOR_CHANNEL_RED) != 0 ? 0xFFu <<  8 : 0) |
        ((c_clear_enabled_data & CLEAR_ENABLED_COLOR_CHANNEL_RED) != 0 ? 0xFFu << 16 : 0) |
        ((c_clear_enabled_data & CLEAR_ENABLED_COLOR_CHANNEL_RED) != 0 ? 0xFFu << 24 : 0) ;
    
    return (color & ~clear_mask) | (clear_color & clear_mask);
}

ushort clear_depth(const ulong c_clear_write_values, const ushort c_clear_enabled_data) {
    return get_clear_depth(c_clear_write_values);
}

uchar clear_stencil(uchar stencil, const ulong c_clear_write_values, const ushort c_clear_enabled_data) {
    uchar clear_stencil = get_clear_stencil(c_clear_write_values);

    uchar clear_mask = get_clear_stencil_mask(c_clear_enabled_data);
    
    return (stencil & ~clear_mask) | (clear_stencil & clear_mask);
}