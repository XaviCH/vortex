
inline uint __attribute__((overloadable)) sub_group_scan_inclusive_add (uint x) {
    uint r;
    asm volatile(
        "{"
        ".reg .pred p;"
        ".reg .b32 dst;"
        "mov.b32 %0, %1;"
        "shfl.sync.up.b32  dst|p, %0, 0x1, 0x0, 0xffffffff;"
        "@p add.u32        %0, dst, %0;"
        "shfl.sync.up.b32  dst|p, %0, 0x2, 0x0, 0xffffffff;"
        "@p add.u32        %0, dst, %0;"
        "shfl.sync.up.b32  dst|p, %0, 0x4, 0x0, 0xffffffff;"
        "@p add.u32        %0, dst, %0;"
        "shfl.sync.up.b32  dst|p, %0, 0x8, 0x0, 0xffffffff;"
        "@p add.u32        %0, dst, %0;"
        "shfl.sync.up.b32  dst|p, %0, 0x10,0x0, 0xffffffff;"
        "@p add.u32        %0, dst, %0;"
        "}"
        : "=r"(r) : "r"(x));
    return r;
}

inline uint __attribute__((overloadable)) sub_group_scan_inclusive_max (uint x) {
    uint r;
    asm volatile(
        "{"
        ".reg .pred p;"
        ".reg .b32 dst;"
        "mov.b32 %0, %1;"
        "shfl.sync.up.b32  dst|p, %0, 0x1, 0x0, 0xffffffff;"
        "@p max.u32        %0, dst, %0;"
        "shfl.sync.up.b32  dst|p, %0, 0x2, 0x0, 0xffffffff;"
        "@p max.u32        %0, dst, %0;"
        "shfl.sync.up.b32  dst|p, %0, 0x4, 0x0, 0xffffffff;"
        "@p max.u32        %0, dst, %0;"
        "shfl.sync.up.b32  dst|p, %0, 0x8, 0x0, 0xffffffff;"
        "@p max.u32        %0, dst, %0;"
        "shfl.sync.up.b32  dst|p, %0, 0x10,0x0, 0xffffffff;"
        "@p max.u32        %0, dst, %0;"
        "}"
        : "=r"(r) : "r"(x));
    return r;
}

inline uint __attribute__((overloadable)) sub_group_scan_inclusive_min (uint x) {
    uint r;
    asm volatile(
        "{"
        ".reg .pred p;"
        ".reg .b32 dst;"
        "mov.b32 %0, %1;"
        "shfl.sync.up.b32  dst|p, %0, 0x1, 0x0, 0xffffffff;"
        "@p min.u32        %0, dst, %0;"
        "shfl.sync.up.b32  dst|p, %0, 0x2, 0x0, 0xffffffff;"
        "@p min.u32        %0, dst, %0;"
        "shfl.sync.up.b32  dst|p, %0, 0x4, 0x0, 0xffffffff;"
        "@p min.u32        %0, dst, %0;"
        "shfl.sync.up.b32  dst|p, %0, 0x8, 0x0, 0xffffffff;"
        "@p min.u32        %0, dst, %0;"
        "shfl.sync.up.b32  dst|p, %0, 0x10,0x0, 0xffffffff;"
        "@p min.u32        %0, dst, %0;"
        "}"
        : "=r"(r) : "r"(x));
    return r;
}
