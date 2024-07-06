kernel void bit16 (
    global ushort *buffer,
    private ushort color,
    private ushort mask
) {
    int gid = get_global_id(0);

    buffer[gid] = 0; // (buffer[gid] & mask) | (color & ~mask);
}

kernel void bit8 (
    global uchar *buffer,
    private uchar color,
    private uchar mask
) {
    int gid = get_global_id(0);

    buffer[gid] = (buffer[gid] & mask) | (color & ~mask);
}