// TODO: better way to define it?? reference glsc2 header file
#define GL_BYTE                           0x1400
#define GL_UNSIGNED_BYTE                  0x1401
#define GL_SHORT                          0x1402
#define GL_UNSIGNED_SHORT                 0x1403
#define GL_INT                            0x1404
#define GL_UNSIGNED_INT                   0x1405
#define GL_FLOAT                          0x1406


void byte_write(
    int size,
    unsigned char normalized,
    constant char *data_in,
	global float4 *data_out
) {
    if (normalized) {
        data_out->x = max((float) data_in[0] / 0x7Fu, -1.f);
        data_out->y = size > 1 ? max((float) data_in[1] / 0x7Fu, -1.0f) : 0.f;
        data_out->z = size > 2 ? max((float) data_in[2] / 0x7Fu, -1.0f) : 0.f;
        data_out->w = size > 3 ? max((float) data_in[3] / 0x7Fu, -1.0f) : 1.f;
        return;
    }
    data_out->x = (float) data_in[0];
    data_out->y = size > 1 ? (float) data_in[1] : 0.f;
    data_out->z = size > 2 ? (float) data_in[2] : 0.f;
    data_out->w = size > 3 ? (float) data_in[3] : 1.f;
}

void ubyte_write(
    int size,
    unsigned char normalized,
    constant unsigned char *data_in,
	global float4 *data_out
) {
    if (normalized) {
        data_out->x = (float) data_in[0] / 0xFFu;
        data_out->y = size > 1 ? (float) data_in[1] / 0xFFu : 0.f;
        data_out->z = size > 2 ? (float) data_in[2] / 0xFFu : 0.f;
        data_out->w = size > 3 ? (float) data_in[3] / 0xFFu : 1.f;
        return;
    }
    data_out->x = (float) data_in[0];
    data_out->y = size > 1 ? (float) data_in[1] : 0.f;
    data_out->z = size > 2 ? (float) data_in[2] : 0.f;
    data_out->w = size > 3 ? (float) data_in[3] : 1.f;
}

void short_write(
    int size,
    unsigned char normalized,
    constant short *data_in,
	global float4 *data_out
) {
    if (normalized) {
        data_out->x = max((float) data_in[0] / 0x7FFFu, -1.f);
        data_out->y = size > 1 ? max((float) data_in[1] / 0x7FFFu, -1.0f) : 0.f;
        data_out->z = size > 2 ? max((float) data_in[2] / 0x7FFFu, -1.0f) : 0.f;
        data_out->w = size > 3 ? max((float) data_in[3] / 0x7FFFu, -1.0f) : 1.f;
        return;
    }
    data_out->x = (float) data_in[0];
    data_out->y = size > 1 ? (float) data_in[1] : 0.f;
    data_out->z = size > 2 ? (float) data_in[2] : 0.f;
    data_out->w = size > 3 ? (float) data_in[3] : 1.f;
}

void ushort_write(
    int size,
    unsigned char normalized,
    constant unsigned short *data_in,
	global float4 *data_out
) {
    if (normalized) {
        data_out->x = (float) data_in[0] / 0xFFFFu;
        data_out->y = size > 1 ? (float) data_in[1] / 0xFFFFu : 0.f;
        data_out->z = size > 2 ? (float) data_in[2] / 0xFFFFu : 0.f;
        data_out->w = size > 3 ? (float) data_in[3] / 0xFFFFu : 1.f;
        return;
    }
    data_out->x = (float) data_in[0];
    data_out->y = size > 1 ? (float) data_in[1] : 0.f;
    data_out->z = size > 2 ? (float) data_in[2] : 0.f;
    data_out->w = size > 3 ? (float) data_in[3] : 1.f;
}


void int_write(
    int size,
    unsigned char normalized,
    constant int *data_in,
	global float4 *data_out
) {
    if (normalized) {
        data_out->x = max((float) data_in[0] / 0x7FFFFFFFu, -1.f);
        data_out->y = size > 1 ? max((float) data_in[1] / 0x7FFFFFFFu, -1.0f) : 0.f;
        data_out->z = size > 2 ? max((float) data_in[2] / 0x7FFFFFFFu, -1.0f) : 0.f;
        data_out->w = size > 3 ? max((float) data_in[3] / 0x7FFFFFFFu, -1.0f) : 1.f;
        return;
    }
    data_out->x = (float) data_in[0];
    data_out->y = size > 1 ? (float) data_in[1] : 0.f;
    data_out->z = size > 2 ? (float) data_in[2] : 0.f;
    data_out->w = size > 3 ? (float) data_in[3] : 1.f;
}

void uint_write(
    int size,
    unsigned char normalized,
    constant unsigned int *data_in,
	global float4 *data_out
) {
    if (normalized) {
        data_out->x = (float) data_in[0] / 0xFFFFFFFFu;
        data_out->y = size > 1 ? (float) data_in[1] / 0xFFFFFFFFu : 0.f;
        data_out->z = size > 2 ? (float) data_in[2] / 0xFFFFFFFFu : 0.f;
        data_out->w = size > 3 ? (float) data_in[3] / 0xFFFFFFFFu : 1.f;
        return;
    }
    data_out->x = (float) data_in[0];
    data_out->y = size > 1 ? (float) data_in[1] : 0.f;
    data_out->z = size > 2 ? (float) data_in[2] : 0.f;
    data_out->w = size > 3 ? (float) data_in[3] : 1.f;
}

void float_write(
    int size,
    constant float *data_in,
	global float4 *data_out
) {
    data_out->x = data_in[0];
    data_out->y = size > 1 ? data_in[1] : 0.f;
    data_out->z = size > 2 ? data_in[2] : 0.f;
    data_out->w = size > 3 ? data_in[3] : 1.f;
}

kernel void gl_strided_write(
    int size,
    int type,
    unsigned char normalized,
    int stride,
	constant void *buff_in,
	global float4 *buff_out
) {
    int gid = get_global_id(0);

    int slice;
    switch (type) {
        case GL_BYTE:
        case GL_UNSIGNED_BYTE:
            slice = 1; break;
        case GL_SHORT:
        case GL_UNSIGNED_SHORT: 
            slice = 2; break;
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_FLOAT:
            slice = 4; break;
    };

    constant void *data_in = buff_in + gid*(size*slice+stride);
    global float4 *data_out = buff_out + gid;

    if (type == GL_BYTE) byte_write(size, normalized, data_in, data_out);
    else if (type == GL_UNSIGNED_BYTE) ubyte_write(size, normalized, data_in, data_out);
    else if (type == GL_SHORT) short_write(size, normalized, data_in, data_out);
    else if (type == GL_UNSIGNED_SHORT) ushort_write(size, normalized, data_in, data_out);
    else if (type == GL_INT) int_write(size, normalized, data_in, data_out);
    else if (type == GL_UNSIGNED_INT) uint_write(size, normalized, data_in, data_out);
    else float_write(size, data_in, data_out);

}