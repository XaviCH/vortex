#define GL_CW       0x0900
#define GL_CCW      0x0901

#define GL_FRONT                          0x0404
#define GL_BACK                           0x0405
#define GL_FRONT_AND_BACK                 0x0408

#define GL_TRIANGLES                      0x0004
#define GL_TRIANGLE_STRIP                 0x0005
#define GL_TRIANGLE_FAN                   0x0006

kernel void gl_rasterization_points(
    /* Vertex buffers */
    global const float4 *gl_Position,
    global const float *gl_PointSize,
    global const float4 *in_varying,
    /* Fragment buffers */
    global float4 *gl_FragCoord,
    global float4 *out_varying,
    global bool *gl_FrontFacing,
    global float2 *gl_PointCoord,
    global uchar *gl_Discard,
    /* Vertex shader data */
    const uint n_varying,
    /* Rasterization data */
    const uint vertex0,
    const uint primitive_id
    /* Rasterization config */
) {
    uint xid = get_global_id(0);
    uint yid = get_global_id(1);
    uint xsize = get_global_size(0);
    uint ysize = get_global_size(1);
    uint gid = xid+yid*xsize;
    uint gsize = xsize*ysize;

    float2 point = (float2){xid, yid} + 0.5f;
    uint primitive_index = vertex0 + primitive_id;

    float2 vertex = gl_Position[primitive_index].xy; 
    float size = gl_PointSize[primitive_index];

    if (distance(vertex, point) >= size) {
        gl_Discard[gid] = 1;
        return;
    }

    gl_FragCoord[gid].xy = point;
    gl_FrontFacing[gid] = 1;
    gl_PointCoord[gid].xy = (float2){
        0.5 + (xid + 0.5 - vertex.x) / size,
        0.5 - (yid + 0.5 - vertex.y) / size
    };
    
    for (uint varying = 0; varying < n_varying; ++varying) {
        uint vertex_subbuff_offset   = primitive_index + gsize * varying;
        uint fragment_subbuff_offset = gid + gsize * varying;

        out_varying[fragment_subbuff_offset] = in_varying[vertex_subbuff_offset];
    }

    gl_Discard[gid] = 0;
}
