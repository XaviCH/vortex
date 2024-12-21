#define GL_CW       0x0900
#define GL_CCW      0x0901

#define GL_FRONT                          0x0404
#define GL_BACK                           0x0405
#define GL_FRONT_AND_BACK                 0x0408

#define GL_TRIANGLES                      0x0004
#define GL_TRIANGLE_STRIP                 0x0005
#define GL_TRIANGLE_FAN                   0x0006

inline float cross2d(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}

inline float cross(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}

inline float3 get_baricentric_coords(float2 p, float4 v0, float4 v1, float4 v2) {
    float3 barycentricCoords;

    float areaABC = cross2d(v2.xy - v0.xy, v1.xy - v0.xy);
    float areaPBC = cross2d(v2.xy - p, v1.xy - p);
    float areaPCA = cross2d(v0.xy - p, v2.xy - p);

    barycentricCoords.x = areaPBC / areaABC;
    barycentricCoords.y = areaPCA / areaABC;
    barycentricCoords.z = 1.f-barycentricCoords.x-barycentricCoords.y;

    return barycentricCoords;
}

#define TRIANGLE_PARAMETERS \
    /* Vertex buffers */ \
    global const float4 *gl_Position, \
    global const float4 *in_varying, \
    /* Fragment buffers */ \
    global float4 *gl_FragCoord, \
    global float4 *out_varying, \
    global bool *gl_FrontFacing, \
    global uchar *gl_Discard, \
    /* Vertex shader data */ \
    const uint n_varying, \
    /* Rasterization data */ \
    const uint vertex0, \
    const uint primitive_id, \
    /* Rasterization config */ \
    const int front_face, \
    const int cull_face, \
    const float polygon_offset_factor, \
    const float polygon_offset_units, \
    const int enabled_culling, \
    const int enabled_polygon_offset_fill

#define BASIC_VARS \
    uint xid = get_global_id(0); \
    uint yid = get_global_id(1); \
    uint xsize = get_global_size(0); \
    uint ysize = get_global_size(1); \
    uint gid = xid+yid*xsize; \
    uint gsize = xsize*ysize; \
    float2 point = (float2){xid, yid} + 0.5f; \
    gl_FragCoord[gid].xy = point;
    

#define CHECK_OUT_OF_POLYGON(v0, v1, v2, abc) \
    { \
        abc = get_baricentric_coords(point, v0, v1, v2); \
        if ((abc.x < -0.00001f || abc.y < -0.00001f || abc.z < -0.00001f)) { \
            gl_Discard[gid] = 2; \
            return; \
        } \
    }

#define CULL_FACE(v0, v1, v2) \
    { \
        float area = 0.5f * cross(v0.xy, v1.xy) + cross(v1.xy, v2.xy) + cross(v2.xy, v0.xy); \
        if (front_face == GL_CCW) area = -area; \
        if (enabled_culling && ( \
                (cull_face == GL_FRONT_AND_BACK     ) || \
                (cull_face == GL_FRONT && area > 0.f) || \
                (cull_face == GL_BACK  && area > 0.f) \
            )) { \
            gl_Discard[gid] = 3; \
            return; \
        } \
        gl_FrontFacing[gid] = area > 0.f ? 1 : 0; \
    }

#define POLYGON_OFFSET(v0, v1, v2, abc) \
    { \
        float depth = abc.x*v0.z + abc.y*v1.z + abc.z*v2.z; \
        if (enabled_polygon_offset_fill) { \
            float m = max(depth / point.x, depth / point.y); \
            float r = 1.0 / 0xFFFFFFFFu; /* implement-dependent min value that may generate a change */ \
            depth += m * polygon_offset_factor + r * polygon_offset_units; \
            depth = min(max(depth, 1.0f), 0.0f); \
        } \
        gl_FragCoord[gid].z = depth; \
    }

kernel void gl_rasterization_triangle_fan(
    TRIANGLE_PARAMETERS
) {
    BASIC_VARS;
    
    // Get primitive vertices
    uint primitive_index = (vertex0 + primitive_id) + 1;

    const float4 v0 = gl_Position[vertex0];
    const float4 v1 = gl_Position[primitive_index + 0];
    const float4 v2 = gl_Position[primitive_index + 1];

    float3 bcoords;
    CHECK_OUT_OF_POLYGON(v0, v1, v2, bcoords);

    CULL_FACE(v0, v1, v2);

    POLYGON_OFFSET(v0, v1, v2, bcoords);

    for (uint varying = 0; varying < n_varying; ++varying) {
        uint vertex_subbuff_offset   = primitive_index + gsize * varying;
        uint fragment_subbuff_offset = gid + gsize * varying;

        float4 var0 = in_varying[gsize * varying + vertex0];
        float4 var1 = in_varying[vertex_subbuff_offset + 0];
        float4 var2 = in_varying[vertex_subbuff_offset + 1];

        var0 = (var0 * bcoords.x) / v0.w;
        var1 = (var1 * bcoords.y) / v1.w;
        var2 = (var2 * bcoords.z) / v2.w;

        float div = (bcoords.x/v0.w + bcoords.y/v1.w + bcoords.z/v2.w);

        out_varying[fragment_subbuff_offset] = (var0 + var1 + var2) / div;
    }

    gl_Discard[gid] = 0;

}

kernel void gl_rasterization_triangle_strip(
    TRIANGLE_PARAMETERS
) {
    BASIC_VARS;

    // Get primitive vertices 
    uint primitive_index = (vertex0 + primitive_id);
    uint oddity = primitive_id%2;

    const float4 v0 = gl_Position[primitive_index + 0 + 2*oddity];
    const float4 v1 = gl_Position[primitive_index + 1];
    const float4 v2 = gl_Position[primitive_index + 2 - 2*oddity];

    float3 bcoords;
    CHECK_OUT_OF_POLYGON(v0, v1, v2, bcoords);

    CULL_FACE(v0, v1, v2);

    POLYGON_OFFSET(v0, v1, v2, bcoords);

    for (uint varying = 0; varying < n_varying; ++varying) {
        uint vertex_subbuff_offset   = primitive_index + gsize * varying;
        uint fragment_subbuff_offset = gid + gsize * varying;

        float4 var0 = in_varying[vertex_subbuff_offset + 0 + 2*oddity];
        float4 var1 = in_varying[vertex_subbuff_offset + 1];
        float4 var2 = in_varying[vertex_subbuff_offset + 2 - 2*oddity];

        var0 = (var0 * bcoords.x) / v0.w;
        var1 = (var1 * bcoords.y) / v1.w;
        var2 = (var2 * bcoords.z) / v2.w;

        float div = (bcoords.x/v0.w + bcoords.y/v1.w + bcoords.z/v2.w);

        out_varying[fragment_subbuff_offset] = (var0 + var1 + var2) / div;
    }

    gl_Discard[gid] = 0;
}

kernel void gl_rasterization_triangles(
    TRIANGLE_PARAMETERS
) {
    BASIC_VARS;
    
    uint primitive_index = (vertex0 + primitive_id) * 3;

    const float4 v0 = gl_Position[primitive_index + 0];
    const float4 v1 = gl_Position[primitive_index + 1];
    const float4 v2 = gl_Position[primitive_index + 2];

    float3 bcoords;
    CHECK_OUT_OF_POLYGON(v0, v1, v2, bcoords);

    CULL_FACE(v0, v1, v2);

    POLYGON_OFFSET(v0, v1, v2, bcoords);
    
    for (uint varying = 0; varying < n_varying; ++varying) {
        uint vertex_subbuff_offset   = primitive_index + gsize * varying;
        uint fragment_subbuff_offset = gid + gsize * varying;

        float4 var0 = in_varying[vertex_subbuff_offset + 0];
        float4 var1 = in_varying[vertex_subbuff_offset + 1];
        float4 var2 = in_varying[vertex_subbuff_offset + 2];

        var0 = (var0 * bcoords.x) / v0.w;
        var1 = (var1 * bcoords.y) / v1.w;
        var2 = (var2 * bcoords.z) / v2.w;

        float div = (bcoords.x/v0.w + bcoords.y/v1.w + bcoords.z/v2.w);

        out_varying[fragment_subbuff_offset] = (var0 + var1 + var2) / div;
    }

    gl_Discard[gid] = 0;
}
