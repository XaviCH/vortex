

inline float cross(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}

#define LINE_PARAMETERS \
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
    const float line_width 

#define BASIC_VARS \
    uint xid = get_global_id(0); \
    uint yid = get_global_id(1); \
    uint xsize = get_global_size(0); \
    uint ysize = get_global_size(1); \
    uint gid = xid+yid*xsize; \
    uint gsize = xsize*ysize; \
    float2 point = (float2){xid, yid} + 0.5f; \
    gl_FragCoord[gid].xy = point;
    

#define LEFT    (float2) {-0.5,0}
#define TOP     (float2) {0, 0.5}
#define RIGHT   (float2) {0.5, 0}
#define BOTTOM  (float2) {0,-0.5}

inline float2 intersect(float2 Pa0, float2 Pa1, float2 Pb0, float2 Pb1) {
    return (float2) {
        cross(Pa0, Pa1)*(Pb0.x-Pb1.x) - (Pa0.x-Pa1.x)*cross(Pb0, Pb1),
        cross(Pa0, Pa1)*(Pb0.y-Pb1.y) - (Pa0.y-Pa1.y)*cross(Pb0, Pb1),
    } / ((Pa0.x - Pa1.x)*(Pb0.y - Pb1.y) - (Pa0.y - Pa1.y)*(Pb0.x - Pb1.x));
}

#define CHECK_IF_INTERSECT(t) \
    { \
        /* Check if line intersects between the diamond shape */ \
        float2 i_tl, i_bl, i_tr, i_br; \
        i_tl = intersect(v0.xy, v1.xy, point + LEFT,  point + TOP); \
        i_bl = intersect(v0.xy, v1.xy, point + LEFT,  point + BOTTOM); \
        i_tr = intersect(v0.xy, v1.xy, point + RIGHT, point + TOP); \
        i_br = intersect(v0.xy, v1.xy, point + RIGHT, point + BOTTOM); \
        /* TODO checkout line width */ \
        if (distance(point,i_tl) > line_width && \
            distance(point,i_bl) > line_width && \
            distance(point,i_tr) > line_width && \
            distance(point,i_br) > line_width \
        ) { \
            gl_Discard[gid] = 1; \
            return; \
        } \
        float r_tl, r_bl, r_tr, r_br; \
        /* Check if v1 is inside diamond shape */ \
        r_tl = cross(TOP - LEFT     , point - TOP); \
        r_tr = cross(TOP - RIGHT    , point - TOP); \
        r_bl = cross(BOTTOM - LEFT  , point - BOTTOM); \
        r_br = cross(BOTTOM - RIGHT , point - BOTTOM); \
        if (r_tl < 0.0 && r_tr < 0.0 && r_bl > 0.0 && r_br > 0.0) { \
            gl_Discard[gid] = 2; \
            return; \
        } \
        float dist_line = distance(v1.xy, v0.xy); \
        t = cross(point - v0.xy, v1.xy - v0.xy) / (dist_line * dist_line); \
        gl_FrontFacing[gid] = 1; \
        gl_FragCoord[gid].z = (1-t)*v0.z + t*v1.z; \
        gl_Discard[gid] = 0; \
    }

kernel void gl_rasterization_line_strip(
    LINE_PARAMETERS
)  {
    BASIC_VARS;

    uint primitive_index = vertex0 + primitive_id;

    const float4 v0 = gl_Position[primitive_index + 0];
    const float4 v1 = gl_Position[primitive_index + 1];

    float coeficient;

    CHECK_IF_INTERSECT(coeficient);

    for (uint varying = 0; varying < n_varying; ++varying) {
        uint vertex_subbuff_offset   = primitive_index + gsize * varying;
        uint fragment_subbuff_offset = gid + gsize * varying;

        float4 var0 = in_varying[vertex_subbuff_offset + 0];
        float4 var1 = in_varying[vertex_subbuff_offset + 1];

        var0 = ((1-coeficient) * var0) / v0.w;
        var1 = (coeficient * var1) / v1.w;

        float div = (1-coeficient)/v0.w + coeficient/v1.w;

        out_varying[fragment_subbuff_offset] = (var0 + var1) / div;
    }

}

kernel void gl_rasterization_line_loop(
    LINE_PARAMETERS,
    const uint last
)  {

    BASIC_VARS;

    uint primitive_index0 = vertex0 + primitive_id;
    uint primitive_index1 = vertex0 * last + (primitive_index0 +1)*(1-last);

    const float4 v0 = gl_Position[primitive_index0];
    const float4 v1 = gl_Position[primitive_index1];

    float coeficient;

    CHECK_IF_INTERSECT(coeficient);

    for (uint varying = 0; varying < n_varying; ++varying) {
        uint vertex_subbuff_offset0   = primitive_index0 + gsize * varying;
        uint vertex_subbuff_offset1   = primitive_index1 + gsize * varying;
        uint fragment_subbuff_offset = gid + gsize * varying;

        float4 var0 = in_varying[vertex_subbuff_offset0];
        float4 var1 = in_varying[vertex_subbuff_offset1];

        var0 = ((1-coeficient) * var0) / v0.w;
        var1 = (coeficient * var1) / v1.w;

        float div = (1-coeficient)/v0.w + coeficient/v1.w;

        out_varying[fragment_subbuff_offset] = (var0 + var1) / div;
    }
}

kernel void gl_rasterization_lines(
    LINE_PARAMETERS
) {
    BASIC_VARS;
    
    uint primitive_index = (vertex0 + primitive_id) * 2;

    const float4 v0 = gl_Position[primitive_index + 0];
    const float4 v1 = gl_Position[primitive_index + 1];

    float coeficient;

    CHECK_IF_INTERSECT(coeficient);

    for (uint varying = 0; varying < n_varying; ++varying) {
        uint vertex_subbuff_offset   = primitive_index + gsize * varying;
        uint fragment_subbuff_offset = gid + gsize * varying;

        float4 var0 = in_varying[vertex_subbuff_offset + 0];
        float4 var1 = in_varying[vertex_subbuff_offset + 1];

        var0 = ((1-coeficient) * var0) / v0.w;
        var1 = (coeficient * var1) / v1.w;

        float div = (1-coeficient)/v0.w + coeficient/v1.w;

        out_varying[fragment_subbuff_offset] = (var0 + var1) / div;
    }

}
