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

float3 get_baricentric_coords(float2 p, float4 v0, float4 v1, float4 v2) {
    float3 barycentricCoords;

    float areaABC = cross2d(v2.xy - v0.xy, v1.xy - v0.xy);
    float areaPBC = cross2d(v2.xy - p, v1.xy - p);
    float areaPCA = cross2d(v0.xy - p, v2.xy - p);

    barycentricCoords.x = areaPBC / areaABC;
    barycentricCoords.y = areaPCA / areaABC;
    barycentricCoords.z = 1.f-barycentricCoords.x-barycentricCoords.y;

    return barycentricCoords;
}

kernel void gl_rasterization_triangles(
    // Vertex buffers 
    global const float4 *gl_Position,
    global const float4 *in_varying,
    // Fragment buffers
    global float4 *gl_FragCoord,
    global float4 *out_varying,
    global bool *gl_FrontFacing,
    global uchar *gl_Discard,
    // Vertex shader data
    const uint n_varying,
    // Rasterization data
    const uint vertex0,
    const uint primitive_id,
    // Rasterization config
    const int front_face,
    const int cull_face,
    const float polygon_offset_factor,
    const float polygon_offset_units,
    const int enabled_culling,
    const int enabled_polygon_offset_fill
) {
    uint xid = get_global_id(0);
    uint yid = get_global_id(1);
    uint xsize = get_global_size(0); 
    uint ysize = get_global_size(1); 
    uint gid = xid*yid;

    float2 point = (float2){xid, yid} + 0.5f;
    
    const float4 v0 = *(gl_Position + (vertex0 + primitive_id) * 3 + 0);
    const float4 v1 = *(gl_Position + (vertex0 + primitive_id) * 3 + 1);
    const float4 v2 = *(gl_Position + (vertex0 + primitive_id) * 3 + 2);

    // Area

    float area = 0.5f * (v0.x*v1.y - v1.x*v0.y + v1.x*v2.y - v2.x*v1.y + v2.x*v0.y - v0.x*v2.y);
    if (front_face == GL_CCW) area = -area;

    if (enabled_culling && (
            (cull_face == GL_FRONT_AND_BACK     ) ||
            (cull_face == GL_FRONT && area > 0.f) ||
            (cull_face == GL_BACK  && area > 0.f))
        ) {
        gl_Discard[gid] = 3;
        return;
    }

    gl_FrontFacing[gid] = area > 0.f ? 1 : 0;

    // Polygon Offset

    float3 abc = get_baricentric_coords(point, v0, v1, v2);
    
    if ((abc.x < -0.00001f) || (abc.y < -0.00001f) || (abc.z < -0.00001f)) {
        gl_Discard[gid] = 2;
        return;
    }

    float depth = abc.x*v0.z + abc.y*v1.z + abc.z*v2.z;
    if (enabled_polygon_offset_fill) {
        float m = max(depth / point.x, depth / point.y);
        float r = 1 / 0xFFFFFFFFu; // implement-dependent min value that may generate a change
        depth += m * polygon_offset_factor + r * polygon_offset_units;
        depth = min(max(depth, 1.0f), 0.0f);
    }

    // Rasterize varying
    /* TODO
    for (uint varying = 0; varying < n_varying; ++varying) {
        float4 var0 = *(in_varying ) 

        out_varying[gid] = (float4) {

        };
    }
    */

    gl_FragCoord[gid].xy = point;
    gl_FragCoord[gid].z = depth;
    gl_Discard[gid] = 0;
}

kernel void gl_rasterization_triangle (
    const int gl_Index, // gl_Index 
    const int width, // 
    const int attributes, //
    global const float4 *gl_Positions,
    global float4 *gl_FragCoords,
    global uchar *gl_Discard,
    global const float4 *gl_Primitives,
    global float4 *gl_Rasterization,
    global bool* facing, // 0: front, 1: back
    const ushort front_face,
    const ushort culling
)
{
    int gid = get_global_id(0);
    int gsize = get_global_size(0); // number of fragments
    // input values
    global const float4 *position = gl_Positions + gl_Index*3;
    global const float4 *primitives = gl_Primitives + gl_Index*3;
    global float4 *fragCoord = gl_FragCoords + gid;
    global float4 *rasterization = gl_Rasterization + gid;

    // base info
    float yf = (float) (gid / width) + 0.5f; // center of the fragment
    float xf = (float) (gid % width) + 0.5f; // center of the fragment

    float4 v0 = position[0];
    float4 v1 = position[1];
    float4 v2 = position[2];

    // area

    float area = 0.5f * (v0.x*v1.y - v1.x*v0.y + v1.x*v2.y - v2.x*v1.y + v2.x*v0.y - v0.x*v2.y);

    
    if (front_face == GL_CCW) area = -area;

    facing[gid] = area < 0.f ? 1 : 0;

    if (area == 0.f || culling == GL_FRONT_AND_BACK || (culling == GL_BACK && area < 0.f) || (culling == GL_FRONT && area > 0.f)) {
        gl_Discard[gid] = 3;
        return;
    }
    
    // barycenter
    float3 abc = get_baricentric_coords((float2) (xf,yf), v0, v1, v2);
    
    if ((abc.x < -0.00001f) || (abc.y < -0.00001f) || (abc.z < -0.00001f)) {
        gl_Discard[gid] = 2;
        return;
    }

    fragCoord->x = xf;
    fragCoord->y = yf;
    fragCoord->z = abc.x*v0.z + abc.y*v1.z + abc.z*v2.z;
    // fragCoord->w = abc.x*v0.w + abc.y*v1.w + abc.z*v2.w; // maybe this is not required

    for(int attribute = 0 ; attribute < attributes; attribute++) {
        global const float4 *p0 = primitives + gsize*attribute;
        global const float4 *p1 = p0 + 1;
        global const float4 *p2 = p0 + 2;
        
        // HW optimization ?? 
        rasterization[gsize*attribute].x = (abc.x*p0->x/v0.w + abc.y*p1->x/v1.w + abc.z*p2->x/v2.w) / (abc.x/v0.w + abc.y/v1.w + abc.z/v2.w);
        rasterization[gsize*attribute].y = (abc.x*p0->y/v0.w + abc.y*p1->y/v1.w + abc.z*p2->y/v2.w) / (abc.x/v0.w + abc.y/v1.w + abc.z/v2.w);
        rasterization[gsize*attribute].z = (abc.x*p0->z/v0.w + abc.y*p1->z/v1.w + abc.z*p2->z/v2.w) / (abc.x/v0.w + abc.y/v1.w + abc.z/v2.w);
        rasterization[gsize*attribute].w = (abc.x*p0->w/v0.w + abc.y*p1->w/v1.w + abc.z*p2->w/v2.w) / (abc.x/v0.w + abc.y/v1.w + abc.z/v2.w);
    }

    // Instruction optimization ?? gl_Discard[gid] = abc.x*abc.y*abc.z; // maybe if abc wasn't float could go faster
    gl_Discard[gid] = false;
}
