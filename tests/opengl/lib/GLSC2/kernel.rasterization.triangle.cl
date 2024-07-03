/*

bool stencilTestPassed(float3 barycentricCoords, __global int* stencilBuffer, int index) {

    int referenceMask = 0xFFFFFFFF;

    int triangleMask = 0xFF; // Supongamos que solo necesitamos 8 bits de stencil

    int maskedStencilValue = stencilBuffer[index] & referenceMask;

    if ((maskedStencilValue & triangleMask) != 0) {
        return true;
    } else {
        return false;
    }
}


void blendFragmentColor(__global float4* framebufferColor, float4 fragmentColor) {
    // Ejemplo de alpha blending
    float alpha = fragmentColor.w;

    framebufferColor->xyz = (1.0f - alpha) * framebufferColor->xyz + alpha * fragmentColor.xyz;
}


*/

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
    barycentricCoords.z = 1.f-barycentricCoords.x-barycentricCoords.y; // What is this??

    return barycentricCoords;
}

__kernel void gl_rasterization_triangle (
    const int gl_Index, // gl_Index 
    const int width, // 
    const int attributes, // 
    global const float4 *gl_Positions,
    global float4 *gl_FragCoords,
    global bool *gl_Discard,
    global const float4 *gl_Primitives,
    global float4 *gl_Rasterization
)
{
    int gid = get_global_id(0);
    int gsize = get_global_size(0); // number of fragments
    // input values
    global const float4 *position = gl_Positions + gl_Index*3;
    global const float4 *primitives = gl_Primitives + gl_Index*3;
    global float4 *fragCoord = gl_FragCoords + gid;
    global float4 *rasterization = gl_Rasterization + gid;

    //frag coords norm
    float xf = (gid % width);
    float yf = (gid / width);

    float4 v0 = position[0];
    float4 v1 = position[1];
    float4 v2 = position[2];
    
    float3 abc = get_baricentric_coords((float2) (xf,yf), v0, v1, v2);
    fragCoord->x = xf;
    fragCoord->y = yf;
    fragCoord->z = abc.x*v0.z + abc.y*v1.z + abc.z*v2.z;
    fragCoord->w = abc.x*v0.w + abc.y*v1.w + abc.z*v2.w;

    for(int attribute = 0 ; attribute < attributes; attribute++) {
        global const float4 *p0 = primitives + gsize*attribute;
        global const float4 *p1 = p0 + 1;
        global const float4 *p2 = p0 + 2;
        
        // HW optimization ?? 
        rasterization[gsize*attribute].x = abc.x*p0->x + abc.y*p1->x + abc.z*p2->x;
        rasterization[gsize*attribute].y = abc.x*p0->y + abc.y*p1->y + abc.z*p2->y;
        rasterization[gsize*attribute].z = abc.x*p0->z + abc.y*p1->z + abc.z*p2->z;
        rasterization[gsize*attribute].w = abc.x*p0->w + abc.y*p1->w + abc.z*p2->w;
    }

    // Instruction optimization ?? gl_Discard[gid] = abc.x*abc.y*abc.z; // maybe if abc wasn't float could go faster
    if ((abc.x >= 0.0f) && (abc.y >= 0.0f) && (abc.z >= 0.0f)) {
        gl_Discard[gid] = false;
    } else {
        gl_Discard[gid] = true;
    }
}
