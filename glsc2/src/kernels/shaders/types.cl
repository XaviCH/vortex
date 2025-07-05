/**
 * Definition of types for easy conversion from OpenCL to GLSL
 * 
 */

typedef float2 vec2;
typedef float3 vec3;
typedef float4 vec4;

typedef float4 mat2;
typedef float16 mat3;
typedef float16 mat4;

#ifdef CONF_FINE_IMAGE_ENABLED
typedef read_only image2d_t image2D_t;
#else
typedef const global uchar* image2D_t;
#endif

