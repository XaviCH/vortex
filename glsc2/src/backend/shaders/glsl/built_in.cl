#ifndef BACKEND_SHADERS_UTILS_BUILT_IN_CL
#define BACKEND_SHADERS_UTILS_BUILT_IN_CL

#ifdef __COMPILER_RELATIVE_PATH__
#include <backend/types.cl>
#include <backend/shaders/glsl/types.cl>
#include "constants.h"
#else
#include "glsc2/src/backend/types.cl"
#include "glsc2/src/backend/shaders/glsl/types.cl"
#include "glsc2/src/constants.h"
#endif



float4 mul(const float16 mat, const float4 vec) {
    
    float4 result = 0;

    for(int i=0; i<16; ++i) {
        result[i%4] += mat[i]*vec[i/4]; 
    }

    return result;
    /*
    float4 *p = (float4*) &mat;
    return (float4){
        dot(p[0], vec),
        dot(p[1], vec),
        dot(p[2], vec),
        dot(p[3], vec),
    };
    */
}

float mod(float x, float y) {
    return x - y * floor(x/y);
}

inline float gl_clamp_tex_coord(float value, uint size) {
    float min = 1.f / (2 * size);
    float max = 1.f - min;
    return clamp(value, min, max);
}

float apply_wrap(uint wrap, float value, uint size) {
    switch(wrap) {
        default:
        case TEXTURE_WRAP_REPEAT:
            return value - floor(value);
        case TEXTURE_WRAP_CLAMP_TO_EDGE:
            return gl_clamp_tex_coord(value, size);
        case TEXTURE_WRAP_MIRRORED_REPEAT:
            {
                float tmp = value - floor(value);
                if ((int)(fabs(floor(value))) % 2 == 1) 
                    tmp = 1.0 - tmp;
                return gl_clamp_tex_coord(tmp, size);
            }
    }
}

uint2 gl_sampl(gl_texture_data_t sampler, float2 coord) {
    uint2 img_coord;
    
    uint wrap_s = get_sampler2D_wrap_s(sampler.sampler2D);
    uint wrap_t = get_sampler2D_wrap_t(sampler.sampler2D);

    float coord_x = apply_wrap(wrap_s, coord.x, sampler.width);
    float coord_y = apply_wrap(wrap_t, coord.y, sampler.height);

    img_coord = (uint2){
        sampler.width * coord_x, 
        sampler.height * coord_y
    };

    return img_coord;
}

float4 texture2D(gl_texture_data_t sampler, ro_texture2d_t image, float2 coord) {
    uint2 img_coord = gl_sampl(sampler, coord);
    
    global const uchar* color = (global const uchar*) image + (img_coord.y*sampler.width + img_coord.x)*4;
    // global const uchar* color = (global const uchar*) image;
    return (float4) ((float)*color / 255, (float)*(color+1) / 255, (float)*(color+2) / 255, (float)*(color+3) / 255);
}

#endif // BACKEND_SHADERS_UTILS_BUILT_IN_CL