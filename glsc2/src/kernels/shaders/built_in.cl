#ifdef __COMPILER_RELATIVE_PATH__
#include "../../constants.h"
#include "../../types.device.h"
#include "types.cl"
#else
#include "glsc2/src/constants.h"
#include "glsc2/src/types.device.h"
#include "glsc2/src/kernels/shaders/types.cl"
#endif



float4 mul(float16 mat, float4 vec) {
    
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


float apply_wrap(uint wrap, float value) {
    switch(wrap) {
        default:
        case TEXTURE_WRAP_REPEAT:
            return value - floor(value);
        case TEXTURE_WRAP_CLAMP_TO_EDGE:
            return clamp(value, 0.f, 1.f);
        case TEXTURE_WRAP_MIRRORED_REPEAT:
            {
                float wrap = value - floor(value);
                if ((int)(floor(value)) % 2 == 0) 
                    wrap = 1 - wrap;
                return clamp(wrap, 0.f, 1.f);
            }
    }
}

uint2 gl_sampl(sampler2D_t sampler, float2 coord) {
    uint2 img_coord;
    
    uint wrap_s = get_sampler2D_wrap_s(sampler);
    uint wrap_t = get_sampler2D_wrap_t(sampler);

    float coord_x = apply_wrap(wrap_s, coord.x);
    float coord_y = apply_wrap(wrap_t, coord.y);

    img_coord = (uint2){
        sampler.width * coord_x, 
        sampler.height * coord_y
    };

    return img_coord;
}

float4 texture2D(sampler2D_t sampler, image2D_t image, float2 coord) {
    uint2 img_coord = gl_sampl(sampler, coord);
    
    global uchar* color = image + (img_coord.y*sampler.width + img_coord.x)*4;
    return (float4) ((float)*color / 255, (float)*(color+1) / 255, (float)*(color+2) / 255, (float)*(color+3) / 255);
}