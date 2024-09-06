
/** HEADER */
/** THIS MUST BE INCLUDED TO SUPPORT TEXTURING **/
#define GL_REPEAT                         0x2901
#define GL_CLAMP_TO_EDGE                  0x812F
#define GL_MIRRORED_REPEAT                0x8370

#define GL_NEAREST                        0x2600
#define GL_LINEAR                         0x2601
#define GL_NEAREST_MIPMAP_NEAREST         0x2700
#define GL_LINEAR_MIPMAP_NEAREST          0x2701
#define GL_NEAREST_MIPMAP_LINEAR          0x2702
#define GL_LINEAR_MIPMAP_LINEAR           0x2703

#define GL_R8                             0x8229
#define GL_RG8                            0x822B
#define GL_RGB8                           0x8051
#define GL_RGBA8                          0x8058
#define GL_RGBA4                          0x8056
#define GL_RGB5_A1                        0x8057
#define GL_RGB565                         0x8D62

typedef global unsigned char* image_t;

typedef struct {
    short s, t, min_filter, mag_filter;
} texture_wraps_t;

typedef struct __attribute__((packed)) {
  uint width, height;
  uint internalformat;
  texture_wraps_t wraps;
} sampler2D_t;

float4 __attribute__((overloadable)) read_imagef(image_t image, sampler2D_t sampler, float2 coord) {
    // TODO: Only works for GL_NEAREST
    int width, height;

    switch (sampler.wraps.s) {
        case GL_REPEAT:
            coord.x = (coord.x - floor(coord.x));
            break;
        case GL_CLAMP_TO_EDGE:
            coord.x = coord.x * min(max(coord.x,0.f), 1.f);
            break;
        case GL_MIRRORED_REPEAT:
            {
                float f_coord = floor(coord.x);
                if ((int) f_coord % 2 == 0) {
                    coord.x = coord.x - f_coord;
                } else {
                    coord.x = 1.f - (coord.x - f_coord);
                }
            }
            break;
        default: break;
    }

    switch (sampler.wraps.t) {
        case GL_REPEAT:
            coord.y = (coord.y - floor(coord.y));
            break;
        case GL_CLAMP_TO_EDGE:
            coord.y = coord.y * min(max(coord.y,0.f), 1.f);
            break;
        case GL_MIRRORED_REPEAT:
            {
                float f_coord = floor(coord.y);
                if ((int) f_coord % 2 == 0) {
                    coord.y = coord.y - f_coord;
                } else {
                    coord.y = 1.f - (coord.y - f_coord);
                }
            }
            break;
        default: break;
    }

    width   = sampler.width * coord.x;
    height  = sampler.height * coord.y;

    // TODO: min and mag filtering
    switch (sampler.wraps.min_filter) {
        case GL_NEAREST:
        case GL_LINEAR:
            break; 
        default: break;
    }
    switch (sampler.wraps.mag_filter) {
        case GL_NEAREST:
        case GL_LINEAR:
            break; 
        default: break;
    }

    // 
    switch (sampler.internalformat) {
        case GL_R8:
            {
                global uchar* color = image + (height*sampler.width + width);
                return (float4) ((float)*color / 255, 0.f, 0.f, 1.f);
            }
        case GL_RG8:
            {
                global uchar* color = image + (height*sampler.width + width)*2;
                return (float4) ((float)*color / 255, (float)*(color+1) / 255, 0.f, 1.f);
            }
        case GL_RGB8:
            {
                global uchar* color = image + (height*sampler.width + width)*3;
                return (float4) ((float)*color / 255, (float)*(color+1) / 255, (float)*(color+2) / 255, 1.f);
            }
        case GL_RGBA8:
            {
                global uchar* color = image + (height*sampler.width + width)*4;
                return (float4) ((float)*color / 255, (float)*(color+1) / 255, (float)*(color+2) / 255, (float)*(color+3) / 255);
            }
        case GL_RGBA4:
            {
                global ushort* color = (global ushort*) image + (height*sampler.width + width);
                return (float4) (
                    (float) ((*color >>  0) & 0x000F) / 15.f,
                    (float) ((*color >>  4) & 0x00F0) / 15.f,
                    (float) ((*color >>  8) & 0x0F00) / 15.f,
                    (float) ((*color >> 12) & 0xF000) / 15.f
                );
            }
        case GL_RGB5_A1:
            {
                global ushort* color = (global ushort*) image + (height*sampler.width + width);
                return (float4) (
                    (float) ((*color >>  0) & 0x001F) / 31.f,
                    (float) ((*color >>  5) & 0x03E0) / 31.f,
                    (float) ((*color >> 10) & 0x7C00) / 31.f,
                    (float) ((*color >> 15) & 0x8000) /  1.f
                );
            }
        case GL_RGB565:
            {
                global ushort* color = (global ushort*) image + (height*sampler.width + width);
                return (float4) (
                    (float) ((*color >>  0) & 0x002F) / 31.f,
                    (float) ((*color >>  4) & 0x07E0) / 63.f,
                    (float) ((*color >>  8) & 0xF800) / 31.f,
                    1.f
                );
            }
        default:
            return (float4) (0.5f);
    }

}


/** VERTEX SHADER **/

kernel void main_vs(
    global const float4* vPosition,
    global const float4* texCoord0,
    global float4* vTexCoord0,  
    global const float4* texCoord1,
    global float4* vTexCoord1,
    global float4* gl_Position
)              
{                        
    int gid = get_global_id(0);
    
    gl_Position[gid] = vPosition[gid]; 
    vTexCoord0[gid]  = texCoord0[gid];
    vTexCoord1[gid]  = texCoord1[gid]; 
}

/** FRAGMENT SHADER **/

float reconstruct_input(sampler2D_t textureUnit, image_t image, float2 vTexCoord) {
    float4 u_split;
    u_split= read_imagef(image, textureUnit, vTexCoord);
    float tmp;
    float reconstructed;
    float sign_value=1.0;

    tmp = floor(256.0*u_split.w - (u_split.w/255.0));
    float exponent = tmp - 127.0;
    tmp = floor(256.0*u_split.z - (u_split.z/255.0));
    reconstructed = (tmp*0.0078125) ;
    if(exponent >= -126.0) if(reconstructed < 1.0) reconstructed += 1.0 ;
    if(tmp > 127.0) sign_value = -1.0 ;
    tmp = floor(256.0*u_split.y - (u_split.y/255.0));
    reconstructed += (tmp*0.000030517578125);
    tmp = floor(256.0*u_split.x - (u_split.x/255.0));
    reconstructed += (tmp*0.00000011920928955078);
    reconstructed = sign_value*reconstructed*exp2(exponent);
    return reconstructed;
}

float4 encode_output(float reconstructed) {
    float sign_value=1.0;
    float exponent;
    float tmp;
    float4 u_split;
    exponent = (floor(log2(fabs(reconstructed))) + 127.0)*step(exp2(-125.0f),fabs(reconstructed)) ;
    u_split.w = ((exponent - 256.0*floor(exponent*0.00390625))*0.00392156862745098) ;
    tmp = clamp(fabs(reconstructed*exp2(-floor(log2(fabs(reconstructed))))) -1.0, 0.0, 1.0);
    tmp = tmp*exp2(23.0);
    if(reconstructed < 0.0 ) sign_value = exp2(23.0);
    u_split.z = (floor(((tmp+sign_value) - 256.0*256.0*256.0*floor((tmp+sign_value)*0.00000005960464477539))*1.52587890625e-05)*0.00392156862745098) ;
    u_split.y = (floor((tmp - 256.0*256.0*floor(tmp*1.52587890625e-05))*0.00390625)*0.00392156862745098) ;
    u_split.x = ((tmp- 256.0*floor(tmp*0.00390625))*0.00392156862745098) ;
    u_split.x = 0.0 ;
    return u_split;
}

kernel void main_fs(
    sampler2D_t textureUnit0,
    image_t image0,
    sampler2D_t textureUnit1,
    image_t image1,
    global const float4* vTexCoord0,  
    global const float4* vTexCoord1,
    global float4* gl_FragColor,
    global const float4* gl_FragCoord
)
{
    int gid = get_global_id(0);

    float reconstructed0;
    float reconstructed1;
    float4 u_split;
    float2 _vTexCoord0; 
    _vTexCoord0.xy = (gl_FragCoord[gid].xy)/16;
    float2 _vTexCoord1 ; 
    _vTexCoord1 = _vTexCoord0; 
    reconstructed0 = reconstruct_input(textureUnit0, image0, _vTexCoord0);
    reconstructed1 = reconstruct_input(textureUnit1, image1, _vTexCoord1);
    reconstructed0 += reconstructed1;
    u_split = encode_output(reconstructed0);
    gl_FragColor[gid] = u_split;
}