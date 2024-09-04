
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


float4 __get_color_from_image_t(image_t image, sampler2D_t sampler, uint x, uint y) {
    switch (sampler.internalformat) {
        case GL_R8:
            {
                global uchar* color = image + (y*sampler.width + x);
                return (float4) ((float)*color / 255, 0.f, 0.f, 1.f);
            }
        case GL_RG8:
            {
                global uchar* color = image + (y*sampler.width + x)*2;
                return (float4) ((float)*color / 255, (float)*(color+1) / 255, 0.f, 1.f);
            }
        case GL_RGB8:
            {
                global uchar* color = image + (y*sampler.width + x)*3;
                return (float4) ((float)*color / 255, (float)*(color+1) / 255, (float)*(color+2) / 255, 1.f);
            }
        case GL_RGBA8:
            {
                global uchar* color = image + (y*sampler.width + x)*4;
                return (float4) ((float)*color / 255, (float)*(color+1) / 255, (float)*(color+2) / 255, (float)*(color+3) / 255);
            }
        case GL_RGBA4:
            {
                global ushort* color = (global ushort*) image + (y*sampler.width + x);
                return (float4) (
                    (float) ((*color >>  0) & 0x000F) / 15.f,
                    (float) ((*color >>  4) & 0x00F0) / 15.f,
                    (float) ((*color >>  8) & 0x0F00) / 15.f,
                    (float) ((*color >> 12) & 0xF000) / 15.f
                );
            }
        case GL_RGB5_A1:
            {
                global ushort* color = (global ushort*) image + (y*sampler.width + x);
                return (float4) (
                    (float) ((*color >>  0) & 0x001F) / 31.f,
                    (float) ((*color >>  5) & 0x03E0) / 31.f,
                    (float) ((*color >> 10) & 0x7C00) / 31.f,
                    (float) ((*color >> 15) & 0x8000) /  1.f
                );
            }
        case GL_RGB565:
            {
                global ushort* color = (global ushort*) image + (y*sampler.width + x);
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

/**
* Overload OpenCL implementation to allow texturing 
*/
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
        default: break; // Code never must reach this point
    }

    switch (sampler.wraps.s) {
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
        default: break; // Code never must reach this point
    }

    float u, v;
    u = coord.x * sampler.width;
    v = coord.y * sampler.height;

    // TODO: decide when min or mag filtering
    switch (sampler.wraps.min_filter) {
        case GL_NEAREST:
            if (coord.x == 1.0) width = sampler.width - 1;
            else width = floor(u);
            if (coord.y == 1.0) height = sampler.height - 1;
            else height = floor(v);
            return __get_color_from_image_t(image, sampler, width, height);
        case GL_LINEAR:
            {
                int i0, j0, i1, j1;
                i0 = (int) floor(u-0.5f);
                j0 = (int) floor(v-0.5f);
                if (sampler.wraps.s == GL_REPEAT) i0 %= sampler.width;
                if (sampler.wraps.t == GL_REPEAT) j0 %= sampler.height;

                i1 = i0+1;
                j1 = j0+1;
                if (sampler.wraps.s == GL_REPEAT) i1 %= sampler.width;
                if (sampler.wraps.t == GL_REPEAT) j1 %= sampler.height;

                float alpha, beta;
                alpha = u - 0.5f;
                alpha -= floor(alpha);
                beta = v - 0.5f;
                beta -= floor(beta);

                float4 color00, color10, color01, color11;
                color00 = __get_color_from_image_t(image, sampler, i0, j0);
                color10 = __get_color_from_image_t(image, sampler, i1, j0);
                color01 = __get_color_from_image_t(image, sampler, i0, j1);
                color11 = __get_color_from_image_t(image, sampler, i1, j1);

                return  (1.0f-alpha) * (1.0f-beta) * color00 *
                        alpha        * (1.0f-beta) * color10 *
                        (1.0f-alpha) * beta        * color01 *
                        alpha        * beta        * color10;
            }
        default: break;
    }
    switch (sampler.wraps.mag_filter) {
        case GL_NEAREST:
            if (coord.x == 1.0) width = sampler.width - 1;
            else width = floor(u);
            if (coord.y == 1.0) height = sampler.height - 1;
            else height = floor(v);
            return __get_color_from_image_t(image, sampler, width, height);
        case GL_LINEAR:
            {
                int i0, j0, i1, j1;
                i0 = (int) floor(u-0.5f);
                j0 = (int) floor(v-0.5f);
                if (sampler.wraps.s == GL_REPEAT) i0 %= sampler.width;
                if (sampler.wraps.t == GL_REPEAT) j0 %= sampler.height;

                i1 = i0+1;
                j1 = j0+1;
                if (sampler.wraps.s == GL_REPEAT) i1 %= sampler.width;
                if (sampler.wraps.t == GL_REPEAT) j1 %= sampler.height;

                float alpha, beta;
                alpha = u - 0.5f;
                alpha -= floor(alpha);
                beta = v - 0.5f;
                beta -= floor(beta);

                float4 color00, color10, color01, color11;
                color00 = __get_color_from_image_t(image, sampler, i0, j0);
                color10 = __get_color_from_image_t(image, sampler, i1, j0);
                color01 = __get_color_from_image_t(image, sampler, i0, j1);
                color11 = __get_color_from_image_t(image, sampler, i1, j1);

                return  (1.0f-alpha) * (1.0f-beta) * color00 *
                        alpha        * (1.0f-beta) * color10 *
                        (1.0f-alpha) * beta        * color01 *
                        alpha        * beta        * color10;
            }
        default: break;
    }

    // 
    

}