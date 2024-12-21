#define GL_R8                             0x8229
#define GL_RG8                            0x822B
#define GL_RGB8                           0x8051
#define GL_RGBA8                          0x8058

#define GL_RGBA4                          0x8056
#define GL_RGB5_A1                        0x8057
#define GL_RGB565                         0x8D62

/**
 * HEADER FOR VORTEX OPENCL FUNCTIONS
 */
float roundf(float x)
{
  int signbit;
  uint w;
  /* Most significant word, least significant word. */
  int exponent_less_127;

  w = *(uint*)&x; // GET_FLOAT_WORD(w, x);

  /* Extract sign bit. */
  signbit = w & 0x80000000;

  /* Extract exponent field. */
  exponent_less_127 = (int)((w & 0x7f800000) >> 23) - 127;

  if (exponent_less_127 < 23)
    {
      if (exponent_less_127 < 0)
        {
          w &= 0x80000000;
          if (exponent_less_127 == -1)
            /* Result is +1.0 or -1.0. */
            w |= ((uint)127 << 23);
        }
      else
        {
          unsigned int exponent_mask = 0x007fffff >> exponent_less_127;
          if ((w & exponent_mask) == 0)
            /* x has an integral value. */
            return x;

          w += 0x00400000 >> exponent_less_127;
          w &= ~exponent_mask;
        }
    }
  else
    {
      if (exponent_less_127 == 128)
        /* x is NaN or infinite. */
        return x + x;
      else
        return x;
    }
  
  x = *(float*)&w; // SET_FLOAT_WORD(x, w);
  return x;
}
/**END */

inline float4 get_color_from_colorbuffer(global void* colorbuffer, uint offset, uint internalformat) {
  float4 color; 

  switch (internalformat) {
  case GL_R8:
    color.x = ((float) *((global uchar*) colorbuffer)) / 255;
    break;
  case GL_RG8:
    {
      global uchar* buffer = (global uchar*) colorbuffer + offset * 2;
      color.x = (float) buffer[0] / 255;
      color.y = (float) buffer[1] / 255;
    }
    break;
  case GL_RGB8:
    {
      global uchar* buffer = (global uchar*) colorbuffer + offset * 3;
      color.x = (float) buffer[0] / 255;
      color.y = (float) buffer[1] / 255;
      color.z = (float) buffer[2] / 255;
    }
    break;
  case GL_RGBA8:
    {
      global uchar* buffer = (global uchar*) colorbuffer + offset * 4;
      color.x = (float) buffer[0] / 255;
      color.y = (float) buffer[1] / 255;
      color.z = (float) buffer[2] / 255;
      color.w = (float) buffer[3] / 255;
    }
    break;
  case GL_RGBA4:
    {
      global ushort* buffer = ((global ushort*) colorbuffer) + offset;
      color.x = (float) ((*buffer & 0x000F) >>  0) / 15;
      color.y = (float) ((*buffer & 0x00F0) >>  4) / 15;
      color.z = (float) ((*buffer & 0x0F00) >>  8) / 15;
      color.w = (float) ((*buffer & 0xF000) >> 12) / 15;
    }
    break;
  case GL_RGB5_A1:
    {
      global ushort* buffer = (global ushort*) colorbuffer + offset;
      color.x = (float) (*buffer & 0x001F >>  0) / 31;
      color.y = (float) (*buffer & 0x03E0 >>  5) / 31;
      color.z = (float) (*buffer & 0x7C00 >> 10) / 31;
      color.w = (float) (*buffer & 0x8000 >> 15) /  1;
    }
    break;
  case GL_RGB565:
    {
      global ushort* buffer = (global ushort*) colorbuffer + offset;
      color.x = (float) (*buffer & 0x001F >>  0) / 31;
      color.y = (float) (*buffer & 0x07E0 >>  5) / 63;
      color.z = (float) (*buffer & 0xF800 >> 11) / 31;
      color.w = 1;
    }
    break;
  default:
    return (float4){1,1,1,1};
  }

  return color;

}

kernel void gl_readnpixels_rgba4(
  global ushort* buffer_out,
  // framebuffer data
  global void* colorbuffer,
  const uint internalformat,
  const uint color_width,
  // readnpixels data
  const int x,
  const int y,
  const uint width,
  const uint height
) {
  int gid = get_global_id(0);

  uint i, j;
  i = gid % width;
  j = gid / width;

  uint offset = x+y*color_width + i+j*color_width;

  float4 color = get_color_from_colorbuffer(colorbuffer, offset, internalformat);
  color *= 15;
  // color = round(color);
  color.x = roundf(color.x);
  color.y = roundf(color.y);
  color.z = roundf(color.z);
  color.w = roundf(color.w);

  ushort color4 = 0x0;
  color4 |= (ushort) color.x <<  0;
  color4 |= (ushort) color.y <<  4;
  color4 |= (ushort) color.z <<  8;
  color4 |= (ushort) color.w << 12;

  buffer_out[gid] = color4;
}

kernel void gl_readnpixels_rgba8(
  global uint* buffer_out,
  // framebuffer data
  global void* colorbuffer,
  const uint internalformat,
  const uint color_width,
  // readnpixels data
  const int x,
  const int y,
  const uint width,
  const uint height 
) {
  int gid = get_global_id(0);

  uint i, j;
  i = gid % width;
  j = gid / width;

  uint offset = x+y*color_width + i+j*color_width;
  float4 color;
  color = get_color_from_colorbuffer(colorbuffer, offset, internalformat);
  color *= 255;
  // color = round(color);
  color.x = roundf(color.x);
  color.y = roundf(color.y);
  color.z = roundf(color.z);
  color.w = roundf(color.w);

  uint color8 = 0x0;
  color8 |= (uint) color.x <<  0;
  color8 |= (uint) color.y <<  8;
  color8 |= (uint) color.z << 16;
  color8 |= (uint) color.w << 24;

  buffer_out[gid] = color8;
}