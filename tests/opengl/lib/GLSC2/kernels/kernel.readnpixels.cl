#define GL_R8                             0x8229
#define GL_RG8                            0x822B
#define GL_RGB8                           0x8051
#define GL_RGBA8                          0x8058

#define GL_RGBA4                          0x8056
#define GL_RGB5_A1                        0x8057
#define GL_RGB565                         0x8D62


__kernel void gl_rgba4_rgba8(
    __global const unsigned short* buf_in,
    __global unsigned int* buf_out,
    const int x,
    const int y,
    const int width,
    const int height
) {
  int gid = get_global_id(0);
  // cordinates of the (i,j)
  int i = gid % width;
  int j = gid / width;

  unsigned short color4 = buf_in[(j+y)*width+(i+x)];

  unsigned int color8 = 0x0;
  color8 |= (unsigned int) (color4 & 0x000F) << 4 | (color4 & 0x000F);
  color8 |= (unsigned int) (color4 & 0x00F0) << 8 | (color4 & 0x00F0) << 4;
  color8 |= (unsigned int) (color4 & 0x0F00) << 12 | (color4 & 0x0F00) << 8;
  color8 |= (unsigned int) (color4 & 0xF000) << 16 | (color4 & 0xF000) << 12;

  buf_out[j*width+i] = color8;
}


inline float4 get_color_from_colorbuffer(constant void* colorbuffer, uint offset, uint internalformat) {
  float4 color; 

  switch (internalformat) {
  case GL_R8:
    color.x = ((float) *((constant uchar*) colorbuffer)) / 255;
    break;
  case GL_RG8:
    {
      constant uchar* buffer = (constant uchar*) colorbuffer + offset * 2;
      color.x = (float) buffer[0] / 255;
      color.y = (float) buffer[1] / 255;
    }
    break;
  case GL_RGB8:
    {
      constant uchar* buffer = (constant uchar*) colorbuffer + offset * 3;
      color.x = (float) buffer[0] / 255;
      color.y = (float) buffer[1] / 255;
      color.z = (float) buffer[2] / 255;
    }
    break;
  case GL_RGBA8:
    {
      constant uchar* buffer = (constant uchar*) colorbuffer + offset * 4;
      color.x = (float) buffer[0] / 255;
      color.y = (float) buffer[1] / 255;
      color.z = (float) buffer[2] / 255;
      color.w = (float) buffer[3] / 255;
    }
    break;
  case GL_RGBA4:
    {
      constant ushort* buffer = ((constant ushort*) colorbuffer) + offset;
      color.x = (float) ((*buffer & 0x000F) >>  0) / 15;
      color.y = (float) ((*buffer & 0x00F0) >>  4) / 15;
      color.z = (float) ((*buffer & 0x0F00) >>  8) / 15;
      color.w = (float) ((*buffer & 0xF000) >> 12) / 15;
    }
    break;
  case GL_RGB5_A1:
    {
      constant ushort* buffer = (constant ushort*) colorbuffer + offset;
      color.x = (float) (*buffer & 0x001F >>  0) / 31;
      color.y = (float) (*buffer & 0x03E0 >>  5) / 31;
      color.z = (float) (*buffer & 0x7C00 >> 10) / 31;
      color.w = (float) (*buffer & 0x8000 >> 15) /  1;
    }
    break;
  case GL_RGB565:
    {
      constant ushort* buffer = (constant ushort*) colorbuffer + offset;
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
  constant void* colorbuffer,
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
  constant void* colorbuffer,
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

  uint color8 = 0x0;
  color8 |= (uint) color.x <<  0;
  color8 |= (uint) color.y <<  8;
  color8 |= (uint) color.z << 16;
  color8 |= (uint) color.w << 24;

  buffer_out[gid] = color8;
}