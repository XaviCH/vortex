// THRESHOLD ALGORITHM
// https://en.wikipedia.org/wiki/Ordered_dithering

#define GL_R8                             0x8229
#define GL_RG8                            0x822B
#define GL_RGB8                           0x8051
#define GL_RGBA8                          0x8058

#define GL_RGBA4                          0x8056
#define GL_RGB5_A1                        0x8057
#define GL_RGB565                         0x8D62


float4 dithering(constant float4* fragColor, constant float4* fragCoord, const uint internalformat) {
  const float M[8][8] = {
    { 0.f/64, 32.f/64,  8.f/64, 40.f/64,  2.f/64, 34.f/64, 10.f/64, 42.f/64},
    {48.f/64, 16.f/64, 56.f/64, 24.f/64, 50.f/64, 18.f/64, 58.f/64, 26.f/64},
    {12.f/64, 44.f/64,  4.f/64, 36.f/64, 14.f/64, 46.f/64,  6.f/64, 38.f/64},
    {60.f/64, 28.f/64, 52.f/64, 20.f/64, 62.f/64, 30.f/64, 54.f/64, 22.f/64},
    { 3.f/64, 35.f/64, 11.f/64, 43.f/64,  1.f/64, 33.f/64,  9.f/64, 41.f/64},
    {51.f/64, 19.f/64, 59.f/64, 27.f/64, 49.f/64, 17.f/64, 57.f/64, 25.f/64},
    {15.f/64, 47.f/64,  7.f/64, 39.f/64, 13.f/64, 45.f/64,  5.f/64, 37.f/64},
    {63.f/64, 31.f/64, 55.f/64, 23.f/64, 61.f/64, 29.f/64, 53.f/64, 21.f/64}
  };

  const float r = 1; // TODO: May change depending on the internal format

  float4 color;
  
  switch (internalformat) {
    case GL_R8:
    case GL_RG8:
    case GL_RGB8:
    case GL_RGBA8:
      color = *fragColor * 0xFFu;
      break;
    case GL_RGBA4:
      color = *fragColor * 0xFu;
      break;
    case GL_RGB5_A1:
      color.xyz = fragColor->xyz * 0x1Fu;
      break;
    case GL_RGB565:
      color.xz = fragColor->xz * 0x1Fu;
      color.y = fragColor->y * 0x3Fu;
      break;
  }

  color.xyz = color.xyz + r*M[(int)fragCoord->x%8][(int)fragCoord->y%8];
  color = round(color);

  switch (internalformat) {
    case GL_R8:
    case GL_RG8:
    case GL_RGB8:
    case GL_RGBA8:
      color = max(color, (float4) {0,0,0,0});
      color = min(color, (float4) {255,255,255,255});
      break;
    case GL_RGBA4:
      color = max(color, (float4) {0,0,0,0});
      color = min(color, (float4) {15,15,15,15});
      break;
    case GL_RGB5_A1:
      color = max(color, (float4) {0,0,0,0});
      color = min(color, (float4) {31,31,31,1});
      break;
    case GL_RGB565:
      color = max(color, (float4) {0,0,0,0});
      color = min(color, (float4) {31,63,31,0});
      break;
  }
  
  return color;
}

kernel void gl_dithering(
  // Fragment data
  constant float4 *gl_FragColor,
  constant float4 *gl_FragCoord,
  constant bool *gl_Discard,
  // Framebuffer data
  global void* colorbuffer,
  const uint internalformat,
  const uchar4 mask,
  // Dither data
  const uchar enabled
) {
  int gid = get_global_id(0);
  if (gl_Discard[gid]) return;
  
  float4 color;

  if (enabled) {
    color = dithering(&gl_FragColor[gid], &gl_FragCoord[gid], internalformat);
  } else {
    switch (internalformat) {
    case GL_R8:
    case GL_RG8:
    case GL_RGB8:
    case GL_RGBA8:
      color = round(gl_FragColor[gid] * 0xFFu);
      break;
    case GL_RGBA4:
      color = round(gl_FragColor[gid] * 0xFu);
      break;
    case GL_RGB5_A1:
      color = gl_FragColor[gid];
      color.xyz *= 0x1Fu;
      color = round(color);
      break;
    case GL_RGB565:
      color.xz = gl_FragColor[gid].xz * 0x1Fu;
      color.y = gl_FragColor[gid].y * 0x3Fu;
      color = round(color);
      break;
    }
  }

  switch (internalformat) {
  case GL_R8:
    if (mask.x)
      ((global uchar*) colorbuffer)[gid] = (uchar) color.x;
    break;
  case GL_RG8:
    {
      global uchar* colorbuffer_ptr = (global uchar*) colorbuffer + gid*2;
      if (mask.x)
        colorbuffer_ptr[0] = (uchar) color.x;
      if (mask.y)
        colorbuffer_ptr[1] = (uchar) color.y;
    }
    break;
  case GL_RGB8:
    {
      global uchar* colorbuffer_ptr = (global uchar*) colorbuffer + gid*3;
      if (mask.x)
        colorbuffer_ptr[0] = (uchar) color.x;
      if (mask.y)
        colorbuffer_ptr[1] = (uchar) color.y;
      if (mask.z)
        colorbuffer_ptr[2] = (uchar) color.z;
    }
    break;
  case GL_RGBA8:
    {
      global uchar* colorbuffer_ptr = (global uchar*) colorbuffer + gid*4;
      if (mask.x)
        colorbuffer_ptr[0] = (uchar) color.x;
      if (mask.y)
        colorbuffer_ptr[1] = (uchar) color.y;
      if (mask.z)
        colorbuffer_ptr[2] = (uchar) color.z;
      if (mask.w)
        colorbuffer_ptr[3] = (uchar) color.w;
    }
    break;
  case GL_RGBA4:
    {
      global ushort* colorbuffer_ptr = (global ushort*) colorbuffer + gid;

      ushort mask_color = *colorbuffer_ptr;
      if (mask.x)
        mask_color = mask_color & ~0x000Fu | (ushort) color.x <<  0 & 0x000Fu;
      if (mask.y)
        mask_color = mask_color & ~0x00F0u | (ushort) color.y <<  4 & 0x00F0u;
      if (mask.z)
        mask_color = mask_color & ~0x0F00u | (ushort) color.z <<  8 & 0x0F00u;
      if (mask.w)
        mask_color = mask_color & ~0xF000u | (ushort) color.w << 12 & 0xF000u;
      
      *colorbuffer_ptr = mask_color;
    }
    break;
  case GL_RGB5_A1:
    {
      global ushort* colorbuffer_ptr = (global ushort*) colorbuffer + gid;
      ushort mask_color = *colorbuffer_ptr;
      if (mask.x)
        mask_color = mask_color & ~0x001Fu | (ushort) color.x <<  0 & 0x001Fu;
      if (mask.y)
        mask_color = mask_color & ~0x03E0u | (ushort) color.y <<  5 & 0x03E0u;
      if (mask.z)
        mask_color = mask_color & ~0x7C00u | (ushort) color.z << 10 & 0x7C00u;
      if (mask.w)
        mask_color = mask_color & ~0x8000u | (ushort) color.w << 15 & 0x8000u;
      
      *colorbuffer_ptr = mask_color;
    }
    break;
  case GL_RGB565:
    {
      global ushort* colorbuffer_ptr = (global ushort*) colorbuffer + gid;
      ushort mask_color = *colorbuffer_ptr;
      if (mask.x)
        mask_color = mask_color & ~0x001Fu | (ushort) color.x <<  0 & 0x001Fu;
      if (mask.y)
        mask_color = mask_color & ~0x07E0u | (ushort) color.y <<  5 & 0x07E0u;
      if (mask.z)
        mask_color = mask_color & ~0xF800u | (ushort) color.z << 11 & 0xF800u;
      
      *colorbuffer_ptr = mask_color;
    }
    break;
  }

}