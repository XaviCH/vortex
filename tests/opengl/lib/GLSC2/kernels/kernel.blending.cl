#define GL_R8                             0x8229
#define GL_RG8                            0x822B
#define GL_RGB8                           0x8051
#define GL_RGBA8                          0x8058
#define GL_RGBA4                          0x8056
#define GL_RGB5_A1                        0x8057
#define GL_RGB565                         0x8D62

#define GL_FUNC_ADD                       0x8006
#define GL_FUNC_SUBTRACT                  0x800A
#define GL_FUNC_REVERSE_SUBTRACT          0x800B

#define GL_ZERO                           0
#define GL_ONE                            1
#define GL_SRC_COLOR                      0x0300
#define GL_ONE_MINUS_SRC_COLOR            0x0301
#define GL_SRC_ALPHA                      0x0302
#define GL_ONE_MINUS_SRC_ALPHA            0x0303
#define GL_DST_ALPHA                      0x0304
#define GL_ONE_MINUS_DST_ALPHA            0x0305
#define GL_DST_COLOR                      0x0306
#define GL_ONE_MINUS_DST_COLOR            0x0307
#define GL_SRC_ALPHA_SATURATE             0x0308
#define GL_CONSTANT_COLOR                 0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR       0x8002
#define GL_CONSTANT_ALPHA                 0x8003
#define GL_ONE_MINUS_CONSTANT_ALPHA       0x8004


kernel void gl_blending (
  // fragment data
  global float4* gl_FragColor,
  global bool *discard,
  // framebuffer data
  constant void* colorbuffer,
  const uint type, 
  // blending data
  const uint modeRGB,
  const uint modeAlpha,
  const uint srcRGB,
  const uint srcAlpha,
  const uint dstRGB,
  const uint dstAlpha,
  const float4 color
) {
  int gid = get_global_id(0);
  if (discard[gid]) return;

  global float4* fragColor = gl_FragColor + gid;

  float4 srcColor, dstColor, wsrcColor, wdstColor, outColor;

  srcColor = *fragColor;

  switch (type) {
    case GL_RGBA4:
      {
        ushort color = ((constant ushort*) colorbuffer)[gid];
        dstColor = (float4) { 
          (float) ((color & 0x000Fu) >>  0) / 15.f, 
          (float) ((color & 0x00F0u) >>  4) / 15.f,
          (float) ((color & 0x0F00u) >>  8) / 15.f,
          (float) ((color & 0xF000u) >> 12) / 15.f
        };
      }
      break;
    case GL_RGB5_A1:
      {
        ushort color = ((constant ushort*) colorbuffer)[gid];
        dstColor = (float4) { 
          (float) ((color & 0x001Fu) >>  0) / 31.f, 
          (float) ((color & 0x03E0u) >>  5) / 31.f,
          (float) ((color & 0x7C00u) >> 10) / 31.f,
          (float) ((color & 0x8000u) >> 15) / 1.f
        };
      }
      break;
    case GL_RGB565:
      {
        ushort color = ((constant ushort*) colorbuffer)[gid];
        dstColor = (float4) { 
          (float) ((color & 0x001Fu) >>  0) / 31.f, 
          (float) ((color & 0x07E0u) >>  5) / 63.f,
          (float) ((color & 0xF800u) >> 11) / 31.f,
          1.f
        };
      }
      break;
  }

  switch (srcRGB) {
    case GL_ZERO:
      wsrcColor.xyz = 0;
      break;
    case GL_ONE:
      wsrcColor.xyz = 1;
      break;
    case GL_SRC_COLOR:
      wsrcColor.xyz = srcColor.xyz;
      break;
    case GL_ONE_MINUS_SRC_COLOR:
      wsrcColor.xyz = 1 - srcColor.xyz;
      break;
    case GL_DST_COLOR:
      wsrcColor.xyz = dstColor.xyz;
      break;
    case GL_ONE_MINUS_DST_COLOR:
      wsrcColor.xyz = 1 - dstColor.xyz;
      break;
    case GL_SRC_ALPHA:
      wsrcColor.xyz = srcColor.w;
      break;
    case GL_ONE_MINUS_SRC_ALPHA:
      wsrcColor.xyz = 1 - srcColor.w;
      break;
    case GL_DST_ALPHA:
      wsrcColor.xyz = dstColor.w;
      break;
    case GL_ONE_MINUS_DST_ALPHA:
      wsrcColor.xyz = 1 - dstColor.w;
      break;
    case GL_CONSTANT_COLOR:
      wsrcColor.xyz = color.xyz;
      break;
    case GL_ONE_MINUS_CONSTANT_COLOR:
      wsrcColor.xyz = 1 - color.xyz;
      break;
    case GL_CONSTANT_ALPHA:
      wsrcColor.xyz = color.w;
      break;
    case GL_ONE_MINUS_CONSTANT_ALPHA:
      wsrcColor.xyz = 1 - color.w;
      break;
    case GL_SRC_ALPHA_SATURATE:
      wsrcColor.xyz = min(srcColor.w, 1 - dstColor.w);
      break;
  }

  switch (srcAlpha) {
    case GL_ZERO:
      wsrcColor.w = 0;
      break;
    case GL_ONE:
      wsrcColor.w = 0;
      break;
    case GL_SRC_COLOR:
      wsrcColor.w = srcColor.w;
      break;
    case GL_ONE_MINUS_SRC_COLOR:
      wsrcColor.w = 1 - srcColor.w;
      break;
    case GL_DST_COLOR:
      wsrcColor.w = dstColor.w;
      break;
    case GL_ONE_MINUS_DST_COLOR:
      wsrcColor.w = 1 - dstColor.w;
      break;
    case GL_SRC_ALPHA:
      wsrcColor.w = srcColor.w;
      break;
    case GL_ONE_MINUS_SRC_ALPHA:
      wsrcColor.w = 1 - srcColor.w;
      break;
    case GL_DST_ALPHA:
      wsrcColor.w = dstColor.w;
      break;
    case GL_ONE_MINUS_DST_ALPHA:
      wsrcColor.w = 1 - dstColor.w;
      break;
    case GL_CONSTANT_COLOR:
      wsrcColor.w = color.w;
      break;
    case GL_ONE_MINUS_CONSTANT_COLOR:
      wsrcColor.w = 1 - color.w;
      break;
    case GL_CONSTANT_ALPHA:
      wsrcColor.w = color.w;
      break;
    case GL_ONE_MINUS_CONSTANT_ALPHA:
      wsrcColor.w = 1 - color.w;
      break;
    case GL_SRC_ALPHA_SATURATE:
      wsrcColor.w = 1;
      break;
  }

  switch (dstRGB) {
    case GL_ZERO:
      wdstColor.xyz = 0;
      break;
    case GL_ONE:
      wdstColor.xyz = 1;
      break;
    case GL_SRC_COLOR:
      wdstColor.xyz = srcColor.xyz;
      break;
    case GL_ONE_MINUS_SRC_COLOR:
      wdstColor.xyz = 1 - srcColor.xyz;
      break;
    case GL_DST_COLOR:
      wdstColor.xyz = dstColor.xyz;
      break;
    case GL_ONE_MINUS_DST_COLOR:
      wdstColor.xyz = 1 - dstColor.xyz;
      break;
    case GL_SRC_ALPHA:
      wdstColor.xyz = srcColor.w;
      break;
    case GL_ONE_MINUS_SRC_ALPHA:
      wdstColor.xyz = 1 - srcColor.w;
      break;
    case GL_DST_ALPHA:
      wdstColor.xyz = dstColor.w;
      break;
    case GL_ONE_MINUS_DST_ALPHA:
      wdstColor.xyz = 1 - dstColor.w;
      break;
    case GL_CONSTANT_COLOR:
      wdstColor.xyz = color.xyz;
      break;
    case GL_ONE_MINUS_CONSTANT_COLOR:
      wdstColor.xyz = 1 - color.xyz;
      break;
    case GL_CONSTANT_ALPHA:
      wdstColor.xyz = color.w;
      break;
    case GL_ONE_MINUS_CONSTANT_ALPHA:
      wdstColor.xyz = 1 - color.w;
      break;
    case GL_SRC_ALPHA_SATURATE:
      wdstColor.xyz = min(srcColor.w, 1 - dstColor.w);
      break;
  }

  switch (dstAlpha) {
    case GL_ZERO:
      wdstColor.w = 0;
      break;
    case GL_ONE:
      wdstColor.w = 0;
      break;
    case GL_SRC_COLOR:
      wdstColor.w = srcColor.w;
      break;
    case GL_ONE_MINUS_SRC_COLOR:
      wdstColor.w = 1 - srcColor.w;
      break;
    case GL_DST_COLOR:
      wdstColor.w = dstColor.w;
      break;
    case GL_ONE_MINUS_DST_COLOR:
      wdstColor.w = 1 - dstColor.w;
      break;
    case GL_SRC_ALPHA:
      wdstColor.w = srcColor.w;
      break;
    case GL_ONE_MINUS_SRC_ALPHA:
      wdstColor.w = 1 - srcColor.w;
      break;
    case GL_DST_ALPHA:
      wdstColor.w = dstColor.w;
      break;
    case GL_ONE_MINUS_DST_ALPHA:
      wdstColor.w = 1 - dstColor.w;
      break;
    case GL_CONSTANT_COLOR:
      wdstColor.w = color.w;
      break;
    case GL_ONE_MINUS_CONSTANT_COLOR:
      wdstColor.w = 1 - color.w;
      break;
    case GL_CONSTANT_ALPHA:
      wdstColor.w = color.w;
      break;
    case GL_ONE_MINUS_CONSTANT_ALPHA:
      wdstColor.w = 1 - color.w;
      break;
    case GL_SRC_ALPHA_SATURATE:
      wdstColor.w = 1;
      break;
  }

  switch (modeRGB) {
    case GL_FUNC_ADD:
      outColor.xyz = srcColor.xyz * wsrcColor.xyz + dstColor.xyz * wdstColor.xyz;
      break;
    case GL_FUNC_SUBTRACT:
      outColor.xyz = srcColor.xyz * wsrcColor.xyz - dstColor.xyz * wdstColor.xyz;
      break;
    case GL_FUNC_REVERSE_SUBTRACT:
      outColor.xyz = dstColor.xyz * wdstColor.xyz - srcColor.xyz * wsrcColor.xyz;
      break;
  }

  switch (modeAlpha) {
    case GL_FUNC_ADD:
      outColor.w = srcColor.w * wsrcColor.w + dstColor.w * wdstColor.w;
      break;
    case GL_FUNC_SUBTRACT:
      outColor.w = srcColor.w * wsrcColor.w - dstColor.w * wdstColor.w;
      break;
    case GL_FUNC_REVERSE_SUBTRACT:
      outColor.w = dstColor.w * wdstColor.w - srcColor.w * wsrcColor.w;
      break;
  }

  *fragColor = outColor;
}