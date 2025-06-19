#ifdef __COMPILER_RELATIVE_PATH__
#include "common.cl"
#else
#include "glsc2/src/kernels/common.cl"
#endif

inline uint get_blending_eq_color   (uint c_blending_data) { return (c_blending_data >>  0) & 0x3u; }
inline uint get_blending_eq_alpha   (uint c_blending_data) { return (c_blending_data >>  2) & 0x3u; }
inline uint get_blending_func_color_src     (uint c_blending_data) { return (c_blending_data >> 16) & 0xfu; }
inline uint get_blending_func_alpha_src     (uint c_blending_data) { return (c_blending_data >> 20) & 0xfu; }
inline uint get_blending_func_color_dst     (uint c_blending_data) { return (c_blending_data >> 24) & 0xfu; }
inline uint get_blending_func_alpha_dst     (uint c_blending_data) { return (c_blending_data >> 28) & 0xfu; }

inline void apply_blend_color_func(uint func, const uint4* src, const uint4* dst, const uint4* color, uint4* out) {
  switch (func) {
    default:
    case GL_ZERO:
      out->xyz = 0;
      break;
    case GL_ONE:
      out->xyz = 1;
      break;
    case GL_SRC_COLOR:
      out->xyz = src->xyz;
      break;
    case GL_ONE_MINUS_SRC_COLOR:
      out->xyz = 1 - src->xyz;
      break;
    case GL_DST_COLOR:
      out->xyz = dst->xyz;
      break;
    case GL_ONE_MINUS_DST_COLOR:
      out->xyz = 1 - dst->xyz;
      break;
    case GL_SRC_ALPHA:
      out->xyz = src->w;
      break;
    case GL_ONE_MINUS_SRC_ALPHA:
      out->xyz = 1 - src->w;
      break;
    case GL_DST_ALPHA:
      out->xyz = dst->w;
      break;
    case GL_ONE_MINUS_DST_ALPHA:
      out->xyz = 1 - dst->w;
      break;
    case GL_CONSTANT_COLOR:
      out->xyz = color->xyz;
      break;
    case GL_ONE_MINUS_CONSTANT_COLOR:
      out->xyz = 1 - color->xyz;
      break;
    case GL_CONSTANT_ALPHA:
      out->xyz = color->w;
      break;
    case GL_ONE_MINUS_CONSTANT_ALPHA:
      out->xyz = 1 - color->w;
      break;
    case GL_SRC_ALPHA_SATURATE:
      out->xyz = min(src->w, 1 - dst->w);
      break;
  }
}

inline void apply_blend_alpha_func(uint func, const uint4* src, const uint4* dst, const uint4* color, uint4* out) {
  switch (func) {
    default:
    case GL_ZERO:
      out->w = 0;
      break;
    case GL_ONE:
      out->w = 0;
      break;
    case GL_SRC_COLOR:
      out->w = src->w;
      break;
    case GL_ONE_MINUS_SRC_COLOR:
      out->w = 1 - src->w;
      break;
    case GL_DST_COLOR:
      out->w = dst->w;
      break;
    case GL_ONE_MINUS_DST_COLOR:
      out->w = 1 - dst->w;
      break;
    case GL_SRC_ALPHA:
      out->w = src->w;
      break;
    case GL_ONE_MINUS_SRC_ALPHA:
      out->w = 1 - src->w;
      break;
    case GL_DST_ALPHA:
      out->w = dst->w;
      break;
    case GL_ONE_MINUS_DST_ALPHA:
      out->w = 1 - dst->w;
      break;
    case GL_CONSTANT_COLOR:
      out->w = color->w;
      break;
    case GL_ONE_MINUS_CONSTANT_COLOR:
      out->w = 1 - color->w;
      break;
    case GL_CONSTANT_ALPHA:
      out->w = color->w;
      break;
    case GL_ONE_MINUS_CONSTANT_ALPHA:
      out->w = 1 - color->w;
      break;
    case GL_SRC_ALPHA_SATURATE:
      out->w = 1;
      break;
  }
}


inline uint blend(
  uint src, uint dst,
  uint c_blending_color, uint c_blending_data
) {
  uint4  srcColor,  dstColor, conColor, 
        wsrcColor, wdstColor, outColor;

  uint blending_func_color_src  = get_blending_func_color_src (c_blending_data);
  uint blending_func_color_dst  = get_blending_func_color_dst (c_blending_data);
  uint blending_func_alpha_src  = get_blending_func_alpha_src (c_blending_data);
  uint blending_func_alpha_dst  = get_blending_func_alpha_dst (c_blending_data);
  uint blending_eq_color    = get_blending_eq_color   (c_blending_data);
  uint blending_eq_alpha    = get_blending_eq_alpha   (c_blending_data);

  srcColor = (uint4){
    (src >>  0) & 0xFFu, 
    (src >>  8) & 0xFFu, 
    (src >> 16) & 0xFFu, 
    (src >> 24) & 0xFFu
  };
  dstColor = (uint4){
    (dst >>  0) & 0xFFu, 
    (dst >>  8) & 0xFFu, 
    (dst >> 16) & 0xFFu, 
    (dst >> 24) & 0xFFu
  };
  conColor = (uint4){
    (c_blending_color >>  0) & 0xFFu, 
    (c_blending_color >>  8) & 0xFFu, 
    (c_blending_color >> 16) & 0xFFu, 
    (c_blending_color >> 24) & 0xFFu
  };

  apply_blend_color_func(blending_func_alpha_src, &srcColor, &dstColor, &conColor, &wsrcColor);
  apply_blend_alpha_func(blending_func_alpha_src, &srcColor, &dstColor, &conColor, &wsrcColor);
  apply_blend_color_func(blending_func_alpha_dst, &srcColor, &dstColor, &conColor, &wdstColor);
  apply_blend_alpha_func(blending_func_alpha_dst, &srcColor, &dstColor, &conColor, &wdstColor);

  switch (blending_eq_color) {
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

  switch (blending_eq_alpha) {
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

  return 
    (outColor.x <<  0) |
    (outColor.y <<  8) |
    (outColor.z << 16) |
    (outColor.w << 24) ;

}