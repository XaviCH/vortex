__kernel void gl_rgba4 (
  const unsigned int gl_Width,
  const unsigned int gl_Height,
  __global unsigned short *gl_ColorBuffer,
  __global const float4 *gl_FragCoord,
  __global bool *gl_Discard,
  __global float4 *gl_FragColor
) {
  int gid = get_global_id(0);
  
  if (gl_Discard[gid]) return;

  float4 color = gl_FragColor[gid];
  unsigned short value = 0;
  value |= (unsigned short) (color.x * 15);
  value |= (unsigned short) (color.y * 15) << 4;
  value |= (unsigned short) (color.z * 15) << 8;
  value |= (unsigned short) (color.w * 15) << 12;

  /*
  unsigned int x, y;
  x = gl_FragCoord[gid][0];
  y = gl_FragCoord[gid][1];

  gl_ColorBuffer[y*gl_Width + x] = value;
  */
  // TODO: Use gl_FragCoord
  gl_ColorBuffer[gid] = value;
}