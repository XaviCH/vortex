// THRESHOLD ALGORITHM
// https://en.wikipedia.org/wiki/Ordered_dithering



__kernel void gl_rgba4 (
  __global unsigned short *gl_Color,
  __global bool *gl_Discard,
  __global float4 *gl_FragColor
) {
  int gid = get_global_id(0);
  
  if (gl_Discard[gid]) return;

  float4 color = gl_FragColor[gid];

  unsigned short value = 0; // HW optimization ??
  value |= (unsigned short) (color.x * 0xFu) << 0;
  value |= (unsigned short) (color.y * 0xFu) << 4;
  value |= (unsigned short) (color.z * 0xFu) << 8;
  value |= (unsigned short) (color.w * 0xFu) << 12;

  gl_Color[gid] = value;
}

kernel void enabled_rgba4(
  global ushort *buffer,
  global float4 *gl_FragColor,
  global bool *gl_Discard,
  global float4 *gl_FragCoord
) {
  int gid = get_global_id(0);
  
  if (gl_Discard[gid]) return;

  const float _M[4][4] = {
    { 0.f/16, 8.f/16, 2.f/16,10.f/16},
    {12.f/16, 4.f/16,14.f/16, 6.f/16},
    { 3.f/16,11.f/16, 1.f/16, 9.f/16},
    {15.f/16, 7.f/16,13.f/16, 5.f/16},
  };

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

  float4 color = gl_FragColor[gid];
  float4 position = gl_FragCoord[gid];
  float r = 1; // 3.75; // 0xFu / 4bits
  color *= 0xFu;
  color.xyz = color.xyz + r*M[(int)position.x%8][(int)position.y%8];
  color.x = round(color.x);
  color.y = round(color.y);
  color.z = round(color.z);

  ushort value = 0;
  if (color.x > 0.f && color.x <= 15.f) value |= (ushort) color.x << 0;
  else if (color.x > 15.f) value |= 0xFu << 0;
  if (color.y > 0.f && color.y <= 15.f) value |= (ushort) color.y << 4;
  else if (color.y > 15.f) value |= 0xFu << 4;
  if (color.z > 0.f && color.z <= 15.f) value |= (ushort) color.z << 8;
  else if (color.z > 15.f) value |= 0xFu << 8;
  value |= (ushort) color.w << 12;

  buffer[gid] = value;
}

kernel void disabled_rgba4(
  global ushort *buffer,
  global float4 *gl_FragColor,
  global bool *gl_Discard
) {
  int gid = get_global_id(0);
  
  if (gl_Discard[gid]) return;

  float4 color = gl_FragColor[gid];
  ushort value;
  value |= (ushort) (color.x * 0xFu) << 0;
  value |= (ushort) (color.y * 0xFu) << 4;
  value |= (ushort) (color.z * 0xFu) << 8;
  value |= (ushort) (color.w * 0xFu) << 12;

  buffer[gid] = value;
}


__kernel void gl_rgba8 (
  __global unsigned int *gl_Color,
  __global bool *gl_Discard,
  __global float4 *gl_FragColor
) {
  int gid = get_global_id(0);
  
  if (gl_Discard[gid]) return;

  float4 color = gl_FragColor[gid];

  unsigned int value = 0; // HW optimization ??
  value |= (unsigned int) (color.x * 0xFFu) << 0;
  value |= (unsigned int) (color.y * 0xFFu) << 8;
  value |= (unsigned int) (color.z * 0xFFu) << 16;
  value |= (unsigned int) (color.w * 0xFFu) << 24;

  gl_Color[gid] = value;
}