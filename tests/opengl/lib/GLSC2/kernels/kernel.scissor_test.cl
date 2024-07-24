kernel void gl_scissor_test (
  // fragment data
  constant float4* gl_FragCood,
  global bool *discard,
  // scissor data
  const int left,
  const int bottom,
  const uint width,
  const uint height
) {
  int gid = get_global_id(0);
  if (discard[gid]) return;

  int x = gl_FragCood[gid].x;
  int y = gl_FragCood[gid].y;

  discard[gid] = left <= x && x < left + width && bottom <= y && y < bottom + height;  
}