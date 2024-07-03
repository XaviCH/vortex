
__kernel void main_vs (
  // uniforms

  // in attributes
  constant float4 *positions,
  constant float4 *in_color,
  // out attributes
  global float4 *out_color,
  // GL required args
  global float4 *gl_Positions
) {
  int gid = get_global_id(0);

  gl_Positions[gid] = positions[gid];
  out_color[gid] = in_color[gid];
}

__kernel void main_fs (
  // uniforms

  // in attributes
  global const float4 *out_color,
  // GL required args
  global float4 *gl_FragColor, // position of the fragment in the window space, z is depth value
  // GL optional args
  global float4 *gl_FragCoord, // out color of the fragment || It is deprecated in OpenGL 3.0 
  global bool *gl_Discard // if discarded
)
{
  int gid = get_global_id(0);

  gl_FragColor[gid] = out_color[gid];// out_color[gid];
}
