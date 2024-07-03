
kernel void main_vs (
  // uniforms
  // in attributes
  global const float4 *positions,
  global const float4 *in_color,
  // out attributes
  global float4 *out_color,
  // GL required args
  global float4 *gl_Position
) {
  int gid = get_global_id(0);

  gl_Position[gid] = positions[gid];
  out_color[gid] = in_color[gid];
}

kernel void main_fs (
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
