
kernel void main_vs (
  global const float4 *position,
  const float4 in_color,
  global float4 *out_color,
  global float4 *gl_Position
) {
  int gid = get_global_id(0);

  gl_Position[gid] = position[gid];
  out_color[gid] = in_color;
}

kernel void main_fs (
  global const float4 *out_color,
  global float4 *gl_FragColor
)
{
  int gid = get_global_id(0);
  
  gl_FragColor[gid] = out_color[gid];
}
