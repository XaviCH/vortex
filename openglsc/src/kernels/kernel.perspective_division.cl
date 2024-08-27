kernel void gl_perspective_division(
	global float4* gl_Positions
) {
  int gid = get_global_id(0);

  gl_Positions[gid].xyz /= gl_Positions[gid].w;
}
