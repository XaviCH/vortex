// TODO: Move to the compiler and added into the vertex shader compiled binary.
kernel void gl_perspective_division(
	global float4* gl_Position
) {
  int gid = get_global_id(0);

  gl_Position[gid].xyz /= gl_Position[gid].w;
}
