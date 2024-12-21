// TODO: Move to the compiler and added into the vertex shader compiled binary.
kernel void gl_viewport_division(
	global float4 *gl_Position,
	private const int4 viewport,
	private const float2 depth_range
) {
  int gid = get_global_id(0);

  int ox = viewport.x+(viewport.z/2);
  int oy = viewport.y+(viewport.w/2);

  gl_Position[gid].x = viewport.z/2 * gl_Position[gid].x + ox; // z == width
  gl_Position[gid].y = viewport.w/2 * gl_Position[gid].y + oy; // w == height
  gl_Position[gid].z = (depth_range.y-depth_range.x)/2 * gl_Position[gid].z + (depth_range.y+depth_range.x)/2;
}
