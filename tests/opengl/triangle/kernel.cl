
__kernel void gl_main_vs (
  // my imp
  __global const float* positions,
  __global const float* color,
  // implementation values
  __global float4 *gl_Positions,
  __global float4 *gl_Primitives
) {
  int gid = get_global_id(0);
  // in out vars
  float x = positions[gid*3];
  float y = positions[gid*3+1];
  float z = positions[gid*3+2];

  float r = color[gid*3];
  float g = color[gid*3+1];
  float b = color[gid*3+2];


  // vertex operations
  gl_Positions[gid] = (float4) (x, y, z, 1.0f);
  gl_Primitives[gid*2] = (float4) (r, g, b, 1.0f);
}

__kernel void gl_main_fs (
  // user values

  // implementation values 
  __global float4 *gl_FragCoord, // position of the fragment in the window space, z is depth value
  __global const float4 *gl_Rasterization,
  __global bool *gl_Discard, // if discarded
  __global float4 *gl_FragColor // out color of the fragment || It is deprecated in OpenGL 3.0 
)
{
  int gid = get_global_id(0);
  // in out vars

  // fragment operations
  gl_FragColor[gid] = gl_Rasterization[gid*2];
}
