float4 mul(float16 mat, float4 vec) {
  float4 result = 0;

  for(int i=0; i<4;++i) {
    float value = 0;
    for(int j=0; j<4; j+=1) {
      value += mat[i*4+j]*vec[j];
    }
    result[i] = value;
  }

  return result;
}

kernel void main_vs (
  constant float16 *perspective,
  constant float16 *view,
  constant float16 *model,
  global const float4 *position,
  global const float4 *in_color,
  global float4 *out_color,
  global float4 *gl_Position
) {
  int gid = get_global_id(0);

  gl_Position[gid] = mul(*perspective,mul(*view, mul(*model,position[gid])));
  out_color[gid] = in_color[gid];
}

kernel void main_fs (
  global const float4 *out_color,
  global float4 *gl_FragColor
)
{
  int gid = get_global_id(0);
  
  gl_FragColor[gid] = out_color[gid];
}
