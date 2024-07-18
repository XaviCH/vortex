#ifndef HOSTGPU
typedef constant unsigned char* image_t;
typedef struct __attribute__((packed)) {
  unsigned int width, height;
  unsigned int internalformat;
  unsigned int flags;
} sampler2D_t;

float4 __attribute__((overloadable)) read_imagef(image_t image, sampler2D_t sampler, float4 coord) {

  int w = (int) (sampler.width * coord.x) % sampler.width;
  int h = (int) (sampler.height * coord.y) % sampler.height;
  image_t color = image + (h*sampler.width + w)*4;
  
  return (float4) ((float)*color / 256, (float)*(color+1) / 256, (float)*(color+2) / 256, (float)*(color+3) / 256);
}
#else
typedef image2d_t image_t;
typedef sampler_t sampler2D_t;
#endif

kernel void main_vs (
  global const float4 *position,
  global const float4 *in_texCoord,
  global float4 *out_texCoord,
  global float4 *gl_Position
) {
  int gid = get_global_id(0);

  gl_Position[gid] = position[gid];
  out_texCoord[gid] = in_texCoord[gid];
}

kernel void main_fs (
  global const float4 *out_texCoord,
  global float4 *gl_FragColor,
  sampler2D_t ourTexture,
  image_t image
)
{
  int gid = get_global_id(0);
  gl_FragColor[gid] = read_imagef(image, ourTexture, out_texCoord[gid]);
}
