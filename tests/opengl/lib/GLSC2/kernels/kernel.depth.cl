#define GL_NEVER                          0x0200
#define GL_LESS                           0x0201
#define GL_EQUAL                          0x0202
#define GL_LEQUAL                         0x0203
#define GL_GREATER                        0x0204
#define GL_NOTEQUAL                       0x0205
#define GL_GEQUAL                         0x0206
#define GL_ALWAYS                         0x0207

#define GL_KEEP                           0x1E00
#define GL_ZERO                           0
#define GL_REPLACE                        0x1E01
#define GL_INCR                           0x1E02
#define GL_DECR                           0x1E03
#define GL_INVERT                         0x150A
#define GL_INCR_WRAP                      0x8507
#define GL_DECR_WRAP                      0x8508

void stencil_operation(uint operation, int ref, uint mask, global uchar* stencil_buffer) {
  switch(operation) {
    case GL_KEEP:
      return;
    case GL_ZERO:
      *stencil_buffer = (*stencil_buffer & ~mask) | (0 & mask);
      return;
    case GL_REPLACE:
      *stencil_buffer = (*stencil_buffer & ~mask) | (ref & mask);
      return;
    case GL_INCR:
      if (*stencil_buffer < 0xFFu) *stencil_buffer = (*stencil_buffer & ~mask) | ((*stencil_buffer + 1) & mask);
      return;
    case GL_DECR:
      if (*stencil_buffer > 0x00u) *stencil_buffer = (*stencil_buffer & ~mask) | ((*stencil_buffer - 1) & mask);
      else *stencil_buffer = mask & (*stencil_buffer);
      return;
    case GL_INVERT:
      *stencil_buffer = (*stencil_buffer & ~mask) | (~(*stencil_buffer) & mask);
      return;
    case GL_INCR_WRAP:
      *stencil_buffer = (*stencil_buffer & ~mask) | ((*stencil_buffer + 1) & mask);
      return;
    case GL_DECR_WRAP:
      *stencil_buffer = (*stencil_buffer & ~mask) | ((*stencil_buffer - 1) & mask);
      return;
  }
}

bool depth_test(uint func, float z, uchar mask, global ushort* depth_buffer) {
  ushort value = z * 0xFFFFu;

  switch(func) {
    case GL_NEVER:
      return false;
    case GL_LESS:
      if (value < *depth_buffer) {
        if (mask) *depth_buffer = value;
        return true;
      } else return false;
    case GL_EQUAL:
      if (value == *depth_buffer) return true;
      else return false;
    case GL_LEQUAL:
      if (value <= *depth_buffer) {
        if (mask) *depth_buffer = value;
        return true;
      } else return false;
    case GL_GREATER:
      if (value > *depth_buffer) {
        if (mask) *depth_buffer = value;
        return true;
      } else return false;
    case GL_NOTEQUAL:
      if (value != *depth_buffer) {
        if (mask) *depth_buffer = value;
        return true;
      } else return false;
    case GL_GEQUAL:
      if (value >= *depth_buffer) {
        if (mask) *depth_buffer = value;
        return true;
      } else return false;
    case GL_ALWAYS:
      if (mask) *depth_buffer = value;
      return true;
  }
}

kernel void gl_depth_test(
  // fragment data
  constant float4* gl_FragCoord,
  constant bool* facing, // 0: front, 1: back
  global bool* discard,
  // framebuffer data
  global ushort* depth_buffer,
  global uchar* stencil_buffer,
  const uchar depth_mask,
  const uint stencil_front_mask,
  const uint stencil_back_mask,
  // depth data
  const uint func,
  // stencil data
  const uint front_ref,
  const uint front_dpfail,
  const uint front_dppass,
  const uint back_ref,
  const uint back_dpfail,
  const uint back_dppass
) {
  int gid = get_global_id(0);
  if (discard[gid]) return;

  bool pass;
  if (depth_buffer == NULL) pass = true; // depth test always pass, NULL if framebuffer does not have any, or is disabled
  else pass = depth_test(func, gl_FragCoord[gid].z, depth_mask, depth_buffer + gid);

  if (stencil_buffer != NULL) {
    bool face = facing[gid];

    if (pass) {
      if (face) stencil_operation(back_dppass, back_ref, stencil_back_mask, stencil_buffer + gid);
      else stencil_operation(front_dppass, front_ref, stencil_front_mask, stencil_buffer + gid);
    } else {
      if (face) stencil_operation(back_dpfail, back_ref, stencil_back_mask, stencil_buffer + gid);
      else stencil_operation(front_dpfail, front_ref, stencil_front_mask, stencil_buffer + gid);
    }
  }

  if (!pass) discard[gid] = true;
}