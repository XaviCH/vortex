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


bool stencil_test(uint func, int ref, uint mask, uchar value) {
  switch(func) {
    case GL_NEVER:
      return false;
    case GL_LESS:
      return (mask & ref) < (mask & value);
    case GL_EQUAL:
      return (mask & ref) == (mask & value);
    case GL_LEQUAL:
      return (mask & ref) <= (mask & value);
    case GL_GREATER:
      return (mask & ref) > (mask & value);
    case GL_NOTEQUAL:
      return (mask & ref) != (mask & value);
    case GL_GEQUAL:
      return (mask & ref) >= (mask & value);
    case GL_ALWAYS:
      return true;
  }
}

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

kernel void gl_stencil_test (
  // fragment data
  global bool* facing, // 0: front, 1: back
  global bool *discard,
  // framebuffer data
  global uchar* stencil_buffer,
  const uint stencil_front_mask,
  const uint stencil_back_mask,
  // stencil data
  const uint front_func,
  const int  front_ref,
  const uint front_mask,
  const uint front_sfail,
  const uint back_func,
  const int  back_ref,
  const uint back_mask,
  const uint back_sfail
) {
  if (stencil_buffer == NULL) return;

  int gid = get_global_id(0);
  if (discard[gid]) return;

  bool face = facing[gid];
  uchar stencil_value = stencil_buffer[gid];

  bool pass;
  if (!face) {
    pass = stencil_test(back_func, back_ref, back_mask, stencil_value);
    if (pass) return;

    stencil_operation(back_sfail, back_ref, stencil_back_mask, stencil_buffer + gid);
  } else {
    pass = stencil_test(front_func, front_ref, front_mask, stencil_value); 
    if(pass) return;
    
    stencil_operation(front_sfail, front_ref, stencil_front_mask, stencil_buffer + gid);
  }

  discard[gid] = true;
}