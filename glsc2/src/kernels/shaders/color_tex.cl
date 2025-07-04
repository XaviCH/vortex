#define ATTRIBUTE_VEC3 position, color
#define ATTRIBUTE_VEC2 tex_coord

#define VARYING_VEC2 out_tex_coord
#define VARYING_VEC3 out_color

#define VS_UNIFORM_MAT4 model, proyection, view

#define FS_UNIFORM_SAMPLER2D mytexture

#ifdef __COMPILER_RELATIVE_PATH__
#include "common_.cl"
#else
#include "glsc2/src/kernels/shaders/common_.cl"
#endif


VS_MAIN({
    gl_Position    = (float4){position, 1};
    out_color      = color;
    out_tex_coord  = tex_coord;
})

FS_MAIN({
    gl_FragColor = TEXTURE2D(mytexture, out_tex_coord);
})