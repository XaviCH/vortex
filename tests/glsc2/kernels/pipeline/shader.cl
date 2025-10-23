#define ATTRIBUTE_VEC2 in_tex
#define ATTRIBUTE_VEC3 position, in_color

#define UNIFORM_MAT4 model, perspective, view
#define UNIFORM_SAMPLER2D samp

#define VS_UNIFORM_MAT4 model, perspective, view

#define VARYING_VEC2 out_tex
#define VARYING_VEC3 out_color

#ifdef __COMPILER_RELATIVE_PATH__
    #include <backend/shaders/wrapper.cl>
#else
    #include "glsc2/src/backend/shaders/wrapper.cl"
#endif

VS_MAIN({
    gl_Position    = mul(perspective, mul(view, mul(model,(vec4){position, 1})));
    out_color = in_color;
    out_tex = in_tex;
})

FS_MAIN({
    gl_FragColor = TEXTURE2D(samp, out_tex);
    // gl_FragColor = (vec4)(out_color, 1);
})

#define SHADER

#ifdef __COMPILER_RELATIVE_PATH__
    #include <backend/pipeline/fine_raster.cl>
#else
    #include "glsc2/src/backend/pipeline/fine_raster.cl"
#endif