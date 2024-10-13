#include <math.h>
#include <string.h>
#include <GLSC2/glsc2.h>
#include "kernel.c" // TODO: may be interesting to extract it to an interface so could be re implementated with CUDA

#include "debug.h"
#include "config.h"
#include "types.h"

#define P99_PROTECT(...) __VA_ARGS__

#define NOT_IMPLEMENTED              \
    {                               \
        printf("Funtion %s at %s:%d is not implemented.\n", __func__, __FILE__, __LINE__); \
        exit(1);                     \
    }

#define INTERNAL_ERROR               \
    ({                               \
        printf("Unexpected reached point in %s at %s:%d.\n",__func__, __FILE__, __LINE__);  \
        exit(1);                     \
    })

GLenum gl_error = GL_NO_ERROR;

#define RETURN_ERROR(error)                     \
    ({                                          \
        if(gl_error == GL_NO_ERROR) {           \
            gl_error = error;                   \
            printf("OpenGL error generated (%d | %x) in %s at %s:%d.\n", gl_error, gl_error, __func__, __FILE__, __LINE__);    \
            exit(error);                        \
        }                                       \
        return;                                 \
    })

#define RETURN_ERROR_WITH_MESSAGE(error, message) \
    ({                                            \
        if(gl_error == GL_NO_ERROR) {           \
            gl_error = error;                   \
            printf("OpenGL error generated (%d | %x) in %s at %s:%d.\n" message "\n", gl_error, gl_error, __func__, __FILE__, __LINE__);    \
            exit(error);                        \
        }                                       \
        return;                                 \
    })    

/************ CONTEXT ************\
 * TODO: Create context using a context manager instead of the library 
 * 
*/

/****** CONFIGURATION INITIAL STATE ******/

kernel_container_t  _kernels; // Filled using constructor attribute
box_t               _viewport; // TODO: Checkout the default initial value

enabled_container_t _enableds = {
    .scissor_test = GL_FALSE,
    .stencil_test = GL_FALSE,
    .depth_test = GL_FALSE,
    .blend = GL_FALSE,
    .dither = GL_TRUE,
    .cull_face = GL_FALSE,
};

mask_container_t _masks = {
    .color = { .r=GL_TRUE, .g=GL_TRUE, .b=GL_TRUE, .a=GL_TRUE },
    .depth = GL_TRUE,
    .stencil = GL_TRUE
};

GLenum _front_face = GL_CCW;
GLenum _cull_face = GL_BACK;

pixel_store_t _pixel_store = { .unpack_aligment = 4 };

cl_command_queue  _vertex_attrib_command_queues[MAX_VERTEX_ATTRIBS];
unsigned char     _vertex_attrib_enable[MAX_VERTEX_ATTRIBS]; // TODO: use it as checker for state
unsigned char     _vertex_attrib_state[MAX_VERTEX_ATTRIBS];
vertex_attrib_u   _vertex_attribs[MAX_VERTEX_ATTRIBS];
uint32_t _current_active_texture = 0;
active_texture_t _active_textures[MAX_COMBINED_TEXTURE_IMAGE_UNITS]; // defines what textures are active

clear_data_t _clear_data = {
    .color = { 0 , 0, 0, 0 },
    .stencil = 0,
    .depth = 1,
};

scissor_data_t _scissor_data = {
    .left = 0,
    .bottom = 0,
    .width = 0, // TODO: If context provider, set to default framebuffer size
    .height = 0, // TODO: If context provider, set to default framebuffer size
};

stencil_data_t _stencil_data = {
    .front = {
        .function = { .func = GL_ALWAYS, .ref = 0, .mask = 1, },
        .operation = { .sfail = GL_KEEP, .dpfail = GL_KEEP, .dppasss = GL_KEEP }
    },
    .back = {
        .function = { .func = GL_ALWAYS, .ref = 0, .mask = 1, },
        .operation = { .sfail = GL_KEEP, .dpfail = GL_KEEP, .dppasss = GL_KEEP }
    },
};

blend_data_t _blend_data = {
    .equation = {
        .modeRGB = GL_FUNC_ADD, .modeAlpha = GL_FUNC_ADD,
    },
    .func = {
        .srcRGB = GL_ZERO, .srcAlpha = GL_ZERO,
        .dstRGB = GL_ONE, .dstAlpha = GL_ONE,
    },
    .color = {
        .red = 0, .green = 0, .blue = 0, .alpha = 0
    }
};

extern unsigned char KERNEL_DEPTH_BIN [];
extern unsigned char KERNEL_SCISSOR_TEST_BIN[];
extern unsigned char KERNEL_STENCIL_TEST_BIN[];
extern unsigned char KERNEL_BLENDING_BIN[];
extern unsigned char KERNEL_DITHER_BIN[];
extern unsigned char KERNEL_RASTERIZATION_BIN[];
extern unsigned char KERNEL_RASTERIZATION_TRIANGLE_FAN_BIN[];
extern unsigned char KERNEL_RASTERIZATION_TRIANGLE_STRIP_BIN[];
extern unsigned char KERNEL_VIEWPORT_DIVISION_BIN[];
extern unsigned char KERNEL_PERSPECTIVE_DIVISION_BIN[];
extern unsigned char KERNEL_READNPIXELS_BIN[];
extern unsigned char KERNEL_STRIDED_WRITE_BIN[];
extern unsigned char KERNEL_CLEAR_BIN[];

__attribute__((constructor))
void __context_constructor__() {
    cl_program  depth_program, stencil_test_program, scissor_test_program, dither_program, rasterization_program, viewport_division_program, 
                perspective_division_program, readnpixels_program, strided_write_program, clear_program, blending_program, rasterization_triangle_fan_program,
                rasterization_triangle_strip_program;


    depth_program                           = createProgramWithBinary(KERNEL_DEPTH_BIN,                            sizeof(KERNEL_DEPTH_BIN));
    scissor_test_program                    = createProgramWithBinary(KERNEL_SCISSOR_TEST_BIN,                     sizeof(KERNEL_SCISSOR_TEST_BIN));
    stencil_test_program                    = createProgramWithBinary(KERNEL_STENCIL_TEST_BIN,                     sizeof(KERNEL_STENCIL_TEST_BIN));
    blending_program                        = createProgramWithBinary(KERNEL_BLENDING_BIN,                         sizeof(KERNEL_BLENDING_BIN));
    dither_program                          = createProgramWithBinary(KERNEL_DITHER_BIN,                           sizeof(KERNEL_DITHER_BIN));
    rasterization_program                   = createProgramWithBinary(KERNEL_RASTERIZATION_BIN,                    sizeof(KERNEL_RASTERIZATION_BIN));
    rasterization_triangle_fan_program      = createProgramWithBinary(KERNEL_RASTERIZATION_TRIANGLE_FAN_BIN,       sizeof(KERNEL_RASTERIZATION_TRIANGLE_FAN_BIN));
    rasterization_triangle_strip_program    = createProgramWithBinary(KERNEL_RASTERIZATION_TRIANGLE_STRIP_BIN,     sizeof(KERNEL_RASTERIZATION_TRIANGLE_STRIP_BIN));
    viewport_division_program               = createProgramWithBinary(KERNEL_VIEWPORT_DIVISION_BIN,                sizeof(KERNEL_VIEWPORT_DIVISION_BIN));
    perspective_division_program            = createProgramWithBinary(KERNEL_PERSPECTIVE_DIVISION_BIN,             sizeof(KERNEL_PERSPECTIVE_DIVISION_BIN));
    readnpixels_program                     = createProgramWithBinary(KERNEL_READNPIXELS_BIN,                      sizeof(KERNEL_READNPIXELS_BIN));
    strided_write_program                   = createProgramWithBinary(KERNEL_STRIDED_WRITE_BIN,                    sizeof(KERNEL_STRIDED_WRITE_BIN));
    clear_program                           = createProgramWithBinary(KERNEL_CLEAR_BIN,                            sizeof(KERNEL_CLEAR_BIN));

    buildProgram(depth_program);
    buildProgram(scissor_test_program);
    buildProgram(stencil_test_program);
    buildProgram(blending_program);
    buildProgram(dither_program);
    buildProgram(rasterization_program);
    buildProgram(rasterization_triangle_fan_program);
    buildProgram(rasterization_triangle_strip_program);
    buildProgram(viewport_division_program);
    buildProgram(perspective_division_program);
    buildProgram(readnpixels_program);
    buildProgram(strided_write_program);
    buildProgram(clear_program);

    _kernels.depth_test     = createKernel(depth_program, "gl_depth_test");
    _kernels.scissor_test   = createKernel(scissor_test_program, "gl_scissor_test");
    _kernels.stencil_test   = createKernel(stencil_test_program, "gl_stencil_test");
    _kernels.blending       = createKernel(blending_program, "gl_blending");

    _kernels.dithering = createKernel(dither_program, "gl_dithering");

    _kernels.rasterization.triangles        = createKernel(rasterization_program, "gl_rasterization_triangle");
    _kernels.rasterization.triangle_fan     = createKernel(rasterization_triangle_fan_program, "gl_rasterization_triangle_fan");
    _kernels.rasterization.triangle_strip   = createKernel(rasterization_triangle_strip_program, "gl_rasterization_triangle_strip");

    _kernels.viewport_division = createKernel(viewport_division_program, "gl_viewport_division");
    _kernels.perspective_division = createKernel(perspective_division_program, "gl_perspective_division");
    _kernels.readnpixels.rgba4 = createKernel(readnpixels_program, "gl_readnpixels_rgba4");
    _kernels.readnpixels.rgba8 = createKernel(readnpixels_program, "gl_readnpixels_rgba8");
    _kernels.strided_write = createKernel(strided_write_program, "gl_strided_write");

    _kernels.clear  = createKernel(clear_program, "gl_clear");

    for (uint32_t queue = 0; queue < MAX_VERTEX_ATTRIBS; ++queue) {
        _vertex_attrib_command_queues[queue] = createCommandQueue(0);
    }
} 

/****** MEMORY HANDLERS ******/
// TODO: 

GLboolean _kernel_load_status;

GLuint _buffer_binding          = 0;
GLuint _framebuffer_binding     = 0;
GLuint _renderbuffer_binding    = 0;
GLuint _texture_binding         = 0;
GLuint _current_program         = 0;

buffer_t        _buffers        [MAX_BUFFER];
framebuffer_t   _framebuffers   [MAX_FRAMEBUFFER];
program_t       _programs       [MAX_PROGRAMS];
renderbuffer_t  _renderbuffers  [MAX_RENDERBUFFER];
texture_t  _textures  [MAX_TEXTURE];

// Depth
GLenum      _depth_func = GL_LESS;
depth_range_t _depth_range = { .n=0.0, .f=1.0};

/****** SHORCUTS MACROS ******/
#define COLOR_ATTACHMENT0 _renderbuffers[_framebuffers[_framebuffer_binding].color_attachment0.position]

#define RENDERBUFFER_COLOR_ATTACHMENT0 _renderbuffers[_framebuffers[_framebuffer_binding].color_attachment0.position]
#define TEXTURE_COLOR_ATTACHMENT0 _textures[_framebuffers[_framebuffer_binding].color_attachment0.position]

#define DEPTH_ATTACHMENT _renderbuffers[_framebuffers[_framebuffer_binding].depth_attachment.position]
#define STENCIL_ATTACHMENT _renderbuffers[_framebuffers[_framebuffer_binding].stencil_attachment.position]

#define FRAMEBUFFER _framebuffers[_framebuffer_binding]
#define CURRENT_PROGRAM _programs[_current_program]


/****** Interface for utils & inline functions ******\
 * Utility or inline function are implemented at the end of the file. 
*/
void* getCommandQueue();

unsigned int size_from_name_type(const char*);
unsigned int type_from_name_type(const char*);

unsigned int sizeof_type(GLenum type) {
    switch (type)
    {
    case GL_BYTE:
    case GL_UNSIGNED_BYTE:
        return 1;
    case GL_SHORT:
    case GL_UNSIGNED_SHORT:
        return 2;
    case GL_INT:
    case GL_UNSIGNED_INT:
    case GL_FLOAT:
        return 4;
    case SAMPLER2D_T:
        return sizeof(cl_sampler);
    case IMAGE_T:
        #ifdef HOSTGPU
        #else
        return sizeof(cl_mem);
        #endif
    }
    NOT_IMPLEMENTED;
} 

/************ OpenGL Interface Implementations ************\
 * 
 * 
*/
GL_APICALL void GL_APIENTRY glActiveTexture (GLenum texture) {
    if (texture < GL_TEXTURE0) RETURN_ERROR(GL_INVALID_ENUM);
    
    uint32_t id = texture - GL_TEXTURE0; 
    if (id >= MAX_COMBINED_TEXTURE_IMAGE_UNITS) RETURN_ERROR(GL_INVALID_OPERATION);
    
    _current_active_texture = id;
}

GL_APICALL void GL_APIENTRY glBindBuffer (GLenum target, GLuint buffer) {
    if (!_buffers[buffer].used) RETURN_ERROR(GL_INVALID_OPERATION);
    
    if (target == GL_ARRAY_BUFFER) {
        _buffer_binding = buffer;
    } else NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glBindFramebuffer (GLenum target, GLuint framebuffer) {
    if (!_framebuffers[framebuffer].used) RETURN_ERROR(GL_INVALID_OPERATION);

    if (target == GL_FRAMEBUFFER) {
        _framebuffer_binding = framebuffer;
    } else NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glBindRenderbuffer (GLenum target, GLuint renderbuffer) {
    if (!_renderbuffers[renderbuffer].used) RETURN_ERROR(GL_INVALID_OPERATION);

    if (target == GL_RENDERBUFFER) {
        _renderbuffer_binding = renderbuffer;
    } else NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glBindTexture (GLenum target, GLuint texture) {
    if (!_textures[texture].used) RETURN_ERROR(GL_INVALID_OPERATION);

    if (target == GL_TEXTURE_2D) {
        _active_textures[_current_active_texture].binding = texture;
        _texture_binding = texture;
    } else NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glBlendColor (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
    _blend_data.color = (blend_color_t) {
        .red = red,
        .green = green,
        .blue = blue,
        .alpha = alpha,
    };
}

GL_APICALL void GL_APIENTRY glBlendEquation (GLenum mode) {
    switch (mode)
    {
    case GL_FUNC_ADD:
    case GL_FUNC_SUBTRACT:
    case GL_FUNC_REVERSE_SUBTRACT:
        _blend_data.equation = (blend_equation_t) {
            .modeRGB = mode,
            .modeAlpha = mode,
        };
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

GL_APICALL void GL_APIENTRY glBlendEquationSeparate (GLenum modeRGB, GLenum modeAlpha) {
    switch (modeRGB)
    {
    case GL_FUNC_ADD:
    case GL_FUNC_SUBTRACT:
    case GL_FUNC_REVERSE_SUBTRACT:
        _blend_data.equation.modeRGB = modeRGB;
        break;
    default:
        NOT_IMPLEMENTED;
    }

    switch (modeAlpha)
    {
    case GL_FUNC_ADD:
    case GL_FUNC_SUBTRACT:
    case GL_FUNC_REVERSE_SUBTRACT:
        _blend_data.equation.modeAlpha = modeAlpha;
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

GL_APICALL void GL_APIENTRY glBlendFunc (GLenum sfactor, GLenum dfactor) {
    // TODO: Enum checking 
    _blend_data.func = (blend_func_t) {
        .srcRGB = sfactor, .srcAlpha = sfactor,
        .dstRGB = dfactor, .dstAlpha = dfactor,
    };
}

GL_APICALL void GL_APIENTRY glBlendFuncSeparate (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha) {
    // TODO: Enum checking 
    _blend_data.func = (blend_func_t) {
        .srcRGB = sfactorRGB, .srcAlpha = sfactorAlpha,
        .dstRGB = dfactorRGB, .dstAlpha = dfactorAlpha,
    };
}

GL_APICALL void GL_APIENTRY glBufferData (GLenum target, GLsizeiptr size, const void *data, GLenum usage) {

    if (target == GL_ARRAY_BUFFER) {
        if (usage == GL_STATIC_DRAW) {
            _buffers[_buffer_binding].mem = createBuffer(MEM_READ_ONLY | MEM_COPY_HOST_PTR, size, data);
        }
        else if (usage == GL_DYNAMIC_DRAW || usage == GL_STREAM_DRAW) {
            _buffers[_buffer_binding].mem = createBuffer(MEM_READ_WRITE | MEM_COPY_HOST_PTR, size, data);
        }
    } else NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glBufferSubData (GLenum target, GLintptr offset, GLsizeiptr size, const void *data) {
    
    if (target == GL_ARRAY_BUFFER) {
        cl_command_queue command_queue = getCommandQueue(); 
        cl_int error = clEnqueueWriteBuffer(command_queue, _buffers[_buffer_binding].mem, CL_TRUE, offset, size, data, 0, NULL, NULL);
        if (error == CL_INVALID_VALUE) RETURN_ERROR(GL_INVALID_VALUE);
    } else NOT_IMPLEMENTED;
}

GL_APICALL GLenum GL_APIENTRY glCheckFramebufferStatus (GLenum target) {
    if (target != GL_FRAMEBUFFER) {
        gl_error = GL_INVALID_ENUM;
        return 0;
    };

    // TODO: if target exist, otherwise return GL_FRAMEBUFFER_UNDEFINED
    
    framebuffer_t *framebuffer = &_framebuffers[_framebuffer_binding];
    
    // TODO: if attachment complete, otherwise return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT
    
    if (   framebuffer->color_attachment0.position == 0 
        && framebuffer->depth_attachment.position == 0 
        && framebuffer->stencil_attachment.position == 0
        ) return GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;

    GLsizei width = 0, height = 0;
    if (framebuffer->color_attachment0.position) {
        if (width == 0) width = COLOR_ATTACHMENT0.width;
        else if (width != COLOR_ATTACHMENT0.width) return GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS;
        if (height == 0) height = COLOR_ATTACHMENT0.height;
        else if (height != COLOR_ATTACHMENT0.height) return GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS;
    }

    if (framebuffer->depth_attachment.position) {
        if (width == 0) width = DEPTH_ATTACHMENT.width;
        else if (width != DEPTH_ATTACHMENT.width) return GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS;
        if (height == 0) height = DEPTH_ATTACHMENT.height;
        else if (height != DEPTH_ATTACHMENT.height) return GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS;
    }

    if (framebuffer->stencil_attachment.position) {
        if (width == 0) width = STENCIL_ATTACHMENT.width;
        else if (width != STENCIL_ATTACHMENT.width) return GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS;
        if (height == 0) height = STENCIL_ATTACHMENT.height;
        else if (height != STENCIL_ATTACHMENT.height) return GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS;
    }

    // TODO: Checkout internal formats

    return GL_FRAMEBUFFER_COMPLETE;
}

GL_APICALL void GL_APIENTRY glClear (GLbitfield mask) {
    if (_enableds.pixel_ownership) NOT_IMPLEMENTED;

    typedef struct {
        cl_mem mem;
        uint32_t width, internalformat;
    } buffer_info_t; 

    buffer_info_t buffer_info;

    if (FRAMEBUFFER.color_attachment0.target == GL_RENDERBUFFER) {

        renderbuffer_t* colorbuffer = &_renderbuffers[FRAMEBUFFER.color_attachment0.position];
        
        buffer_info = (buffer_info_t) {
            .mem = colorbuffer->mem,
            .width = colorbuffer->width,
            .internalformat = colorbuffer->internalformat,
        };
    } else {
        texture_t* texture = &_textures[FRAMEBUFFER.color_attachment0.position];
        
        buffer_info = (buffer_info_t) {
            .mem = texture->mem,
            .width = texture->width,
            .internalformat = texture->internalformat,
        };
    }

    cl_uint scissor_enabled, mask_color_r, mask_color_g, mask_color_b, mask_color_a, mask_depth;

    scissor_enabled = _enableds.scissor_test;
    mask_color_r = _masks.color.r;
    mask_color_g = _masks.color.g;
    mask_color_b = _masks.color.b;
    mask_color_a = _masks.color.a;
    mask_depth = _masks.depth;

    setKernelArg(_kernels.clear, 0, sizeof(cl_mem), &buffer_info.mem);
    setKernelArg(_kernels.clear, 1, sizeof(cl_mem), &DEPTH_ATTACHMENT.mem);
    setKernelArg(_kernels.clear, 2, sizeof(cl_mem), &STENCIL_ATTACHMENT.mem);
    setKernelArg(_kernels.clear, 3, sizeof(cl_uint), &buffer_info.internalformat);
    setKernelArg(_kernels.clear, 4, sizeof(cl_uint), &buffer_info.width);
    setKernelArg(_kernels.clear, 5, sizeof(cl_uint), &mask);
    setKernelArg(_kernels.clear, 6, sizeof(cl_float4), &_clear_data.color);
    setKernelArg(_kernels.clear, 7, sizeof(cl_float), &_clear_data.depth);
    setKernelArg(_kernels.clear, 8, sizeof(cl_int), &_clear_data.stencil);
    setKernelArg(_kernels.clear, 9, sizeof(cl_uint), &scissor_enabled);
    setKernelArg(_kernels.clear,10, sizeof(cl_int), &_scissor_data.left);
    setKernelArg(_kernels.clear,11, sizeof(cl_int), &_scissor_data.bottom);
    setKernelArg(_kernels.clear,12, sizeof(cl_uint), &_scissor_data.width);
    setKernelArg(_kernels.clear,13, sizeof(cl_uint), &_scissor_data.height);
    setKernelArg(_kernels.clear,14, sizeof(cl_uint), &mask_color_r);
    setKernelArg(_kernels.clear,15, sizeof(cl_uint), &mask_color_g);
    setKernelArg(_kernels.clear,16, sizeof(cl_uint), &mask_color_b);
    setKernelArg(_kernels.clear,17, sizeof(cl_uint), &mask_color_a);
    setKernelArg(_kernels.clear,18, sizeof(cl_uint), &mask_depth);
    setKernelArg(_kernels.clear,19, sizeof(cl_uint), &_masks.stencil);
    enqueueNDRangeKernel(getCommandQueue(), _kernels.clear, COLOR_ATTACHMENT0.height * COLOR_ATTACHMENT0.width);
}

GL_APICALL void GL_APIENTRY glClearColor (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
    _clear_data.color = (clear_color_t) {
        .red = red,
        .green = green,
        .blue = blue,
        .alpha = alpha
    };
}

GL_APICALL void GL_APIENTRY glClearDepthf (GLfloat d) {
    _clear_data.depth = d;
}

GL_APICALL void GL_APIENTRY glClearStencil (GLint s) {
    _clear_data.stencil = s;
}

GL_APICALL void GL_APIENTRY glColorMask (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
    _masks.color = (color_mask_t) {
        .r = red,
        .g = green,
        .b = blue,
        .a = alpha,
    };
}

GL_APICALL void GL_APIENTRY glCompressedTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data) {
    NOT_IMPLEMENTED;
}

GL_APICALL GLuint GL_APIENTRY glCreateProgram (void){
    static GLuint program = 1; // Pointer to the next available program.

    if (program < MAX_PROGRAMS) return program++;

    // TODO: Checkout documentation about this behaviour
    printf("No more programs available.\n");
    exit(program);
}

GL_APICALL void GL_APIENTRY glCullFace (GLenum mode) {
    switch (mode)
    {
    case GL_FRONT:
    case GL_BACK:
    case GL_FRONT_AND_BACK:
        _cull_face = mode;
        return;
    default:
        NOT_IMPLEMENTED;
    }
}

GL_APICALL void GL_APIENTRY glDepthFunc (GLenum func) {
    _depth_func = func;
}

GL_APICALL void GL_APIENTRY glDepthMask (GLboolean flag) {
    _masks.depth = flag;
}

GL_APICALL void GL_APIENTRY glDepthRangef (GLfloat n, GLfloat f) {
    _depth_range = (depth_range_t) {.n=n, .f=f};
}

GL_APICALL void GL_APIENTRY glDisable (GLenum cap) {
    switch (cap)
    {
    case GL_SCISSOR_TEST:
        _enableds.scissor_test = GL_FALSE;
        break;
    case GL_STENCIL_TEST:
        _enableds.stencil_test = GL_FALSE;
        break;
    case GL_DEPTH_TEST:
        _enableds.depth_test = GL_FALSE;
        break;
    case GL_BLEND:
        _enableds.blend = GL_FALSE;
        break;
    case GL_DITHER:
        _enableds.dither = GL_FALSE;
        break;
    case GL_CULL_FACE:
        _enableds.cull_face = GL_FALSE;
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

GL_APICALL void GL_APIENTRY glDisableVertexAttribArray (GLuint index) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) RETURN_ERROR(GL_INVALID_VALUE);

    _vertex_attrib_enable[index] = 0;
}

GL_APICALL void GL_APIENTRY glDrawArrays (GLenum mode, GLint first, GLsizei count) {
    if (!_current_program) RETURN_ERROR(GL_INVALID_OPERATION);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) RETURN_ERROR(GL_INVALID_FRAMEBUFFER_OPERATION);

    if (first < 0) RETURN_ERROR(GL_INVALID_VALUE);

    if (first != 0) NOT_IMPLEMENTED;
    if (mode == GL_POINTS || mode == GL_LINE_STRIP || mode == GL_LINE_LOOP || mode == GL_LINES) NOT_IMPLEMENTED;

    /* ---- General vars ---- */
    
    cl_int tmp_err; 
    cl_buffer_region region;

    /* -- Get Framebuffer Data -- */
    typedef struct {
        cl_mem mem;
        GLenum internalformat;
    } buffer_data_t; 
    
    typedef struct {
        buffer_data_t color, depth, stencil;
        GLsizei width, height;
    } framebuffer_data_t; 

    framebuffer_data_t framebuffer;

    {
        attachment_t *color, *depth, *stencil;
        color   = &FRAMEBUFFER.color_attachment0;
        depth   = &FRAMEBUFFER.depth_attachment;
        stencil = &FRAMEBUFFER.stencil_attachment;
        
        if (color->position) {
            if (color->target == GL_RENDERBUFFER) {
                renderbuffer_t *colorbuffer = &_renderbuffers[color->position];
                framebuffer.color = (buffer_data_t) {
                    .mem = colorbuffer->mem,
                    .internalformat = colorbuffer->internalformat,
                };
                framebuffer.width = colorbuffer->width;
                framebuffer.height = colorbuffer->height; 
            } else {
                texture_t *colorbuffer = &_textures[color->position];
                framebuffer.color = (buffer_data_t) {
                    .mem = colorbuffer->mem,
                    .internalformat = colorbuffer->internalformat,
                };
                framebuffer.width = colorbuffer->width;
                framebuffer.height = colorbuffer->height; 
            }
        }
        if (depth->position) {
            if (depth->target == GL_RENDERBUFFER) {
                renderbuffer_t *depthbuffer = &_renderbuffers[depth->position];
                framebuffer.depth = (buffer_data_t) {
                    .mem = depthbuffer->mem,
                    .internalformat = depthbuffer->internalformat,
                };
                framebuffer.width = depthbuffer->width;
                framebuffer.height = depthbuffer->height; 
            } else {
                texture_t *depthbuffer = &_textures[depth->position];
                framebuffer.depth = (buffer_data_t) {
                    .mem = depthbuffer->mem,
                    .internalformat = depthbuffer->internalformat,
                };
                framebuffer.width = depthbuffer->width;
                framebuffer.height = depthbuffer->height; 
            }
        }
        if (stencil->position) {
            if (stencil->target == GL_RENDERBUFFER) {
                renderbuffer_t *stencilbuffer = &_renderbuffers[stencil->position];
                framebuffer.stencil = (buffer_data_t) {
                    .mem = stencilbuffer->mem,
                    .internalformat = stencilbuffer->internalformat,
                };
                framebuffer.width = stencilbuffer->width;
                framebuffer.height = stencilbuffer->height; 
            } else {
                texture_t *stencilbuffer = &_textures[stencil->position];
                framebuffer.stencil = (buffer_data_t) {
                    .mem = stencilbuffer->mem,
                    .internalformat = stencilbuffer->internalformat,
                };
                framebuffer.width = stencilbuffer->width;
                framebuffer.height = stencilbuffer->height; 
            }
        }
    }

    /* -- Pipeline data -- */
    
    GLsizei num_vertices, num_fragments;
    
    num_vertices = count-first;
    num_fragments = framebuffer.width * framebuffer.height;

    GLsizei num_primitives = num_vertices;
    if (mode==GL_LINES) num_primitives /= 2;
    else if (mode==GL_TRIANGLES) num_primitives /= 3;

    /* ---- Setup Per-Vertex Buffers ---- */

    // User input defined vertex array
    cl_mem vertex_array_mem[MAX_VERTEX_ATTRIBS];
    cl_mem temp_mem[MAX_VERTEX_ATTRIBS];

    for (int attrib=0; attrib < CURRENT_PROGRAM.active_vertex_attribs; ++attrib) {

        if (_vertex_attrib_enable[attrib]) {
            vertex_attrib_pointer_t *pointer = &_vertex_attribs[attrib].pointer;
            size_t buffer_in_size = num_vertices*sizeof_type(pointer->type)*pointer->size+pointer->stride;

            //cl_command_queue command_queue = createCommandQueue(0);
            
            if (pointer->binding) {
                region.origin = (uint64_t) pointer->pointer, 
                region.size = buffer_in_size;
                temp_mem[attrib] = clCreateSubBuffer(_buffers[pointer->binding].mem, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &tmp_err);
            } else {
                temp_mem[attrib] = createBuffer(MEM_READ_WRITE | MEM_COPY_HOST_PTR, buffer_in_size, pointer->pointer);
            }
            vertex_array_mem[attrib] = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_vertices, NULL);

            cl_uint normalized = pointer->normalized;

            setKernelArg(_kernels.strided_write, 0, sizeof(pointer->size),       &pointer->size);
            setKernelArg(_kernels.strided_write, 1, sizeof(pointer->type),       &pointer->type);
            setKernelArg(_kernels.strided_write, 2, sizeof(cl_uint),             &normalized);
            setKernelArg(_kernels.strided_write, 3, sizeof(pointer->stride),     &pointer->stride);
            setKernelArg(_kernels.strided_write, 4, sizeof(cl_mem),              &temp_mem[attrib]);
            setKernelArg(_kernels.strided_write, 5, sizeof(cl_mem),              &vertex_array_mem[attrib]);
            
            enqueueNDRangeKernel(_vertex_attrib_command_queues[attrib], _kernels.strided_write, num_vertices);

        }
    }
    
    

    cl_mem vertex_out_buffer    = createBuffer(CL_MEM_READ_WRITE,   sizeof(float[4])*num_vertices*CURRENT_PROGRAM.varying_size, NULL);
    cl_mem gl_Positions         = createBuffer(MEM_READ_WRITE,      sizeof(float[4])*num_vertices,                              NULL);

    /* ---- Set Up Per-Vertex Kernels ---- */
    //printf("Vertex Kernel Set Up.");
    cl_kernel vertex_kernel = CURRENT_PROGRAM.vertex_kernel;

    for(int uniform = 0; uniform < CURRENT_PROGRAM.active_uniforms; ++uniform) {
        if (CURRENT_PROGRAM.uniforms_data[uniform].vertex_location != -1) {
            setKernelArg(vertex_kernel, 
                CURRENT_PROGRAM.uniforms_data[uniform].vertex_location, 
                sizeof(CURRENT_PROGRAM.uniforms_mem[uniform]),
                &CURRENT_PROGRAM.uniforms_mem[uniform]
            );
        }
    }
    for(int attrib = 0; attrib < CURRENT_PROGRAM.active_vertex_attribs; ++attrib) {
        if (_vertex_attrib_state[attrib] == VEC4) {
            setKernelArg(vertex_kernel, CURRENT_PROGRAM.vertex_attribs_data[attrib].vertex_location, sizeof(vertex_attrib_t), &_vertex_attribs[attrib]);
        } else {
            setKernelArg(vertex_kernel, CURRENT_PROGRAM.vertex_attribs_data[attrib].vertex_location, sizeof(cl_mem), &vertex_array_mem[attrib]);
        }
    }

    // TODO: Locations of out are consecutive and are the last ones, change it to accept more types 
    unsigned int vertex_out_location = CURRENT_PROGRAM.active_uniforms + CURRENT_PROGRAM.active_vertex_attribs;

    region.size = num_vertices*sizeof(float[4]);
    for(int out=0; out < CURRENT_PROGRAM.varying_size; ++out) {
        region.origin = sizeof(float[4])*num_vertices*out;
        cl_mem subbuffer = clCreateSubBuffer(vertex_out_buffer,CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &tmp_err);
        setKernelArg(vertex_kernel, CURRENT_PROGRAM.varying_data[out].vertex_location, sizeof(cl_mem), &subbuffer);
    }
    setKernelArg(vertex_kernel, CURRENT_PROGRAM.gl_position_location, sizeof(gl_Positions), &gl_Positions);

    // Perspective Division Kernel Set Up
    setKernelArg(_kernels.perspective_division, 0, sizeof(gl_Positions), &gl_Positions);

    // Viewport Division Kernel Set Up
    setKernelArg(_kernels.viewport_division, 0, sizeof(gl_Positions), &gl_Positions);
    setKernelArg(_kernels.viewport_division, 1, sizeof(_viewport),    &_viewport);
    setKernelArg(_kernels.viewport_division, 2, sizeof(_depth_range), &_depth_range);

    /* ---- Set Up Per-Fragment Buffers ---- */
    cl_mem fragment_in_buffer, gl_FragCoord, gl_Discard, gl_FragColor, facing_buffer;
    
    fragment_in_buffer  = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments*CURRENT_PROGRAM.varying_size, NULL);
    gl_FragCoord        = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments,                              NULL);
    gl_Discard          = createBuffer(MEM_READ_WRITE, sizeof(uint8_t)*num_fragments,                               NULL);
    gl_FragColor        = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments,                              NULL);
    facing_buffer       = createBuffer(MEM_READ_WRITE, sizeof(uint8_t)*num_fragments,                               NULL);

    /* ---- Set Up Per-Fragment Kernels ---- */
    cl_kernel fragment_kernel;
    // Rasterization Kernel Set Up
    uint16_t cull_face = 0;
    if (_enableds.cull_face) cull_face = _cull_face;

    cl_kernel rasterization_kernel;
    switch (mode)
    {
    case GL_TRIANGLES:      rasterization_kernel = _kernels.rasterization.triangles;        break;
    case GL_TRIANGLE_FAN:   rasterization_kernel = _kernels.rasterization.triangle_fan;     break;
    case GL_TRIANGLE_STRIP: rasterization_kernel = _kernels.rasterization.triangle_strip;   break;
    default: NOT_IMPLEMENTED;
    }
    // arg 0 is reserved for the primitive index
    setKernelArg(_kernels.rasterization.triangles, 1, sizeof(int),      &framebuffer.width);
    setKernelArg(_kernels.rasterization.triangles, 2, sizeof(int),      &CURRENT_PROGRAM.varying_size);
    setKernelArg(_kernels.rasterization.triangles, 3, sizeof(cl_mem),   &gl_Positions);
    setKernelArg(_kernels.rasterization.triangles, 4, sizeof(cl_mem),   &gl_FragCoord);
    setKernelArg(_kernels.rasterization.triangles, 5, sizeof(cl_mem),   &gl_Discard);
    setKernelArg(_kernels.rasterization.triangles, 6, sizeof(cl_mem),   &vertex_out_buffer);
    setKernelArg(_kernels.rasterization.triangles, 7, sizeof(cl_mem),   &fragment_in_buffer);
    setKernelArg(_kernels.rasterization.triangles, 8, sizeof(cl_mem),   &facing_buffer);
    setKernelArg(_kernels.rasterization.triangles, 9, sizeof(uint16_t), &_front_face);
    setKernelArg(_kernels.rasterization.triangles,10, sizeof(uint16_t), &cull_face);

    // Fragment Kernel Set Up
    fragment_kernel = CURRENT_PROGRAM.fragment_kernel;
    for(int uniform = 0; uniform < CURRENT_PROGRAM.active_uniforms; ++uniform) {
        if (CURRENT_PROGRAM.uniforms_data[uniform].fragment_location != -1) {
            setKernelArg(fragment_kernel, 
                CURRENT_PROGRAM.uniforms_data[uniform].fragment_location, 
                sizeof(CURRENT_PROGRAM.uniforms_mem[uniform]),
                &CURRENT_PROGRAM.uniforms_mem[uniform]
            );
        }
    }
    for(int sampler = 0; sampler < CURRENT_PROGRAM.texture_unit_size; ++sampler) {

        typedef struct __attribute__((packed)) {
            unsigned int width, height;
            unsigned int internalformat;
            texture_wraps_t wraps;
        } sampler2D_t;

        texture_t *texture_unit = _textures + _active_textures[CURRENT_PROGRAM.sampler_value[sampler]].binding;

        texture_unit->wraps.s;

        sampler2D_t sampler2D = {
            .width = texture_unit->width,
            .height = texture_unit->height,
            .internalformat = texture_unit->internalformat,
            .wraps = texture_unit->wraps,
        };

        setKernelArg(fragment_kernel,
            CURRENT_PROGRAM.sampler_data[sampler].fragment_location,
            sizeof(sampler2D_t),
            &sampler2D
        );

        setKernelArg(fragment_kernel,
            CURRENT_PROGRAM.image_data[sampler].fragment_location,
            sizeof(texture_unit->mem),
            &texture_unit->mem
        );
    }

    // Texture vars
    region.size = num_fragments*sizeof(float[4]);
    for(int in=0; in < CURRENT_PROGRAM.varying_size; ++in) {
        region.origin = sizeof(float[4])*num_fragments*in;
        cl_mem subbuffer = clCreateSubBuffer(fragment_in_buffer,CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &tmp_err);
        setKernelArg(fragment_kernel, CURRENT_PROGRAM.varying_data[in].fragment_location, sizeof(cl_mem), &subbuffer);
    }
    // Required fragment args
    setKernelArg(fragment_kernel, CURRENT_PROGRAM.gl_fragcolor_location, sizeof(gl_FragColor), &gl_FragColor);
    // Optional fragment args
    if (CURRENT_PROGRAM.gl_fragcoord_location) 
        setKernelArg(fragment_kernel, CURRENT_PROGRAM.gl_fragcoord_location, sizeof(gl_FragCoord), &gl_FragCoord);
    if (CURRENT_PROGRAM.gl_discard_location) 
        setKernelArg(fragment_kernel, CURRENT_PROGRAM.gl_discard_location, sizeof(gl_Discard),   &gl_Discard);
    
    /* Scissor Test Kernel Set Up */
    cl_kernel scissor_kernel = _kernels.scissor_test;
    setKernelArg(scissor_kernel, 0, sizeof(cl_mem),   &gl_FragCoord);
    setKernelArg(scissor_kernel, 1, sizeof(cl_mem),   &gl_Discard);
    setKernelArg(scissor_kernel, 2, sizeof(cl_int),   &_scissor_data.left);
    setKernelArg(scissor_kernel, 3, sizeof(cl_int),   &_scissor_data.bottom);
    setKernelArg(scissor_kernel, 4, sizeof(cl_uint),  &_scissor_data.width);
    setKernelArg(scissor_kernel, 5, sizeof(cl_uint),  &_scissor_data.height);

    // Dummy objects set up
    cl_int cl_error;
    cl_mem dummy_stencil_mem = clCreateBuffer(_getContext(), CL_MEM_READ_WRITE, 200, NULL, &cl_error);
    CHECK_CL(cl_error);
    cl_mem dummy_depth_mem = clCreateBuffer(_getContext(), CL_MEM_READ_WRITE, 200, NULL, &cl_error);
    CHECK_CL(cl_error);

    // Stencil Kernel Set Up
    cl_kernel stencil_kernel = _kernels.stencil_test;
    setKernelArg(stencil_kernel, 0, sizeof(cl_mem),  &facing_buffer);
    setKernelArg(stencil_kernel, 1, sizeof(cl_mem),  &gl_Discard);
    setKernelArg(stencil_kernel, 2, sizeof(cl_mem),  _enableds.stencil_test ? &framebuffer.stencil.mem : &dummy_stencil_mem);
    setKernelArg(stencil_kernel, 3, sizeof(cl_uint), &_masks.stencil.front);
    setKernelArg(stencil_kernel, 4, sizeof(cl_uint), &_masks.stencil.back);
    setKernelArg(stencil_kernel, 5, sizeof(cl_uint), &_stencil_data.front.function.func);
    setKernelArg(stencil_kernel, 6, sizeof(cl_int),  &_stencil_data.front.function.ref);
    setKernelArg(stencil_kernel, 7, sizeof(cl_uint), &_stencil_data.front.function.mask);
    setKernelArg(stencil_kernel, 8, sizeof(cl_uint), &_stencil_data.front.operation.sfail);
    setKernelArg(stencil_kernel, 9, sizeof(cl_uint), &_stencil_data.back.function.func);
    setKernelArg(stencil_kernel,10, sizeof(cl_int),  &_stencil_data.back.function.ref);
    setKernelArg(stencil_kernel,11, sizeof(cl_uint), &_stencil_data.back.function.mask);
    setKernelArg(stencil_kernel,12, sizeof(cl_uint), &_stencil_data.back.operation.sfail);

    // Depth Kernel Set Up
    cl_uint depth_test, stencil_test;
    depth_test = _enableds.depth_test;
    stencil_test = _enableds.stencil_test;

    cl_kernel depth_kernel = _kernels.depth_test;
    setKernelArg(depth_kernel, 0, sizeof(cl_mem), &gl_FragCoord);
    setKernelArg(depth_kernel, 1, sizeof(cl_mem), &facing_buffer);
    setKernelArg(depth_kernel, 2, sizeof(cl_mem), &gl_Discard);
    setKernelArg(depth_kernel, 3, sizeof(cl_mem), _enableds.depth_test ? &framebuffer.depth.mem : &dummy_depth_mem);
    setKernelArg(depth_kernel, 4, sizeof(cl_mem), _enableds.stencil_test ? &framebuffer.stencil.mem : &dummy_stencil_mem);
    setKernelArg(depth_kernel, 5, sizeof(cl_uint), &depth_test);
    setKernelArg(depth_kernel, 6, sizeof(cl_uint), &stencil_test);
    setKernelArg(depth_kernel, 7, sizeof(cl_uchar), &_masks.depth);
    setKernelArg(depth_kernel, 8, sizeof(cl_uint), &_masks.stencil.front);
    setKernelArg(depth_kernel, 9, sizeof(cl_uint), &_masks.stencil.back);
    setKernelArg(depth_kernel,10, sizeof(cl_uint), &_depth_func);
    setKernelArg(depth_kernel,11, sizeof(cl_int),  &_stencil_data.front.function.ref);
    setKernelArg(depth_kernel,12, sizeof(cl_uint), &_stencil_data.front.operation.dpfail);
    setKernelArg(depth_kernel,13, sizeof(cl_uint), &_stencil_data.front.operation.dppasss);
    setKernelArg(depth_kernel,14, sizeof(cl_int),  &_stencil_data.back.function.ref);
    setKernelArg(depth_kernel,15, sizeof(cl_uint), &_stencil_data.back.operation.dpfail);
    setKernelArg(depth_kernel,16, sizeof(cl_uint), &_stencil_data.back.operation.dppasss);

    /* Blending Kernel Set Up */
    cl_kernel blending_kernel = _kernels.blending;
    setKernelArg(blending_kernel,  0, sizeof(cl_mem), &gl_FragColor);
    setKernelArg(blending_kernel,  1, sizeof(cl_mem), &gl_Discard);
    setKernelArg(blending_kernel,  2, sizeof(cl_mem), &framebuffer.color.mem);
    setKernelArg(blending_kernel,  3, sizeof(cl_uint), &framebuffer.color.internalformat);
    setKernelArg(blending_kernel,  4, sizeof(cl_uint), &_blend_data.equation.modeRGB);
    setKernelArg(blending_kernel,  5, sizeof(cl_uint), &_blend_data.equation.modeAlpha);
    setKernelArg(blending_kernel,  6, sizeof(cl_uint), &_blend_data.func.srcRGB);
    setKernelArg(blending_kernel,  7, sizeof(cl_uint), &_blend_data.func.srcAlpha);
    setKernelArg(blending_kernel,  8, sizeof(cl_uint), &_blend_data.func.dstRGB);
    setKernelArg(blending_kernel,  9, sizeof(cl_uint), &_blend_data.func.dstAlpha);
    setKernelArg(blending_kernel, 10, sizeof(cl_float4), &_blend_data.color);

    // Color Kernel Set Up
    cl_kernel dithering_kernel = _kernels.dithering;
    setKernelArg(dithering_kernel, 0, sizeof(cl_mem),    &gl_FragColor);
    setKernelArg(dithering_kernel, 1, sizeof(cl_mem),    &gl_FragCoord);
    setKernelArg(dithering_kernel, 2, sizeof(cl_mem),    &gl_Discard);
    setKernelArg(dithering_kernel, 3, sizeof(cl_mem),    &framebuffer.color.mem);
    setKernelArg(dithering_kernel, 4, sizeof(cl_uint),   &framebuffer.color.internalformat);
    setKernelArg(dithering_kernel, 5, sizeof(cl_uchar4), &_masks.color);
    setKernelArg(dithering_kernel, 6, sizeof(cl_uchar),  &_enableds.dither);

    /* ---- Enqueue Kernels ---- */
    cl_command_queue command_queue = getCommandQueue();

    //PRINT_BUFFER_F(vertex_array_mem[0], num_vertices, float, 4);
    
    enqueueNDRangeKernel(command_queue, vertex_kernel,                  num_vertices);
    enqueueNDRangeKernel(command_queue, _kernels.perspective_division,   num_vertices);
    enqueueNDRangeKernel(command_queue, _kernels.viewport_division,      num_vertices);

    //PRINT_BUFFER_F(gl_Positions, num_vertices, float, 4);

    for(GLsizei primitive=0; primitive < num_primitives; ++primitive) {
        setKernelArg(_kernels.rasterization.triangles, 0, sizeof(primitive), &primitive);

        //PRINT_BUFFER_I(gl_Discard, num_fragments, uint8_t, 1);

        enqueueNDRangeKernel(command_queue, _kernels.rasterization.triangles,   num_fragments);

        //PRINT_BUFFER_I(gl_Discard, num_fragments, uint8_t, 1);

        enqueueNDRangeKernel(command_queue, fragment_kernel,                    num_fragments);

        // TODO: Pixel Ownership, is controled by the context provider
        if (_enableds.scissor_test) 
            enqueueNDRangeKernel(command_queue, scissor_kernel, num_fragments);
        // TODO: Multisample Pre Operations, is controlled by the context provider
        if (_enableds.stencil_test) {
            //PRINT_BUFFER_I(framebuffer.stencil.mem, num_fragments, uint8_t, 1);
            enqueueNDRangeKernel(command_queue, stencil_kernel, num_fragments); // Also manage write on stencilbuffer
            //PRINT_BUFFER_I(framebuffer.stencil.mem, num_fragments, uint8_t, 1);
        }
        if (_enableds.stencil_test || _enableds.depth_test) 
            enqueueNDRangeKernel(command_queue, depth_kernel, num_fragments); // Also manage write on depthbuffer and on stencilbuffer
        if (_enableds.blend)
            enqueueNDRangeKernel(command_queue, blending_kernel, num_fragments);

        enqueueNDRangeKernel(command_queue, dithering_kernel, num_fragments); // Also manage write on colorbuffer
        // TODO: Multisample Post Operations, is controlled by the context provider

        // TODO: Write on buffers (now is handled using other kernels )
    }

    //PRINT_BUFFER_F(gl_FragColor, num_fragments, float, 4);  
    //PRINT_BUFFER_I(framebuffer.color.mem, num_fragments, uint8_t, 2);  
    CHECK_CL(clReleaseMemObject(vertex_out_buffer));
    CHECK_CL(clReleaseMemObject(fragment_in_buffer));
    CHECK_CL(clReleaseMemObject(gl_Positions));
    CHECK_CL(clReleaseMemObject(gl_FragCoord));
    CHECK_CL(clReleaseMemObject(gl_Discard));
    CHECK_CL(clReleaseMemObject(gl_FragColor));
    CHECK_CL(clReleaseMemObject(facing_buffer));
    CHECK_CL(clReleaseMemObject(dummy_stencil_mem));
    CHECK_CL(clReleaseMemObject(dummy_depth_mem));

    for (int attrib=0; attrib < CURRENT_PROGRAM.active_vertex_attribs; ++attrib) {
        if (_vertex_attrib_enable[attrib]) {
            CHECK_CL(clReleaseMemObject(temp_mem[attrib]));
            CHECK_CL(clReleaseMemObject(vertex_array_mem[attrib]));
        }
    }

}

GL_APICALL void GL_APIENTRY glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices) NOT_IMPLEMENTED;

GL_APICALL void GL_APIENTRY glEnable (GLenum cap) {
    switch (cap)
    {
    case GL_SCISSOR_TEST:
        _enableds.scissor_test = GL_TRUE;
        break;
    case GL_STENCIL_TEST:
        _enableds.stencil_test = GL_TRUE;
        break;
    case GL_DEPTH_TEST:
        _enableds.depth_test = GL_TRUE;
        break;
    case GL_BLEND:
        _enableds.blend = GL_TRUE;
        break;
    case GL_DITHER:
        _enableds.dither = GL_TRUE;
        break;
    case GL_CULL_FACE:
        _enableds.cull_face = GL_TRUE;
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

GL_APICALL void GL_APIENTRY glEnableVertexAttribArray (GLuint index) {
    if (index >= MAX_VERTEX_ATTRIBS) RETURN_ERROR(GL_INVALID_VALUE);
    
    _vertex_attrib_enable[index] = 1;
}

GL_APICALL void GL_APIENTRY glFinish (void) {
    CHECK_CL(clFinish(getCommandQueue()));
}

GL_APICALL void GL_APIENTRY glFlush (void) {
    CHECK_CL(clFlush(getCommandQueue()));
}

GL_APICALL void GL_APIENTRY glFramebufferRenderbuffer (GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) {
    if (target != GL_FRAMEBUFFER) NOT_IMPLEMENTED;
    if (_framebuffer_binding == 0) RETURN_ERROR(GL_INVALID_OPERATION);

    if (renderbuffer != 0) {
        // TODO: Check if texture exist and with a target == textarget, if not RETURN_ERROR(GL_INVALID_OPERATION); 
        if (renderbuffertarget != GL_RENDERBUFFER) NOT_IMPLEMENTED;
    }

    attachment_t *attachment_ptr;
    switch (attachment)
    {
    case GL_COLOR_ATTACHMENT0: 
        attachment_ptr = &_framebuffers[_framebuffer_binding].color_attachment0;
        break;
    case GL_DEPTH_ATTACHMENT: 
        attachment_ptr = &_framebuffers[_framebuffer_binding].depth_attachment;
        break;
    case GL_STENCIL_ATTACHMENT: 
        attachment_ptr = &_framebuffers[_framebuffer_binding].stencil_attachment;
        break;
    default:
        NOT_IMPLEMENTED;
    }

    *attachment_ptr = (attachment_t) {
        .target = renderbuffertarget,
        .position = renderbuffer,
    };

}

GL_APICALL void GL_APIENTRY glFramebufferTexture2D (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
    if (target != GL_FRAMEBUFFER) NOT_IMPLEMENTED;
    if (_framebuffer_binding == 0) RETURN_ERROR(GL_INVALID_OPERATION);

    if (texture != 0) {
        // TODO: Check if texture exist and with a target == textarget, if not RETURN_ERROR(GL_INVALID_OPERATION); 
        if (level != 0) RETURN_ERROR(GL_INVALID_VALUE);
        if (textarget != GL_TEXTURE_2D) NOT_IMPLEMENTED; // TODO checkout the error;
    }

    attachment_t *attachment_ptr;
    switch (attachment)
    {
    case GL_COLOR_ATTACHMENT0: 
        attachment_ptr = &_framebuffers[_framebuffer_binding].color_attachment0;
        break;
    case GL_DEPTH_ATTACHMENT: 
        attachment_ptr = &_framebuffers[_framebuffer_binding].depth_attachment;
        break;
    case GL_STENCIL_ATTACHMENT: 
        attachment_ptr = &_framebuffers[_framebuffer_binding].stencil_attachment;
        break;
    default:
        NOT_IMPLEMENTED;
    }

    *attachment_ptr = (attachment_t) {
        .target = textarget,
        .position = texture,
    };

}

GL_APICALL void GL_APIENTRY glFrontFace (GLenum mode) {
    switch (mode)
    {
    case GL_CCW:
    case GL_CW:
        _front_face = mode;
        return;
    default:
        NOT_IMPLEMENTED;
    }
}

GL_APICALL void GL_APIENTRY glGenBuffers (GLsizei n, GLuint *buffers) {
    GLuint _id = 1; // _id = 0 is reserved for ARRAY_BUFFER

    while(n > 0 && _id < 256) {
        if (!_buffers[_id].used) {
            _buffers[_id].used = GL_TRUE;
            *buffers = _id;

            buffers += 1; 
            n -= 1;
        }
        _id += 1;
    }
}

GL_APICALL void GL_APIENTRY glGenerateMipmap (GLenum target) NOT_IMPLEMENTED;

GL_APICALL void GL_APIENTRY glGenFramebuffers (GLsizei n, GLuint *framebuffers) {
    GLuint id = 1; // _id = 0 is reserved for ARRAY_BUFFER

    while(n > 0 && id < MAX_FRAMEBUFFER) {
        if (!_framebuffers[id].used) {
            _framebuffers[id].used = GL_TRUE;
            
            *framebuffers = id;
            framebuffers += 1; 
            n -= 1;
        }
        id += 1;
    }
}

GL_APICALL void GL_APIENTRY glGenRenderbuffers (GLsizei n, GLuint *renderbuffers) {
    GLuint id = 1; // id = 0 is reserved for ARRAY_BUFFER

    while(n > 0 && id < MAX_RENDERBUFFER) {
        if (!_renderbuffers[id].used) {
            _renderbuffers[id].used = GL_TRUE;
            *renderbuffers = id;

            renderbuffers += 1; 
            n -= 1;
        }
        id += 1;
    }
}

GL_APICALL void GL_APIENTRY glGenTextures (GLsizei n, GLuint *textures) {
    GLuint id = 1;

    while(n > 0 && id < MAX_TEXTURE) {
        if (!_textures[id].used) {
            _textures[id] = (texture_t) {
                .used = GL_TRUE,
                .wraps = {
                    .s = GL_REPEAT,
                    .t = GL_REPEAT,
                    .min_filter = GL_NEAREST, // TODO: Check if is default
                    .mag_filter = GL_NEAREST, // TODO: Check if is default
                },
            };

            *textures = id;
            textures += 1; 
            n -= 1;
        }
        id += 1;
    }
}

GL_APICALL GLint GL_APIENTRY glGetAttribLocation (GLuint program, const GLchar *name) {
    if (!program) NOT_IMPLEMENTED;

    program_t* program_ptr = &_programs[program];

    if (!program_ptr->last_load_attempt) RETURN_ERROR(GL_INVALID_OPERATION);
    
    for(size_t attrib=0; attrib<program_ptr->active_vertex_attribs; ++attrib) {
        if (strcmp(name, program_ptr->vertex_attribs_data[attrib].name) == 0) return attrib;
    }

    return -1;
}

GL_APICALL void GL_APIENTRY glGetBooleanv (GLenum pname, GLboolean *data) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glGetBufferParameteriv (GLenum target, GLenum pname, GLint *params) {
    NOT_IMPLEMENTED;
}

GL_APICALL GLenum GL_APIENTRY glGetError (void) {
    GLenum error = gl_error;
    gl_error = GL_NO_ERROR;
    return error;
}

GL_APICALL void GL_APIENTRY glGetFloatv (GLenum pname, GLfloat *data) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glGetFramebufferAttachmentParameteriv (GLenum target, GLenum attachment, GLenum pname, GLint *params) {
    NOT_IMPLEMENTED;
}

// TODO:
GL_APICALL GLenum GL_APIENTRY glGetGraphicsResetStatus (void) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glGetIntegerv (GLenum pname, GLint *data) {
    switch (pname)
    {
    case GL_IMPLEMENTATION_COLOR_READ_FORMAT:
        *data = GL_RGBA;
        break;
    case GL_IMPLEMENTATION_COLOR_READ_TYPE:
        *data = GL_UNSIGNED_SHORT_4_4_4_4;
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

GL_APICALL void GL_APIENTRY glGetProgramiv (GLuint program, GLenum pname, GLint *params) {
    switch (pname)
    {
    case GL_LINK_STATUS:
        *params = _programs[program].last_load_attempt;
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

GL_APICALL void GL_APIENTRY glGetRenderbufferParameteriv (GLenum target, GLenum pname, GLint *params) {
    NOT_IMPLEMENTED;
}

GL_APICALL const GLubyte *GL_APIENTRY glGetString (GLenum name) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glGetTexParameterfv (GLenum target, GLenum pname, GLfloat *params) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glGetTexParameteriv (GLenum target, GLenum pname, GLint *params) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glGetnUniformfv (GLuint program, GLint location, GLsizei bufSize, GLfloat *params) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glGetnUniformiv (GLuint program, GLint location, GLsizei bufSize, GLint *params) {
    NOT_IMPLEMENTED;
}

GL_APICALL GLint GL_APIENTRY glGetUniformLocation (GLuint program, const GLchar *name) {
    if (!_current_program) RETURN_ERROR(GL_INVALID_OPERATION);
    
    for(size_t uniform=0; uniform<CURRENT_PROGRAM.active_uniforms; ++uniform) {
        if (strcmp(name, CURRENT_PROGRAM.uniforms_data[uniform].name) == 0) return uniform;
    }

    for(size_t sampler=0; sampler<CURRENT_PROGRAM.texture_unit_size; ++sampler) {
        if (strcmp(name, CURRENT_PROGRAM.sampler_data[sampler].name) == 0) return CURRENT_PROGRAM.active_uniforms+sampler;
    }

    return -1;
}

GL_APICALL void GL_APIENTRY glGetVertexAttribfv (GLuint index, GLenum pname, GLfloat *params) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glGetVertexAttribiv (GLuint index, GLenum pname, GLint *params) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glGetVertexAttribPointerv (GLuint index, GLenum pname, void **pointer) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glHint (GLenum target, GLenum mode) {
    NOT_IMPLEMENTED;
}

GL_APICALL GLboolean GL_APIENTRY glIsEnabled (GLenum cap) NOT_IMPLEMENTED;

GL_APICALL void GL_APIENTRY glLineWidth (GLfloat width) NOT_IMPLEMENTED;

GL_APICALL void GL_APIENTRY glPixelStorei (GLenum pname, GLint param) {
    switch (pname)
    {
    case GL_UNPACK_ALIGNMENT:
        if (param != 1 && param != 2 && param != 4 && param != 8) RETURN_ERROR(GL_INVALID_VALUE);
        _pixel_store.unpack_aligment = param;
        return;
    default:
        RETURN_ERROR(GL_INVALID_ENUM);
    }
}

GL_APICALL void GL_APIENTRY glPolygonOffset (GLfloat factor, GLfloat units) NOT_IMPLEMENTED;

GL_APICALL void GL_APIENTRY glProgramBinary (GLuint program, GLenum binaryFormat, const void *binary, GLsizei length){

    if (_programs[program].program) RETURN_ERROR(GL_INVALID_OPERATION);
    // TODO: Check binaryFormat
    if (binaryFormat == POCL_BINARY) {
        _programs[program].program=createProgramWithBinary(binary, length); // TODO: NOT HERE, but check to send Host OpenCL programs
        buildProgram(_programs[program].program);
        // TODO: Check this logic
        _programs[program].last_load_attempt        = GL_TRUE;
        _programs[program].last_validation_attempt  = GL_TRUE;
        _programs[program].vertex_kernel            = createKernel(_programs[program].program, VERTEX_SHADER_FNAME);
        _programs[program].fragment_kernel          = createKernel(_programs[program].program, FRAGMENT_SHADER_FNAME);

        cl_uint kernel_num_args;

        clGetKernelInfo(_programs[program].vertex_kernel,CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &kernel_num_args, NULL);
        
        for(cl_uint arg=0; arg < kernel_num_args; ++arg) {
            cl_kernel_arg_address_qualifier addr_qualifier;
            cl_kernel_arg_type_qualifier type_qualifier;
            cl_kernel_arg_access_qualifier access_qualifier;
            char name[ARG_NAME_SIZE];
            char type_name[32];
            uint32_t size, type;

            clGetKernelArgInfo(_programs[program].vertex_kernel, arg, CL_KERNEL_ARG_ADDRESS_QUALIFIER,  sizeof(addr_qualifier),     &addr_qualifier,    NULL);
            clGetKernelArgInfo(_programs[program].vertex_kernel, arg, CL_KERNEL_ARG_TYPE_QUALIFIER,     sizeof(type_qualifier),     &type_qualifier,    NULL);
            clGetKernelArgInfo(_programs[program].vertex_kernel, arg, CL_KERNEL_ARG_ACCESS_QUALIFIER,   sizeof(access_qualifier),   &access_qualifier,  NULL);
            clGetKernelArgInfo(_programs[program].vertex_kernel, arg, CL_KERNEL_ARG_NAME,               sizeof(name),               &name,              NULL);
            clGetKernelArgInfo(_programs[program].vertex_kernel, arg, CL_KERNEL_ARG_TYPE_NAME,          sizeof(type_name),          &type_name,         NULL);
            size = size_from_name_type(type_name);
            type = type_from_name_type(type_name);

            printf("name=%s type=%s size=%d type=%x addr_q=%x type_q=%x\n",name, type_name, size, type, addr_qualifier, type_qualifier);

            if (strncmp(GL_POSITION, name, sizeof(GL_POSITION)) == 0) {
                _programs[program].gl_position_location = arg;
                continue;
            };

            arg_data_t *arg_data;

            if (addr_qualifier == CL_KERNEL_ARG_ADDRESS_CONSTANT) {
                arg_data = _programs[program].uniforms_data + _programs[program].active_uniforms;
                _programs[program].uniforms_mem[_programs[program].active_uniforms] = createBuffer(CL_MEM_READ_ONLY, sizeof_type(type)*size, NULL);
                _programs[program].uniforms_array_size[_programs[program].active_uniforms] = 1; // TODO: Not supported more than one
                _programs[program].active_uniforms += 1;
            } else if (type == GL_FLOAT && (
                       addr_qualifier == CL_KERNEL_ARG_ADDRESS_GLOBAL && type_qualifier == CL_KERNEL_ARG_TYPE_CONST
                    || addr_qualifier == CL_KERNEL_ARG_ADDRESS_PRIVATE)) {
                arg_data = _programs[program].vertex_attribs_data + _programs[program].active_vertex_attribs;
                _programs[program].active_vertex_attribs += 1;
            } else if (type == GL_FLOAT 
                    && addr_qualifier == CL_KERNEL_ARG_ADDRESS_GLOBAL 
                    && type_qualifier == CL_KERNEL_ARG_TYPE_NONE) {
                arg_data = _programs[program].varying_data + _programs[program].varying_size;
                _programs[program].varying_size += 1;
            } else {
                _programs[program].last_load_attempt = GL_FALSE;
                strcpy(_programs[program].log, "Failed load attempt");
                return;
            }

            *arg_data = (arg_data_t) {
                .vertex_location    = arg,
                .fragment_location  = -1,
                .size               = size,
                .type               = type, 
            };
            strcpy(&arg_data->name, &name);
            printf("names: %s\n", arg_data->name); 
        }

        clGetKernelInfo(_programs[program].fragment_kernel,CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &kernel_num_args, NULL);
        printf("kernel args frag: %d", kernel_num_args);
        uint32_t n_images = 0, n_samplers = 0;

        for(cl_uint arg=0; arg < kernel_num_args; ++arg) {
            cl_kernel_arg_address_qualifier addr_qualifier;
            cl_kernel_arg_type_qualifier type_qualifier;
            cl_kernel_arg_access_qualifier access_qualifier;
            char name[ARG_NAME_SIZE];
            char type_name[32];
            uint32_t size, type;

            clGetKernelArgInfo(_programs[program].fragment_kernel, arg, CL_KERNEL_ARG_ADDRESS_QUALIFIER,  sizeof(addr_qualifier),     &addr_qualifier,    NULL);
            clGetKernelArgInfo(_programs[program].fragment_kernel, arg, CL_KERNEL_ARG_TYPE_QUALIFIER,     sizeof(type_qualifier),     &type_qualifier,    NULL);
            clGetKernelArgInfo(_programs[program].fragment_kernel, arg, CL_KERNEL_ARG_ACCESS_QUALIFIER,   sizeof(access_qualifier),   &access_qualifier,  NULL);
            clGetKernelArgInfo(_programs[program].fragment_kernel, arg, CL_KERNEL_ARG_NAME,               sizeof(name),               &name,              NULL);
            clGetKernelArgInfo(_programs[program].fragment_kernel, arg, CL_KERNEL_ARG_TYPE_NAME,          sizeof(type_name),          &type_name,         NULL);
            size = size_from_name_type(type_name);
            type = type_from_name_type(type_name);

            printf("name=%s type=%s size=%d type=%x addr_q=%x type_q=%x\n",name, type_name, size, type, addr_qualifier, type_qualifier);

            if (strcmp(GL_FRAGCOLOR, name) == 0) {
                _programs[program].gl_fragcolor_location = arg;
                continue;
            };
            if (strcmp(GL_FRAGCOORD, name) == 0) {
                _programs[program].gl_fragcoord_location = arg;
                continue;
            };
            if (strcmp(GL_DISCARD, name) == 0) {
                _programs[program].gl_discard_location = arg;
                continue;
            };

            arg_data_t *arg_data;
            int exist = 0;

            if (addr_qualifier == CL_KERNEL_ARG_ADDRESS_CONSTANT && type != IMAGE_T) {
                int find = -1;
                for(int uniform; _programs[program].active_uniforms; ++uniform) {
                    if (strcmp(name, _programs[program].uniforms_data[uniform].name) == 0) {
                        find = uniform;
                        break;
                    }
                }
                if (find == -1) {
                    arg_data = _programs[program].uniforms_data + _programs[program].active_uniforms;
                    _programs[program].uniforms_mem[_programs[program].active_uniforms] = createBuffer(CL_MEM_READ_ONLY, sizeof_type(type)*size, NULL);
                    _programs[program].uniforms_array_size[_programs[program].active_uniforms] = 1; // TODO: Not supported more than one
                    _programs[program].active_uniforms += 1;
                } else {
                    arg_data = _programs[program].uniforms_data + find; 
                    exist = 1;                   
                    // TODO: type checker
                }
            } else if (type == GL_FLOAT && addr_qualifier == CL_KERNEL_ARG_ADDRESS_GLOBAL && type_qualifier == CL_KERNEL_ARG_TYPE_CONST) {
                int find = -1;
                for(int varying=0; varying <_programs[program].varying_size; ++varying) {
                    if (strcmp(name, _programs[program].varying_data[varying].name) == 0) {
                        find = varying;
                        break;
                    }
                }
                if (find == -1) {
                    _programs[program].last_load_attempt = GL_FALSE;
                    strcpy(_programs[program].log, "Failed load attempt");
                    return;
                }
                arg_data = _programs[program].varying_data + find;
                exist = 1;
            } else if (type == SAMPLER2D_T) {
                arg_data = _programs[program].sampler_data + n_samplers;
                ++n_samplers;
                ++_programs[program].texture_unit_size;
            } else if (type == IMAGE_T) {
                arg_data = _programs[program].image_data + n_images;
                ++n_images;
            } else {
                _programs[program].last_load_attempt = GL_FALSE;
                strcpy(_programs[program].log, "Failed load attempt");
                return;
            }

            if (exist) {
                arg_data->fragment_location = arg;
            } else {
                *arg_data = (arg_data_t) {
                    .vertex_location    = -1,
                    .fragment_location  = arg,
                    .size               = size,
                    .type               = type, 
                };
                strcpy(arg_data->name, name); 
            }
        }
        if (n_images != n_samplers) {
            _programs[program].last_load_attempt = GL_FALSE;
            strcpy(_programs[program].log, "Failed load attempt");
            return;
        }
    } else NOT_IMPLEMENTED;

}

GL_APICALL void GL_APIENTRY glReadnPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data) {
    // TODO: Check behaviour for non 4 component internal formats
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) RETURN_ERROR(GL_INVALID_FRAMEBUFFER_OPERATION);
    if (!_framebuffer_binding) NOT_IMPLEMENTED; // Context related

    cl_kernel readnpixels_kernel;
    cl_int cl_error;
    size_t num_pixels;

    if (format == GL_RGBA && type == GL_UNSIGNED_BYTE) {
        num_pixels = bufSize / sizeof(uint32_t);
        readnpixels_kernel = _kernels.readnpixels.rgba8;
    } else if (format == GL_RGBA && type == GL_UNSIGNED_SHORT_4_4_4_4) {
        num_pixels = bufSize / sizeof(uint16_t);
        readnpixels_kernel = _kernels.readnpixels.rgba4;
    } else RETURN_ERROR(GL_INVALID_OPERATION);

    if (!FRAMEBUFFER.color_attachment0.position) NOT_IMPLEMENTED; // No texture or texture buffer attached, maybe throw GL_INVALID_OPERATION?

    typedef struct {
        cl_mem mem;
        uint32_t width, internalformat;
    } buffer_info_t; 

    buffer_info_t buffer_info;

    if (FRAMEBUFFER.color_attachment0.target == GL_RENDERBUFFER) {

        renderbuffer_t* colorbuffer = &_renderbuffers[FRAMEBUFFER.color_attachment0.position];
        
        if (x+width > colorbuffer->width || y+height > colorbuffer->height) UNDEFINED_BEHAVIOUR;

        buffer_info = (buffer_info_t) {
            .mem = colorbuffer->mem,
            .width = colorbuffer->width,
            .internalformat = colorbuffer->internalformat,
        };
    } else {
        texture_t* texture = &_textures[FRAMEBUFFER.color_attachment0.position];
        
        if (x+width > texture->width || y+height > texture->height) UNDEFINED_BEHAVIOUR;

        buffer_info = (buffer_info_t) {
            .mem = texture->mem,
            .width = texture->width,
            .internalformat = texture->internalformat,
        };
    }

    cl_mem dst_buff = clCreateBuffer(_getContext(), CL_MEM_WRITE_ONLY, bufSize, NULL, &cl_error);
    // cl_mem dst_buff = clCreateBuffer(_getContext(), CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, bufSize, data, &cl_error); DO NOT SUPPORTED ON VORTEX
    CHECK_CL(cl_error);

    setKernelArg(readnpixels_kernel, 0, sizeof(cl_mem), &dst_buff);
    setKernelArg(readnpixels_kernel, 1, sizeof(cl_mem), &buffer_info.mem);
    setKernelArg(readnpixels_kernel, 2, sizeof(cl_uint), &buffer_info.internalformat);
    setKernelArg(readnpixels_kernel, 3, sizeof(cl_uint), &buffer_info.width);
    setKernelArg(readnpixels_kernel, 4, sizeof(int), &x);
    setKernelArg(readnpixels_kernel, 5, sizeof(int), &y);
    setKernelArg(readnpixels_kernel, 6, sizeof(int), &width);
    setKernelArg(readnpixels_kernel, 7, sizeof(int), &height);

    void *command_queue = getCommandQueue();
    size_t global_work_size[1] = { num_pixels };
    enqueueNDRangeKernel(command_queue, readnpixels_kernel, global_work_size[0]);
    // clEnqueueNDRangeKernel(command_queue, readnpixels_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

    // TODO: Check out how to make the device access to host ptr.
    cl_error = clEnqueueReadBuffer(command_queue, dst_buff, CL_TRUE, 0, bufSize, data, 0, NULL, NULL); 
    CHECK_CL(cl_error);
    cl_error = clReleaseMemObject(dst_buff);
    CHECK_CL(cl_error);
}

GL_APICALL void GL_APIENTRY glRenderbufferStorage (GLenum target, GLenum internalformat, GLsizei width, GLsizei height) {
    if (width > MAX_RENDERBUFFER_SIZE || height > MAX_RENDERBUFFER_SIZE) RETURN_ERROR(GL_INVALID_VALUE);
    if (_renderbuffers[_renderbuffer_binding].mem != NULL) RETURN_ERROR(GL_INVALID_OPERATION); 
    if (target != GL_RENDERBUFFER) NOT_IMPLEMENTED; // Check what error throw

    size_t size;

    switch (internalformat)
    {
    case GL_DEPTH_COMPONENT16:
    case GL_RGBA4:
    case GL_RGB5_A1:
    case GL_RGB565:
        size = sizeof(uint16_t[width][height]);
        break;
    case GL_STENCIL_INDEX8:
        size = sizeof(uint8_t[width][height]);
        break;
    default:
        NOT_IMPLEMENTED; // Check what error throw
    }

    // TODO: if (size > remaining_memory) RETURN_ERROR(GL_OUT_OF_MEMORY);

    _renderbuffers[_renderbuffer_binding].mem               = createBuffer(MEM_READ_WRITE, size, NULL);
    _renderbuffers[_renderbuffer_binding].internalformat    = internalformat;
    _renderbuffers[_renderbuffer_binding].width             = width;
    _renderbuffers[_renderbuffer_binding].height            = height;
}

GL_APICALL void GL_APIENTRY glSampleCoverage (GLfloat value, GLboolean invert) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glScissor (GLint x, GLint y, GLsizei width, GLsizei height) {
    if (width < 0 || height < 0) RETURN_ERROR(GL_INVALID_VALUE);
    
    _scissor_data = (scissor_data_t) {
        .left = x,
        .bottom = y,
        .width = width,
        .height = height,
    };
}

GL_APICALL void GL_APIENTRY glStencilFunc (GLenum func, GLint ref, GLuint mask) {
    _stencil_data.front.function = (stencil_function_t) {
        .func = func,
        .ref = ref,
        .mask = mask,
    };
    _stencil_data.back.function = (stencil_function_t) {
        .func = func,
        .ref = ref,
        .mask = mask,
    };
}

GL_APICALL void GL_APIENTRY glStencilFuncSeparate (GLenum face, GLenum func, GLint ref, GLuint mask) {
    switch (face)
    {
    case GL_FRONT:
        _stencil_data.front.function = (stencil_function_t) {
            .func = func,
            .ref = ref,
            .mask = mask,
        };
        break;
    case GL_BACK:
        _stencil_data.back.function = (stencil_function_t) {
            .func = func,
            .ref = ref,
            .mask = mask,
        };
        break;
    case GL_FRONT_AND_BACK:
        _stencil_data.front.function = (stencil_function_t) {
            .func = func,
            .ref = ref,
            .mask = mask,
        };
        _stencil_data.back.function = (stencil_function_t) {
            .func = func,
            .ref = ref,
            .mask = mask,
        };
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

GL_APICALL void GL_APIENTRY glStencilMask (GLuint mask) {
    _masks.stencil = (stencil_mask_t) {
        .front  = mask,
        .back   = mask,
    };
}

GL_APICALL void GL_APIENTRY glStencilMaskSeparate (GLenum face, GLuint mask) {
    switch (face) {
    case GL_FRONT:
        _masks.stencil.front = mask;
        break;
    case GL_BACK:
        _masks.stencil.back = mask;
        break;
    case GL_FRONT_AND_BACK:
        _masks.stencil = (stencil_mask_t) {
            .front  = mask,
            .back   = mask,
        };
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

GL_APICALL void GL_APIENTRY glStencilOp (GLenum fail, GLenum zfail, GLenum zpass) {
    _stencil_data.front.operation = (stencil_operation_t) {
        .sfail = fail,
        .dpfail = zfail,
        .dppasss = zpass
    };
    _stencil_data.back.operation = (stencil_operation_t) {
        .sfail = fail,
        .dpfail = zfail,
        .dppasss = zpass
    };
}

GL_APICALL void GL_APIENTRY glStencilOpSeparate (GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass) {
    switch (face)
    {
    case GL_FRONT:
        _stencil_data.front.operation = (stencil_operation_t) {
            .sfail = sfail,
            .dpfail = dpfail,
            .dppasss = dppass
        };
        break;
    case GL_BACK:
        _stencil_data.back.operation = (stencil_operation_t) {
            .sfail = sfail,
            .dpfail = dpfail,
            .dppasss = dppass
        };
        break;
    case GL_FRONT_AND_BACK:
        _stencil_data.front.operation = (stencil_operation_t) {
            .sfail = sfail,
            .dpfail = dpfail,
            .dppasss = dppass
        };
        _stencil_data.back.operation = (stencil_operation_t) {
            .sfail = sfail,
            .dpfail = dpfail,
            .dppasss = dppass
        };
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

#define max(a, b) (a >= b ? a : b)
#define IS_POWER_OF_2(a) !(a & 0x1u) 

GL_APICALL void GL_APIENTRY glTexStorage2D (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) {
	if (target != GL_TEXTURE_2D) return; // Unknown behaviour

    if (!_texture_binding) RETURN_ERROR(GL_INVALID_OPERATION);
    
    if (width < 1 || height < 1 || levels < 1) RETURN_ERROR(GL_INVALID_VALUE);

    uint32_t pixel_size;

    switch (internalformat) {
    case GL_RGBA8:
        pixel_size = sizeof(uint8_t[4]);
        break;
    case GL_RGB8:
        pixel_size = sizeof(uint8_t[3]);
        break;
    case GL_RG8:
    case GL_RGBA4:
    case GL_RGB5_A1:
    case GL_RGB565:
        pixel_size = sizeof(uint8_t[2]);
        break;
    case GL_R8:
        pixel_size = sizeof(uint8_t[1]);
        break;
    default:
        RETURN_ERROR(GL_INVALID_ENUM);
    }
    
    if (levels > (int) log2f(max(width,height)) + 1) RETURN_ERROR(GL_INVALID_OPERATION);
    
    if (levels != 1 && (IS_POWER_OF_2(width) || IS_POWER_OF_2(height))) RETURN_ERROR(GL_INVALID_OPERATION);
    
    if (_textures[_texture_binding].mem) RETURN_ERROR(GL_INVALID_OPERATION); 
    
    #ifndef HOSTGPU

    texture_t *texture = _textures + _active_textures[_current_active_texture].binding;

    texture->width = width;
    texture->height = height;
    texture->internalformat = internalformat;

    GLsizei level = 0;
    size_t total_pixels = 0;
    while (level++ < levels) { 
        total_pixels += width * height;
        width /= 2;
        height /= 2;
    }

    texture->mem = createBuffer(MEM_READ_ONLY, total_pixels*pixel_size, NULL);
    #else 
    // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_mapping_to_external_image_formats
    cl_image_format format; 
    cl_image_desc desc;

    switch (internalformat) {
        case GL_RGBA8:
            format.image_channel_order = CL_RGBA;
            format.image_channel_data_type = CL_UNORM_INT8;
            break;
        case GL_RGB8:
            format.image_channel_order = CL_RGB;
            format.image_channel_data_type = CL_UNORM_INT8;
            break;
        case GL_RG8:
            format.image_channel_order = CL_RG;
            format.image_channel_data_type = CL_UNORM_INT8;
            break;
        case GL_R8:
            format.image_channel_order = CL_R;
            format.image_channel_data_type = CL_UNORM_INT8;
            break;
        case GL_RGBA4:
            NOT_IMPLEMENTED;
        case GL_RGB5_A1:
            format.image_channel_order = CL_RGBA;
            format.image_channel_data_type = CL_UNORM_SHORT_555;
        case GL_RGB565:
            format.image_channel_order = CL_RGB;
            format.image_channel_data_type = CL_UNORM_SHORT_565;
            break;
        default:
            RETURN_ERROR(GL_INVALID_ENUM);
    }

    desc.image_width = width;
    desc.image_height = height;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.num_mip_levels = levels - 1;

    //[_texture_binding].mem = createImage(MEM_READ_ONLY, &format, &desc, NULL);
    #endif
}

GL_APICALL void GL_APIENTRY glTexParameterf (GLenum target, GLenum pname, GLfloat param) NOT_IMPLEMENTED;
GL_APICALL void GL_APIENTRY glTexParameterfv (GLenum target, GLenum pname, const GLfloat *params) NOT_IMPLEMENTED;
GL_APICALL void GL_APIENTRY glTexParameteri (GLenum target, GLenum pname, GLint param) {
    // Chekc if value is correct
    switch (pname)
    {
    case GL_TEXTURE_WRAP_S:
    case GL_TEXTURE_WRAP_T:
        if (param != GL_CLAMP_TO_EDGE && param != GL_REPEAT && param != GL_MIRRORED_REPEAT) NOT_IMPLEMENTED;
        break;
    case GL_TEXTURE_MAG_FILTER:
        if (param != GL_NEAREST && param != GL_LINEAR) NOT_IMPLEMENTED;
        break;
    case GL_TEXTURE_MIN_FILTER:
        if (param != GL_NEAREST                && param != GL_LINEAR && 
            param != GL_NEAREST_MIPMAP_NEAREST && param != GL_NEAREST_MIPMAP_LINEAR && 
            param != GL_LINEAR_MIPMAP_NEAREST  && param != GL_LINEAR_MIPMAP_LINEAR
            ) NOT_IMPLEMENTED;
        break;
    default:
        NOT_IMPLEMENTED;
    }
    // Set value
    switch (pname)
    {
    case GL_TEXTURE_WRAP_S:
        _textures[_active_textures[_current_active_texture].binding].wraps.s          = param; return;        
    case GL_TEXTURE_WRAP_T:
        _textures[_active_textures[_current_active_texture].binding].wraps.t          = param; return;
    case GL_TEXTURE_MIN_FILTER:
        _textures[_active_textures[_current_active_texture].binding].wraps.min_filter = param; return;
    case GL_TEXTURE_MAG_FILTER:
        _textures[_active_textures[_current_active_texture].binding].wraps.mag_filter = param; return;
    }
};
GL_APICALL void GL_APIENTRY glTexParameteriv (GLenum target, GLenum pname, const GLint *params) NOT_IMPLEMENTED;

GL_APICALL void GL_APIENTRY glTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) {
    
    texture_t *texture_unit = _textures + _active_textures[_current_active_texture].binding;
    
    if (level < 0 || level > (int) log2f(max(texture_unit->width,texture_unit->height)))
        RETURN_ERROR(GL_INVALID_VALUE);

    if (xoffset < 0 || yoffset < 0 || xoffset + width > texture_unit->width || yoffset + height > texture_unit->height)
        RETURN_ERROR(GL_INVALID_VALUE);

    // TODO subImage2d kernel
    #ifndef HOSTGPU
    if (texture_unit->internalformat == GL_RGBA8 && format == GL_RGBA && type == GL_UNSIGNED_BYTE && xoffset == 0 && yoffset == 0) {
        enqueueWriteBuffer(getCommandQueue(),texture_unit->mem, CL_TRUE, 0, width*height*sizeof(uint8_t[4]), pixels);
    } else NOT_IMPLEMENTED;
    #else
    size_t origin[2], region[2], pixel_size;
    origin[0] = xoffset;
    origin[1] = yoffset;
    region[0] = width;
    region[1] = height;

    if (format == GL_RGBA && type == GL_UNSIGNED_BYTE) pixel_size = sizeof(uint8_t[4]);
    if (format == GL_RGBA && type == GL_UNSIGNED_SHORT_5_5_5_1) pixel_size = sizeof(uint8_t[2]);
    if (format == GL_RGBA && type == GL_UNSIGNED_SHORT_5_6_5) pixel_size = sizeof(uint8_t[2]);
    else if (format == GL_RGBA && type == GL_UNSIGNED_SHORT) pixel_size = sizeof(uint8_t[2]);
    else if (format == GL_RGB && type == GL_UNSIGNED_BYTE) pixel_size = sizeof(uint8_t[3]);
    else if (format == GL_RG && type == GL_UNSIGNED_BYTE) pixel_size = sizeof(uint8_t[2]);
    else if (format == GL_RED && type == GL_UNSIGNED_BYTE) pixel_size = sizeof(uint8_t[1]);
    else RETURN_ERROR(GL_INVALID_ENUM);

    //enqueueWriteImage(getCommandQueue(), _textures[_texture_binding].mem, &origin, &region, pixel_size*width, 0, pixels);
    #endif
}

#define ERROR_CHECKER(_COUNT, _SIZE, _TYPE) ({                                                                          \
    if (!_current_program) RETURN_ERROR_WITH_MESSAGE(GL_INVALID_OPERATION, "No current program setted.");                                                          \
    if (CURRENT_PROGRAM.uniforms_data[location].size != _SIZE) RETURN_ERROR_WITH_MESSAGE(GL_INVALID_OPERATION, "Size do not match.");                      \
    if (   CURRENT_PROGRAM.uniforms_data[location].type != GL_BYTE                                                      \
        && CURRENT_PROGRAM.uniforms_data[location].type != _TYPE)                                                       \
        RETURN_ERROR_WITH_MESSAGE(GL_INVALID_OPERATION, "Type do not match.");                                                                             \
    if (location >= CURRENT_PROGRAM.active_uniforms) RETURN_ERROR_WITH_MESSAGE(GL_INVALID_OPERATION, "Location equal or greater than active uniforms.");                                \
    if (_COUNT < -1 || _COUNT > CURRENT_PROGRAM.uniforms_array_size[location]) RETURN_ERROR_WITH_MESSAGE(GL_INVALID_OPERATION, "Array size is less than -1 or greater than expected.");      \
    if (location == -1) return;                                                                                         \
    })

#define GENERIC_UNIFORM(_SIZE, _TYPE, _ARRAY_TYPE, _ARRAY) ({                                                   \
    ERROR_CHECKER(1, _SIZE, _TYPE);                                                                             \
    if (CURRENT_PROGRAM.uniforms_data[location].type == GL_BYTE) NOT_IMPLEMENTED;                               \
    _ARRAY_TYPE value[] = _ARRAY;                                                                               \
    enqueueWriteBuffer(getCommandQueue(), CURRENT_PROGRAM.uniforms_mem[location], GL_TRUE, 0, sizeof(value), value);      \
    })

#define GENERIC_UNIFORM_V(_SIZE, _TYPE, _ARRAY_TYPE) ({                                                                         \
    ERROR_CHECKER(count, _SIZE, _TYPE);                                                                                         \
    if (CURRENT_PROGRAM.uniforms_data[location].type == GL_BYTE) NOT_IMPLEMENTED;                                               \
    enqueueWriteBuffer(getCommandQueue(), CURRENT_PROGRAM.uniforms_mem[location], GL_TRUE, 0, sizeof(_ARRAY_TYPE[count][_SIZE]), value);  \
    })


GL_APICALL void GL_APIENTRY glUniform1f (GLint location, GLfloat v0) {
    GENERIC_UNIFORM(1, GL_FLOAT, GLfloat,P99_PROTECT({v0}));
}
GL_APICALL void GL_APIENTRY glUniform1fv (GLint location, GLsizei count, const GLfloat *value) {
    GENERIC_UNIFORM_V(1, GL_FLOAT, GLfloat);
}
GL_APICALL void GL_APIENTRY glUniform1i (GLint location, GLint v0) { // TODO: refactor uniforms with samplers
    GLuint sampler_loc = location - CURRENT_PROGRAM.active_uniforms;
    if (sampler_loc < CURRENT_PROGRAM.texture_unit_size)
        CURRENT_PROGRAM.sampler_value[sampler_loc] = v0;
    else
        GENERIC_UNIFORM(1, GL_INT, GLint, P99_PROTECT({v0}));
}
GL_APICALL void GL_APIENTRY glUniform1iv (GLint location, GLsizei count, const GLint *value) {
    GENERIC_UNIFORM_V(1, GL_INT, GLint);
}
GL_APICALL void GL_APIENTRY glUniform2f (GLint location, GLfloat v0, GLfloat v1) {
    GENERIC_UNIFORM(2, GL_FLOAT, GLfloat, P99_PROTECT({v0, v1}));
}
GL_APICALL void GL_APIENTRY glUniform2fv (GLint location, GLsizei count, const GLfloat *value) {
    GENERIC_UNIFORM_V(2, GL_FLOAT, GLfloat);
}
GL_APICALL void GL_APIENTRY glUniform2i (GLint location, GLint v0, GLint v1) {
    GENERIC_UNIFORM(2, GL_INT, GLint, P99_PROTECT({v0, v1}));
}
GL_APICALL void GL_APIENTRY glUniform2iv (GLint location, GLsizei count, const GLint *value) {
    GENERIC_UNIFORM_V(2, GL_INT, GLint);
}
GL_APICALL void GL_APIENTRY glUniform3f (GLint location, GLfloat v0, GLfloat v1, GLfloat v2) {
    GENERIC_UNIFORM(3, GL_FLOAT, GLfloat, P99_PROTECT({v0, v1, v2}));
}
GL_APICALL void GL_APIENTRY glUniform3fv (GLint location, GLsizei count, const GLfloat *value) {
    GENERIC_UNIFORM_V(3, GL_FLOAT, GLfloat);
}
GL_APICALL void GL_APIENTRY glUniform3i (GLint location, GLint v0, GLint v1, GLint v2) {
    GENERIC_UNIFORM(3, GL_INT, GLint, P99_PROTECT({v0, v1, v2}));
}
GL_APICALL void GL_APIENTRY glUniform3iv (GLint location, GLsizei count, const GLint *value) {
    GENERIC_UNIFORM_V(3, GL_INT, GLint);
}
GL_APICALL void GL_APIENTRY glUniform4f (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
    GENERIC_UNIFORM(4, GL_FLOAT, GLfloat, P99_PROTECT({v0, v1, v2, v3}));
}
GL_APICALL void GL_APIENTRY glUniform4fv (GLint location, GLsizei count, const GLfloat *value) {
    GENERIC_UNIFORM_V(4, GL_FLOAT, GLfloat);
}
GL_APICALL void GL_APIENTRY glUniform4i (GLint location, GLint v0, GLint v1, GLint v2, GLint v3) {
    GENERIC_UNIFORM(4, GL_INT, GLint, P99_PROTECT({v0, v1, v2, v3}));
}
GL_APICALL void GL_APIENTRY glUniform4iv (GLint location, GLsizei count, const GLint *value) {
    GENERIC_UNIFORM_V(4, GL_INT, GLint);
}
GL_APICALL void GL_APIENTRY glUniformMatrix2fv (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    if (transpose != GL_FALSE) RETURN_ERROR(GL_INVALID_VALUE);

    GENERIC_UNIFORM_V(8,GL_FLOAT,GLfloat);
}
GL_APICALL void GL_APIENTRY glUniformMatrix3fv (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    if (transpose != GL_FALSE) RETURN_ERROR(GL_INVALID_VALUE);

    GENERIC_UNIFORM_V(16,GL_FLOAT,GLfloat);
}
GL_APICALL void GL_APIENTRY glUniformMatrix4fv (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    if (transpose != GL_FALSE) RETURN_ERROR(GL_INVALID_VALUE);

    GENERIC_UNIFORM_V(16,GL_FLOAT,GLfloat);
}

GL_APICALL void GL_APIENTRY glUseProgram (GLuint program){
    if (program && _programs[program].last_load_attempt == GL_FALSE) 
        RETURN_ERROR(GL_INVALID_OPERATION); // TODO: Check if this emits the error or draw function has to
    
    _current_program=program;
}


#define SET_VERTEX_ATTRIB(index, x, y, z, w) ({                         \
    if (index>=MAX_VERTEX_ATTRIBS) RETURN_ERROR(GL_INVALID_VALUE);      \
    _vertex_attrib_state[index] = VEC4;                                 \
    _vertex_attribs[index].vec4.values[0] = x;                          \
    _vertex_attribs[index].vec4.values[1] = y;                          \
    _vertex_attribs[index].vec4.values[2] = z;                          \
    _vertex_attribs[index].vec4.values[3] = w;                          \
    })

GL_APICALL void GL_APIENTRY glVertexAttrib1f (GLuint index, GLfloat x) {
    SET_VERTEX_ATTRIB(index, x, 0.f, 0.f, 1.f);
}
GL_APICALL void GL_APIENTRY glVertexAttrib1fv (GLuint index, const GLfloat *v) {
    SET_VERTEX_ATTRIB(index, v[0], 0.f, 0.f, 1.f);
}
GL_APICALL void GL_APIENTRY glVertexAttrib2f (GLuint index, GLfloat x, GLfloat y) {
    SET_VERTEX_ATTRIB(index, x, y, 0.f, 1.f);
}
GL_APICALL void GL_APIENTRY glVertexAttrib2fv (GLuint index, const GLfloat *v) {
    SET_VERTEX_ATTRIB(index, v[0], v[1], 0.f, 1.f);
}
GL_APICALL void GL_APIENTRY glVertexAttrib3f (GLuint index, GLfloat x, GLfloat y, GLfloat z) {
    SET_VERTEX_ATTRIB(index, x, y, z, 1.f);
}
GL_APICALL void GL_APIENTRY glVertexAttrib3fv (GLuint index, const GLfloat *v) {
    SET_VERTEX_ATTRIB(index, v[0], v[1], v[2], 1.f);
}
GL_APICALL void GL_APIENTRY glVertexAttrib4f (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
    SET_VERTEX_ATTRIB(index, x, y, z, w);
}
GL_APICALL void GL_APIENTRY glVertexAttrib4fv (GLuint index, const GLfloat *v) {
    SET_VERTEX_ATTRIB(index, v[0], v[1], v[2], v[3]);
}

GL_APICALL void GL_APIENTRY glVertexAttribPointer (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) {
    if (index >= MAX_VERTEX_ATTRIBS) RETURN_ERROR(GL_INVALID_VALUE);
    if (size > 4 || size <=0) RETURN_ERROR(GL_INVALID_VALUE);
    if (stride < 0) RETURN_ERROR(GL_INVALID_VALUE);
    if (type < GL_BYTE || type > GL_FLOAT) RETURN_ERROR(GL_INVALID_VALUE);

    _vertex_attribs[index].pointer = (vertex_attrib_pointer_t) {
        .size = size,
        .type = type,
        .normalized = normalized,
        .stride = stride,
        .pointer = pointer,
        .binding = _buffer_binding
    };
    _vertex_attrib_state[index] = POINTER;
}
GL_APICALL void GL_APIENTRY glViewport (GLint x, GLint y, GLsizei width, GLsizei height){
    _viewport.x=x;
    _viewport.y=y;
    _viewport.width=width;
    _viewport.height=height;
}

/**** Utils & inline functions ****\
 *
*/

void* getCommandQueue() {
    static void* command_queue;
    if (command_queue == NULL) command_queue = createCommandQueue(0);
    return command_queue;
}

unsigned int size_from_name_type(const char* name_type) {
    #define RETURN_IF_SIZE_FROM(_TYPE)                              \
        if (strncmp(name_type, _TYPE, sizeof(_TYPE) - 1) == 0) {    \
            substr_size = name_type + sizeof(_TYPE) - 1;            \
            if (*substr_size == '*' || *substr_size == '\0') return 1;                     \
            return atoi(substr_size);                               \
        }

    const char* substr_size;
    RETURN_IF_SIZE_FROM("float");
    RETURN_IF_SIZE_FROM("int");
    RETURN_IF_SIZE_FROM("short");
    RETURN_IF_SIZE_FROM("char");
    RETURN_IF_SIZE_FROM("bool");
    #undef RETURN_IF_SIZE_FROM

    // OpenGL - OpenCL special types
    if (strcmp(name_type, "sampler2D_t") == 0) return 1;
    #ifdef HOSTGPU
    if (strcmp(name_type, "image_t") == 0) return 1;
    #else
    if (strncmp(name_type, "uchar", sizeof("uchar") -1) == 0) return 1;
    #endif
    printf("%s\n", name_type);
    NOT_IMPLEMENTED;

}
unsigned int type_from_name_type(const char* name_type) {
    if (strncmp(name_type, "float",  sizeof("float") -1)  == 0) return GL_FLOAT;
    if (strncmp(name_type, "int",    sizeof("int")   -1)  == 0) return GL_INT;
    if (strncmp(name_type, "short",  sizeof("short") -1)  == 0) return GL_SHORT;
    if (strncmp(name_type, "char",   sizeof("char")  -1)  == 0) return GL_BYTE;
    if (strncmp(name_type, "bool",   sizeof("bool")  -1)  == 0) return GL_BYTE;
    
    // OpenGL - OpenCL special types
    if (strcmp(name_type, "sampler2D_t") == 0) return SAMPLER2D_T;
    if (strncmp(name_type, "uchar*", sizeof("uchar*")-1)  == 0) return IMAGE_T;

    // 
    #ifdef DEBUG
    printf("%s\n", name_type);
    NOT_IMPLEMENTED;
    #else
    return 0;
    #endif
}
