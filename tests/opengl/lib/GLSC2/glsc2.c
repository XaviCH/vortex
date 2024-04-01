#include <GLSC2/glsc2.h>
#include "kernel.c"

#define VERTEX_SHADER "kernel.vert.cl"
#define PERS_DIV "kernel-persp-div.cl"
#define VIEWPORT_TRANS "kernel-viewport-trans.cl"

#define MAX_PROGRAMS 255 // TODO
#define _MAX_VERTEX_ATTRIBS 255 // TODO update GL_MAX_VERTEX_ATTRIBS

/****** GENERIC objects ******\
 * 
 * 
*/
typedef struct {
    GLint x;
    GLint y;
    GLsizei width;
    GLsizei height;
} BOX;



BOX viewportTransform;

//entiendo que
// binary = kernel_bin
// length = kernel_length
typedef struct {
    GLboolean created;
    cl_program binary;
    GLsizei length;
}PROGRAM_OBJECT;

PROGRAM_OBJECT programs [MAX_PROGRAMS];
GLboolean _no_program = GL_TRUE;
PROGRAM_OBJECT _current_program;

/****** BUFFER objects ******\
 * 
 * 
*/

typedef struct {
    GLboolean used;
    GLenum target;
    cl_mem mem;
} BUFFER;

typedef struct {
    GLboolean enable, normalized;
    GLuint index;
    GLint size;
    GLenum type;
    GLsizei stride;
    const void *pointer;
} VERTEX_ATTRIB;

BUFFER _buffers[255];
GLuint _buffer_binding;
VERTEX_ATTRIB vertex_attrib[_MAX_VERTEX_ATTRIBS]; 

/****** FRAMEBUFFER objects ******\
 * 
 * 
*/

typedef struct {
    cl_mem mem;
    GLuint color_attachment0, depth_attachment, stencil_attachment;
    GLboolean used;
} FRAMEBUFFER;

FRAMEBUFFER _framebuffers[255];
GLuint _framebuffer_binding;

/****** RENDERBUFFER objects ******\
 * 
 * 
*/

typedef struct {
    cl_mem mem;
    GLenum internalformat;
    GLsizei width, height;
    GLboolean used;
} RENDERBUFFER;

RENDERBUFFER _renderbuffers[255];
GLuint _renderbuffer_binding;

/****** PER-FRAGMENT objects ******\
 * 
 * 
*/

// Color
typedef struct { GLboolean red, green, blue, alpha } COLOR_MASK;

GLboolean _color_enabled = 1;
COLOR_MASK _color_mask = {1, 1, 1, 1};
// Depth
typedef struct { GLfloat n, f } DEPTH_RANGE; // z-near & z-far

GLboolean   _depth_enabled = 0;
GLboolean   _depth_mask = 1;
GLenum      _depth_func = GL_LESS;
DEPTH_RANGE _depth_range = { 0.0, 1.0};
// Scissor

GLuint _scissor_enabled = 0;
BOX _scissor_box;
// Stencil
typedef struct { GLboolean front, back } STENCIL_MASK;

GLuint _stencil_enabled = 0;
STENCIL_MASK _stencil_mask = {1, 1};
// TODO blending & dithering

/****** OpenGL Interface Implementations ******\
 * 
 * 
*/

GL_APICALL void GL_APIENTRY glBindBuffer (GLenum target, GLuint buffer) {
    if (!_buffers[buffer].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    
    _buffer_binding = buffer;
    _buffers[buffer].target = target;
}
GL_APICALL void GL_APIENTRY glBindFramebuffer (GLenum target, GLuint framebuffer) {
    if (!_framebuffers[framebuffer].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    if (target == GL_FRAMEBUFFER) {
        _framebuffer_binding = framebuffer;
    }
}

GL_APICALL void GL_APIENTRY glBufferData (GLenum target, GLsizeiptr size, const void *data, GLenum usage) {

    if (target == GL_ARRAY_BUFFER) {
        if (usage == GL_STATIC_DRAW) {
            _buffers[_buffer_binding].mem = clCreateBuffer(_context, CL_MEM_READ_ONLY, size, data, &_err);
        }
        else if (usage == GL_DYNAMIC_DRAW || usage == GL_STREAM_DRAW) {
            _buffers[_buffer_binding].mem = clCreateBuffer(_context, CL_MEM_READ_WRITE, size, data, &_err);
        }
    }
}

GL_APICALL void GL_APIENTRY glClear (GLbitfield mask) {
    if(mask & GL_COLOR_BUFFER_BIT) glClearColor(0.0,0.0,0.0,0.0);
    if(mask & GL_DEPTH_BUFFER_BIT) glClearDepthf(1.0);
    if(mask & GL_STENCIL_BUFFER_BIT) glClearStencil(0);
}

GL_APICALL void GL_APIENTRY glClearColor (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
    RENDERBUFFER color_attachment0 = _renderbuffers[_framebuffers[_framebuffer_binding].color_attachment0];

    if (color_attachment0.internalformat == GL_RGBA4) {
        unsigned short color;
        color |= (unsigned short) (15*red);
        color |= (unsigned short) (15*green) << 4;
        color |= (unsigned short) (15*blue) << 8;
        color |= (unsigned short) (15*alpha) << 12;
        fill(color_attachment0.mem, color_attachment0.width*color_attachment0.height*2, &color, 2);
    }
}
GL_APICALL void GL_APIENTRY glClearDepthf (GLfloat d) {
    RENDERBUFFER depth_attachment = _renderbuffers[_framebuffers[_framebuffer_binding].depth_attachment];

    if (depth_attachment.internalformat == GL_DEPTH_COMPONENT16) {
        unsigned short value = 65535*d;
        fill(depth_attachment.mem, depth_attachment.width*depth_attachment.height*2, &value, 2);
    }
}
GL_APICALL void GL_APIENTRY glClearStencil (GLint s) {
    RENDERBUFFER stencil_attachment = _renderbuffers[_framebuffers[_framebuffer_binding].stencil_attachment];

    if (stencil_attachment.internalformat == GL_STENCIL_INDEX8) {
        fill(stencil_attachment.mem, stencil_attachment.width*stencil_attachment.height, &s, 1);
    }
}

GL_APICALL void GL_APIENTRY glColorMask (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
    _color_mask.red = red;
    _color_mask.green = green;
    _color_mask.blue = blue;
    _color_mask.alpha = alpha;
}

// TODO move this to another file
inline void gl_pipeline(GLint first, GLsizei count){
    //pipeline
    unsigned int numVerts = count-first;
    //vertex shader
    float primitives[4*GL_MAX_VERTEX_ATTRIBS];
    if(!_no_program == GL_TRUE)
        vertex_shader(first, count, primitives);
    //clip coord
    perspective_division(numVerts, primitives);
    //normalized-device-coords
    viewport_transformation(numVerts, primitives);
    //rasterization
    float fragments[4*viewportTransform.w*viewportTransform.h];//color
    rasterization(numVerts, primitives, fragments, viewportTransform.w*viewportTransform.h);
    //fragment-shader
    fragment_shader();
    //per-vertex-ops
    per_vertex_operations();
    //entiendo que aqui escribe en frame buff
}

// TODO move this to another file
void _glDrawArraysTriangles(GLint first, GLsizei count) {
    gl_pipeline(first, count);
}

GL_APICALL void GL_APIENTRY glDepthFunc (GLenum func) {
    _depth_func = func;
}

GL_APICALL void GL_APIENTRY glDepthMask (GLboolean flag) {
    _depth_mask = flag;
}

GL_APICALL void GL_APIENTRY glDepthRangef (GLfloat n, GLfloat f) {
    _depth_range.n=n;
    _depth_range.f=f;
}

GL_APICALL void GL_APIENTRY glDrawArrays (GLenum mode, GLint first, GLsizei count) {
    if (first <0){
        _err= GL_INVALID_VALUE;
        return;
    }
    if (mode==GL_POINTS); // TODO
    else if (mode==GL_LINES); // TODO
    else if (mode==GL_TRIANGLES) 
        _glDrawArraysTriangles(first, count);
}

GL_APICALL void GL_APIENTRY glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices);

GL_APICALL void GL_APIENTRY glDisable (GLenum cap) {
    if (cap == GL_SCISSOR_TEST)
        _scissor_enabled = 0;
    else if (cap == GL_DEPTH_TEST)
        _depth_enabled = 0;
    else if (cap == GL_STENCIL_TEST)
        _stencil_enabled = 0;
}

GL_APICALL void GL_APIENTRY glDisableVertexAttribArray (GLuint index) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) {
        _err = GL_INVALID_VALUE;
        return;
    }

    vertex_attrib[index].enable = 0;
}

GL_APICALL void GL_APIENTRY glEnable (GLenum cap) {
    if (cap == GL_SCISSOR_TEST)
        _scissor_enabled = 1;
    else if (cap == GL_DEPTH_TEST)
        _depth_enabled = 1;
    else if (cap == GL_STENCIL_TEST)
        _stencil_enabled = 1;
}

GL_APICALL void GL_APIENTRY glEnableVertexAttribArray (GLuint index) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) {
        _err = GL_INVALID_VALUE;
        return;
    }

    vertex_attrib[index].enable = 1;
}

GL_APICALL void GL_APIENTRY glFinish (void);

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

GL_APICALL void GL_APIENTRY glGenFramebuffers (GLsizei n, GLuint *framebuffers) {
    GLuint _id = 1; // _id = 0 is reserved for ARRAY_BUFFER

    while(n > 0 && _id < 256) {
        if (!_framebuffers[_id].used) {
            _framebuffers[_id].used = GL_TRUE;
            *framebuffers = _id;

            framebuffers += 1; 
            n -= 1;
        }
        _id += 1;
    }
}

// TODO check if this has sense does not convice me using CL word or #define it would never be included from the client file 
#define CL_PROGRAM 0


GL_APICALL GLuint GL_APIENTRY glCreateProgram (void){
    for (int i=1; i<MAX_PROGRAMS; i++)
        if (!programs[i].created)
        {
            programs[i].created=GL_TRUE;
            return i;
        }
    return 0;
}

GL_APICALL void GL_APIENTRY glProgramBinary (GLuint program, GLenum binaryFormat, const void *binary, GLsizei length){
    if(!programs[program].binary == (void*)0)
        _err = GL_INVALID_OPERATION;
        return;
    if (binaryFormat == CL_PROGRAM){
        // CREATE BINARY
        cl_program triangle_cl = clCreateProgramWithSource(
            _context, 1, (const char**)&binary, &length, &_err); // vertex + fragment

        programs[program].binary=(*(cl_program*)triangle_cl);
        programs[program].length=length;
    }


}
GL_APICALL void GL_APIENTRY glReadnPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data) {
    if (format == GL_RGBA && type == GL_UNSIGNED_BYTE) {
        if (_framebuffer_binding) {
            RENDERBUFFER color_attachment0 = _renderbuffers[_framebuffers[_framebuffer_binding].color_attachment0];

            // TODO use width and height to get a cutted version of the image
            // TODO this may need a program to convert 4b to 8b rgba
            read(color_attachment0.mem, data, bufSize, 0);
        }
    }
}

GL_APICALL void GL_APIENTRY glScissor (GLint x, GLint y, GLsizei width, GLsizei height) {
    if (width < 0 || height < 0) {
        _err = GL_INVALID_VALUE;
        return;
    }
    _scissor_box.x = x;
    _scissor_box.y = y;
    _scissor_box.width = width;
    _scissor_box.height = height;
}

GL_APICALL void GL_APIENTRY glStencilMask (GLuint mask) {
    _stencil_mask.front = mask;
    _stencil_mask.back = mask;
}
GL_APICALL void GL_APIENTRY glStencilMaskSeparate (GLenum face, GLuint mask) {
    if (GL_FRONT == face) 
        _stencil_mask.front = mask;
    else if (GL_BACK == face) 
        _stencil_mask.back = mask;
    else if (GL_FRONT_AND_BACK == face) {
        _stencil_mask.front = mask;
        _stencil_mask.back = mask;
    }
}

GL_APICALL void GL_APIENTRY glUseProgram (GLuint program){
    if (program=0){
        _no_program=GL_TRUE;
        return;
    }
    if(programs[program].binary==(void*)0){
        _err = GL_INVALID_OPERATION;
        return;
    }
    _no_program=GL_FALSE;
    _current_program=programs[program];
}

GL_APICALL void GL_APIENTRY glVertexAttribPointer (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) {
        _err = GL_INVALID_VALUE;
        return;
    }

    if (size > 4 || size <=0) {
        _err = GL_INVALID_VALUE;
        return;
    }
    
    if (stride < 0) {
        _err = GL_INVALID_VALUE;
        return;
    }

    //check type
    if (type != GL_BYTE || type != GL_UNSIGNED_BYTE || type != GL_SHORT || type != GL_UNSIGNED_SHORT || type != GL_FLOAT){
        _err=GL_INVALID_VALUE;
        return;
    }

    if (normalized == GL_TRUE){
        //normalizar integers
    }

    vertex_attrib[index].size = size;
    vertex_attrib[index].type = type;
    vertex_attrib[index].normalized = normalized;
    vertex_attrib[index].stride = stride;
    vertex_attrib[index].pointer = pointer;
}
GL_APICALL void GL_APIENTRY glViewport (GLint x, GLint y, GLsizei width, GLsizei height){
    viewportTransform.x=x;
    viewportTransform.y=y;
    viewportTransform.width=width;
    viewportTransform.height=height;
}

