#include <math.h>
#include <string.h>
#include <GLSC2/glsc2.h>
#include "kernel.c" // TODO: may be interesting to extract it to an interface so could be re implementated with CUDA
#include "binary.c" // TODO: maybe with extern

#define P99_PROTECT(...) __VA_ARGS__

#define NOT_IMPLEMENTED              \
    {                               \
        printf("NOT_IMPLEMENTED: %s:%d\n", __FILE__, __LINE__); \
        exit(0);                     \
    }

#define INTERNAL_ERROR               \
    ({                               \
        printf("INTERNAL_ERROR\n");  \
        exit(0);                     \
    })

GLenum gl_error = 0;

#define RETURN_ERROR(error)                 \
    ({                                      \
        if(gl_error == GL_NO_ERROR)         \
            gl_error = error;               \
        return;                             \
    })

// Our definitions
#define MAX_BUFFER 256
#define MAX_FRAMEBUFFER 256
#define MAX_RENDERBUFFER 256
#define MAX_TEXTURE 256
#define MAX_PROGRAMS 256
#define MAX_UNIFORM_VECTORS 16
#define MAX_NAME_SIZE 64
#define MAX_INFO_SIZE 256
#define MAX_UNIFORM_SIZE sizeof(float[4][4]) // Limited to a matf4x4

#define VERTEX_SHADER_FNAME "main_vs"
#define PERSPECTIVE_DIVISION_SHADER_FNAME "gl_perspective_division"
#define VIEWPORT_DIVISION_SHADER_FNAME "gl_viewport_division"
#define FRAGMENT_SHADER_FNAME "main_fs"

#define POCL_BINARY 0x0

// OpenGL required definitions
#define MAX_VERTEX_ATTRIBS 16
#define MAX_VERTEX_UNIFORM_VECTORS sizeof(float[4])         // atMost MAX_UNIFORM_VECTORS
#define MAX_FRAGMENT_UNIFORM_VECTORS sizeof(float[4])       // atMost MAX_UNIFORM_VECTORS
#define MAX_RENDERBUFFER_SIZE sizeof(uint16_t[1920][1080])  // TODO: Maybe another number

#define GL_POSITION "gl_Position"
#define GL_FRAGCOLOR "gl_FragColor"
#define GL_FRAGCOORD "gl_FragCoord"
#define GL_DISCARD "gl_Discard"

/****** DTO objects ******\
 * TODO: externalize to could be imported from kernel cl programs
*/

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

typedef struct {
    unsigned char info[MAX_INFO_SIZE];
    int length;
} LOG;

BOX _viewport;

/****** PROGRAM objects ******\
 * 
 * 
*/

typedef struct {
    float values[4]; // default: {0.0, 0.0, 0.0, 1.0} 
} vertex_attrib_t;

typedef struct {
    GLint size;
    GLenum type;
    GLboolean normalized;
    GLsizei stride;
    void *pointer;
    GLuint binding;
} vertex_attrib_pointer_t;

typedef union {
    vertex_attrib_t vec4;
    vertex_attrib_pointer_t pointer;
} vertex_attrib_u;

// States of attributes
#define VEC4 0
#define POINTER 1
#define BUFFEROBJECT 2

unsigned char     _vertex_attrib_enable[MAX_VERTEX_ATTRIBS]; // TODO: use it as checker for state
unsigned char     _vertex_attrib_state[MAX_VERTEX_ATTRIBS];
vertex_attrib_u   _vertex_attribs[MAX_VERTEX_ATTRIBS];


#define PROGRAM_LOG_SIZE 256
#define ARG_NAME_SIZE 128
#define MAX_VARYING 16

typedef struct {
    unsigned int vertex_location, fragment_location, size, type;
    unsigned char name[ARG_NAME_SIZE]; 
} arg_data_t;

typedef struct { 
    GLboolean last_load_attempt, last_validation_attempt;
    unsigned char log[PROGRAM_LOG_SIZE];
} program_data_t;

typedef struct {
    // Program data
    GLboolean               last_load_attempt; 
    GLboolean               last_validation_attempt;
    unsigned char           log[PROGRAM_LOG_SIZE];
    unsigned int            log_length;
    cl_program              program;
    // Kernel data
    cl_kernel               vertex_kernel;
    cl_kernel               fragment_kernel;
    // Uniform data
    unsigned int            active_uniforms;
    arg_data_t              uniforms_data[MAX_UNIFORM_SIZE];
    unsigned int            uniforms_array_size[MAX_UNIFORM_SIZE];
    cl_mem                  uniforms_mem[MAX_UNIFORM_SIZE];
    // Vertex data
    unsigned int            active_vertex_attribs;
    arg_data_t              vertex_attribs_data[MAX_VERTEX_ATTRIBS];
    unsigned int            gl_position_location;
    // Varying data
    unsigned int            varying_size;
    arg_data_t              varying_data[16];
    // Fragment data
    unsigned int            gl_fragcolor_location;
    int                     gl_fragcoord_location;
    int                     gl_discard_location;
} program_t;

unsigned int            _current_program;
program_t               _programs[MAX_PROGRAMS];

typedef struct {
    cl_kernel never, always, less, lequal, equal, greater, gequal, notequal;
} depth_kernel_container_t;

typedef struct {
    cl_kernel points, line_stip, line_loop, lines, triangle_strip, triangle_fan, triangles;
} rasterization_kernel_container_t;

typedef struct {
    cl_kernel enabled_rgba4, disabled_rgba4, rgb5_a1, rgb565;
} dithering_kernel_container_t;

typedef struct {
    cl_kernel bit16, bit8;
} clear_kernel_container_t;

typedef struct {
    dithering_kernel_container_t dithering;
    depth_kernel_container_t depth;
    clear_kernel_container_t clear;
    rasterization_kernel_container_t rasterization;
    cl_kernel viewport_division, perspective_division, readnpixels, strided_write;
} kernel_container_t;

kernel_container_t _kernels; 

typedef struct {
    GLboolean depth, stencil, scissor, pixel_ownership, dithering, bleding, culling
} enabled_container_t;

enabled_container_t _enableds = {
    .dithering = GL_TRUE,
    .culling = GL_FALSE,
};

typedef struct {
    GLboolean r,g,b,a;
} color_mask_t;

typedef struct {
    GLboolean front, back;
} stencil_mask_t;

typedef struct {
    color_mask_t color;
    GLboolean depth;
    stencil_mask_t stencil;
} mask_container_t;

mask_container_t _masks = {
    .color = { .r=GL_FALSE, .g=GL_FALSE, .b=GL_FALSE, .a=GL_FALSE },
    .depth = GL_FALSE,
    .stencil = GL_FALSE
};

GLenum _front_face = GL_CCW;
GLenum _cull_face = GL_BACK;

typedef struct {
    void *less;
} DEPTH_KERNEL;

typedef struct {
    void *rgba4, *rgba8;
} COLOR_KERNEL;

typedef struct {
    cl_kernel triangles
} rasterization_kernels_t;

GLboolean _kernel_load_status;
DEPTH_KERNEL _depth_kernel;
COLOR_KERNEL _color_kernel;
rasterization_kernels_t _rasterization_kernels;
void *_viewport_division_kernel;
void *_perspective_division_kernel;
void *_readnpixels_kernel;
void *_strided_write_kernel;

/******s BUFFER objects ******\
 * TODOs: Re think this, I think it is actually more tricky than the first though. 
 * Seams that the program object holds also the vertex attributes, and the VAO is on 
 * server side.
 * 
*/

typedef struct {
    GLboolean used;
    GLenum target;
    void* mem;
} buffer_t;


buffer_t _buffers[MAX_BUFFER];
GLuint _buffer_binding;

/****** FRAMEBUFFER objects ******\
 * 
 * 
*/

typedef struct {
    GLuint color_attachment0, depth_attachment, stencil_attachment;
    GLboolean used;
} framebuffer_t;

framebuffer_t _framebuffers[MAX_FRAMEBUFFER];
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
} renderbuffer_t;

renderbuffer_t _renderbuffers[MAX_RENDERBUFFER];
GLuint _renderbuffer_binding;

/****** TEXTURE 2D objects ******\
 * 
 * 
*/

typedef struct {
    cl_mem mem;
    GLenum internalformat;
    GLsizei width, height;
    GLboolean used;
    #ifdef IMAGE_SUPPORT
    cl_sampler sampler;
    #endif
} TEXTURE_2D;

TEXTURE_2D _textures[MAX_TEXTURE];
GLuint _texture_binding;

/****** PER-FRAGMENT objects ******\
 * 
 * 
*/

// Color
typedef struct { GLboolean red, green, blue, alpha; } COLOR_MASK;

COLOR_MASK _color_mask = {1, 1, 1, 1};
// Depth
typedef struct { GLfloat n, f; } DEPTH_RANGE; // z-near & z-far

GLboolean   _depth_enabled = 0;
GLboolean   _depth_mask = 1;
GLenum      _depth_func = GL_LESS;
DEPTH_RANGE _depth_range = {0.0, 1.0};
// Scissor

GLuint _scissor_enabled = 0;
BOX _scissor_box;
// Stencil
typedef struct { GLboolean front, back; } STENCIL_MASK;

GLuint _stencil_enabled = 0;
STENCIL_MASK _stencil_mask = {1, 1};
// TODO blending & dithering

/****** SHORCUTS MACROS ******/
#define COLOR_ATTACHMENT0 _renderbuffers[_framebuffers[_framebuffer_binding].color_attachment0]
#define DEPTH_ATTACHMENT _renderbuffers[_framebuffers[_framebuffer_binding].depth_attachment]
#define STENCIL_ATTACHMENT _renderbuffers[_framebuffers[_framebuffer_binding].stencil_attachment]
#define CURRENT_PROGRAM _programs[_current_program]


/****** Interface for utils & inline functions ******\
 * Utility or inline function are implemented at the end of the file. 
*/
void* getCommandQueue();

unsigned int size_from_name_type(const char*);
unsigned int type_from_name_type(const char*);

cl_kernel getDepthKernel();
cl_kernel getDitherKernel();

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
    }
} 

/****** OpenGL Interface Implementations ******\
 * 
 * 
*/
GL_APICALL void GL_APIENTRY glActiveTexture (GLenum texture) NOT_IMPLEMENTED;

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
        _texture_binding = texture;
    } else NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glBlendColor (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) NOT_IMPLEMENTED;
GL_APICALL void GL_APIENTRY glBlendEquation (GLenum mode) NOT_IMPLEMENTED;
GL_APICALL void GL_APIENTRY glBlendEquationSeparate (GLenum modeRGB, GLenum modeAlpha) NOT_IMPLEMENTED;
GL_APICALL void GL_APIENTRY glBlendFunc (GLenum sfactor, GLenum dfactor) NOT_IMPLEMENTED;
GL_APICALL void GL_APIENTRY glBlendFuncSeparate (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha) NOT_IMPLEMENTED;

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

GL_APICALL void GL_APIENTRY glClear (GLbitfield mask) {
    if (_enableds.pixel_ownership || _enableds.scissor) NOT_IMPLEMENTED;
    if (mask & GL_COLOR_BUFFER_BIT)     glClearColor(0,0,0,0);
    if (mask & GL_DEPTH_BUFFER_BIT)     glClearDepthf(1);
    if (mask & GL_STENCIL_BUFFER_BIT)   glClearStencil(0);
    if (mask & (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) == 0) RETURN_ERROR(GL_INVALID_VALUE);
}

GL_APICALL void GL_APIENTRY glClearColor (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
    if (_masks.color.r && _masks.color.g && _masks.color.b && _masks.color.a) return; // No paint is done

    uint16_t color16 = 0;
    uint16_t mask16 = 0;

    if (COLOR_ATTACHMENT0.internalformat == GL_RGBA4) {
        color16 |= (unsigned int) (red   * 0xFu) << 0;
        color16 |= (unsigned int) (green * 0xFu) << 4;
        color16 |= (unsigned int) (blue  * 0xFu) << 8;
        color16 |= (unsigned int) (alpha * 0xFu) << 12;
        if (_masks.color.r) mask16 |= 0x000Fu;
        if (_masks.color.g) mask16 |= 0x00F0u;
        if (_masks.color.b) mask16 |= 0x0F00u;
        if (_masks.color.a) mask16 |= 0xF000u;
    } else if(COLOR_ATTACHMENT0.internalformat == GL_RGBA8) {
        NOT_IMPLEMENTED;
    } else if(COLOR_ATTACHMENT0.internalformat == GL_RGB8) {
        NOT_IMPLEMENTED;
    } else if(COLOR_ATTACHMENT0.internalformat == GL_RG8) {
        NOT_IMPLEMENTED;
    } else if(COLOR_ATTACHMENT0.internalformat == GL_R8) {
        NOT_IMPLEMENTED;
    } else if(COLOR_ATTACHMENT0.internalformat == GL_RGB5_A1) {
        color16 |= (unsigned int) (red   * 0x1Fu) << 0;
        color16 |= (unsigned int) (green * 0x1Fu) << 5;
        color16 |= (unsigned int) (blue  * 0x1Fu) << 10;
        color16 |= (unsigned int) (alpha * 0x1u)   << 15;
        if (_masks.color.r) mask16 |= 0x1Fu << 0;
        if (_masks.color.g) mask16 |= 0x1Fu << 5; 
        if (_masks.color.b) mask16 |= 0x1Fu << 10;
        if (_masks.color.a) mask16 |= 0x1u  << 15;
    } else if(COLOR_ATTACHMENT0.internalformat == GL_RGB565) {
        color16 |= (unsigned int) (red   * 0x1Fu) << 0;
        color16 |= (unsigned int) (green * 0x3Fu) << 5;
        color16 |= (unsigned int) (blue  * 0x1Fu) << 11;
        if (_masks.color.r) mask16 |= 0x1Fu << 0;
        if (_masks.color.g) mask16 |= 0x3Fu << 5; 
        if (_masks.color.b) mask16 |= 0x1Fu << 11;
    } else RETURN_ERROR(GL_INVALID_OPERATION);

    setKernelArg(_kernels.clear.bit16, 0, sizeof(cl_mem),  &COLOR_ATTACHMENT0.mem);
    setKernelArg(_kernels.clear.bit16, 1, sizeof(color16), &color16);
    setKernelArg(_kernels.clear.bit16, 2, sizeof(mask16),  &mask16);
    enqueueNDRangeKernel(getCommandQueue(), _kernels.clear.bit16, COLOR_ATTACHMENT0.width*COLOR_ATTACHMENT0.height);

}

GL_APICALL void GL_APIENTRY glClearDepthf (GLfloat d) {
    if (_masks.depth) return; // no paint is done
    
    uint16_t depth16 = 0;
    uint16_t mask16 = 0;

    if (DEPTH_ATTACHMENT.internalformat == GL_DEPTH_COMPONENT16) {
        depth16 = d*0xFFFFu;
    } else NOT_IMPLEMENTED;

    setKernelArg(_kernels.clear.bit16, 0, sizeof(cl_mem),   &DEPTH_ATTACHMENT.mem);
    setKernelArg(_kernels.clear.bit16, 1, sizeof(depth16),  depth16);
    setKernelArg(_kernels.clear.bit16, 2, sizeof(mask16),   mask16);
    enqueueNDRangeKernel(getCommandQueue(), _kernels.clear.bit16, DEPTH_ATTACHMENT.width*DEPTH_ATTACHMENT.height);

}

GL_APICALL void GL_APIENTRY glClearStencil (GLint s) {
    NOT_IMPLEMENTED;
    if (_masks.stencil.front && _masks.stencil.back) return; // no paint is done
    
    uint8_t stencil8 = 0;
    uint8_t mask8 = 0;

    if (STENCIL_ATTACHMENT.internalformat == GL_STENCIL_INDEX8) {
        stencil8 = s*0xFFu;
    } else NOT_IMPLEMENTED;

    setKernelArg(_kernels.clear.bit8, 0, sizeof(cl_mem),    &STENCIL_ATTACHMENT.mem);
    setKernelArg(_kernels.clear.bit8, 1, sizeof(stencil8),  stencil8);
    setKernelArg(_kernels.clear.bit8, 2, sizeof(mask8),     mask8);
    enqueueNDRangeKernel(getCommandQueue(), _kernels.clear.bit8, STENCIL_ATTACHMENT.width*STENCIL_ATTACHMENT.height);

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
    _depth_range = (DEPTH_RANGE) {.n=n, .f=f};
}

GL_APICALL void GL_APIENTRY glDisable (GLenum cap) {
    if (cap == GL_SCISSOR_TEST)
        _enableds.scissor = GL_FALSE;
    else if (cap == GL_DEPTH_TEST)
        _enableds.depth = GL_FALSE;
    else if (cap == GL_STENCIL_TEST)
        _enableds.stencil = GL_FALSE;
    else if (cap == GL_CULL_FACE)
        _enableds.culling = GL_FALSE;
    else if (cap == GL_DITHER)
        _enableds.dithering = GL_TRUE;
    else NOT_IMPLEMENTED; // RETURN_ERROR(GL_INVALID_ENUM);
}

GL_APICALL void GL_APIENTRY glDisableVertexAttribArray (GLuint index) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) RETURN_ERROR(GL_INVALID_VALUE);

    _vertex_attrib_enable[index] = 0;
}

GL_APICALL void GL_APIENTRY glDrawArrays (GLenum mode, GLint first, GLsizei count) {
    if (!_current_program) RETURN_ERROR(GL_INVALID_OPERATION);
    if (first <0) RETURN_ERROR(GL_INVALID_VALUE);
    if (first != 0) NOT_IMPLEMENTED;
    /* ---- General vars ---- */
    cl_int tmp_err; 
    cl_buffer_region region;

    GLsizei num_vertices = count-first;
    GLsizei num_fragments = COLOR_ATTACHMENT0.width * COLOR_ATTACHMENT0.height;
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

            cl_command_queue command_queue = createCommandQueue(0);
            
            if (pointer->binding) {
                region.origin = (uint64_t) pointer->pointer, 
                region.size = buffer_in_size;
                temp_mem[attrib] = clCreateSubBuffer(_buffers[pointer->binding].mem, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &tmp_err);
            } else {
                temp_mem[attrib] = createBuffer(MEM_READ_WRITE | MEM_COPY_HOST_PTR, buffer_in_size, pointer->pointer);
            }
            vertex_array_mem[attrib] = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_vertices, NULL);

            setKernelArg(_strided_write_kernel, 0, sizeof(pointer->size),       &pointer->size);
            setKernelArg(_strided_write_kernel, 1, sizeof(pointer->type),       &pointer->type);
            setKernelArg(_strided_write_kernel, 2, sizeof(pointer->normalized), &pointer->normalized);
            setKernelArg(_strided_write_kernel, 3, sizeof(pointer->stride),     &pointer->stride);
            setKernelArg(_strided_write_kernel, 4, sizeof(cl_mem),              &temp_mem[attrib]);
            setKernelArg(_strided_write_kernel, 5, sizeof(cl_mem),              &vertex_array_mem[attrib]);
            
            enqueueNDRangeKernel(command_queue, _strided_write_kernel, num_vertices);
        }
    }
    
    cl_mem vertex_out_buffer    = createBuffer(CL_MEM_READ_WRITE,   sizeof(float[4])*num_vertices*CURRENT_PROGRAM.varying_size, NULL);
    cl_mem gl_Positions         = createBuffer(MEM_READ_WRITE,      sizeof(float[4])*num_vertices,                              NULL);

    /* ---- Set Up Per-Vertex Kernels ---- */
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
    setKernelArg(_perspective_division_kernel, 0, sizeof(gl_Positions), &gl_Positions);

    // Viewport Division Kernel Set Up
    setKernelArg(_viewport_division_kernel, 0, sizeof(gl_Positions), &gl_Positions);
    setKernelArg(_viewport_division_kernel, 1, sizeof(_viewport),    &_viewport);
    setKernelArg(_viewport_division_kernel, 2, sizeof(_depth_range), &_depth_range);

    /* ---- Set Up Per-Fragment Buffers ---- */
    cl_mem fragment_in_buffer, gl_FragCoord, gl_Discard, gl_FragColor;
    
    fragment_in_buffer  = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments*CURRENT_PROGRAM.varying_size, NULL);
    gl_FragCoord        = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments,                              NULL);
    gl_Discard          = createBuffer(MEM_READ_WRITE, sizeof(uint8_t)*num_fragments,                               NULL);
    gl_FragColor        = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments,                              NULL);

    /* ---- Set Up Per-Fragment Kernels ---- */
    cl_kernel fragment_kernel;
    // Rasterization Kernel Set Up
    uint16_t cull_face = 0;
    if (_enableds.culling) cull_face = _cull_face;

    if (mode==GL_TRIANGLES) {
        // arg 0 is reserved for the primitive index
        setKernelArg(_rasterization_kernels.triangles, 1, sizeof(int),      &COLOR_ATTACHMENT0.width);
        setKernelArg(_rasterization_kernels.triangles, 2, sizeof(int),      &CURRENT_PROGRAM.varying_size);
        setKernelArg(_rasterization_kernels.triangles, 3, sizeof(cl_mem),   &gl_Positions);
        setKernelArg(_rasterization_kernels.triangles, 4, sizeof(cl_mem),   &gl_FragCoord);
        setKernelArg(_rasterization_kernels.triangles, 5, sizeof(cl_mem),   &gl_Discard);
        setKernelArg(_rasterization_kernels.triangles, 6, sizeof(cl_mem),   &vertex_out_buffer);
        setKernelArg(_rasterization_kernels.triangles, 7, sizeof(cl_mem),   &fragment_in_buffer);
        setKernelArg(_rasterization_kernels.triangles, 8, sizeof(uint16_t), &_front_face);
        setKernelArg(_rasterization_kernels.triangles, 9, sizeof(uint16_t), &cull_face);
    } else NOT_IMPLEMENTED;

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

    // Texture vars
    // TODO: Texture support, this is 
    int active_textures = _texture_binding != 0;
    // In fragment args
    int fragment_in_out_location = CURRENT_PROGRAM.active_uniforms + active_textures*2; // sample_t + image_t 
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
    
    // Depth Kernel Set Up
    cl_kernel depth_kernel = NULL;
    if (_enableds.depth) {
        depth_kernel = getDepthKernel();
        setKernelArg(depth_kernel, 0, sizeof(DEPTH_ATTACHMENT.mem), &DEPTH_ATTACHMENT.mem);
        setKernelArg(depth_kernel, 1, sizeof(gl_Discard),           &gl_Discard);
        setKernelArg(depth_kernel, 2, sizeof(gl_FragCoord),         &gl_FragCoord);
    }

    // Color Kernel Set Up
    cl_kernel dither_kenel = getDitherKernel();
    setKernelArg(dither_kenel, 0, sizeof(COLOR_ATTACHMENT0.mem),    &COLOR_ATTACHMENT0.mem);
    setKernelArg(dither_kenel, 1, sizeof(gl_FragColor),             &gl_FragColor);
    setKernelArg(dither_kenel, 2, sizeof(gl_Discard),               &gl_Discard);
    if (_enableds.dithering)
        setKernelArg(dither_kenel, 3, sizeof(gl_FragCoord),         &gl_FragCoord);

    /* ---- Enqueue Kernels ---- */
    cl_command_queue command_queue = getCommandQueue();

    enqueueNDRangeKernel(command_queue, vertex_kernel,                  num_vertices);
    enqueueNDRangeKernel(command_queue, _perspective_division_kernel,   num_vertices);
    enqueueNDRangeKernel(command_queue, _viewport_division_kernel,      num_vertices);

    for(GLsizei primitive=0; primitive < num_primitives; ++primitive) {
        setKernelArg(_rasterization_kernels.triangles, 0, sizeof(primitive), &primitive);

        enqueueNDRangeKernel(command_queue, _rasterization_kernels.triangles,   num_fragments);   
        enqueueNDRangeKernel(command_queue, fragment_kernel,                    num_fragments);

        if (depth_kernel) enqueueNDRangeKernel(command_queue, depth_kernel, num_fragments);

        enqueueNDRangeKernel(command_queue, dither_kenel, num_fragments);
    }
}

GL_APICALL void GL_APIENTRY glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices) NOT_IMPLEMENTED;

GL_APICALL void GL_APIENTRY glEnable (GLenum cap) {
    if (cap == GL_SCISSOR_TEST)
        _enableds.scissor = GL_TRUE;
    else if (cap == GL_DEPTH_TEST)
        _enableds.depth = GL_TRUE;
    else if (cap == GL_STENCIL_TEST)
        _enableds.stencil = GL_TRUE;
    else if (cap == GL_CULL_FACE)
        _enableds.culling = GL_TRUE;
    else if (cap == GL_DITHER)
        _enableds.dithering = GL_TRUE;
    else NOT_IMPLEMENTED; // RETURN_ERROR(GL_INVALID_ENUM);
}

GL_APICALL void GL_APIENTRY glEnableVertexAttribArray (GLuint index) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) RETURN_ERROR(GL_INVALID_VALUE);

    _vertex_attrib_enable[index] = 1;
}

GL_APICALL void GL_APIENTRY glFinish (void) {
    finish(getCommandQueue());
    // TODO: Checkout multiqueuing
}

GL_APICALL void GL_APIENTRY glFlush (void) {
    clFlush(getCommandQueue());
    // TODO: Checkout multiqueuing
}

GL_APICALL void GL_APIENTRY glFramebufferRenderbuffer (GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) {
    if (attachment == GL_COLOR_ATTACHMENT0)
        _framebuffers[_framebuffer_binding].color_attachment0=renderbuffer;
    else if (attachment == GL_DEPTH_ATTACHMENT)
        _framebuffers[_framebuffer_binding].depth_attachment=renderbuffer;
    else if (attachment == GL_STENCIL_ATTACHMENT)
        _framebuffers[_framebuffer_binding].stencil_attachment=renderbuffer;
    else NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glFramebufferTexture2D (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
    NOT_IMPLEMENTED;
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

GL_APICALL void GL_APIENTRY glGenerateMipmap (GLenum target) {
    NOT_IMPLEMENTED;
}

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
            _textures[id].used = GL_TRUE;
            *textures = id;

            textures += 1; 
            n -= 1;
        }
        id += 1;
    }
}

GL_APICALL GLint GL_APIENTRY glGetAttribLocation (GLuint program, const GLchar *name) {
    if (!_current_program) RETURN_ERROR(GL_INVALID_OPERATION);
    
    for(size_t attrib=0; attrib<CURRENT_PROGRAM.active_vertex_attribs; ++attrib) {
        if (strcmp(name, CURRENT_PROGRAM.vertex_attribs_data[attrib].name) == 0) return attrib;
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

GL_APICALL void GL_APIENTRY glGetProgramiv (GLuint program, GLenum pname, GLint *params) {
    NOT_IMPLEMENTED;
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

GL_APICALL GLboolean GL_APIENTRY glIsEnabled (GLenum cap) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glLineWidth (GLfloat width) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glPixelStorei (GLenum pname, GLint param) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glPolygonOffset (GLfloat factor, GLfloat units) {
    NOT_IMPLEMENTED;
}

#ifdef HOSTGPU
#include "common.h"

#define PATH_OF(_FILE) "/home/xavier/repositories/vortex/tests/opengl/lib/GLSC2/" _FILE
#endif

GL_APICALL void GL_APIENTRY glProgramBinary (GLuint program, GLenum binaryFormat, const void *binary, GLsizei length){
    printf("glProgramBinary() program=%d, binaryFormat=%d\n",program,binaryFormat);
    if(!_kernel_load_status) { // TODO: maybe on a __constructor__ function??
        cl_program gl_program;

        #ifdef HOSTGPU
        file_t file;
        cl_int error;
        #endif

        #ifdef HOSTGPU
        if (0 != read_file(PATH_OF("kernel.depth.cl"), &file))
            return -1;
        gl_program = clCreateProgramWithSource(_getContext(), 1, (const char**)&file.data, &file.size, &error);
        free(file.data);
        #else
        gl_program = createProgramWithBinary(GLSC2_kernel_depth_pocl, sizeof(GLSC2_kernel_depth_pocl));
        #endif
        buildProgram(gl_program);
        _kernels.depth.always   = createKernel(gl_program, "gl_always");
        _kernels.depth.never    = createKernel(gl_program, "gl_never");
        _kernels.depth.less     = createKernel(gl_program, "gl_less");
        _kernels.depth.lequal   = createKernel(gl_program, "gl_lequal");
        _kernels.depth.equal    = createKernel(gl_program, "gl_equal");
        _kernels.depth.greater  = createKernel(gl_program, "gl_greater");
        _kernels.depth.gequal   = createKernel(gl_program, "gl_gequal");
        _kernels.depth.notequal = createKernel(gl_program, "gl_notequal");

        #ifdef HOSTGPU
        if (0 != read_file(PATH_OF("kernel.color.cl"), &file))
            return -1;
        gl_program = clCreateProgramWithSource(_getContext(), 1, (const char**)&file.data, &file.size, &error);
        free(file.data);
        #else
        gl_program = createProgramWithBinary(GLSC2_kernel_dithering_pocl, sizeof(GLSC2_kernel_dithering_pocl));
        #endif
        buildProgram(gl_program);
        _kernels.dithering.enabled_rgba4  = createKernel(gl_program, "enabled_rgba4");
        _kernels.dithering.disabled_rgba4 = createKernel(gl_program, "disabled_rgba4");

        #ifdef HOSTGPU
        if (0 != read_file(PATH_OF("kernel.rasterization.triangle.cl"), &file))
            return -1;
        gl_program = clCreateProgramWithSource(_getContext(), 1, (const char**)&file.data, &file.size, &error);
        free(file.data);
        #else
        gl_program = createProgramWithBinary(GLSC2_kernel_rasterization_triangle_pocl, sizeof(GLSC2_kernel_rasterization_triangle_pocl));
        #endif
        buildProgram(gl_program);
        _rasterization_kernels.triangles = createKernel(gl_program, "gl_rasterization_triangle");

        #ifdef HOSTGPU
        if (0 != read_file(PATH_OF("kernel.viewport_division.cl"), &file))
            return -1;
        gl_program = clCreateProgramWithSource(_getContext(), 1, (const char**)&file.data, &file.size, &error);
        free(file.data);
        #else
        gl_program = createProgramWithBinary(GLSC2_kernel_viewport_division_pocl, sizeof(GLSC2_kernel_viewport_division_pocl));
        #endif
        buildProgram(gl_program);
        _viewport_division_kernel = createKernel(gl_program, "gl_viewport_division");

        #ifdef HOSTGPU
        if (0 != read_file(PATH_OF("kernel.perspective_division.cl"), &file))
            return -1;
        gl_program = clCreateProgramWithSource(_getContext(), 1, (const char**)&file.data, &file.size, &error);
        free(file.data);
        #else
        gl_program = createProgramWithBinary(GLSC2_kernel_perspective_division_pocl, sizeof(GLSC2_kernel_perspective_division_pocl));
        #endif
        buildProgram(gl_program);
        _perspective_division_kernel = createKernel(gl_program, "gl_perspective_division");

        #ifdef HOSTGPU
        if (0 != read_file(PATH_OF("kernel.readnpixels.cl"), &file))
            return -1;
        gl_program = clCreateProgramWithSource(_getContext(), 1, (const char**)&file.data, &file.size, &error);
        free(file.data);
        #else
        gl_program = createProgramWithBinary(GLSC2_kernel_readnpixels_pocl, sizeof(GLSC2_kernel_readnpixels_pocl));
        #endif
        buildProgram(gl_program);
        _readnpixels_kernel = createKernel(gl_program, "gl_rgba4_rgba8");

        #ifdef HOSTGPU
        if (0 != read_file(PATH_OF("kernel.strided_write.cl"), &file))
            return -1;
        gl_program = clCreateProgramWithSource(_getContext(), 1, (const char**)&file.data, &file.size, &error);
        free(file.data);
        #else
        gl_program = createProgramWithBinary(GLSC2_kernel_strided_write_pocl, sizeof(GLSC2_kernel_strided_write_pocl));
        #endif
        buildProgram(gl_program);
        _strided_write_kernel = createKernel(gl_program, "gl_strided_write");

        gl_program = createProgramWithBinary(GLSC2_kernel_clear_pocl, sizeof(GLSC2_kernel_clear_pocl));
        buildProgram(gl_program);
        _kernels.clear.bit16 = createKernel(gl_program, "bit16");
        _kernels.clear.bit8  = createKernel(gl_program, "bit8");

        _kernel_load_status = 1;
    }

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
            
            if (addr_qualifier == CL_KERNEL_ARG_ADDRESS_CONSTANT) {
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
    } else NOT_IMPLEMENTED;

}

GL_APICALL void GL_APIENTRY glReadnPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data) {
    if (format == GL_RGBA && type == GL_UNSIGNED_BYTE) {
        if (_framebuffer_binding) {

            if (COLOR_ATTACHMENT0.internalformat == GL_RGBA8) {
                enqueueReadBuffer(getCommandQueue(), COLOR_ATTACHMENT0.mem, bufSize, data);
                return;
            }

            if (COLOR_ATTACHMENT0.internalformat != GL_RGBA4) NOT_IMPLEMENTED;
            void *dst_buff = createBuffer(MEM_WRITE_ONLY, bufSize, NULL);

            setKernelArg(_readnpixels_kernel, 0, sizeof(COLOR_ATTACHMENT0.mem), &COLOR_ATTACHMENT0.mem);
            setKernelArg(_readnpixels_kernel, 1, sizeof(void*), &dst_buff);
            setKernelArg(_readnpixels_kernel, 2, sizeof(int), &x);
            setKernelArg(_readnpixels_kernel, 3, sizeof(int), &y);
            setKernelArg(_readnpixels_kernel, 4, sizeof(int), &width);
            setKernelArg(_readnpixels_kernel, 5, sizeof(int), &height);

            void *command_queue = getCommandQueue();
            size_t global_work_size = bufSize/4; // 4 bytes x color
            enqueueNDRangeKernel(command_queue, _readnpixels_kernel, global_work_size);
            enqueueReadBuffer(command_queue, dst_buff, bufSize, data);
        }
    }
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
    if (width < 0 || height < 0) {
        _err = GL_INVALID_VALUE;
        return;
    }
    _scissor_box.x = x;
    _scissor_box.y = y;
    _scissor_box.width = width;
    _scissor_box.height = height;
}

GL_APICALL void GL_APIENTRY glStencilFunc (GLenum func, GLint ref, GLuint mask) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glStencilFuncSeparate (GLenum face, GLenum func, GLint ref, GLuint mask) {
    NOT_IMPLEMENTED;
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
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glStencilOpSeparate (GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass) {
    NOT_IMPLEMENTED;
}

#define max(a, b) (a >= b ? a : b)
#define IS_POWER_OF_2(a) !(a & 0x1u) 

GL_APICALL void GL_APIENTRY glTexStorage2D (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) {
	if (target != GL_TEXTURE_2D) return; // Unknown behaviour

    if (!_texture_binding) 
        RETURN_ERROR(GL_INVALID_OPERATION);
    if (width < 1 || height < 1 || levels < 1) // not sure of this
        RETURN_ERROR(GL_INVALID_VALUE);
    if (levels > (int) log2f(max(width,height)) + 1)
        RETURN_ERROR(GL_INVALID_OPERATION);
    if (levels != 1 && (IS_POWER_OF_2(width) || IS_POWER_OF_2(height)))
        RETURN_ERROR(GL_INVALID_OPERATION);
    if (_textures[_texture_binding].width || _textures[_texture_binding].height)
        RETURN_ERROR(GL_INVALID_OPERATION);
    
    #ifndef IMAGE_SUPPORT
    uint32_t pixel_size;

    switch (internalformat) {
    case GL_RGBA8:
        pixel_size = sizeof(uint8_t[4]);
        break;
    case GL_RGB8:
        pixel_size = sizeof(uint8_t[3]);
        break;
    case GL_RG8:
        pixel_size = sizeof(uint8_t[2]);
        break;
    case GL_R8:
        pixel_size = sizeof(uint8_t[1]);
        break;
    case GL_RGBA4:
    case GL_RGB5_A1:
    case GL_RGB565:
        pixel_size = sizeof(uint8_t[2]);
        break;
    default:
        RETURN_ERROR(GL_INVALID_ENUM);
    }

    _textures[_texture_binding].width = width;
    _textures[_texture_binding].height = height;
    _textures[_texture_binding].internalformat = internalformat;

    GLsizei level = 0;
    size_t total_pixels = 0;
    while (level++<levels) { 
        total_pixels += width * height;
        width /= 2;
        height /= 2;
    }
    _textures[_texture_binding].mem = createBuffer(MEM_READ_ONLY, total_pixels*pixel_size, NULL);
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

    _textures[_texture_binding].mem = createImage(MEM_READ_ONLY, &format, &desc, NULL);
    #endif
}

GL_APICALL void GL_APIENTRY glTexParameterf (GLenum target, GLenum pname, GLfloat param) NOT_IMPLEMENTED;
GL_APICALL void GL_APIENTRY glTexParameterfv (GLenum target, GLenum pname, const GLfloat *params) NOT_IMPLEMENTED;
GL_APICALL void GL_APIENTRY glTexParameteri (GLenum target, GLenum pname, GLint param) NOT_IMPLEMENTED;
GL_APICALL void GL_APIENTRY glTexParameteriv (GLenum target, GLenum pname, const GLint *params) NOT_IMPLEMENTED;

GL_APICALL void GL_APIENTRY glTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) {
    
    if (level < 0 || level > (int) log2f(max(_textures[_texture_binding].width,_textures[_texture_binding].height)))
        RETURN_ERROR(GL_INVALID_VALUE);

    if (xoffset < 0 || yoffset < 0 || xoffset + width > _textures[_texture_binding].width || yoffset + height > _textures[_texture_binding].height)
        RETURN_ERROR(GL_INVALID_VALUE);

    // TODO subImage2d kernel
    #ifndef IMAGE_SUPPORT
    if (_textures[_texture_binding].internalformat == GL_RGBA8 && format == GL_RGBA && type == GL_UNSIGNED_BYTE && xoffset == 0 && yoffset == 0) {
        enqueueWriteBuffer(getCommandQueue(),_textures[_texture_binding].mem, CL_TRUE, 0, width*height*sizeof(uint8_t[4]), pixels);
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

    enqueueWriteImage(getCommandQueue(), _textures[_texture_binding].mem, &origin, &region, pixel_size*width, 0, pixels);
    #endif
}

#define ERROR_CHECKER(_COUNT, _SIZE, _TYPE) ({                                                                          \
    if (!_current_program) RETURN_ERROR(GL_INVALID_OPERATION);                                                          \
    if (CURRENT_PROGRAM.uniforms_data[location].size != _SIZE) RETURN_ERROR(GL_INVALID_OPERATION);                      \
    if (   CURRENT_PROGRAM.uniforms_data[location].type != GL_BYTE                                                      \
        && CURRENT_PROGRAM.uniforms_data[location].type != _TYPE)                                                       \
        RETURN_ERROR(GL_INVALID_OPERATION);                                                                             \
    if (location >= CURRENT_PROGRAM.active_uniforms) RETURN_ERROR(GL_INVALID_OPERATION);                                \
    if (_COUNT < -1 || _COUNT > CURRENT_PROGRAM.uniforms_array_size[location]) RETURN_ERROR(GL_INVALID_OPERATION);      \
    if (location == -1) return;                                                                                         \
    })

#define GENERIC_UNIFORM(_SIZE, _TYPE, _ARRAY_TYPE, _ARRAY) ({                                                   \
    ERROR_CHECKER(1, _SIZE, _TYPE);                                                                             \
    if (CURRENT_PROGRAM.uniforms_data[location].type == GL_BYTE) NOT_IMPLEMENTED;                               \
    _ARRAY_TYPE value[] = _ARRAY;                                                                               \
    enqueueWriteBuffer(getCommandQueue(), CURRENT_PROGRAM.uniforms_mem, GL_TRUE, 0, sizeof(value), value);      \
    })

#define GENERIC_UNIFORM_V(_SIZE, _TYPE, _ARRAY_TYPE) ({                                                                         \
    ERROR_CHECKER(count, _SIZE, _TYPE);                                                                                         \
    if (CURRENT_PROGRAM.uniforms_data[location].type == GL_BYTE) NOT_IMPLEMENTED;                                               \
    enqueueWriteBuffer(getCommandQueue(), CURRENT_PROGRAM.uniforms_mem, GL_TRUE, 0, sizeof(_ARRAY_TYPE[count][_SIZE]), value);  \
    })


GL_APICALL void GL_APIENTRY glUniform1f (GLint location, GLfloat v0) {
    GENERIC_UNIFORM(1, GL_FLOAT, GLfloat,P99_PROTECT({v0}));
}
GL_APICALL void GL_APIENTRY glUniform1fv (GLint location, GLsizei count, const GLfloat *value) {
    GENERIC_UNIFORM_V(1, GL_FLOAT, GLfloat);
}
GL_APICALL void GL_APIENTRY glUniform1i (GLint location, GLint v0) {
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

    GENERIC_UNIFORM_V(2,GL_FLOAT,GLfloat[2]);
}
GL_APICALL void GL_APIENTRY glUniformMatrix3fv (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    if (transpose != GL_FALSE) RETURN_ERROR(GL_INVALID_VALUE);

    GENERIC_UNIFORM_V(3,GL_FLOAT,GLfloat[3]);
}
GL_APICALL void GL_APIENTRY glUniformMatrix4fv (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    if (transpose != GL_FALSE) RETURN_ERROR(GL_INVALID_VALUE);

    GENERIC_UNIFORM_V(4,GL_FLOAT,GLfloat[4]);
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

cl_kernel getDepthKernel() {
    switch (_depth_func) {
        case GL_ALWAYS:
            return _kernels.depth.always;
        case GL_NEVER:
            return _kernels.depth.never;
        case GL_LESS:
            return _kernels.depth.less;
        case GL_LEQUAL:
            return _kernels.depth.lequal;
        case GL_EQUAL:
            return _kernels.depth.equal;
        case GL_GREATER:
            return _kernels.depth.greater;
        case GL_GEQUAL:
            return _kernels.depth.gequal;
        case GL_NOTEQUAL:
            return _kernels.depth.notequal;
        
        // TODO add all cases
    }
    NOT_IMPLEMENTED;
}

cl_kernel getDitherKernel() {
    switch (COLOR_ATTACHMENT0.internalformat) {
        case GL_RGBA4:
            return _enableds.dithering ? _kernels.dithering.enabled_rgba4 : _kernels.dithering.disabled_rgba4;
        default:
            NOT_IMPLEMENTED;
    }
}

unsigned int size_from_name_type(const char* name_type) {
    #define RETURN_IF_SIZE_FROM(_TYPE)                              \
        if (strncmp(name_type, _TYPE, sizeof(_TYPE) - 1) == 0) {    \
            substr_size = name_type + sizeof(_TYPE) - 1;            \
            if (*substr_size == '\0') return 1;                     \
            return atoi(substr_size);                               \
        }

    const char* substr_size;
    RETURN_IF_SIZE_FROM("float");
    RETURN_IF_SIZE_FROM("int");
    RETURN_IF_SIZE_FROM("short");
    RETURN_IF_SIZE_FROM("char");
    RETURN_IF_SIZE_FROM("bool");
    printf("%s\n", name_type);
    NOT_IMPLEMENTED;

    #undef RETURN_IF_SIZE_FROM
}
unsigned int type_from_name_type(const char* name_type) {
    if (strncmp(name_type, "float",  sizeof("float") -1)  == 0) return GL_FLOAT;
    if (strncmp(name_type, "int",    sizeof("int")   -1)  == 0) return GL_INT;
    if (strncmp(name_type, "short",  sizeof("short") -1)  == 0) return GL_SHORT;
    if (strncmp(name_type, "char",   sizeof("char")  -1)  == 0) return GL_BYTE;
    if (strncmp(name_type, "bool",   sizeof("bool")  -1)  == 0) return GL_BYTE;
    NOT_IMPLEMENTED;
}
