#include <math.h>
#include <GLSC2/glsc2.h>
#include "kernel.c" // TODO: may be interesting to extract it to an interface so could be re implementated with CUDA
#include "binary.c" // TODO: maybe with extern

#define NOT_IMPLEMENTED              \
    ({                               \
        printf("NOT_IMPLEMENTED\n"); \
        exit(0);                     \
    })

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

// OpenGL required definitions
#define MAX_VERTEX_ATTRIBS 16
#define MAX_VERTEX_UNIFORM_VECTORS 16 // atMost MAX_UNIFORM_VECTORS
#define MAX_FRAGMENT_UNIFORM_VECTORS 16 // atMost MAX_UNIFORM_VECTORS


/****** DTO objects ******\
 * TODO: externalize to could be imported from kernel cl programs
*/
typedef struct { 
    int type; // byte, ubyte, short, ushort, float 
    int size; 
    void* mem; 
} _attribute_pointer;

typedef struct { int values[4]; } _attribute_int;

typedef struct { float values[4]; } _attribute_float;

typedef union {
    _attribute_int  int4; // 0
    _attribute_float  float4; // 1
    _attribute_pointer pointer; // 2
} _attribute;

typedef struct __attribute__ ((packed)) {
    int type; // 0, 1, 2
    _attribute attribute;
} attribute;

typedef struct {

} FRAGMENT_DATA;

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
    GLint location, size, type;
    unsigned char name[MAX_NAME_SIZE]; 
    uint8_t data[MAX_UNIFORM_SIZE]; // TODO: Use union types
} UNIFORM;

typedef struct {
    GLint location, size, type;
    unsigned char name[MAX_NAME_SIZE]; 
    attribute data;
} ATTRIBUTE;

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
#define ARG_NAME_SIZE 256
#define MAX_VARYING 16

typedef struct {
    unsigned int location, size, type;
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
    unsigned int            varying_size; // out - in connections TODO: now is expected to be provided in the same order
    // Uniform data
    unsigned int            active_uniforms;
    arg_data_t              uniforms_data[MAX_UNIFORM_SIZE];
    cl_mem                  uniforms_mem[MAX_UNIFORM_SIZE];
    // Vertex data
    unsigned int            active_vertex_attribs;
    arg_data_t              vertex_attribs_data[MAX_VERTEX_ATTRIBS];
} program_t;

unsigned int            _current_program;
program_t               _programs[MAX_PROGRAMS];

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
} BUFFER;

typedef struct {
    GLboolean enable, normalized;
    GLuint index;
    GLint size;
    GLenum type;
    GLsizei stride;
    const void *pointer;
} VERTEX_ATTRIB;

BUFFER _buffers[MAX_BUFFER];
GLuint _buffer_binding;
VERTEX_ATTRIB _vertex_attrib[MAX_VERTEX_ATTRIBS];

/****** FRAMEBUFFER objects ******\
 * 
 * 
*/

typedef struct {
    GLuint color_attachment0, depth_attachment, stencil_attachment;
    GLboolean used;
} FRAMEBUFFER;

FRAMEBUFFER _framebuffers[MAX_FRAMEBUFFER];
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

RENDERBUFFER _renderbuffers[MAX_RENDERBUFFER];
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

/****** Interface for utils & inline functions ******\
 * Utility or inline function are implemented at the end of the file. 
*/
#define COLOR_ATTACHMENT0 _renderbuffers[_framebuffers[_framebuffer_binding].color_attachment0]
#define DEPTH_ATTACHMENT _renderbuffers[_framebuffers[_framebuffer_binding].depth_attachment]
#define STENCIL_ATTACHMENT _renderbuffers[_framebuffers[_framebuffer_binding].stencil_attachment]
#define CURRENT_PROGRAM _programs[_current_program]

void* getCommandQueue();

void* getPerspectiveDivisionKernel(GLenum mode, GLint first, GLsizei count);
void* getViewportDivisionKernel(GLenum mode, GLint first, GLsizei count);
void* getRasterizationTriangleKernel(GLenum mode, GLint first, GLsizei count);
cl_kernel getDepthKernel();
cl_kernel getColorKernel();

void* createVertexKernel(GLenum mode, GLint first, GLsizei count);
void* createFragmentKernel(GLenum mode, GLint first, GLsizei count);

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

GL_APICALL void GL_APIENTRY glBindBuffer (GLenum target, GLuint buffer) {
    if (!_buffers[buffer].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    
    _buffer_binding = buffer;
    _buffers[buffer].target = target;
}
GL_APICALL void GL_APIENTRY glBindFramebuffer (GLenum target, GLuint framebuffer) {
    printf("glBindFramebuffer(framebuffer: %d)\n", framebuffer);

    if (!_framebuffers[framebuffer].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    if (target == GL_FRAMEBUFFER) {
        _framebuffer_binding = framebuffer;
    }
}
GL_APICALL void GL_APIENTRY glBindRenderbuffer (GLenum target, GLuint renderbuffer) {
    if (!_renderbuffers[renderbuffer].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    if (target == GL_RENDERBUFFER) {
        _renderbuffer_binding = renderbuffer;
    }
}
GL_APICALL void GL_APIENTRY glBindTexture (GLenum target, GLuint texture) {
    if (!_textures[texture].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    if (target == GL_TEXTURE_2D) {
        _texture_binding = texture;
    }
}

GL_APICALL void GL_APIENTRY glRenderbufferStorage (GLenum target, GLenum internalformat, GLsizei width, GLsizei height) {

    if (internalformat == GL_RGBA4) {
        _renderbuffers[_renderbuffer_binding].mem = createBuffer(MEM_READ_WRITE, width*height*2, NULL);
    } else if (internalformat == GL_DEPTH_COMPONENT16) {
        _renderbuffers[_renderbuffer_binding].mem = createBuffer(MEM_READ_WRITE, width*height*2, NULL);
    } else if (internalformat == GL_STENCIL_INDEX8) {
        _renderbuffers[_renderbuffer_binding].mem = createBuffer(MEM_READ_WRITE, width*height*1, NULL);
    } else {
        printf("NOT IMPLEMENTED\n");
        exit(0);
    }
    _renderbuffers[_renderbuffer_binding].internalformat = internalformat;
    _renderbuffers[_renderbuffer_binding].width = width;
    _renderbuffers[_renderbuffer_binding].height = height;
    
}


GL_APICALL void GL_APIENTRY glBufferData (GLenum target, GLsizeiptr size, const void *data, GLenum usage) {

    if (target == GL_ARRAY_BUFFER) {
        if (usage == GL_STATIC_DRAW) {
            _buffers[_buffer_binding].mem = createBuffer(MEM_READ_ONLY | MEM_COPY_HOST_PTR, size, data);
        }
        else if (usage == GL_DYNAMIC_DRAW || usage == GL_STREAM_DRAW) {
            _buffers[_buffer_binding].mem = createBuffer(MEM_READ_WRITE | MEM_COPY_HOST_PTR, size, data);
        }
    }
}

GL_APICALL void GL_APIENTRY glClear (GLbitfield mask) {
    // TODO: Check behaviour on the specs
    if(mask & GL_COLOR_BUFFER_BIT) glClearColor(0.0,0.0,0.0,1.0);
    if(mask & GL_DEPTH_BUFFER_BIT) glClearDepthf(1.0);
    if(mask & GL_STENCIL_BUFFER_BIT) glClearStencil(0);
}

GL_APICALL void GL_APIENTRY glClearColor (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
    uint32_t pattern = 0;
    uint32_t pixel_size = 0;

    if (COLOR_ATTACHMENT0.internalformat == GL_RGBA4) {
        pattern |= (unsigned int) (red   * 0xFu) << 0;
        pattern |= (unsigned int) (green * 0xFu) << 4;
        pattern |= (unsigned int) (blue  * 0xFu) << 8;
        pattern |= (unsigned int) (alpha * 0xFu) << 12;
        pattern |= pattern << 16;
        pixel_size = sizeof(uint8_t[2]);
    } else if(COLOR_ATTACHMENT0.internalformat == GL_RGBA8) {
        pattern |= (unsigned int) (red   * 0xFFu) << 0;
        pattern |= (unsigned int) (green * 0xFFu) << 8;
        pattern |= (unsigned int) (blue  * 0xFFu) << 16;
        pattern |= (unsigned int) (alpha * 0xFFu) << 24;
        pixel_size = sizeof(uint8_t[4]);
    } else if(COLOR_ATTACHMENT0.internalformat == GL_RGB8) {
        NOT_IMPLEMENTED; // TODO: Update enqueueFillBuffer
    } else if(COLOR_ATTACHMENT0.internalformat == GL_RG8) {
        pattern |= (unsigned int) (red   * 0xFFu) << 0;
        pattern |= (unsigned int) (green * 0xFFu) << 8;
        pattern |= pattern << 16;
        pixel_size = sizeof(uint8_t[2]);
    } else if(COLOR_ATTACHMENT0.internalformat == GL_R8) {
        pattern |= (unsigned int) (red   * 0xFFu) << 0;
        pattern |= pattern << 8;
        pattern |= pattern << 16;
        pattern |= pattern << 24;
        pixel_size = sizeof(uint8_t[1]);
    } else if(COLOR_ATTACHMENT0.internalformat == GL_RGB5_A1) {
        pattern |= (unsigned int) (red   * 0x1Fu) << 0;
        pattern |= (unsigned int) (green * 0x1Fu) << 5;
        pattern |= (unsigned int) (blue  * 0x1Fu) << 10;
        pattern |= (unsigned int) (alpha * 0x1u)   << 15;
        pattern |= pattern << 16;
        pixel_size = sizeof(uint8_t[2]);
    } else if(COLOR_ATTACHMENT0.internalformat == GL_RGB565) {
        pattern |= (unsigned int) (red   * 0x1Fu) << 0;
        pattern |= (unsigned int) (green * 0x3Fu) << 5;
        pattern |= (unsigned int) (blue  * 0x1Fu) << 11;
        pattern |= pattern << 16;
        pixel_size = sizeof(uint8_t[2]);
    } else RETURN_ERROR(GL_INVALID_OPERATION);
    
    enqueueFillBuffer(getCommandQueue(), COLOR_ATTACHMENT0.mem, &pattern, 4, 0, COLOR_ATTACHMENT0.width*COLOR_ATTACHMENT0.height*pixel_size);
}

GL_APICALL void GL_APIENTRY glClearDepthf (GLfloat d) {
    uint32_t pattern = 0;

    if (DEPTH_ATTACHMENT.internalformat == GL_DEPTH_COMPONENT16) {
        pattern = d*0xFFFFu;
        pattern |= pattern << 16;
    } else NOT_IMPLEMENTED;

    enqueueFillBuffer(getCommandQueue(), DEPTH_ATTACHMENT.mem, &pattern, 4, 0, DEPTH_ATTACHMENT.width*DEPTH_ATTACHMENT.height*2);
}

GL_APICALL void GL_APIENTRY glClearStencil (GLint s) {
    uint32_t pattern = 0;

    if (STENCIL_ATTACHMENT.internalformat == GL_STENCIL_INDEX8) {
        pattern = s*0xFFu;
        pattern |= pattern << 8;
        pattern |= pattern << 16;
        pattern |= pattern << 24;
    } else NOT_IMPLEMENTED;

    enqueueFillBuffer(getCommandQueue(), STENCIL_ATTACHMENT.mem, &pattern, 4, 0, STENCIL_ATTACHMENT.width*STENCIL_ATTACHMENT.height*1);
}

GL_APICALL void GL_APIENTRY glColorMask (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
    _color_mask.red = red;
    _color_mask.green = green;
    _color_mask.blue = blue;
    _color_mask.alpha = alpha;
}

GL_APICALL GLuint GL_APIENTRY glCreateProgram (void){
    static GLuint program = 1; // Pointer to the next available program.

    if (program < MAX_PROGRAMS) return program++;

    // TODO: Checkout documentation about this behaviour
    printf("No more programs available.\n");
    exit(program);
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

/**
 * TODO: first is expected to be 0
*/
GL_APICALL void GL_APIENTRY glDrawArrays (GLenum mode, GLint first, GLsizei count) {
    if (!_current_program) RETURN_ERROR(GL_INVALID_OPERATION);
    if (first <0) RETURN_ERROR(GL_INVALID_VALUE);
    
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

            cl_command_queue command_queue = createCommandQueue(0);
            
            setKernelArg(_strided_write_kernel, 0, sizeof(pointer->size),       &pointer->size);
            setKernelArg(_strided_write_kernel, 1, sizeof(pointer->type),       &pointer->type);
            setKernelArg(_strided_write_kernel, 2, sizeof(pointer->normalized), &pointer->normalized);
            setKernelArg(_strided_write_kernel, 3, sizeof(pointer->stride),     &pointer->stride);
            if (pointer->binding) {
                cl_int tmp_err;
                cl_buffer_region region = {
                    .origin = (uint64_t) pointer->pointer,
                    .size = num_vertices*sizeof_type(pointer->type)*pointer->size+pointer->stride
                };
                temp_mem[attrib] = clCreateSubBuffer(_buffers[pointer->binding].mem, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &tmp_err);
                printf("pointer value: %i, error=%i\n", (uint64_t) pointer->pointer, tmp_err);
                setKernelArg(_strided_write_kernel, 4, sizeof(cl_mem), &temp_mem[attrib]);
            } else {
                size_t slice_size = sizeof_type(pointer->type)*pointer->size+pointer->stride;
                temp_mem[attrib] = createBuffer(MEM_READ_WRITE | MEM_COPY_HOST_PTR, slice_size*num_vertices, pointer->pointer);
                setKernelArg(_strided_write_kernel, 4, sizeof(cl_mem), &temp_mem[attrib]);
            }
            vertex_array_mem[attrib] = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_vertices, NULL);
            setKernelArg(_strided_write_kernel, 5, sizeof(cl_mem), &vertex_array_mem[attrib]);
            enqueueNDRangeKernel(command_queue, _strided_write_kernel, num_vertices);
            
            // printf("attrib %i.\n", attrib);
            // float input_subbuffer[num_vertices][3], data_in[num_vertices][3], data_out[num_vertices][4];
            // enqueueReadBuffer(command_queue, _buffers[pointer->binding].mem,sizeof(float[3])*num_vertices,input_subbuffer);
            // enqueueReadBuffer(command_queue, temp_mem[attrib],sizeof(float[3])*num_vertices,data_in);
            // enqueueReadBuffer(command_queue, vertex_array_mem[attrib],sizeof(float[4])*num_vertices,data_out);
            // for (int i = 0; i < num_vertices; i+=1) {
            //     printf("input_subbuffer %d, x=%f, y=%f, z=%f\n", i, input_subbuffer[i][0],input_subbuffer[i][1],input_subbuffer[i][2]);
            //     printf("data_in %d, x=%f, y=%f, z=%f\n", i, data_in[i][0],data_in[i][1],data_in[i][2]);
            //     printf("data_out %d, x=%f, y=%f, z=%f, w=%f\n", i, data_out[i][0],data_out[i][1],data_out[i][2], data_out[i][3]);
            // }
        }
    }
    /*
    for (int attrib=0; attrib < CURRENT_PROGRAM.active_vertex_attribs; ++attrib) {
        if (temp_mem[attrib] && !_vertex_attribs[attrib].pointer.pointer) clReleaseMemObject(temp_mem[attrib]);
    }*/
    
    
    // OpenGL output required vertex array
    void *gl_Positions = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_vertices, NULL);
    // TODO: size of gl_Primitives is compiler dependence, for now we use active_vertex_attribs as a reference
    cl_mem vertex_out_buffer = createBuffer(CL_MEM_READ_WRITE, sizeof(float[4])*num_vertices*CURRENT_PROGRAM.varying_size-1, NULL); // TODO varying_size

    /* ---- Set Up Per-Vertex Kernels ---- */
    cl_kernel vertex_kernel;

    // Vertex Kernel Set Up
    vertex_kernel = createKernel(CURRENT_PROGRAM.program, VERTEX_SHADER_FNAME);

    for(int uniform = 0; uniform < CURRENT_PROGRAM.active_uniforms; ++uniform) {
        setKernelArg(vertex_kernel, 
            CURRENT_PROGRAM.uniforms_data[uniform].location, 
            sizeof(CURRENT_PROGRAM.uniforms_mem[uniform]),
            &CURRENT_PROGRAM.uniforms_mem[uniform]
        );
    }
    for(int attrib = 0; attrib < CURRENT_PROGRAM.active_vertex_attribs; ++attrib) {
        if (_vertex_attrib_state[attrib] == VEC4) {
            setKernelArg(vertex_kernel,
                CURRENT_PROGRAM.active_uniforms + attrib, 
                sizeof(vertex_attrib_t), &_vertex_attrib[attrib]
            );
        } else {
            setKernelArg(vertex_kernel,
                CURRENT_PROGRAM.active_uniforms + attrib, 
                sizeof(cl_mem), &vertex_array_mem[attrib]
            );
        }
    }

    // TODO: Locations of out are consecutive and are the last ones, change it to accept more types 
    unsigned int vertex_out_location = CURRENT_PROGRAM.active_uniforms + CURRENT_PROGRAM.active_vertex_attribs;
    printf("in: %i, out: %i, uni: %i\n", CURRENT_PROGRAM.active_vertex_attribs, CURRENT_PROGRAM.varying_size, CURRENT_PROGRAM.active_uniforms);
    cl_int tmp_err; cl_mem out_subbuffer;
    for(int out=0; out < CURRENT_PROGRAM.varying_size-1; ++out) {
        cl_buffer_region region = {
            .origin = sizeof(float[4])*num_vertices*out,
            .size = num_vertices*sizeof(float[4])
        };
        out_subbuffer = clCreateSubBuffer(vertex_out_buffer,CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &tmp_err);
        setKernelArg(vertex_kernel, vertex_out_location++, sizeof(cl_mem), &out_subbuffer);
    }
    setKernelArg(vertex_kernel,
        vertex_out_location,
        sizeof(gl_Positions), &gl_Positions
    );

    // Perspective Division Kernel Set Up
    setKernelArg(_perspective_division_kernel, 0, sizeof(gl_Positions), &gl_Positions);

    // Viewport Division Kernel Set Up
    setKernelArg(_viewport_division_kernel, 0, sizeof(gl_Positions), &gl_Positions);
    setKernelArg(_viewport_division_kernel, 1, sizeof(_viewport),    &_viewport);
    setKernelArg(_viewport_division_kernel, 2, sizeof(_depth_range), &_depth_range);

    /* ---- Set Up Per-Fragment Buffers ---- */
    cl_mem fragment_in_buffer, gl_FragCoord, gl_Discard, gl_FragColor;
    
    fragment_in_buffer  = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments*CURRENT_PROGRAM.varying_size-1, NULL);
    gl_FragCoord        = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments,                              NULL);
    gl_Discard          = createBuffer(MEM_READ_WRITE, sizeof(uint8_t)*num_fragments,                               NULL);
    gl_FragColor        = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments,                              NULL);

    /* ---- Set Up Per-Fragment Kernels ---- */
    cl_kernel fragment_kernel;
    int tmp_varying_size = CURRENT_PROGRAM.varying_size-1;
    // Rasterization Kernel Set Up
    if (mode==GL_TRIANGLES) {
        // arg 0 is reserved for the primitive index
        setKernelArg(_rasterization_kernels.triangles, 1, sizeof(int),      &COLOR_ATTACHMENT0.width);
        setKernelArg(_rasterization_kernels.triangles, 2, sizeof(int),      &tmp_varying_size);
        setKernelArg(_rasterization_kernels.triangles, 3, sizeof(cl_mem),   &gl_Positions);
        setKernelArg(_rasterization_kernels.triangles, 4, sizeof(cl_mem),   &gl_FragCoord);
        setKernelArg(_rasterization_kernels.triangles, 5, sizeof(cl_mem),   &gl_Discard);
        setKernelArg(_rasterization_kernels.triangles, 6, sizeof(cl_mem),   &vertex_out_buffer);
        setKernelArg(_rasterization_kernels.triangles, 7, sizeof(cl_mem),   &fragment_in_buffer);
    } else NOT_IMPLEMENTED;

    // Fragment Kernel Set Up
    fragment_kernel = createKernel(CURRENT_PROGRAM.program, FRAGMENT_SHADER_FNAME);
    for(int uniform = 0; uniform < CURRENT_PROGRAM.active_uniforms; ++uniform) {
        setKernelArg(fragment_kernel, 
            CURRENT_PROGRAM.uniforms_data[uniform].location, 
            sizeof(CURRENT_PROGRAM.uniforms_mem[uniform]),
            &CURRENT_PROGRAM.uniforms_mem[uniform]
        );
    }

    // Texture vars
    // TODO: Texture support, this is 
    int active_textures = _texture_binding != 0;
    // In fragment vars
    int fragment_in_out_location = CURRENT_PROGRAM.active_uniforms + active_textures*2; // sample_t + image_t 
    for(int in=0; in < CURRENT_PROGRAM.varying_size-1; ++in) { // TODO use varying_size
        cl_int tmp_err;
        cl_buffer_region region = {
            .origin = sizeof(float[4])*num_fragments*in,
            .size = num_fragments*sizeof(float[4])
        };
        cl_mem subbuffer = clCreateSubBuffer(fragment_in_buffer,CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &tmp_err);
        printf("Fragment subbuffer: %i\n", tmp_err);
        setKernelArg(fragment_kernel, fragment_in_out_location++, sizeof(cl_mem), &subbuffer);
    }

    // Required fragment vars
    setKernelArg(fragment_kernel, fragment_in_out_location++, sizeof(gl_FragColor), &gl_FragColor);
    // Optional fragment vars
    setKernelArg(fragment_kernel, fragment_in_out_location++, sizeof(gl_FragCoord), &gl_FragCoord);
    setKernelArg(fragment_kernel, fragment_in_out_location++, sizeof(gl_Discard), &gl_Discard);
    // OUT fragment vars
    
    // Depth Kernel Set Up
    cl_kernel depth_kernel = NULL;
    if (_depth_enabled) {
        depth_kernel = getDepthKernel();
        setKernelArg(depth_kernel, 0, sizeof(DEPTH_ATTACHMENT.mem), &DEPTH_ATTACHMENT.mem);
        setKernelArg(depth_kernel, 1, sizeof(gl_Discard), &gl_Discard);
        setKernelArg(depth_kernel, 2, sizeof(gl_FragCoord), &gl_FragCoord);
    }

    // Color Kernel Set Up
    cl_kernel color_kernel = getColorKernel();
    setKernelArg(color_kernel, 0, sizeof(COLOR_ATTACHMENT0.mem), &COLOR_ATTACHMENT0.mem);
    setKernelArg(color_kernel, 1, sizeof(gl_Discard), &gl_Discard);
    setKernelArg(color_kernel, 2, sizeof(gl_FragColor), &gl_FragColor);

    /* ---- Enqueue Kernels ---- */

    cl_command_queue command_queue = getCommandQueue();
    // Vertex
    enqueueNDRangeKernel(command_queue, vertex_kernel, num_vertices);
    // TODO: Make it for debug mode
    // float positions[num_vertices][4]; 
    // enqueueReadBuffer(command_queue, vertex_array_mem[0],sizeof(float[4])*num_vertices,positions);
    // for (int i = 0; i < num_vertices; i+=1) {
    //     printf("vertex %d, x=%f, y=%f, z=%f, w=%f\n", i, positions[i][0],positions[i][1],positions[i][2], positions[i][3]);
    // }

    enqueueNDRangeKernel(command_queue, _perspective_division_kernel, num_vertices);
    
    enqueueNDRangeKernel(command_queue, _viewport_division_kernel, num_vertices);

    for(GLsizei primitive=0; primitive < num_primitives; ++primitive) {
        // Rasterization
        setKernelArg(_rasterization_kernels.triangles, 0, sizeof(primitive), &primitive);

        enqueueNDRangeKernel(command_queue, _rasterization_kernels.triangles, num_fragments);   

        enqueueNDRangeKernel(command_queue, fragment_kernel, num_fragments);

        if (depth_kernel) enqueueNDRangeKernel(command_queue, depth_kernel, num_fragments);

        enqueueNDRangeKernel(command_queue, color_kernel, num_fragments);
    }
}

GL_APICALL void GL_APIENTRY glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices) {}

GL_APICALL void GL_APIENTRY glDisable (GLenum cap) {
    if (cap == GL_SCISSOR_TEST)
        _scissor_enabled = 0;
    else if (cap == GL_DEPTH_TEST)
        _depth_enabled = 0;
    else if (cap == GL_STENCIL_TEST)
        _stencil_enabled = 0;
    else RETURN_ERROR(GL_INVALID_ENUM);
}

GL_APICALL void GL_APIENTRY glDisableVertexAttribArray (GLuint index) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) RETURN_ERROR(GL_INVALID_VALUE);

    _vertex_attrib_enable[index] = 0;
}

GL_APICALL void GL_APIENTRY glEnable (GLenum cap) {
    if (cap == GL_SCISSOR_TEST)
        _scissor_enabled = 1;
    else if (cap == GL_DEPTH_TEST)
        _depth_enabled = 1;
    else if (cap == GL_STENCIL_TEST)
        _stencil_enabled = 1;
    else RETURN_ERROR(GL_INVALID_ENUM);
}

GL_APICALL void GL_APIENTRY glEnableVertexAttribArray (GLuint index) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) RETURN_ERROR(GL_INVALID_VALUE);

    _vertex_attrib_enable[index] = 1;
}

GL_APICALL void GL_APIENTRY glFinish (void) {
    finish(getCommandQueue());
}

GL_APICALL void GL_APIENTRY glFramebufferRenderbuffer (GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) {
    if (attachment == GL_COLOR_ATTACHMENT0)
        _framebuffers[_framebuffer_binding].color_attachment0=renderbuffer;
    else if (attachment == GL_DEPTH_ATTACHMENT)
        _framebuffers[_framebuffer_binding].depth_attachment=renderbuffer;
    else if (attachment == GL_STENCIL_ATTACHMENT)
        _framebuffers[_framebuffer_binding].stencil_attachment=renderbuffer;
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

GL_APICALL GLenum GL_APIENTRY glGetError (void) {
    GLenum error = gl_error;
    gl_error = GL_NO_ERROR;
    return error;
}

// TODO:
GL_APICALL GLenum GL_APIENTRY glGetGraphicsResetStatus (void) {

    return GL_NO_ERROR;
}

#define POCL_BINARY 0x0

GL_APICALL void GL_APIENTRY glProgramBinary (GLuint program, GLenum binaryFormat, const void *binary, GLsizei length){
    printf("glProgramBinary() program=%d, binaryFormat=%d\n",program,binaryFormat);
    if(!_kernel_load_status) {
        void *gl_program;

        gl_program = createProgramWithBinary(GLSC2_kernel_depth_pocl, sizeof(GLSC2_kernel_depth_pocl));
        buildProgram(gl_program);
        _depth_kernel.less = createKernel(gl_program, "gl_less");

        gl_program = createProgramWithBinary(GLSC2_kernel_color_pocl, sizeof(GLSC2_kernel_color_pocl));
        buildProgram(gl_program);
        _color_kernel.rgba4 = createKernel(gl_program, "gl_rgba4");
        _color_kernel.rgba8 = createKernel(gl_program, "gl_rgba8");

        gl_program = createProgramWithBinary(GLSC2_kernel_rasterization_triangle_pocl, sizeof(GLSC2_kernel_rasterization_triangle_pocl));
        buildProgram(gl_program);
        _rasterization_kernels.triangles = createKernel(gl_program, "gl_rasterization_triangle");
        gl_program = createProgramWithBinary(GLSC2_kernel_viewport_division_pocl, sizeof(GLSC2_kernel_viewport_division_pocl));
        buildProgram(gl_program);
        _viewport_division_kernel = createKernel(gl_program, "gl_viewport_division");
        gl_program = createProgramWithBinary(GLSC2_kernel_perspective_division_pocl, sizeof(GLSC2_kernel_perspective_division_pocl));
        buildProgram(gl_program);
        _perspective_division_kernel = createKernel(gl_program, "gl_perspective_division");
        gl_program = createProgramWithBinary(GLSC2_kernel_readnpixels_pocl, sizeof(GLSC2_kernel_readnpixels_pocl));
        buildProgram(gl_program);
        _readnpixels_kernel = createKernel(gl_program, "gl_rgba4_rgba8");

        gl_program = createProgramWithBinary(GLSC2_kernel_strided_write_pocl, sizeof(GLSC2_kernel_strided_write_pocl));
        buildProgram(gl_program);
        _strided_write_kernel = createKernel(gl_program, "gl_strided_write");

        _kernel_load_status = 1;
    }

    if (_programs[program].program) RETURN_ERROR(GL_INVALID_OPERATION);
    // TODO: Check binaryFormat
    if (binaryFormat == POCL_BINARY) {
        _programs[program].program=createProgramWithBinary(binary, length);
        buildProgram(_programs[program].program);
        // TODO: Check this logic
        _programs[program].last_load_attempt = GL_TRUE;
        _programs[program].last_validation_attempt = GL_TRUE;
        _programs[program].vertex_kernel = createKernel(_programs[program].program, VERTEX_SHADER_FNAME);
        _programs[program].fragment_kernel = createKernel(_programs[program].program, FRAGMENT_SHADER_FNAME);

        cl_uint kernel_num_args;
        clGetKernelInfo(_programs[program].vertex_kernel,CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &kernel_num_args, NULL);
        for(cl_uint arg=0; arg < kernel_num_args; ++arg) {
            cl_kernel_arg_address_qualifier addr_qualifier;
            cl_kernel_arg_type_qualifier type_qualifier;
            clGetKernelArgInfo(_programs[program].vertex_kernel, arg, CL_KERNEL_ARG_ADDRESS_QUALIFIER, sizeof(addr_qualifier), &addr_qualifier, NULL);
            clGetKernelArgInfo(_programs[program].vertex_kernel, arg, CL_KERNEL_ARG_TYPE_QUALIFIER, sizeof(type_qualifier), &type_qualifier, NULL);

            if (addr_qualifier == CL_KERNEL_ARG_ADDRESS_CONSTANT) { // UNIFORM
                _programs[program].active_vertex_attribs += 1;
            } else if (addr_qualifier == CL_KERNEL_ARG_ADDRESS_GLOBAL && type_qualifier == CL_KERNEL_ARG_TYPE_CONST) { // IN attrib pointer
                _programs[program].active_vertex_attribs += 1;
            } else if (addr_qualifier == CL_KERNEL_ARG_ADDRESS_GLOBAL && type_qualifier == CL_KERNEL_ARG_TYPE_NONE) { // VARYING attrib pointer
                _programs[program].varying_size += 1;
            } else if (addr_qualifier == CL_KERNEL_ARG_ADDRESS_PRIVATE) { // IN attrib
                _programs[program].active_vertex_attribs += 1;
            }
        }
    }
}

GL_APICALL void GL_APIENTRY glReadnPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data) {
    if (format == GL_RGBA && type == GL_UNSIGNED_BYTE) {
        if (_framebuffer_binding) {

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

GL_APICALL void GL_APIENTRY glTexParameterf (GLenum target, GLenum pname, GLfloat param) {
    if (target != GL_TEXTURE_2D) return; // Unknown behaviour
    NOT_IMPLEMENTED;
}
GL_APICALL void GL_APIENTRY glTexParameterfv (GLenum target, GLenum pname, const GLfloat *params) {
    if (target != GL_TEXTURE_2D) return; // Unknown behaviour
    NOT_IMPLEMENTED;
}
GL_APICALL void GL_APIENTRY glTexParameteri (GLenum target, GLenum pname, GLint param) {
    if (target != GL_TEXTURE_2D) return; // Unknown behaviour
    NOT_IMPLEMENTED;
}
GL_APICALL void GL_APIENTRY glTexParameteriv (GLenum target, GLenum pname, const GLint *params) {
    if (target != GL_TEXTURE_2D) return; // Unknown behaviour
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) {
    if (target != GL_TEXTURE_2D) return; // Unknown behaviour
    
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

#define UMAT4 0x0;

GL_APICALL void GL_APIENTRY glUniformMatrix4fv (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    if (transpose) NOT_IMPLEMENTED;
    if (!_current_program) NOT_IMPLEMENTED;
    if (count > 4) NOT_IMPLEMENTED;
    if (count < 1) NOT_IMPLEMENTED;

    GLint uniform_id = _programs[_current_program].active_uniforms;
    if (location < uniform_id) {
       uniform_id = location;
    } else {
       _programs[_current_program].active_uniforms += 1;
    }
    _programs[_current_program].uniforms_data[uniform_id].location = location;
    _programs[_current_program].uniforms_data[uniform_id].size = sizeof(float[4])*count;
    _programs[_current_program].uniforms_data[uniform_id].type = UMAT4;

    // TODO:
    /*
    float *data_ptr = (float*) _programs[_current_program].uniforms[uniform_id].data;
    for(GLsizei i=0; i<count; ++i) {
        data_ptr[0] = *(value + 4*i);
        data_ptr[1] = *(value + 4*i + 1);
        data_ptr[2] = *(value + 4*i + 2);
        data_ptr[3] = *(value + 4*i + 3);
        data_ptr +=4;
    }
    */
}

GL_APICALL void GL_APIENTRY glUseProgram (GLuint program){
    printf("glUseProgram() program=%d\n", program);
    if (program) {
        if (!_programs[program].last_load_attempt){
            printf("\tERROR last_load_attempt=%d\n", _programs[program].last_load_attempt);

            RETURN_ERROR(GL_INVALID_OPERATION);
        }
        // TODO install program
    }
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


void* createVertexKernel(GLenum mode, GLint first, GLsizei count) {
    /*
    void *kernel = createKernel(_programs[_current_program].program, "gl_main_vs");
    // VAO locations
    GLuint attribute = 0;
    while(attribute < _programs[_current_program].active_attributes) {
        
        if(_programs[_current_program].attributes[attribute].data.type == 0x2) {
            setKernelArg(
                kernel, 
                _programs[_current_program].attributes[attribute].location,
                _programs[_current_program].attributes[attribute].size,
                &_programs[_current_program].attributes[attribute].data.attribute.pointer.mem
            );
        } else {
            NOT_IMPLEMENTED;
            setKernelArg(
                kernel, 
                _programs[_current_program].attributes[attribute].location,
                _programs[_current_program].attributes[attribute].size,
                _programs[_current_program].attributes[attribute].data.attribute.int4.values // TODO: 
            );
        }
        ++attribute;
    }
    // Uniform locations
    GLuint uniform = 0;
    GLuint active_attributes = _programs[_current_program].active_attributes; 
    while(uniform < _programs[_current_program].active_uniforms) {
        setKernelArg(
            kernel, 
            _programs[_current_program].uniforms[uniform].location + active_attributes,
            _programs[_current_program].uniforms[uniform].size, 
            &_programs[_current_program].uniforms[uniform].data
            );
        ++uniform;
    }
    
    */
    return NULL;
}

void* getPerspectiveDivisionKernel(GLenum mode, GLint first, GLsizei count) {
    return _perspective_division_kernel;
}

void* getViewportDivisionKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel = _viewport_division_kernel;

    setKernelArg(kernel, 1,
        sizeof(_viewport), &_viewport
    );
    setKernelArg(kernel, 2,
        sizeof(_depth_range), &_depth_range
    );

    return kernel;
}

void* getRasterizationTriangleKernel(GLenum mode, GLint first, GLsizei count) {
    /*
    void *kernel = _rasterization_kernel;

    setKernelArg(kernel, 1,
        sizeof(COLOR_ATTACHMENT0.width), &COLOR_ATTACHMENT0.width
    );
//    setKernelArg(kernel, 2,
//        sizeof(COLOR_ATTACHMENT0.height), &COLOR_ATTACHMENT0.height
//    );
    setKernelArg(kernel, 2,
        sizeof(_programs[_current_program].active_attributes), &_programs[_current_program].active_attributes
    );

    return kernel;
    */
    return NULL;
}

void* createFragmentKernel(GLenum mode, GLint first, GLsizei count) {
    /*
    void *kernel = createKernel(_programs[_current_program].program, "gl_main_fs");
    // Uniform locations
    GLuint uniform = 0;
    while(uniform < _programs[_current_program].active_uniforms) {
        setKernelArg(
            kernel, 
            _programs[_current_program].uniforms[uniform].location,
            _programs[_current_program].uniforms[uniform].size, 
            &_programs[_current_program].uniforms[uniform].data
            );
        ++uniform;
    }
    GLuint texture = 0;
    while(texture < 1) { // TODO: Add support for more than one texture
        #ifndef IMAGE_SUPPORT
        int size[2];
        size[0] = _textures[_texture_binding].width;
        size[1] = _textures[_texture_binding].height;

        setKernelArg(
            kernel, 
            _programs[_current_program].active_uniforms,
            sizeof(size), 
            &size
            );
        #else
        void *sampler = createSampler(CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST);
	    setKernelArg(
            kernel, 
            _programs[_current_program].active_uniforms,
            sizeof(sampler), 
            &sampler
        );
        #endif
        setKernelArg(
            kernel, 
            _programs[_current_program].active_uniforms + 1,
            sizeof(_textures[_texture_binding].mem), 
            &_textures[_texture_binding].mem
        );
        ++texture;
    }

    return kernel;
    */
    return NULL;
}
cl_kernel getDepthKernel() {
    switch (_depth_func) {
        case GL_LESS:
            return _depth_kernel.less;
        // TODO add all cases
    }
    NOT_IMPLEMENTED;
}

cl_kernel getColorKernel() {
    switch (COLOR_ATTACHMENT0.internalformat) {
        case GL_RGBA8:
            return _color_kernel.rgba8;
        case GL_RGBA4:
            return _color_kernel.rgba4;
        // TODO: add all cases
    }
    NOT_IMPLEMENTED;
}
