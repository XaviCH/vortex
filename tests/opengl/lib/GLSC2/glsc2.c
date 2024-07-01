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

int gl_error = 0;

#define RETURN_ERROR(error)          \
    ({                               \
        gl_error = error;            \
        return;                      \
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
    GLboolean used;
    GLuint object_name;
    GLboolean load_status, validation_status;
    LOG log;
    GLuint active_uniforms, active_attributes;
    UNIFORM uniforms[MAX_UNIFORM_VECTORS];
    ATTRIBUTE attributes[16];
    cl_program program;
} PROGRAM;

PROGRAM _programs[MAX_PROGRAMS];
GLuint _current_program; // ZERO is reserved for NULL program

typedef struct {
    void *less;
} DEPTH_KERNEL;

typedef struct {
    void *rgba4, *rgba8;
} COLOR_KERNEL;

GLboolean _kernel_load_status;
DEPTH_KERNEL _depth_kernel;
COLOR_KERNEL _color_kernel;
void *_rasterization_kernel;
void *_viewport_division_kernel;
void *_perspective_division_kernel;
void *_readnpixels_kernel;

/****** BUFFER objects ******\
 * TODO: Re think this, I think it is actually more tricky than the first though. 
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

void* getCommandQueue();

void* getPerspectiveDivisionKernel(GLenum mode, GLint first, GLsizei count);
void* getViewportDivisionKernel(GLenum mode, GLint first, GLsizei count);
void* getRasterizationTriangleKernel(GLenum mode, GLint first, GLsizei count);
void* getDepthKernel(GLenum mode, GLint first, GLsizei count);
void* getColorKernel(GLenum mode, GLint first, GLsizei count);

void* createVertexKernel(GLenum mode, GLint first, GLsizei count);
void* createFragmentKernel(GLenum mode, GLint first, GLsizei count);

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
    GLuint program = 1; // ZERO is reserved
    while(program < MAX_PROGRAMS) {
        if (!_programs[program].used) {
            _programs[program].used=GL_TRUE;
            return program;
        } 
        ++program;
    }
    return 0; // TODO maybe throw some error ??
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

    if (first <0){
        _err= GL_INVALID_VALUE;
        return;
    }
    
    GLsizei num_vertices = count-first;
    GLsizei num_fragments = COLOR_ATTACHMENT0.width * COLOR_ATTACHMENT0.height;
    GLsizei num_primitives = num_vertices;
    
    if (mode==GL_LINES) num_primitives /= 2;
    else if (mode==GL_TRIANGLES) num_primitives /= 3;

    // Build memory buffers
    void *gl_Positions = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_vertices, NULL);
    void *gl_Primitives = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_vertices*_programs[_current_program].active_attributes, NULL);
    void *gl_Rasterization = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments*_programs[_current_program].active_attributes, NULL);
    void *gl_FragCoord = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments, NULL);
    void *gl_Discard = createBuffer(MEM_READ_WRITE, sizeof(uint8_t)*num_fragments, NULL);
    void *gl_FragColor = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments, NULL);

    // Set up kernels
    printf("SetUp vertex");
    void* vertex_kernel = createVertexKernel(mode, first, count);
    setKernelArg(vertex_kernel,
        _programs[_current_program].active_attributes + _programs[_current_program].active_uniforms,
        sizeof(gl_Positions), &gl_Positions
    );
    setKernelArg(vertex_kernel,
        _programs[_current_program].active_attributes + _programs[_current_program].active_uniforms + 1,
        sizeof(gl_Primitives), &gl_Primitives
    );

    void* perspective_division_kernel = getPerspectiveDivisionKernel(mode, first, count);
    setKernelArg(perspective_division_kernel, 0,
        sizeof(gl_Positions), &gl_Positions
    );
    void* viewport_division_kernel = getViewportDivisionKernel(mode, first, count);
    setKernelArg(viewport_division_kernel, 0,
        sizeof(gl_Positions), &gl_Positions
    );
    void *rasterization_kernel;
    if (mode==GL_TRIANGLES) {
        rasterization_kernel = getRasterizationTriangleKernel(mode, first, count);
        setKernelArg(rasterization_kernel, 3,
            sizeof(gl_FragCoord), &gl_Positions
        );
        setKernelArg(rasterization_kernel, 4,
            sizeof(gl_Primitives), &gl_Primitives
        );
        setKernelArg(rasterization_kernel, 5,
            sizeof(gl_FragCoord), &gl_FragCoord
        );
        setKernelArg(rasterization_kernel, 6,
            sizeof(gl_Rasterization), &gl_Rasterization
        );
        setKernelArg(rasterization_kernel, 7,
            sizeof(gl_Discard), &gl_Discard
        );
    } else NOT_IMPLEMENTED;

    void* fragment_kernel = createFragmentKernel(mode, first, count);
    int active_textures = _texture_binding != 0;
    setKernelArg(fragment_kernel, 
        _programs[_current_program].active_uniforms + active_textures*2,
        sizeof(gl_FragCoord), &gl_FragCoord
    );
    setKernelArg(fragment_kernel, 
        _programs[_current_program].active_uniforms + active_textures*2 + 1,
        sizeof(gl_Rasterization), &gl_Rasterization
    );
    setKernelArg(fragment_kernel, 
        _programs[_current_program].active_uniforms + active_textures*2 + 2,
        sizeof(gl_Discard), &gl_Discard
    );
    setKernelArg(fragment_kernel, 
        _programs[_current_program].active_uniforms + active_textures*2 + 3,
        sizeof(gl_FragColor), &gl_FragColor
    );

    void *depth_kernel = NULL; 
    if (_depth_enabled) {
        depth_kernel = getDepthKernel(mode, first, count);
        setKernelArg(depth_kernel, 1, sizeof(gl_Discard), &gl_Discard);
        setKernelArg(depth_kernel, 2, sizeof(gl_FragCoord), &gl_FragCoord);
    }

    void *color_kernel = getColorKernel(mode, first, count);
    setKernelArg(color_kernel, 1, sizeof(gl_Discard), &gl_Discard);
    setKernelArg(color_kernel, 2, sizeof(gl_FragColor), &gl_FragColor);

    // Enqueue kernels
    void *command_queue = getCommandQueue();
    // Vertex
    enqueueNDRangeKernel(command_queue, vertex_kernel, num_vertices);
    // Post-Vertex
      float _gl_Positions[num_vertices][4]; 
    enqueueReadBuffer(command_queue, gl_Positions,sizeof(float[4])*num_vertices,_gl_Positions);
    for (int i = 0; i < num_vertices; i+=1) {
        printf("vertex %d, x=%f, y=%f, z=%f, w=%f\n", i, _gl_Positions[i][0],_gl_Positions[i][1],_gl_Positions[i][2], _gl_Positions[i][3]);
    }
    
    enqueueNDRangeKernel(command_queue, perspective_division_kernel, num_vertices);
    enqueueReadBuffer(command_queue, gl_Positions,sizeof(float[4])*num_vertices,_gl_Positions);
    for (int i = 0; i < num_vertices; i+=1) {
        printf("vertex %d, x=%f, y=%f, z=%f, w=%f\n", i, _gl_Positions[i][0],_gl_Positions[i][1],_gl_Positions[i][2], _gl_Positions[i][3]);
    }
    
    enqueueNDRangeKernel(command_queue, viewport_division_kernel, num_vertices);
    enqueueReadBuffer(command_queue, gl_Positions,sizeof(float[4])*num_vertices,_gl_Positions);
    for (int i = 0; i < num_vertices; i+=1) {
        printf("vertex %d, x=%f, y=%f, z=%f, w=%f\n", i, _gl_Positions[i][0],_gl_Positions[i][1],_gl_Positions[i][2], _gl_Positions[i][3]);
    }

    for(GLsizei primitive=0; primitive < num_primitives; ++primitive) {
        // Rasterization
        setKernelArg(rasterization_kernel, 0, sizeof(primitive), &primitive);
        enqueueNDRangeKernel(command_queue, rasterization_kernel, num_fragments);   
        // Fragment
        enqueueNDRangeKernel(command_queue, fragment_kernel, num_fragments);   
	    // Post-Fragment
        if (depth_kernel != NULL) {
            enqueueNDRangeKernel(command_queue, depth_kernel, num_fragments);
        }
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
}

GL_APICALL void GL_APIENTRY glDisableVertexAttribArray (GLuint index) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) {
        _err = GL_INVALID_VALUE;
        return;
    }

    _vertex_attrib[index].enable = 0;
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

    _vertex_attrib[index].enable = 1;
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
        _rasterization_kernel = createKernel(gl_program, "gl_rasterization_triangle");
        gl_program = createProgramWithBinary(GLSC2_kernel_viewport_division_pocl, sizeof(GLSC2_kernel_viewport_division_pocl));
        buildProgram(gl_program);
        _viewport_division_kernel = createKernel(gl_program, "gl_viewport_division");
        gl_program = createProgramWithBinary(GLSC2_kernel_perspective_division_pocl, sizeof(GLSC2_kernel_perspective_division_pocl));
        buildProgram(gl_program);
        _perspective_division_kernel = createKernel(gl_program, "gl_perspective_division");
        gl_program = createProgramWithBinary(GLSC2_kernel_readnpixels_pocl, sizeof(GLSC2_kernel_readnpixels_pocl));
        buildProgram(gl_program);
        _readnpixels_kernel = createKernel(gl_program, "gl_rgba4_rgba8");

        _kernel_load_status = 1;
    }
    if(_programs[program].program) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    if (binaryFormat == POCL_BINARY) {
        _programs[program].program=createProgramWithBinary(binary, length);
        buildProgram(_programs[program].program);
        // TODO: Check this logic
        _programs[program].load_status = GL_TRUE;
        _programs[program].validation_status = GL_TRUE;
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
        enqueueWriteBuffer(getCommandQueue(),_textures[_texture_binding].mem,width*height*sizeof(uint8_t[4]),pixels);
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
    _programs[_current_program].uniforms[uniform_id].location = location;
    _programs[_current_program].uniforms[uniform_id].size = sizeof(float[4])*count;
    _programs[_current_program].uniforms[uniform_id].type = UMAT4;

    float *data_ptr = (float*) _programs[_current_program].uniforms[uniform_id].data;
    for(GLsizei i=0; i<count; ++i) {
        data_ptr[0] = *(value + 4*i);
        data_ptr[1] = *(value + 4*i + 1);
        data_ptr[2] = *(value + 4*i + 2);
        data_ptr[3] = *(value + 4*i + 3);
        data_ptr +=4;
    }
}

GL_APICALL void GL_APIENTRY glUseProgram (GLuint program){
    printf("glUseProgram() program=%d\n", program);
    if (program) {
        if (!_programs[program].load_status){
            printf("\tERROR load_status=%d\n", _programs[program].load_status);

            _err = GL_INVALID_OPERATION;
            return;
        }
        // TODO install program
    }
    _current_program=program;
}

GL_APICALL void GL_APIENTRY glVertexAttribPointer (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) {
    if (index >= MAX_VERTEX_ATTRIBS) {
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

    if (type != GL_BYTE && type != GL_UNSIGNED_BYTE && type != GL_SHORT && type != GL_UNSIGNED_SHORT && type != GL_FLOAT){
        _err=GL_INVALID_VALUE;
        return;
    }
    // TODO: normalized & strid
    if (!_current_program) {
        // TODO:
    } else {
        if (_buffer_binding) {
            _programs[_current_program].attributes[_programs[_current_program].active_attributes].data.attribute.pointer.mem = _buffers[_buffer_binding].mem;
            _programs[_current_program].attributes[_programs[_current_program].active_attributes].data.type = 0x2;
        }
        _programs[_current_program].attributes[_programs[_current_program].active_attributes].data.attribute.pointer.size = size;
        _programs[_current_program].attributes[_programs[_current_program].active_attributes].data.attribute.pointer.type = type;
        _programs[_current_program].attributes[_programs[_current_program].active_attributes].location = index;
        _programs[_current_program].attributes[_programs[_current_program].active_attributes].size = sizeof(void*);
        _programs[_current_program].attributes[_programs[_current_program].active_attributes].type = type;
        
        _programs[_current_program].active_attributes += 1;
    
    }
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
    

    return kernel;
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
}

void* createFragmentKernel(GLenum mode, GLint first, GLsizei count) {
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
}
void* getDepthKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel;

    if (DEPTH_ATTACHMENT.internalformat != GL_DEPTH_COMPONENT16)
        NOT_IMPLEMENTED;

    switch (_depth_func) {
        case GL_LESS:
            kernel = _depth_kernel.less;
            break;
        // TODO add more depth kernels
        default:
            INTERNAL_ERROR;
    }

    setKernelArg(kernel, 0, sizeof(DEPTH_ATTACHMENT.mem), &DEPTH_ATTACHMENT.mem);

    return kernel;
}

void* getColorKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel;
    switch (COLOR_ATTACHMENT0.internalformat) {
        case GL_RGBA8:
            kernel = _color_kernel.rgba8;
            break;
        case GL_RGBA4:
            kernel = _color_kernel.rgba4;
            break;
        // TODO add more color kernels
        default:
            INTERNAL_ERROR;
    }

    setKernelArg(kernel, 0, sizeof(COLOR_ATTACHMENT0.mem), &COLOR_ATTACHMENT0.mem);

    return kernel;
}
