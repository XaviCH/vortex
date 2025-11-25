#include <math.h>
#include <string.h>
#include <GLSC2/glsc2.h>
#include "kernel.c" // TODO: may be interesting to extract it to an interface so could be re implementated with CUDA

#include "debug.h"
#include "constants.device.h"
#include "config.device.h"
#include <frontend/types.h>
#include <frontend/device.h>
#include "types.device.h"

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

#define RETURN_ERROR_WITH_VALUE(error, value)                     \
    ({                                          \
        if(gl_error == GL_NO_ERROR) {           \
            gl_error = error;                   \
            printf("OpenGL error generated (%d | %x) in %s at %s:%d.\n", gl_error, gl_error, __func__, __FILE__, __LINE__);    \
            exit(error);                        \
        }                                       \
        return value;                                 \
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

#define MIN(_A,_B) (((_A)<(_B))? (_A):(_B))
#define MAX(_A,_B) (((_A)>(_B))? (_A):(_B))

/************ CONTEXT ************\
 * TODO: Create context using a context manager instead of the library 
 * 
*/

/****** CONFIGURATION INITIAL STATE ******/

box_t               _viewport; // TODO: Checkout the default initial value

enabled_container_t _enableds = {
    .scissor_test = GL_FALSE,
    .stencil_test = GL_FALSE,
    .depth_test = GL_FALSE,
    .blend = GL_FALSE,
    .dither = GL_TRUE,
    .cull_face = GL_FALSE,
    .polygon_offset_fill = GL_FALSE,
};

mask_container_t _masks = {
    .color = { .r=GL_TRUE, .g=GL_TRUE, .b=GL_TRUE, .a=GL_TRUE },
    .depth = GL_TRUE,
    .stencil = { .back=0xFFu , .front=0xFFu },
};

pixel_store_t _pixel_store = { .unpack_aligment = 4 };

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
        .function = { .func = STENCIL_FUNC_ALWAYS, .ref = 0, .mask = 1, },
        .operation = { .sfail = STENCIL_OP_KEEP, .dpfail = STENCIL_OP_KEEP, .dpass = STENCIL_OP_KEEP }
    },
    .back = {
        .function = { .func = STENCIL_FUNC_ALWAYS, .ref = 0, .mask = 1, },
        .operation = { .sfail = STENCIL_OP_KEEP, .dpfail = STENCIL_OP_KEEP, .dpass = STENCIL_OP_KEEP }
    },
};

blend_data_t _blend_data = {
    .equation = {
        .modeRGB = BLEND_FUNC_ADD, .modeAlpha = BLEND_FUNC_ADD,
    },
    .func = {
        .srcRGB = BLEND_ZERO, .srcAlpha = BLEND_ZERO,
        .dstRGB = BLEND_ONE, .dstAlpha = BLEND_ONE,
    },
    .color = {
        .red = 0, .green = 0, .blue = 0, .alpha = 0
    }
};

rasterization_data_t _rasterization_data = {
    .polygon_offset = {
        .factor = GL_ZERO,
        .units = GL_ZERO,
    },
    .cull_face = GL_BACK,
    .front_face = GL_CCW,
    .line_width = GL_ONE,
};


rasterization_program_container_t _rasterization_programs;
rasterization_mem_container_t _rasterization_mem;

cl_ulong c_clear_write_values;
cl_ushort c_clear_enabled_data = 0;
active_deferred_clear_t _deferred_clear;

// Host State
static uint8_t                      vertex_attibute_updated;
static cl_float4                    vertex_attributes           [DEVICE_VERTEX_ATTRIBUTE_SIZE];
static vertex_attribute_data_t      vertex_attribute_datas      [DEVICE_VERTEX_ATTRIBUTE_SIZE];
static vertex_attribute_binding_t   vertex_attribute_binding    [DEVICE_VERTEX_ATTRIBUTE_SIZE];

static uint8_t                  texture_unit_updated;
static size_t                   active_texture_unit;
static size_t                   texture_unit_bindings    [DEVICE_TEXTURE_UNITS];
static size_t                   _texture_sz;
static size_t                   _texture_binding;
static texture_t                _textures                [HOST_TEXTURES_SIZE];

static uint8_t                  rop_config_updated;
static uint32_t                 rop_config_count;
static rop_config_t             rop_config;

static uint8_t                  _uniform_data_updated;
static size_t                   _current_program;
static size_t                   _program_sz;
static program_t                _programs                [HOST_PROGRAMS_SIZE];

static uint32_t                 _framebuffer_updated;
static size_t                   _framebuffer_binding;
static size_t                   _framebuffer_sz;
static framebuffer_t            _framebuffers   [HOST_FRAMEBUFFER_SIZE];

static size_t                   _buffer_binding;
static size_t                   _buffer_sz;
static buffer_t                 _buffers        [HOST_BUFFERS_SIZE];

static size_t                   _renderbuffer_binding;
static size_t                   _renderbuffer_sz;
static renderbuffer_t           _renderbuffers  [HOST_RENDERBUFFERS_SIZE];

static GLenum _hint;
static GLenum _error;

// Device State
static device_shared_objects_t      device_shared_objects;
static size_t                       context_sz;
static device_context_t*            contexts[HOST_PROGRAMS_SIZE];

__attribute__((constructor))
void __context_constructor__() 
{
    vertex_attibute_updated = 0;
    for(size_t i = 0; i<DEVICE_VERTEX_ATTRIBUTE_SIZE; ++i) 
    {
        vertex_attributes[i] = (cl_float4) {0,0,0,1};
        vertex_attribute_datas[i].misc = 0;
    }

    texture_unit_updated = 0;
    active_texture_unit = 0;
    for (size_t i = 0; i < DEVICE_TEXTURE_UNITS; ++i) 
    {
        texture_unit_bindings[i] = 0;
    }

    rop_config_updated = 0;
    rop_config_count = 0;

    _program_sz = 0;

    _framebuffer_updated = 0;
    _framebuffer_binding = 0;
    _framebuffer_sz      = 0;

    _texture_sz = 0;
    _texture_binding = 0;

    _buffer_binding = 0;
    _buffer_sz = 0;

    _renderbuffer_binding = 0;
    _renderbuffer_sz = 0;

    _hint = GL_DONT_CARE;
    _error = GL_NO_ERROR;

    device_create_shared_objects(&device_shared_objects);
    context_sz = 0;
} 

/****** MEMORY HANDLERS ******/
// TODO: 

GLboolean _kernel_load_status;

// Depth
GLenum      _depth_func = DEPTH_FUNC_LESS;
depth_range_t _depth_range = { .n=0.0, .f=1.0};

/****** Interface for utils & inline functions ******\
 * Utility or inline function are implemented at the end of the file. 
*/


#ifdef DEBUG
    #define SET_GL_ERROR(error) \
        { \
            printf("DEBUG: %s throws a "#error" at %s:%d\n", __func__, __FILE__, __LINE__); \
            exit(error); \
        }
#else
    #define SET_GL_ERROR(error) \
        { _error = error; }
#endif

static void flush_device_context(device_context_t* context);
static uint32_t is_valid_program(GLuint program);
static GLboolean* get_enabled_pointer(GLenum cap);
static render_mode_t get_render_mode_flags(GLenum mode);
static size_t sizeof_vertex_attribute_type(GLenum type);



// TODO: Dependant texture throws this if (id >= MAX_COMBINED_TEXTURE_IMAGE_UNITS) RETURN_ERROR(GL_INVALID_OPERATION);
GL_APICALL void GL_APIENTRY glActiveTexture (GLenum texture) 
{
    uint32_t texture_unit = texture - GL_TEXTURE0;

    if (texture_unit >= DEVICE_TEXTURE_UNITS) RETURN_ERROR(GL_INVALID_VALUE); // TODO: MAX_COMBINED_TEXTURE_IMAGE_UNITS

    active_texture_unit = texture_unit;
    _texture_binding = texture_unit_bindings[active_texture_unit];
}

GL_APICALL void GL_APIENTRY glBindBuffer (GLenum target, GLuint buffer) 
{
    if (target != GL_ARRAY_BUFFER) RETURN_ERROR(GL_INVALID_ENUM);

    if (buffer >= _buffer_sz) RETURN_ERROR(GL_INVALID_OPERATION);

    _buffer_binding = buffer;
}

GL_APICALL void GL_APIENTRY glBindFramebuffer (GLenum target, GLuint framebuffer) 
{
    if (target != GL_FRAMEBUFFER) RETURN_ERROR(GL_INVALID_ENUM);

    if (framebuffer >= _framebuffer_sz) RETURN_ERROR(GL_INVALID_OPERATION);

    if (_current_program > 0)
    {
        flush_device_context(contexts[_current_program-1]);
    }

    _framebuffer_binding = framebuffer;
}

GL_APICALL void GL_APIENTRY glBindRenderbuffer (GLenum target, GLuint renderbuffer) 
{
    if (target != GL_RENDERBUFFER) RETURN_ERROR(GL_INVALID_ENUM);

    if (renderbuffer >= _renderbuffer_sz) RETURN_ERROR(GL_INVALID_OPERATION);

    _renderbuffer_binding = renderbuffer;
}

GL_APICALL void GL_APIENTRY glBindTexture (GLenum target, GLuint texture) 
{
    if (target != GL_TEXTURE_2D) RETURN_ERROR(GL_INVALID_ENUM);

    if (texture >= _texture_sz) RETURN_ERROR(GL_INVALID_OPERATION);

    texture_unit_updated = 1;
    texture_unit_bindings[active_texture_unit] = texture;
    _texture_binding = texture;
}

GL_APICALL void GL_APIENTRY glBlendColor (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) 
{
    _blend_data.color = (blend_color_t) 
    {
        .red = red,
        .green = green,
        .blue = blue,
        .alpha = alpha,
    };
}

static inline GLenum gl_equation_to_blend_equation (GLenum mode) 
{
    switch (mode)
    {
        case GL_FUNC_ADD:               return BLEND_FUNC_ADD;
        case GL_FUNC_SUBTRACT:          return BLEND_FUNC_SUBTRACT;
        case GL_FUNC_REVERSE_SUBTRACT:  return BLEND_FUNC_REVERSE_SUBTRACT;
        default:                        SET_GL_ERROR(GL_INVALID_ENUM); return mode;
    }
}

static GLboolean is_valid_blend_equation(GLenum mode)
{
    switch (mode)
    {
        default:
            return GL_FALSE;
        case GL_FUNC_ADD:
        case GL_FUNC_SUBTRACT:
        case GL_FUNC_REVERSE_SUBTRACT:
            return GL_TRUE;
    }
}

GL_APICALL void GL_APIENTRY glBlendEquation (GLenum mode) 
{
    glBlendEquationSeparate(mode, mode);
}

GL_APICALL void GL_APIENTRY glBlendEquationSeparate (GLenum modeRGB, GLenum modeAlpha) 
{
    if (is_valid_blend_equation(modeRGB) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);

    if (is_valid_blend_equation(modeAlpha) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);
    
    _blend_data.equation = (blend_equation_t) 
    {
        .modeRGB    = modeRGB,
        .modeAlpha  = modeAlpha,
    };
}

inline GLenum gl_function_to_blend_function (GLenum factor) 
{
    switch (factor)
    {
        case GL_ZERO:                       return BLEND_ZERO;
        case GL_ONE:                        return BLEND_ONE;                            
        case GL_SRC_COLOR:                  return BLEND_SRC_COLOR;                      
        case GL_ONE_MINUS_SRC_COLOR:        return BLEND_ONE_MINUS_SRC_COLOR;            
        case GL_SRC_ALPHA:                  return BLEND_SRC_ALPHA;                      
        case GL_ONE_MINUS_SRC_ALPHA:        return BLEND_ONE_MINUS_SRC_ALPHA;            
        case GL_DST_ALPHA:                  return BLEND_DST_ALPHA;                      
        case GL_ONE_MINUS_DST_ALPHA:        return BLEND_ONE_MINUS_DST_ALPHA;            
        case GL_DST_COLOR:                  return BLEND_DST_COLOR;                      
        case GL_ONE_MINUS_DST_COLOR:        return BLEND_ONE_MINUS_DST_COLOR;            
        case GL_SRC_ALPHA_SATURATE:         return BLEND_SRC_ALPHA_SATURATE;             
        case GL_CONSTANT_COLOR:             return BLEND_CONSTANT_COLOR;                 
        case GL_ONE_MINUS_CONSTANT_COLOR:   return BLEND_ONE_MINUS_CONSTANT_COLOR;       
        case GL_CONSTANT_ALPHA:             return BLEND_CONSTANT_ALPHA;                 
        case GL_ONE_MINUS_CONSTANT_ALPHA:   return BLEND_ONE_MINUS_CONSTANT_ALPHA;       
        default:                            SET_GL_ERROR(GL_INVALID_ENUM); return factor;
    }
}

static GLboolean is_valid_blend_function(GLenum mode)
{
    switch (mode)
    {
        default:
            return GL_FALSE;
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA_SATURATE:             
        case GL_CONSTANT_COLOR:       
        case GL_ONE_MINUS_CONSTANT_COLOR:       
        case GL_CONSTANT_ALPHA:       
        case GL_ONE_MINUS_CONSTANT_ALPHA:
            return GL_TRUE;
    }
}

GL_APICALL void GL_APIENTRY glBlendFunc (GLenum sfactor, GLenum dfactor) 
{
    glBlendFuncSeparate(sfactor, dfactor, sfactor, dfactor);
}

GL_APICALL void GL_APIENTRY glBlendFuncSeparate (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha) 
{
    if (is_valid_blend_function(sfactorRGB) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);

    if (is_valid_blend_function(dfactorRGB) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);
    
    if (is_valid_blend_function(sfactorAlpha) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);
    
    if (is_valid_blend_function(dfactorAlpha) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);
    
    _blend_data.func = (blend_func_t)
    {
        .srcRGB     = sfactorRGB, 
        .srcAlpha   = sfactorAlpha,
        .dstRGB     = dfactorRGB, 
        .dstAlpha   = dfactorAlpha,
    };
}

GL_APICALL void GL_APIENTRY glBufferData (GLenum target, GLsizeiptr size, const void *data, GLenum usage) 
{
    if (target != GL_ARRAY_BUFFER) RETURN_ERROR(GL_INVALID_ENUM);

    if (_buffer_binding == 0) RETURN_ERROR(GL_INVALID_OPERATION);

    if (size == 0) RETURN_ERROR(GL_INVALID_VALUE); // TODO: Is size==0 valid?

    buffer_t* buffer = &_buffers[_buffer_binding-1];
    uint32_t is_inizialized = buffer->size != 0;
    device_shared_objects_t *shared = &device_shared_objects;

    if (is_inizialized) 
    {
        if (buffer->size != size || buffer->usage != usage) RETURN_ERROR(GL_INVALID_OPERATION);
    }
    else 
    {
        buffer->size = size;
        buffer->usage = usage;
        buffer->id = device_create_ro_buffer(shared, size);
        // TODO: Set CL_OUT_OF_RESOURCES if no memory available
    }

    if (data != NULL)
    {
        device_write_on_buffer(shared, buffer->id, 0, size, data);
    }
}

GL_APICALL void GL_APIENTRY glBufferSubData (GLenum target, GLintptr offset, GLsizeiptr size, const void *data) 
{
    if (target != GL_ARRAY_BUFFER) RETURN_ERROR(GL_INVALID_ENUM);

    if (_buffer_binding == 0) RETURN_ERROR(GL_INVALID_OPERATION);

    if (offset < 0 || size < 0 ) RETURN_ERROR(GL_INVALID_VALUE);

    buffer_t *buffer = &_buffers[_buffer_binding-1];

    if (offset + size > buffer->size) RETURN_ERROR(GL_INVALID_VALUE);

    device_write_on_buffer(&device_shared_objects, buffer->id, offset, size, data);
}

typedef struct { uint32_t width, height; } attachment_size_t;

static attachment_size_t get_attachment_size(attachment_t* attachment) 
{
    attachment_size_t size;

    if (attachment->target == GL_RENDERBUFFER) 
    {
        renderbuffer_t *renderbuffer = &_renderbuffers[attachment->binding-1];
        size = (attachment_size_t) {
            .width  = renderbuffer->width,
            .height = renderbuffer->height,
        };
    } 
    else // GL_TEXTURE_2D 
    {
        texture_t *texture = &_textures[attachment->binding-1];
        size = (attachment_size_t) {
            .width  = texture->width,
            .height = texture->height,
        };
    }

    return size;
}

GL_APICALL GLenum GL_APIENTRY glCheckFramebufferStatus (GLenum target) 
{
    if (target != GL_FRAMEBUFFER) RETURN_ERROR_WITH_VALUE(GL_INVALID_ENUM, 0);

    if (_buffer_binding == 0) NOT_IMPLEMENTED;

    framebuffer_t *framebuffer = &_framebuffers[_framebuffer_binding-1];
    
    uint32_t is_color_attached = framebuffer->color_attachment0.binding;
    uint32_t is_depth_attached = framebuffer->depth_attachment.binding;
    uint32_t is_stencil_attached = framebuffer->stencil_attachment.binding;
    
    uint32_t any_attachment = 
        is_color_attached   ||
        is_depth_attached   ||
        is_stencil_attached ;

    if (!any_attachment) return GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;

    attachment_size_t common_size = {0,0};
    attachment_size_t tmp_size = {0,0};

    if (is_color_attached) 
    {
        common_size = get_attachment_size(&framebuffer->color_attachment0);
    }

    if (is_depth_attached) 
    {
        tmp_size = get_attachment_size(&framebuffer->depth_attachment);
        if (common_size.width == 0 && common_size.height == 0) 
        {
            common_size = tmp_size;
        } 
        else if (common_size.width != tmp_size.width || common_size.height != tmp_size.height) 
        {
            return GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS;
        }
    }

    if (is_stencil_attached) 
    {
        tmp_size = get_attachment_size(&framebuffer->stencil_attachment);
        if (common_size.width == 0 && common_size.height == 0) 
        {
            common_size = tmp_size;
        } 
        else if (common_size.width != tmp_size.width || common_size.height != tmp_size.height) 
        {
            return GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS;
        }
    }

    return GL_FRAMEBUFFER_COMPLETE;
}

static uint32_t get_clear_color();
static uint16_t get_clear_depth();
static uint8_t get_clear_stencil();

static cl_ulong get_clear_write_values() {
    cl_ulong clear_write_values;

    cl_ulong clear_color = get_clear_color();
    cl_ulong clear_depth = get_clear_depth();
    cl_ulong clear_stencil = get_clear_stencil();

    clear_write_values = 
        clear_color      <<  0 |
        clear_depth      << 32 |
        clear_stencil    << 48 ;

    return clear_write_values;
};

static cl_ushort get_clear_enabled_data(GLbitfield mask) {
    cl_ushort clear_enabled_data;

    cl_ushort enabled_color = 0;
    cl_ushort enabled_depth = 0;
    cl_ushort enabled_stencil = 0;

    if ((mask & GL_COLOR_BUFFER_BIT) != 0) {
        enabled_color = 
            _masks.color.r ? CLEAR_ENABLED_COLOR_CHANNEL_RED    : 0 |
            _masks.color.g ? CLEAR_ENABLED_COLOR_CHANNEL_GREEN  : 0 |
            _masks.color.b ? CLEAR_ENABLED_COLOR_CHANNEL_BLUE   : 0 |
            _masks.color.a ? CLEAR_ENABLED_COLOR_CHANNEL_ALPHA  : 0 ;
    }
    
    if ((mask & GL_DEPTH_BUFFER_BIT) != 0)
        enabled_depth = _masks.depth ? CLEAR_ENABLED_DEPTH_CHANNEL : 0;

    if ((mask & GL_STENCIL_BUFFER_BIT) != 0)
        enabled_stencil = _masks.stencil.front & 0xFFu;

    clear_enabled_data = enabled_color | enabled_depth | enabled_stencil;

    return clear_enabled_data;
};

/*
void clear_framebuffer() {
    printf("DEBUG: Forced framebuffer clear.\n");

    cl_kernel kernel = _kernels.force_clear;
    cl_command_queue command_queue = getCommandQueue();

    framebuffer_data_t framebuffer_data = get_framebuffer_data();
    
    cl_uint count = 0; 
    setKernelArg(kernel, count++, sizeof(cl_mem), &framebuffer_data.color.mem);
    setKernelArg(kernel, count++, sizeof(cl_mem), &framebuffer_data.depth.mem);
    setKernelArg(kernel, count++, sizeof(cl_mem), &framebuffer_data.stencil.mem);
    #ifndef DEVICE_IMAGE_ENABLED
    cl_uint colorbuffer_type = framebuffer_data.color.internalformat;
    cl_uint buffer_width = framebuffer_data.width;
    setKernelArg(kernel, count++, sizeof(colorbuffer_type), &colorbuffer_type);
    setKernelArg(kernel, count++, sizeof(buffer_width), &buffer_width);
    #endif
    setKernelArg(kernel, count++, sizeof(c_clear_write_values), &c_clear_write_values);
    setKernelArg(kernel, count++, sizeof(c_clear_enabled_data), &c_clear_enabled_data);

    size_t global_work_offset[2], global_work_size[2];
    if (_enableds.scissor_test) {
        global_work_offset[0] = MAX(_scissor_data.left, 0);
        global_work_offset[1] = MAX(_scissor_data.bottom, 0);
        global_work_size[0] = MIN(_scissor_data.width, framebuffer_data.width);
        global_work_size[1] = MIN(_scissor_data.height, framebuffer_data.height);

        if (global_work_offset[0] > global_work_size[0] || global_work_offset[1] > global_work_size[1]) return;
    } else {
        global_work_offset[0] = 0;
        global_work_offset[1] = 0;
        global_work_size[0] = framebuffer_data.width;
        global_work_size[1] = framebuffer_data.height;
    }

    CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, global_work_offset, global_work_size, NULL, 0, NULL, NULL));
} 
*/

GL_APICALL void GL_APIENTRY glClear (GLbitfield mask) 
{
    GLbitfield valid_mask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;

    if (mask & valid_mask == 0) RETURN_ERROR(GL_INVALID_VALUE);
    
    if (_current_program == 0) NOT_IMPLEMENTED;

    device_context_t *context = &contexts[_current_program-1];

    flush_device_context(context);

    device_clear_framebuffer(
        context,
        get_clear_write_values(),
        get_clear_enabled_data(mask),
        1
    );
}

GL_APICALL void GL_APIENTRY glClearColor (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) 
{
    _clear_data.color = (clear_color_t) {
        .red = red,
        .green = green,
        .blue = blue,
        .alpha = alpha
    };
}

GL_APICALL void GL_APIENTRY glClearDepthf (GLfloat d) 
{
    _clear_data.depth = d;
}

GL_APICALL void GL_APIENTRY glClearStencil (GLint s) 
{
    _clear_data.stencil = s;
}

GL_APICALL void GL_APIENTRY glColorMask (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) 
{
    _masks.color = (color_mask_t) {
        .r = red,
        .g = green,
        .b = blue,
        .a = alpha,
    };
}

GL_APICALL void GL_APIENTRY glCompressedTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data)
{
    NOT_IMPLEMENTED;
}

GL_APICALL GLuint GL_APIENTRY glCreateProgram (void)
{
    if (context_sz >= HOST_PROGRAMS_SIZE) RETURN_ERROR_WITH_VALUE(GL_OUT_OF_MEMORY, 0);

    contexts[context_sz] = (device_context_t*) malloc(sizeof(device_context_t));
    
    device_create_context(contexts[context_sz], &device_shared_objects);
    
    context_sz += 1;
    
    return context_sz;
}

GL_APICALL void GL_APIENTRY glCullFace (GLenum mode) 
{
    switch (mode)
    {
        default: 
            RETURN_ERROR(GL_INVALID_ENUM);
        case GL_FRONT:
        case GL_BACK:
        case GL_FRONT_AND_BACK:
    }

    _rasterization_data.cull_face = mode;
}

static inline GLenum gl_depth_function_to_device_depth_function (GLenum func) {
    switch (func)
    {
        case GL_NEVER:      return DEPTH_FUNC_NEVER;
        case GL_LESS:       return DEPTH_FUNC_LESS;
        case GL_EQUAL:      return DEPTH_FUNC_EQUAL;
        case GL_LEQUAL:     return DEPTH_FUNC_LEQUAL;
        case GL_GREATER:    return DEPTH_FUNC_GREATER;
        case GL_NOTEQUAL:   return DEPTH_FUNC_NOTEQUAL;
        case GL_GEQUAL:     return DEPTH_FUNC_GEQUAL;
        case GL_ALWAYS:     return DEPTH_FUNC_ALWAYS;
        default:            SET_GL_ERROR(GL_INVALID_ENUM); return func;
    }
} 

GL_APICALL void GL_APIENTRY glDepthFunc (GLenum func) 
{
    switch (func)
    {
        default:            
            RETURN_ERROR(GL_INVALID_ENUM);
        case GL_NEVER:
        case GL_LESS:
        case GL_EQUAL:
        case GL_LEQUAL:
        case GL_GREATER:
        case GL_NOTEQUAL:
        case GL_GEQUAL:
        case GL_ALWAYS:
    }

    _depth_func = func;
}

GL_APICALL void GL_APIENTRY glDepthMask (GLboolean flag) 
{
    _masks.depth = flag;
}

GL_APICALL void GL_APIENTRY glDepthRangef (GLfloat n, GLfloat f) 
{
    _depth_range = (depth_range_t) 
    {
        .n=n,
        .f=f
    };
}

GL_APICALL void GL_APIENTRY glDisable (GLenum cap) 
{
    GLboolean* enabled_ptr = get_enabled_pointer(cap);

    if (enabled_ptr == NULL) RETURN_ERROR(GL_INVALID_ENUM);

    *enabled_ptr = GL_FALSE;
}

GL_APICALL void GL_APIENTRY glDisableVertexAttribArray (GLuint index) 
{
    if (index >= GL_MAX_VERTEX_ATTRIBS) RETURN_ERROR(GL_INVALID_VALUE);

    vertex_attibute_updated = 1;
    vertex_attribute_datas[index].misc &= ~VERTEX_ATTRIBUTE_ACTIVE_POINTER;
}

#define IS_DRAW_MODE(_MODE) \
    (   \
        _MODE == GL_POINTS          || \
        _MODE == GL_LINES           || \
        _MODE == GL_LINE_LOOP       || \
        _MODE == GL_LINE_STRIP      || \
        _MODE == GL_TRIANGLES       || \
        _MODE == GL_TRIANGLE_STRIP  || \
        _MODE == GL_TRIANGLE_FAN       \
    )

inline uint32_t get_num_primitives(GLenum mode, uint32_t count) {
    switch (mode)
    {
    case GL_POINTS: return count;
    case GL_LINES: return count / 2;
    case GL_LINE_LOOP: return count;
    case GL_LINE_STRIP: return count - 1;
    case GL_TRIANGLES: return count / 3;
    case GL_TRIANGLE_STRIP: return count <= 2 ? 0 : count - 2;
    case GL_TRIANGLE_FAN: return count <= 2 ? 0 : count - 2;
    default: UNDEFINED_BEHAVIOUR; return 0; 
    }
}

inline uint32_t get_num_primitives_assembled(GLenum mode, size_t offset, size_t size) {
    switch (mode)
    {
    case GL_POINTS: return size - offset;
    case GL_LINES: return (size - offset) / 2;
    case GL_LINE_LOOP: return size - offset;
    case GL_LINE_STRIP: return size - offset - 1;
    case GL_TRIANGLES: return (size - offset) / 3;
    case GL_TRIANGLE_STRIP: return size - offset - 2;
    case GL_TRIANGLE_FAN: return size - offset - 2;
    default: UNDEFINED_BEHAVIOUR; return 0;
    }
};


inline int32_t get_primitive_offset(GLenum mode) {
    switch (mode)
    {
    case GL_POINTS:
    case GL_LINES:
    case GL_TRIANGLES: 
        return 0;
    case GL_LINE_LOOP: 
    case GL_LINE_STRIP:
    case GL_TRIANGLE_FAN:
        return -1;
    case GL_TRIANGLE_STRIP:
        return -2;
    default: UNDEFINED_BEHAVIOUR; return 0; 
    }
} 

#define NUM_GROUP 32 // TODO: Check this value
cl_context _context;

// VS data
#define MAX_ELEMENTS_INDICES 256 // max element indices that can be move to server
cl_mem _indices_mem_obj; // size == MAX_
#define VERTEX_OUT_BUFFER_SIZE 256 // bytes
cl_mem _vertex_out_buffer; // size ==
// programs create their own subbuffers required for varying
// programs also creates their displaced _vertex_out_subbuffer
cl_mem _vertex_out_subbuffer; //

#define VS_VARYING_INDEX_0_RESERVED (1 << 0)
#define VS_LAST_AS_FIRST            (1 << 1)



inline uint32_t get_deferred_clear() {
    return c_clear_enabled_data != 0;
}

static inline uint32_t rgba_to_uint32(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
    return 
        ((uint32_t)(r * 0xFF) <<  0) |
        ((uint32_t)(g * 0xFF) <<  8) |
        ((uint32_t)(b * 0xFF) << 16) |
        ((uint32_t)(a * 0xFF) << 24) ;
}

static inline uint32_t get_blending_color() {
    return rgba_to_uint32(
        _blend_data.color.red, 
        _blend_data.color.green, 
        _blend_data.color.blue, 
        _blend_data.color.alpha
    );
}
inline uint32_t get_blending_data() {
    uint32_t blend_data = 0;

    blend_data |= (_blend_data.equation.modeRGB     & 0x3u) << 0;
    blend_data |= (_blend_data.equation.modeAlpha   & 0x3u) << 2;
    blend_data |= (_blend_data.func.srcRGB          & 0xFu) << 16;
    blend_data |= (_blend_data.func.srcAlpha        & 0xFu) << 20;
    blend_data |= (_blend_data.func.dstRGB          & 0xFu) << 24;
    blend_data |= (_blend_data.func.dstAlpha        & 0xFu) << 28;

    return blend_data;
}

static inline uint32_t get_clear_color() 
{
    return rgba_to_uint32(
        _clear_data.color.red, 
        _clear_data.color.green, 
        _clear_data.color.blue, 
        _clear_data.color.alpha
    );
}
inline uint16_t get_clear_depth() {
    return (uint16_t)(_clear_data.depth*0xFFFFu);
}
inline uint8_t  get_clear_stencil() {
    return _clear_data.stencil;
}

inline uint32_t get_depth_data() {
    uint32_t depth_data = 0;

    depth_data |= (_depth_func  &  0xFFFFu) <<  0;
    depth_data |= (_masks.depth &  0xFFFFu) << 16;

    return depth_data;
}

inline uint32_t get_num_tris(GLenum mode, GLsizei count) {
    switch (mode)
    {
        default:
        case GL_TRIANGLES: return count / 3;
        case GL_TRIANGLE_FAN: return count - 2;
        case GL_TRIANGLE_STRIP: return count - 2;
    }
}

inline cl_ushort get_enabled_data(framebuffer_data_t framebuffer) {
    cl_ushort enabled_data;

    cl_ushort enabled_color = 0;
    cl_ushort enabled_depth = 0;
    cl_ushort enabled_stencil = 0;

    enabled_color = 
        (_masks.color.r ? ENABLED_COLOR_CHANNEL_RED    : 0) |
        (_masks.color.g ? ENABLED_COLOR_CHANNEL_GREEN  : 0) |
        (_masks.color.b ? ENABLED_COLOR_CHANNEL_BLUE   : 0) |
        (_masks.color.a ? ENABLED_COLOR_CHANNEL_ALPHA  : 0) ;
    
    if (framebuffer.depth.mem != _rasterization_mem.globals.depthbuffer) 
        enabled_depth = _masks.depth ? ENABLED_DEPTH_CHANNEL : 0;
    
    if (framebuffer.stencil.mem != _rasterization_mem.globals.stencilbuffer) 
        enabled_stencil = _masks.stencil.front & 0xFFu;

    enabled_data = enabled_color | enabled_depth | enabled_stencil;

    printf("ecolor=%x, edepth=%x, estencil=%x\n", enabled_color, enabled_depth, enabled_stencil);

    return enabled_data;
}

inline uint32_t get_stencil_data() {
    uint32_t stencil_data = 0;

    stencil_data |= (_stencil_data.front.function.func      &  0xFu) <<  0;
    stencil_data |= (_stencil_data.front.operation.sfail    &  0xFu) <<  4;
    stencil_data |= (_stencil_data.front.operation.dpass    &  0xFu) <<  8;
    stencil_data |= (_stencil_data.front.operation.dpfail   &  0xFu) << 12;
    stencil_data |= (_stencil_data.front.function.mask      & 0xFFu) << 16;
    stencil_data |= (_stencil_data.front.function.ref       & 0xFFu) << 24;

    return stencil_data;
}

static uint32_t require_flush_context(device_context_t* context) {
    return 0;
}



static render_mode_t get_render_mode() {
    render_mode_t render_mode;

    return render_mode;
}

static rop_config_t get_rop_config(GLenum mode) {
    rop_config_t rop_config;
    
    rop_config = (rop_config_t) {
        .render_mode = get_render_mode(mode),
        .blending_data = get_blending_data(),
        .blending_color = get_blending_color(),
        .stencil_data = get_stencil_data(),
        .depth_data = get_depth_data(),
        .enabled_data = get_enabled_data(),
    };
    
    return rop_config;
}

static size_t update_current_context() 
{
    device_context_t* device_context = contexts[_current_program-1];

    uint32_t new_config_required = rop_config_updated || _uniform_data_updated;

    rop_config_count = (rop_config_count + new_config_required) % TRIANGLE_PRIMITIVE_CONFIGS;
    
    uint32_t flush_required = 
        texture_unit_updated    ||
        (
            new_config_required && 
            device_config_is_full(device_context)
        );

    if (flush_required)
    {
        flush_device_context(device_context);
    }

    if (vertex_attibute_updated)
    {
        device_load_vertex_attributes(device_context, vertex_attributes, vertex_attribute_datas);
        vertex_attibute_updated = 0;
    }

    if (texture_unit_updated) {
        gl_texture_data_t texture_datas[DEVICE_TEXTURE_UNITS];

        for (size_t i = 0; i < DEVICE_TEXTURE_UNITS; ++i)
        {
            texture_t *texture = &_textures[texture_unit_bindings[i]];
            gl_texture_data_t *texture_data = &texture_datas[i];
            sampler2D_t *sampler2D = &texture_data->sampler2D;

            set_sampler2D_internalformat(sampler2D, texture->internalformat);
            set_sampler2D_mag_filter(sampler2D, texture->wraps.mag_filter);
            set_sampler2D_min_filter(sampler2D, texture->wraps.min_filter);
            set_sampler2D_wrap_s(sampler2D, texture->wraps.s);
            set_sampler2D_wrap_t(sampler2D, texture->wraps.t);

            #ifndef DEVICE_IMAGE_ENABLED
            {
                texture_data->height = texture->height;
                texture_data->width = texture->width;
            }
            #endif
        }

        device_load_texture_datas(device_context, texture_datas);
    }

    if (new_config_required)
    {
        device_load_config(device_context, rop_config_count, &rop_config, &_programs[_current_program].uniform_data);
    }

    vertex_attibute_updated = 0;
    texture_unit_updated = 0;
    rop_config_updated = 0;
    _uniform_data_updated = 0;

    return rop_config_count;
}

GL_APICALL void GL_APIENTRY glDrawArrays (GLenum mode, GLint first, GLsizei count) 
{
    if (_current_program == 0) RETURN_ERROR(GL_INVALID_OPERATION);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) RETURN_ERROR(GL_INVALID_FRAMEBUFFER_OPERATION);

    if (first < 0) RETURN_ERROR(GL_INVALID_VALUE);

    if (first != 0) NOT_IMPLEMENTED;

    if (mode == GL_POINTS || mode == GL_LINE_STRIP || mode == GL_LINE_LOOP || mode == GL_LINES) NOT_IMPLEMENTED;

    device_context_t* device_context = contexts[_current_program-1];

    size_t config_id = update_current_context();
    
    device_launch_arrays_triangle_assembly(device_context, config_id, count - first, count);
}

GL_APICALL void GL_APIENTRY glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices)
{
    if (!IS_DRAW_MODE(mode)) RETURN_ERROR(GL_INVALID_ENUM);

    if (count < 0) RETURN_ERROR(GL_INVALID_VALUE);
    
    if (end < start) RETURN_ERROR(GL_INVALID_VALUE);
    
    if (type != GL_UNSIGNED_SHORT) RETURN_ERROR(GL_INVALID_ENUM);

    if (start != 0) NOT_IMPLEMENTED;
    
    if (count == 0) return;

    device_context_t* device_context = contexts[_current_program-1];

    size_t config_id = update_current_context();

    device_launch_range_triangle_assembly(device_context, end - start, count, indices, config_id);
}

GL_APICALL void GL_APIENTRY glEnable (GLenum cap) 
{
    GLboolean* enabled_ptr = get_enabled_pointer(cap);

    if (enabled_ptr == NULL) RETURN_ERROR(GL_INVALID_ENUM);

    *enabled_ptr = GL_TRUE;
}

GL_APICALL void GL_APIENTRY glEnableVertexAttribArray (GLuint index) 
{
    if (index >= DEVICE_VERTEX_ATTRIBUTE_SIZE) RETURN_ERROR(GL_INVALID_VALUE);
    
    vertex_attibute_updated = 1;
    vertex_attribute_datas[index].misc |= VERTEX_ATTRIBUTE_ACTIVE_POINTER;
}

GL_APICALL void GL_APIENTRY glFinish (void) 
{
    glFlush();

    if (_current_program > 0) 
    {
        device_finish(contexts[_current_program-1]);
    }
    else NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glFlush (void) 
{
    if (_current_program > 0) 
    {
        flush_device_context(contexts[_current_program-1]);
    }
    else NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glFramebufferRenderbuffer (GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) 
{
    if (target != GL_FRAMEBUFFER) RETURN_ERROR(GL_INVALID_ENUM);

    if (_framebuffer_binding == 0) RETURN_ERROR(GL_INVALID_OPERATION);

    if (renderbuffertarget != GL_RENDERBUFFER) RETURN_ERROR(GL_INVALID_ENUM);

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

    _framebuffer_updated = 1;

    *attachment_ptr = (attachment_t) {
        .target = GL_RENDERBUFFER,
        .binding = renderbuffer,
    };

}

GL_APICALL void GL_APIENTRY glFramebufferTexture2D (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) 
{
    if (target != GL_FRAMEBUFFER) NOT_IMPLEMENTED;

    if (_framebuffer_binding == 0) RETURN_ERROR(GL_INVALID_OPERATION);

    if (textarget != GL_TEXTURE_2D) RETURN_ERROR(GL_INVALID_ENUM);

    if (level != 0) RETURN_ERROR(GL_INVALID_VALUE);

    framebuffer_t *framebuffer = &_framebuffers[_framebuffer_binding-1];
    
    attachment_t *attachment_ptr;
    switch (attachment)
    {
        case GL_COLOR_ATTACHMENT0:
            attachment_ptr = &framebuffer->color_attachment0;
            break;
        case GL_DEPTH_ATTACHMENT:
            attachment_ptr = &framebuffer->depth_attachment;
            break;
        case GL_STENCIL_ATTACHMENT:
            attachment_ptr = &framebuffer->stencil_attachment;
            break;
        default:
            NOT_IMPLEMENTED;
    }

    _framebuffer_updated = 1;
    
    *attachment_ptr = (attachment_t) 
    {
        .target = textarget,
        .binding = texture,
    };

}

GL_APICALL void GL_APIENTRY glFrontFace (GLenum mode) {
    switch (mode)
    {
    case GL_CCW:
    case GL_CW:
        _rasterization_data.front_face = mode;
        return;
    default:
        RETURN_ERROR(GL_INVALID_ENUM); // TODO: Checkout error
    }
}

GL_APICALL void GL_APIENTRY glGenBuffers (GLsizei n, GLuint *buffers)
{
    for (size_t i=0; i<n && _buffer_sz < HOST_BUFFERS_SIZE; ++i) 
    {
        _buffers[_buffer_sz].size = 0;

        buffers[i] = _buffer_sz + 1;
        
        _buffer_sz += 1;
    }
}

GL_APICALL void GL_APIENTRY glGenerateMipmap (GLenum target) NOT_IMPLEMENTED;

GL_APICALL void GL_APIENTRY glGenFramebuffers (GLsizei n, GLuint *framebuffers) 
{
    for (size_t i=0; i<n && _framebuffer_sz < HOST_FRAMEBUFFER_SIZE; ++i)
    {
        _framebuffers[_framebuffer_sz] = (framebuffer_t) {
            .color_attachment0 = {
                .binding = 0,
                .target = 0,
            },
            .depth_attachment = {
                .binding = 0,
                .target = 0,
            },
            .stencil_attachment = {
                .binding = 0,
                .target = 0,
            },
        };

        framebuffers[i] = _framebuffer_sz + 1; // framebuffer id start at 1
        _framebuffer_sz += 1;
    }
}

GL_APICALL void GL_APIENTRY glGenRenderbuffers (GLsizei n, GLuint *renderbuffers)
{
    for (size_t i=0; i<n && _renderbuffer_sz < HOST_RENDERBUFFERS_SIZE; ++i) 
    {
        renderbuffer_t *renderbuffer = &_renderbuffers[_renderbuffer_sz];

        renderbuffer->width = 0;
        renderbuffer->height = 0;

        renderbuffers[i] = _renderbuffer_sz + 1;
        
        _renderbuffer_sz += 1;
    }
}

GL_APICALL void GL_APIENTRY glGenTextures (GLsizei n, GLuint *textures) 
{
    for (size_t i=0; i<n && _texture_sz < HOST_TEXTURES_SIZE; ++i) 
    {
        _textures[_texture_sz] = (texture_t) {
            .wraps = (texture_wraps_t) {
                .s = TEXTURE_WRAP_REPEAT,
                .t = TEXTURE_WRAP_REPEAT,
                .min_filter = TEXTURE_FILTER_NEAREST,
                .mag_filter = TEXTURE_FILTER_NEAREST,
            },
            .width = 0, 
            .height = 0,
        };

        textures[i] = _texture_sz + 1; // texture id start at 1
        _texture_sz += 1;
    }
}

GL_APICALL GLint GL_APIENTRY glGetAttribLocation (GLuint program, const GLchar *name) 
{
    if (program == 0) RETURN_ERROR(GL_INVALID_OPERATION);

    program_t* program_ptr = &_programs[program-1];

    // TOOD: if (!program_ptr->last_load_attempt) RETURN_ERROR(GL_INVALID_OPERATION);
    
    for(size_t attrib=0; attrib<program_ptr->vertex_attrib_sz; ++attrib) 
    {
        if (strcmp(name, program_ptr->vertex_attrib_arg_datas[attrib].name) == 0) return attrib;
    }

    return -1;
}

GL_APICALL void GL_APIENTRY glGetBooleanv (GLenum pname, GLboolean *data) 
{
    switch (pname)
    {
    case GL_DEPTH_WRITEMASK:
        *data = _masks.depth;
        return;
    default:
        RETURN_ERROR(GL_INVALID_ENUM);
    }
}

GL_APICALL void GL_APIENTRY glGetBufferParameteriv (GLenum target, GLenum pname, GLint *params) {
    NOT_IMPLEMENTED;
}

GL_APICALL GLenum GL_APIENTRY glGetError (void) 
{
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

GL_APICALL GLenum GL_APIENTRY glGetGraphicsResetStatus (void) {
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glGetIntegerv (GLenum pname, GLint *data) 
{
    switch (pname)
    {
    case GL_IMPLEMENTATION_COLOR_READ_FORMAT:
        *data = GL_RGBA;
        return;
    case GL_IMPLEMENTATION_COLOR_READ_TYPE:
        *data = GL_UNSIGNED_SHORT_4_4_4_4;
        return;
    case GL_FRAMEBUFFER_BINDING: 
        *data = _framebuffer_binding; 
        return;
    case GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS:
        *data = DEVICE_VERTEX_TEXTURE_UNITS;
        return;
    case GL_MAX_TEXTURE_IMAGE_UNITS:
        *data = DEVICE_TEXTURE_UNITS;
        return;
    case GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS:
        *data = DEVICE_VERTEX_TEXTURE_UNITS + DEVICE_TEXTURE_UNITS;
        return;
    case GL_DEPTH_FUNC:
        *data = _depth_func;
        return;
    default:
        RETURN_ERROR(GL_INVALID_ENUM);
    }
}

GL_APICALL void GL_APIENTRY glGetProgramiv (GLuint program, GLenum pname, GLint *params) 
{
    if (is_valid_program(program) == GL_FALSE) RETURN_ERROR(GL_INVALID_OPERATION);

    switch (pname)
    {
    case GL_LINK_STATUS:
        *params = _programs[program-1].last_load_attempt;
        break;
    default:
        RETURN_ERROR(GL_INVALID_ENUM);
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

GL_APICALL GLint GL_APIENTRY glGetUniformLocation (GLuint program, const GLchar *name) 
{
    if (is_valid_program(program) == GL_FALSE) RETURN_ERROR_WITH_VALUE(GL_INVALID_OPERATION, -1);

    program_t *program_ptr = &_programs[program-1];

    // TOOD: if (!program_ptr->last_load_attempt) RETURN_ERROR(GL_INVALID_OPERATION);

    for(size_t uniform=0; uniform < program_ptr->uniform_sz; ++uniform) {
        if (strcmp(name, program_ptr->uniform_arg_datas[uniform].name) == 0) return uniform;
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


GL_APICALL void GL_APIENTRY glHint (GLenum target, GLenum mode) 
{
    if (target != GL_GENERATE_MIPMAP_HINT) RETURN_ERROR(GL_INVALID_ENUM);

    switch (mode)
    {
    default:
        RETURN_ERROR(GL_INVALID_ENUM);
    case GL_FASTEST:
    case GL_NICEST:
    case GL_DONT_CARE:
    }

    _hint = mode;
}

GL_APICALL GLboolean GL_APIENTRY glIsEnabled (GLenum cap) 
{
    GLboolean *enabled_ptr = get_enabled_pointer(cap);

    if (enabled_ptr == NULL) RETURN_ERROR_WITH_VALUE(GL_INVALID_ENUM, GL_FALSE);

    return *enabled_ptr;
}

GL_APICALL void GL_APIENTRY glLineWidth (GLfloat width) 
{
    if (width <= (GLfloat) 0) RETURN_ERROR(GL_INVALID_VALUE);

    _rasterization_data.line_width = width;
};

GL_APICALL void GL_APIENTRY glPixelStorei (GLenum pname, GLint param) 
{
    switch (pname)
    {
        case GL_UNPACK_ALIGNMENT:
            switch (param)
            {
                default:
                    RETURN_ERROR(GL_INVALID_VALUE);
                case 1:
                case 2:
                case 4:
                case 8:
                    _pixel_store.unpack_aligment = param;
            }
        default:
            RETURN_ERROR(GL_INVALID_ENUM);
    }
}

GL_APICALL void GL_APIENTRY glPolygonOffset (GLfloat factor, GLfloat units) 
{
    _rasterization_data.polygon_offset = (polygon_offset_t) {
        .factor = factor,
        .units = units
    };
}

GL_APICALL void GL_APIENTRY glProgramBinary (GLuint program, GLenum binaryFormat, const void *binary, GLsizei length)
{
    if (!is_valid_program(program)) RETURN_ERROR(GL_INVALID_OPERATION);

    program_t *program_ptr = &_programs[program-1];
    device_shared_objects_t* shared = &device_shared_objects;

    program_ptr->program_id = device_create_program_from_binary(shared, length, binary);
    program_ptr->uniform_sz = device_get_program_uniform_size(shared, program_ptr->program_id);

    for(size_t i=0; i<program_ptr->uniform_sz; ++i) 
    {
        device_get_program_uniform_arg_data(shared, program_ptr->program_id, i, &program_ptr->uniform_arg_datas[i]);
    }

    program_ptr->vertex_attrib_sz = device_get_program_vertex_attrib_size(shared, program_ptr->program_id);

    for(size_t i=0; i<program_ptr->vertex_attrib_sz; ++i) 
    {
        device_get_program_vertex_attrib_arg_data(shared, program_ptr->program_id, i, &program_ptr->vertex_attrib_arg_datas[i]);
    }
}

static void update_colorbuffer() 
{
    device_context_t *context = contexts[_current_program-1]; 
    
    framebuffer_t *framebuffer = &_framebuffers[_framebuffer_binding-1];
    
    if (framebuffer->color_attachment0.target == GL_RENDERBUFFER) {
        renderbuffer_t *renderbuffer = &_renderbuffers[framebuffer->color_attachment0.binding-1]; 
        
        device_load_renderbuffer_to_colorbuffer(
            context, 
            renderbuffer->id, 
            get_texture_mode_from_internalformat(renderbuffer->internalformat)
        );
    } 
    else
    {
        texture_t *texture = &_textures[framebuffer->color_attachment0.binding-1];
        
        device_load_texture_to_colorbuffer(
            context, 
            texture->id,
            get_texture_mode_from_internalformat(texture->internalformat)
        );
    }
}

static void update_depthbuffer() 
{
    device_context_t *context = contexts[_current_program-1]; 
    
    framebuffer_t *framebuffer = &_framebuffers[_framebuffer_binding-1];

    if (framebuffer->depth_attachment.target == GL_RENDERBUFFER) {
        renderbuffer_t *renderbuffer = &_renderbuffers[framebuffer->depth_attachment.binding-1]; 
        
        device_load_renderbuffer_to_depthbuffer(
            context, 
            renderbuffer->id, 
            get_texture_mode_from_internalformat(renderbuffer->internalformat)
        );
    } 
    else
    {
        texture_t *texture = &_textures[framebuffer->depth_attachment.binding-1];
        
        device_load_texture_to_depthbuffer(
            context, 
            texture->id,
            get_texture_mode_from_internalformat(texture->internalformat)
        );
    }
}

static void update_stencilbuffer()
{
    device_context_t *context = contexts[_current_program-1]; 
    
    framebuffer_t *framebuffer = &_framebuffers[_framebuffer_binding-1];
    
    if (framebuffer->stencil_attachment.target == GL_RENDERBUFFER) {
        renderbuffer_t *renderbuffer = &_renderbuffers[framebuffer->stencil_attachment.binding-1]; 
        
        device_load_renderbuffer_to_stencilbuffer(
            context, 
            renderbuffer->id, 
            get_texture_mode_from_internalformat(renderbuffer->internalformat)
        );
    } 
    else
    {
        texture_t *texture = &_textures[framebuffer->stencil_attachment.binding-1];
        
        device_load_texture_to_stencilbuffer(
            context, 
            texture->id,
            get_texture_mode_from_internalformat(texture->internalformat)
        );
    }
}

static void update_framebuffer() 
{
    device_context_t *context = contexts[_current_program]; 
    
    framebuffer_t *framebuffer = &_framebuffers[_framebuffer_binding];
    
    update_colorbuffer();

    update_depthbuffer();

    update_stencilbuffer();    
}

GL_APICALL void GL_APIENTRY glReadnPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data) 
{
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) RETURN_ERROR(GL_INVALID_FRAMEBUFFER_OPERATION);

    if (!_framebuffer_binding) NOT_IMPLEMENTED; // TODO: Context related operation

    framebuffer_t *framebuffer = &_framebuffers[_framebuffer_binding-1];

    if (framebuffer->color_attachment0.binding == 0) RETURN_ERROR(GL_INVALID_OPERATION);

    device_context_t *context = contexts[_current_program];

    flush_device_context(context);
    if (device_has_triangles_enqueued(context)) 
    {
        device_launch_triangle_rasterization(context);
    } 
    else if (device_has_clear_pending(context))
    {
        device_launch_clear_framebuffer(context);
    }

    device_launch_read_pixels(contexts, 0, 0, width, height, data);

    // TODO: multiple formats and types support
    /*
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
    */
}

static GLboolean is_valid_renderbuffer_internalformat(GLenum internalformat) 
{
    switch (internalformat)
    {
        case GL_DEPTH_COMPONENT16:
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_RGB565:
        case GL_STENCIL_INDEX8:
            return 1;
        default:
            return 0;
    }
}

GL_APICALL void GL_APIENTRY glRenderbufferStorage (GLenum target, GLenum internalformat, GLsizei width, GLsizei height) 
{
    if (target != GL_RENDERBUFFER) RETURN_ERROR(GL_INVALID_ENUM);
    
    if (is_valid_renderbuffer_internalformat(internalformat) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);

    if (width > CR_MAXVIEWPORT_SIZE || height > CR_MAXVIEWPORT_SIZE) RETURN_ERROR(GL_INVALID_VALUE);
    
    if (width == 0 || height == 0) NOT_IMPLEMENTED;

    if (_renderbuffer_binding == 0) RETURN_ERROR(GL_INVALID_OPERATION);

    renderbuffer_t *renderbuffer = &_renderbuffers[_renderbuffer_binding-1];
    
    if (renderbuffer->width != 0 || renderbuffer->height != 0) RETURN_ERROR(GL_INVALID_OPERATION); 

    renderbuffer->id = device_create_renderbuffer(&device_shared_objects, internalformat, width, height);
    renderbuffer->width = width;
    renderbuffer->height = height;
}

GL_APICALL void GL_APIENTRY glSampleCoverage (GLfloat value, GLboolean invert) 
{
    // TODO for multisampling
    NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glScissor (GLint x, GLint y, GLsizei width, GLsizei height) 
{
    if (width < 0 || height < 0) RETURN_ERROR(GL_INVALID_VALUE);
    
    _scissor_data = (scissor_data_t) {
        .left = x,
        .bottom = y,
        .width = width,
        .height = height,
    };
}

/**
 * @post: if function does not exist GL_INVALID_ENUM is set up
 */
inline GLenum gl_function_to_stencil_function(GLenum func) {
    switch (func)
    {
        case GL_NEVER:     return STENCIL_FUNC_NEVER;
        case GL_LESS:      return STENCIL_FUNC_LESS;
        case GL_EQUAL:     return STENCIL_FUNC_EQUAL;
        case GL_LEQUAL:    return STENCIL_FUNC_LEQUAL;
        case GL_GREATER:   return STENCIL_FUNC_GREATER;
        case GL_NOTEQUAL:  return STENCIL_FUNC_NOTEQUAL;
        case GL_GEQUAL:    return STENCIL_FUNC_GEQUAL;
        case GL_ALWAYS:    return STENCIL_FUNC_ALWAYS;
    }
    SET_GL_ERROR(GL_INVALID_ENUM);
    return -1;
}

static void set_stencil_function(stencil_function_t *stencil_function, GLenum func, GLint ref, GLuint mask)
{
    *stencil_function = (stencil_function_t) 
    {
        .func = func,
        .ref = ref,
        .mask = mask,
    };
}

static GLboolean is_valid_face(GLenum face) 
{
    switch (face)
    {
        default: 
            return GL_FALSE;
        case GL_FRONT:
        case GL_BACK:
        case GL_FRONT_AND_BACK:
    }

    return GL_TRUE;
}

GL_APICALL void GL_APIENTRY glStencilFunc (GLenum func, GLint ref, GLuint mask) 
{
    glStencilFuncSeparate (GL_FRONT_AND_BACK, func, ref, mask);
}

GL_APICALL void GL_APIENTRY glStencilFuncSeparate (GLenum face, GLenum func, GLint ref, GLuint mask) 
{
    if (is_valid_face(face) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);

    switch (func)
    {
        default: RETURN_ERROR(GL_INVALID_ENUM);
        case GL_NEVER:
        case GL_LESS:
        case GL_EQUAL:
        case GL_LEQUAL:
        case GL_GREATER:
        case GL_NOTEQUAL:
        case GL_GEQUAL:
        case GL_ALWAYS:
    }

    switch (face)
    {
        case GL_FRONT:
        case GL_FRONT_AND_BACK:
            set_stencil_function(&_stencil_data.front.function, func, ref, mask);
        default:
    }

    switch (face)
    {
        case GL_BACK:
        case GL_FRONT_AND_BACK:
            set_stencil_function(&_stencil_data.back.function, func, ref, mask);
        default:
    }
}

GL_APICALL void GL_APIENTRY glStencilMask (GLuint mask) 
{
    glStencilMaskSeparate(GL_FRONT_AND_BACK, mask);
}

GL_APICALL void GL_APIENTRY glStencilMaskSeparate (GLenum face, GLuint mask) 
{
    if (is_valid_face(face) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);

    switch (face)
    {
        case GL_FRONT:
        case GL_FRONT_AND_BACK:
            _masks.stencil.front = mask;
        default:
    }

    switch (face)
    {
        case GL_BACK:
        case GL_FRONT_AND_BACK:
            _masks.stencil.back = mask;
        default:
    }
}

/**
 * @post: if function does not exist GL_INVALID_ENUM is set up
 */
inline GLenum gl_operation_to_stencil_operation(GLenum operation) {
    switch (operation)
    {
        case GL_KEEP:       return STENCIL_OP_KEEP;
        case GL_ZERO:       return STENCIL_OP_ZERO;
        case GL_REPLACE:    return STENCIL_OP_REPLACE;
        case GL_INCR:       return STENCIL_OP_INCR;
        case GL_DECR:       return STENCIL_OP_DECR;
        case GL_INVERT:     return STENCIL_OP_INVERT;
        case GL_INCR_WRAP:  return STENCIL_OP_INCR_WRAP;
        case GL_DECR_WRAP:  return STENCIL_OP_DECR_WRAP;
    }
    SET_GL_ERROR(GL_INVALID_ENUM);
    return -1;
} 

static void set_stencil_operation(stencil_operation_t *stencil_operation, GLenum sfail, GLenum dfail, GLenum dpass)
{
    *stencil_operation = (stencil_operation_t) 
    {
        .sfail = sfail,
        .dpfail = dfail,
        .dpass = dpass
    };
}

static GLboolean is_valid_stencil_operation(GLenum operation)
{
    switch (operation)
    {
        default:
            return GL_FALSE;
        case GL_KEEP:
        case GL_ZERO:
        case GL_REPLACE:
        case GL_INCR:
        case GL_DECR:
        case GL_INVERT:
        case GL_INCR_WRAP:
        case GL_DECR_WRAP:
    }

    return GL_TRUE;
}

GL_APICALL void GL_APIENTRY glStencilOp (GLenum fail, GLenum zfail, GLenum zpass)
{
    glStencilOpSeparate(GL_FRONT_AND_BACK, fail, zfail, zpass);
}


GL_APICALL void GL_APIENTRY glStencilOpSeparate (GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass) 
{
    if (is_valid_face(face) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);

    if (is_valid_stencil_operation(sfail) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);
    
    if (is_valid_stencil_operation(dpfail) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);

    if (is_valid_stencil_operation(dppass) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM);

    switch (face)
    {
        case GL_FRONT:
        case GL_FRONT_AND_BACK:
            set_stencil_operation(&_stencil_data.front.operation, sfail, dpfail, dppass);
        default:
    }

    switch (face)
    {
        case GL_BACK:
        case GL_FRONT_AND_BACK:
            set_stencil_operation(&_stencil_data.back.operation, sfail, dpfail, dppass);
        default:
    }
}

#define max(a, b) (a >= b ? a : b)
#define IS_POWER_OF_2(a) !(a & 0x1u) 

static int get_texture_mode_from_internalformat(GLenum internalformat) 
{
    switch (internalformat) {
    case GL_RGBA8:
        return TEX_RGBA8;
    case GL_RGB8:
        return TEX_RGB8;
    case GL_RG8:
        return TEX_RG8;
    case GL_RGBA4:
        return TEX_RGBA4;
    case GL_RGB5_A1:
        return TEX_RGB5_A1;
    case GL_RGB565:
        return TEX_RGB565;
    case GL_R8:
        return TEX_R8;
    default:
        SET_GL_ERROR(GL_INVALID_ENUM);
        return -1;
    }
}

GL_APICALL void GL_APIENTRY glTexStorage2D (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) 
{
	if (target != GL_TEXTURE_2D) RETURN_ERROR(GL_INVALID_ENUM);

    if (!_texture_binding) RETURN_ERROR(GL_INVALID_OPERATION);
    
    if (width < 1 || height < 1 || levels < 1) RETURN_ERROR(GL_INVALID_VALUE);
    
    if (levels > (int) log2f(max(width,height)) + 1) RETURN_ERROR(GL_INVALID_OPERATION);
    
    if (levels != 1 && (IS_POWER_OF_2(width) || IS_POWER_OF_2(height))) RETURN_ERROR(GL_INVALID_OPERATION);
    
    texture_t* texture = &_textures[_texture_binding-1];

    if (texture->width || texture->height) RETURN_ERROR(GL_INVALID_OPERATION); 
    
    if (levels != 1) NOT_IMPLEMENTED; // Mipmaps not supported yet

    int texture_mode = get_texture_mode_from_internalformat(internalformat);
    texture->id = device_create_2d_texture(contexts[_current_program], width, height, texture_mode);
    texture->width = width;
    texture->height = height;
    texture->internalformat = internalformat;
    texture->levels = levels;
}

GL_APICALL void GL_APIENTRY glTexParameterf (GLenum target, GLenum pname, GLfloat param) 
{
    RETURN_ERROR(GL_INVALID_ENUM);
}

GL_APICALL void GL_APIENTRY glTexParameterfv (GLenum target, GLenum pname, const GLfloat *params)
{
    RETURN_ERROR(GL_INVALID_ENUM);
}

static GLboolean gl_tex_parameter_to_tex_wrap(GLint param, uint32_t* result) {
    switch (param)
    {
    case GL_REPEAT: 
        *result = TEXTURE_WRAP_REPEAT;
        break;
    case GL_CLAMP_TO_EDGE: 
        *result = TEXTURE_WRAP_CLAMP_TO_EDGE;
        break;
    case GL_MIRRORED_REPEAT: 
        *result = TEXTURE_WRAP_MIRRORED_REPEAT;
        break;
    default: return GL_FALSE;
    }
    return GL_TRUE;
}

static GLboolean is_tex_parameter_valid(GLenum pname, GLint param)
{
    switch (pname)
    {
        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T:
            switch (param)
            {
                default: return GL_FALSE;
                case GL_CLAMP_TO_EDGE:
                case GL_REPEAT:
                case GL_MIRRORED_REPEAT:
            }
            break;
        case GL_TEXTURE_MAG_FILTER:
            switch (param)
            {
                default: return GL_FALSE;
                case GL_NEAREST:
                case GL_LINEAR:
            }
            break;
        case GL_TEXTURE_MIN_FILTER:
            switch (param)
            {
                default: return GL_FALSE;
                case GL_NEAREST:
                case GL_LINEAR:
                case GL_NEAREST_MIPMAP_NEAREST:
                case GL_NEAREST_MIPMAP_LINEAR:
                case GL_LINEAR_MIPMAP_NEAREST:
                case GL_LINEAR_MIPMAP_LINEAR:
            }
            break;
        default: return GL_FALSE;
    }

    return GL_TRUE;
}

static GLint* get_tex_parameter_pointer(GLenum pname) 
{
    texture_t *texture = &_textures[_texture_binding-1];

    switch (pname)
    {
        case GL_TEXTURE_WRAP_S:
            return &texture->wraps.s;        
        case GL_TEXTURE_WRAP_T:
            return &texture->wraps.t; 
        case GL_TEXTURE_MIN_FILTER:
            return &texture->wraps.min_filter; 
        case GL_TEXTURE_MAG_FILTER:
            return &texture->wraps.mag_filter; 
        default:
            return NULL;
    }
}

GL_APICALL void GL_APIENTRY glTexParameteri (GLenum target, GLenum pname, GLint param) 
{
    if (target != GL_TEXTURE_2D) RETURN_ERROR(GL_INVALID_ENUM);

    if (is_tex_parameter_valid(pname, param) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM); 

    if (!_texture_binding) RETURN_ERROR(GL_INVALID_OPERATION);
    
    GLint* tex_parameter = get_tex_parameter_pointer(pname);

    *tex_parameter = param;
}

GL_APICALL void GL_APIENTRY glTexParameteriv (GLenum target, GLenum pname, const GLint *params) 
{
    if (target != GL_TEXTURE_2D) RETURN_ERROR(GL_INVALID_ENUM);

    if (is_tex_parameter_valid(pname, *params) == GL_FALSE) RETURN_ERROR(GL_INVALID_ENUM); 

    if (!_texture_binding) RETURN_ERROR(GL_INVALID_OPERATION);
    
    GLint* tex_parameter = get_tex_parameter_pointer(pname);

    *tex_parameter = *params;
}

GL_APICALL void GL_APIENTRY glTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) 
{
    if (target != GL_TEXTURE_2D) RETURN_ERROR(GL_INVALID_ENUM);
    
    if (!_texture_binding) RETURN_ERROR(GL_INVALID_OPERATION);

    texture_t *texture = &_textures[_texture_binding-1];
    
    if (level < 0 || level > (int) log2f(max(texture->width,texture->height)))
        RETURN_ERROR(GL_INVALID_VALUE);

    if (xoffset < 0 || yoffset < 0 || xoffset + width > texture->width || yoffset + height > texture->height)
        RETURN_ERROR(GL_INVALID_VALUE);

    if (level != 0) NOT_IMPLEMENTED; // Mipmaps not supported yet

    if (xoffset != 0 || yoffset != 0) NOT_IMPLEMENTED; // Subtextures not supported yet
    
    if (texture->internalformat == GL_RGBA8 && format == GL_RGBA && type == GL_UNSIGNED_BYTE ) NOT_IMPLEMENTED; // Only RGBA8 supported yet

    int texture_mode = get_texture_mode_from_internalformat(texture->internalformat);

    device_write_2d_texture(contexts[_current_program], texture->id, width, height, texture_mode, pixels);
}

static void set_uniform_data(GLint location, size_t size, void* data)
{
    program_t *program = &_programs[_current_program-1];

    memcpy(program->uniform_data + program->uniform_arg_datas[location].offset, data, size);
}

static GLboolean is_correct_type(arg_data_t* arg_data, size_t size, GLenum type)
{
    return arg_data->type == type && arg_data->size == size ? GL_TRUE : GL_FALSE;
}

static GLboolean is_valid_uniform_data(GLint location, size_t size, GLenum type)
{
    if (_current_program == 0) return GL_FALSE;

    program_t *program = &_programs[_current_program-1];

    if (location >= program->uniform_sz) return GL_FALSE;

    arg_data_t *arg_data = &program->uniform_arg_datas[location];

    if (is_correct_type(arg_data, 1, GL_FLOAT) == GL_FALSE) return GL_FALSE;

    return GL_TRUE;
}

#define GENERIC_UNIFORM(_SIZE, _TYPE, _LOCATION, ...) \
{ \
    if (is_valid_uniform_data(_LOCATION, _SIZE, _TYPE) == GL_FALSE) RETURN_ERROR(GL_INVALID_OPERATION); \
    GLfloat array[] = {__VA_ARGS__}; \
    set_uniform_data(location, sizeof(array), array); \
}

#define GENERIC_UNIFORM_V(_SIZE, _TYPE, _LOCATION, _ARRAY) \
{ \
    if (is_valid_uniform_data(_LOCATION, _SIZE, _TYPE) == GL_FALSE) RETURN_ERROR(GL_INVALID_OPERATION); \
    set_uniform_data(location, sizeof(_ARRAY[0])*_SIZE, _ARRAY); \
}

GL_APICALL void GL_APIENTRY glUniform1f (GLint location, GLfloat v0)
{
    GENERIC_UNIFORM(1, GL_FLOAT, location, v0);
}

GL_APICALL void GL_APIENTRY glUniform1fv (GLint location, GLsizei count, const GLfloat *value) 
{
    GENERIC_UNIFORM_V(1, GL_FLOAT, location, value);
}

GL_APICALL void GL_APIENTRY glUniform1i (GLint location, GLint v0)
{
    GENERIC_UNIFORM(1, GL_INT, location, v0);
}

GL_APICALL void GL_APIENTRY glUniform1iv (GLint location, GLsizei count, const GLint *value) 
{
    GENERIC_UNIFORM_V(1, GL_INT, location, value);
}

GL_APICALL void GL_APIENTRY glUniform2f (GLint location, GLfloat v0, GLfloat v1)
{
    GENERIC_UNIFORM(2, GL_FLOAT, location, v0, v1);
}

GL_APICALL void GL_APIENTRY glUniform2fv (GLint location, GLsizei count, const GLfloat *value)
{
    GENERIC_UNIFORM_V(2, GL_FLOAT, location, value);
}

GL_APICALL void GL_APIENTRY glUniform2i (GLint location, GLint v0, GLint v1)
{
    GENERIC_UNIFORM(2, GL_INT, location, v0, v1);
}

GL_APICALL void GL_APIENTRY glUniform2iv (GLint location, GLsizei count, const GLint *value)
{
    GENERIC_UNIFORM_V(2, GL_INT, location, value);
}

GL_APICALL void GL_APIENTRY glUniform3f (GLint location, GLfloat v0, GLfloat v1, GLfloat v2)
{
    GENERIC_UNIFORM(3, GL_FLOAT, location, v0, v1, v2);
}

GL_APICALL void GL_APIENTRY glUniform3fv (GLint location, GLsizei count, const GLfloat *value)
{
    GENERIC_UNIFORM_V(3, GL_FLOAT, location, value);
}

GL_APICALL void GL_APIENTRY glUniform3i (GLint location, GLint v0, GLint v1, GLint v2)
{
    GENERIC_UNIFORM(3, GL_INT, location, v0, v1, v2);
}

GL_APICALL void GL_APIENTRY glUniform3iv (GLint location, GLsizei count, const GLint *value)
{
    GENERIC_UNIFORM_V(3, GL_INT, location, value);
}

GL_APICALL void GL_APIENTRY glUniform4f (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)
{
    GENERIC_UNIFORM(4, GL_FLOAT, location, v0, v1, v2, v3);
}

GL_APICALL void GL_APIENTRY glUniform4fv (GLint location, GLsizei count, const GLfloat *value) 
{
    GENERIC_UNIFORM_V(4, GL_FLOAT, location, value);
}

GL_APICALL void GL_APIENTRY glUniform4i (GLint location, GLint v0, GLint v1, GLint v2, GLint v3)
{
    GENERIC_UNIFORM(4, GL_INT, location, v0, v1, v2, v3);
}

GL_APICALL void GL_APIENTRY glUniform4iv (GLint location, GLsizei count, const GLint *value)
{
    GENERIC_UNIFORM_V(4, GL_INT, location, value);
}

GL_APICALL void GL_APIENTRY glUniformMatrix2fv (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    if (transpose != GL_FALSE) RETURN_ERROR(GL_INVALID_VALUE);

    GENERIC_UNIFORM_V(8, GL_FLOAT, location, value);
}

GL_APICALL void GL_APIENTRY glUniformMatrix3fv (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    if (transpose != GL_FALSE) RETURN_ERROR(GL_INVALID_VALUE);

    GENERIC_UNIFORM_V(16, GL_FLOAT, location, value);
}
GL_APICALL void GL_APIENTRY glUniformMatrix4fv (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    if (transpose != GL_FALSE) RETURN_ERROR(GL_INVALID_VALUE);

    GENERIC_UNIFORM_V(16, GL_FLOAT, location, value);
}

GL_APICALL void GL_APIENTRY glUseProgram (GLuint program)
{
    if (program == _current_program) return; // No change

    if (program == 0) 
    {
        flush_device_context(contexts[_current_program-1]);

        _current_program = 0;

        return;
    }
        
    if (is_valid_program(program) == GL_FALSE) RETURN_ERROR(GL_INVALID_OPERATION);

    if (_current_program != 0) 
    {
        device_context_t *current_context = contexts[_current_program-1]; 

        flush_device_context(current_context);
        device_link_contexts(current_context, contexts[program-1]);
    }

    _current_program=program;
}

static inline void set_vertex_attribute(GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w) 
{
    if (index >= DEVICE_VERTEX_ATTRIBUTE_SIZE) RETURN_ERROR(GL_INVALID_VALUE);

    vertex_attibute_updated = 1;
    vertex_attributes[index] = (cl_float4){x, y, z, w};
}

GL_APICALL void GL_APIENTRY glVertexAttrib1f (GLuint index, GLfloat x) 
{
    set_vertex_attribute(index, x, 0.f, 0.f, 1.f);
}

GL_APICALL void GL_APIENTRY glVertexAttrib1fv (GLuint index, const GLfloat *v) 
{
    set_vertex_attribute(index, v[0], 0.f, 0.f, 1.f);
}

GL_APICALL void GL_APIENTRY glVertexAttrib2f (GLuint index, GLfloat x, GLfloat y) 
{
    set_vertex_attribute(index, x, y, 0.f, 1.f);
}

GL_APICALL void GL_APIENTRY glVertexAttrib2fv (GLuint index, const GLfloat *v) 
{
    set_vertex_attribute(index, v[0], v[1], 0.f, 1.f);
}

GL_APICALL void GL_APIENTRY glVertexAttrib3f (GLuint index, GLfloat x, GLfloat y, GLfloat z) 
{
    set_vertex_attribute(index, x, y, z, 1.f);
}

GL_APICALL void GL_APIENTRY glVertexAttrib3fv (GLuint index, const GLfloat *v) 
{
    set_vertex_attribute(index, v[0], v[1], v[2], 1.f);
}

GL_APICALL void GL_APIENTRY glVertexAttrib4f (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w) 
{
    set_vertex_attribute(index, x, y, z, w);
}

GL_APICALL void GL_APIENTRY glVertexAttrib4fv (GLuint index, const GLfloat *v) 
{
    set_vertex_attribute(index, v[0], v[1], v[2], v[3]);
}

GLboolean gl_type_to_vertex_attribute_type(GLenum gl_type, unsigned int* va_type) 
{
    #define CASE_GL_TO_VERTEX_ATTRIBUTE_TYPE(type) \
        case GL_##type: *va_type = VERTEX_ATTRIBUTE_TYPE_##type; break;

    switch (gl_type)
    {
        CASE_GL_TO_VERTEX_ATTRIBUTE_TYPE(BYTE);
        CASE_GL_TO_VERTEX_ATTRIBUTE_TYPE(UNSIGNED_BYTE);
        CASE_GL_TO_VERTEX_ATTRIBUTE_TYPE(SHORT);
        CASE_GL_TO_VERTEX_ATTRIBUTE_TYPE(UNSIGNED_SHORT);
        CASE_GL_TO_VERTEX_ATTRIBUTE_TYPE(FLOAT);
        default:
            return GL_FALSE;
    }

    #undef CASE_GL_TO_VERTEX_ATTRIBUTE_TYPE

    return GL_TRUE;
};

GLboolean gl_size_to_vertex_attribute_size(GLint gl_size, unsigned int* va_size)
{
    #define CASE_GL_TO_VERTEX_ATTRIBUTE_SIZE(size) \
        case size: *va_size = VERTEX_ATTRIBUTE_SIZE_##size << 3; break;

    switch (gl_size)
    {
        CASE_GL_TO_VERTEX_ATTRIBUTE_SIZE(1);
        CASE_GL_TO_VERTEX_ATTRIBUTE_SIZE(2);
        CASE_GL_TO_VERTEX_ATTRIBUTE_SIZE(3);
        CASE_GL_TO_VERTEX_ATTRIBUTE_SIZE(4);
        default:
            return GL_FALSE;
    }

    #undef CASE_GL_TO_VERTEX_ATTRIBUTE_SIZE

    return GL_TRUE;
}

GL_APICALL void GL_APIENTRY glVertexAttribPointer (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) 
{
    unsigned int va_size, va_type, va_active_pointer;

    if (index >= DEVICE_VERTEX_ATTRIBUTE_SIZE) {
        SET_GL_ERROR(GL_INVALID_VALUE);
        return;
    }
    
    if (gl_size_to_vertex_attribute_size(size, &va_size) == GL_FALSE) {
        SET_GL_ERROR(GL_INVALID_VALUE);
        return;
    }

    if (gl_type_to_vertex_attribute_type(type, &va_type) == GL_FALSE) {
        SET_GL_ERROR(GL_INVALID_ENUM);
        return;
    }

    if (stride < 0) RETURN_ERROR(GL_INVALID_VALUE);
    
    if (stride == 0) 
    {
        stride = sizeof_vertex_attribute_type(type)*size;
    }

    va_active_pointer = vertex_attribute_datas[index].misc & VERTEX_ATTRIBUTE_ACTIVE_POINTER;

    uint32_t is_buffer_bound = _buffer_binding;

    vertex_attibute_updated = 1;
    vertex_attribute_datas[index] = (vertex_attribute_data_t) {
        .stride = stride,
        .offset = is_buffer_bound ? (unsigned int) pointer : 0,
        .misc = 
            va_type |
            va_size |
            va_active_pointer,
    };
    vertex_attribute_binding[index] = (vertex_attribute_binding_t) {
        .binding = _buffer_binding,
        .pointer = pointer,
    };
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

// impl utility functions

static void flush_device_context(device_context_t* context)
{
    if (device_has_triangles_enqueued(context)) 
    {
        device_launch_triangle_rasterization(context);
    } 
    else if (device_has_clear_pending(context))
    {
        device_launch_clear_framebuffer(context);
    }
}

static uint32_t is_valid_program(GLuint program)
{
    return program < _program_sz && program != 0;
}

static GLboolean* get_enabled_pointer(GLenum cap)
{
    switch (cap)
    {
    case GL_SCISSOR_TEST:
        return &_enableds.scissor_test;
    case GL_STENCIL_TEST:
        return &_enableds.stencil_test;
    case GL_DEPTH_TEST:
        return &_enableds.depth_test;
    case GL_BLEND:
        return &_enableds.blend;
    case GL_DITHER:
        return &_enableds.dither;
    case GL_CULL_FACE:
        return &_enableds.cull_face;
    default:
        return NULL;
    }
}

static render_mode_t get_render_mode_flags(GLenum mode)
{
    render_mode_t render_mode_flags;

    render_mode_flags.flags = RENDER_MODE_FLAG_ENABLE_LERP; // always lerp enabled

    if (_enableds.depth_test) render_mode_flags.flags |= RENDER_MODE_FLAG_ENABLE_DEPTH;

    if (_enableds.stencil_test) render_mode_flags.flags |= RENDER_MODE_FLAG_ENABLE_STENCIL;

    if (_enableds.blend) render_mode_flags.flags |= RENDER_MODE_FLAG_ENABLE_BLENDER;

    if (_enableds.cull_face) 
    {
        switch (_rasterization_data.cull_face)
        {
            case GL_FRONT:
            case GL_FRONT_AND_BACK:
                render_mode_flags.flags |= RENDER_MODE_FLAG_ENABLE_CULL_FRONT;
            default:
        }

        switch (_rasterization_data.cull_face)
        {
            case GL_BACK:
            case GL_FRONT_AND_BACK:
                render_mode_flags.flags |= RENDER_MODE_FLAG_ENABLE_CULL_BACK;
            default:
        }
    }
    
    switch (mode)
    {
        case GL_TRIANGLE_FAN:
            render_mode_flags.flags |= RENDER_MODE_FLAG_TRIANGLE_FAN;
            break;
        case GL_TRIANGLE_STRIP:
            render_mode_flags.flags |= RENDER_MODE_FLAG_TRIANGLE_STRIP;
            break;
        default:
    }

    return render_mode_flags;
}

static size_t sizeof_vertex_attribute_type(GLenum type)
{
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
        default:
            return 0;
    }
}
