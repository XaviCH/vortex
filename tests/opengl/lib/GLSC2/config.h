#ifndef __config_h__
#define __config_h__ 1

#include <CL/opencl.h>
#include <GLSC2/glsc2.h>

// Extern compiled kernels
#include "kernels/kernel.color.c"
#include "kernels/kernel.scissor_test.c"
#include "kernels/kernel.blending.c"
#include "kernels/kernel.dither.c"
#include "kernels/kernel.clear.c"
#include "kernels/kernel.depth.c"
#include "kernels/kernel.stencil_test.c"
#include "kernels/kernel.perspective_division.c"
#include "kernels/kernel.viewport_division.c"
#include "kernels/kernel.rasterization.c"
#include "kernels/kernel.rasterization.triangle_fan.c"
#include "kernels/kernel.rasterization.triangle_strip.c"
#include "kernels/kernel.readnpixels.c"
#include "kernels/kernel.strided_write.c"


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
#define FRAGMENT_SHADER_FNAME "main_fs"
#define PERSPECTIVE_DIVISION_SHADER_FNAME "gl_perspective_division"
#define VIEWPORT_DIVISION_SHADER_FNAME "gl_viewport_division"

#define POCL_BINARY 0x0
#define SAMPLER2D_T GL_FLOAT+1
#define IMAGE_T SAMPLER2D_T+1

// OpenGL required definitions
#define MAX_VERTEX_ATTRIBS 16
#define MAX_VERTEX_UNIFORM_VECTORS sizeof(float[4])         // atMost MAX_UNIFORM_VECTORS
#define MAX_FRAGMENT_UNIFORM_VECTORS sizeof(float[4])       // atMost MAX_UNIFORM_VECTORS
#define MAX_RENDERBUFFER_SIZE sizeof(uint16_t[1920][1080])  // TODO: Maybe another number
#define MAX_COMBINED_TEXTURE_IMAGE_UNITS 16

#define GL_POSITION "gl_Position"
#define GL_FRAGCOLOR "gl_FragColor"
#define GL_FRAGCOORD "gl_FragCoord"
#define GL_DISCARD "gl_Discard"

#define VEC4 0
#define POINTER 1
#define BUFFEROBJECT 2

#define PROGRAM_LOG_SIZE 256
#define ARG_NAME_SIZE 128
#define MAX_VARYING 16

#endif