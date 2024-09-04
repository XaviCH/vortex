#ifndef __config_h__
#define __config_h__ 1

#include <CL/opencl.h>
#include <GLSC2/glsc2.h>

#ifndef C_OPENCL_HOST
#ifndef C_OPENCL_VORTEX
#define C_OPENCL_VORTEX
#endif
#endif

// Extern compiled kernels
#ifdef C_OPENCL_VORTEX
#include "kernels/kernel.color.pocl.c"
#include "kernels/kernel.scissor_test.pocl.c"
#include "kernels/kernel.blending.pocl.c"
#include "kernels/kernel.dither.pocl.c"
#include "kernels/kernel.clear.pocl.c"
#include "kernels/kernel.depth.pocl.c"
#include "kernels/kernel.stencil_test.pocl.c"
#include "kernels/kernel.perspective_division.pocl.c"
#include "kernels/kernel.viewport_division.pocl.c"
#include "kernels/kernel.rasterization.pocl.c"
#include "kernels/kernel.rasterization.triangle_fan.pocl.c"
#include "kernels/kernel.rasterization.triangle_strip.pocl.c"
#include "kernels/kernel.readnpixels.pocl.c"
#include "kernels/kernel.strided_write.pocl.c"

#define KERNEL_DEPTH_BIN                            kernel_depth_pocl
#define KERNEL_STENCIL_TEST_BIN                     kernel_stencil_test_pocl
#define KERNEL_SCISSOR_TEST_BIN                     kernel_scissor_test_pocl
#define KERNEL_BLENDING_BIN                         kernel_blending_pocl
#define KERNEL_DITHER_BIN                           kernel_dither_pocl
#define KERNEL_RASTERIZATION_BIN                    kernel_rasterization_pocl
#define KERNEL_RASTERIZATION_TRIANGLE_FAN_BIN       kernel_rasterization_triangle_fan_pocl
#define KERNEL_RASTERIZATION_TRIANGLE_STRIP_BIN     kernel_rasterization_triangle_strip_pocl
#define KERNEL_VIEWPORT_DIVISION_BIN                kernel_viewport_division_pocl
#define KERNEL_PERSPECTIVE_DIVISION_BIN             kernel_perspective_division_pocl
#define KERNEL_READNPIXELS_BIN                      kernel_readnpixels_pocl
#define KERNEL_STRIDED_WRITE_BIN                    kernel_strided_write_pocl
#define KERNEL_CLEAR_BIN                            kernel_clear_pocl
#endif

#ifdef C_OPENCL_HOST
#include "kernels/kernel.color.ocl.c"
#include "kernels/kernel.scissor_test.ocl.c"
#include "kernels/kernel.blending.ocl.c"
#include "kernels/kernel.dither.ocl.c"
#include "kernels/kernel.clear.ocl.c"
#include "kernels/kernel.depth.ocl.c"
#include "kernels/kernel.stencil_test.ocl.c"
#include "kernels/kernel.perspective_division.ocl.c"
#include "kernels/kernel.viewport_division.ocl.c"
#include "kernels/kernel.rasterization.ocl.c"
#include "kernels/kernel.rasterization.triangle_fan.ocl.c"
#include "kernels/kernel.rasterization.triangle_strip.ocl.c"
#include "kernels/kernel.readnpixels.ocl.c"
#include "kernels/kernel.strided_write.ocl.c"

#define KERNEL_DEPTH_BIN                            kernel_depth_ocl
#define KERNEL_STENCIL_TEST_BIN                     kernel_stencil_test_ocl
#define KERNEL_SCISSOR_TEST_BIN                     kernel_scissor_test_ocl
#define KERNEL_BLENDING_BIN                         kernel_blending_ocl
#define KERNEL_DITHER_BIN                           kernel_dither_ocl
#define KERNEL_RASTERIZATION_BIN                    kernel_rasterization_ocl
#define KERNEL_RASTERIZATION_TRIANGLE_FAN_BIN       kernel_rasterization_triangle_fan_ocl
#define KERNEL_RASTERIZATION_TRIANGLE_STRIP_BIN     kernel_rasterization_triangle_strip_ocl
#define KERNEL_VIEWPORT_DIVISION_BIN                kernel_viewport_division_ocl
#define KERNEL_PERSPECTIVE_DIVISION_BIN             kernel_perspective_division_ocl
#define KERNEL_READNPIXELS_BIN                      kernel_readnpixels_ocl
#define KERNEL_STRIDED_WRITE_BIN                    kernel_strided_write_ocl
#define KERNEL_CLEAR_BIN                            kernel_clear_ocl
#endif



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