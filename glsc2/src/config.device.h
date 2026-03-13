#ifndef CONFIG_DEVICE_H
#define CONFIG_DEVICE_H

// -----------
// DEVICE ARCH
// -----------
// Byte size of local memory available for core
#define DEVICE_LOCAL_MEM_SIZE 0xc000u

#define DEVICE_NUM_CORES 36

#define DEVICE_LOCAL_THREADS_LOG2 10

// if disable no images nor texture units where used
#ifndef DEVICE_IMAGE_SUPPORT
#define DEVICE_IMAGE_SUPPORT 0
#endif

#ifndef DEVICE_RW_IMAGE_SUPPORT
#define DEVICE_RW_IMAGE_SUPPORT 0
#endif

#define DEVICE_SUB_GROUP_THREADS_LOG2 5

// if disable the threads are executed as each local thread is independant of each other
#ifndef DEVICE_SUB_GROUP_SUPPORT
#define DEVICE_SUB_GROUP_SUPPORT 1
#endif

// if enabled, sub group threads are executed in locksteps.
#ifndef DEVICE_SUB_GROUP_LOCKSTEP
#define DEVICE_SUB_GROUP_LOCKSTEP 0
#endif

// if enabled, sub group threads read and writes instructions are visible to all threads. 
#ifndef DEVICE_SUB_GROUP_RAW_COHERENCE
#define DEVICE_SUB_GROUP_RAW_COHERENCE 0
#endif

// if enabled the device supports sub group intra-register operations as ballot, all, any, scan or reduce. 
#ifndef DEVICE_SUB_GROUP_INTRINSICTS_SUPPORT
#define DEVICE_SUB_GROUP_INTRINSICTS_SUPPORT 1
#endif

// ------
// RENDER CONFIGURATIONS
// ------
#define DEVICE_CONTEXT_NUMBER 1
#define DEVICE_BIN_QUEUE_SIZE 1

#define DEVICE_VERTEX_ATTRIBUTE_SIZE 16
#define DEVICE_UNIFORM_CAPACITY (2*1024) // 2 KB for uniforms

#define DEVICE_VERTEX_TEXTURE_UNITS 0 // max number of vertex texture units
#define DEVICE_TEXTURE_UNITS 8 // max number of active texture units

#define TRIANGLE_PRIMITIVE_CONFIGS_LOG2 4
#define DEVICE_VERTEX_COMMAND_QUEUE_SIZE 8

// max number of triangles could be rasterized
#define DEVICE_MAX_NUMBER_TRIANGLES (1UL << 15) 
#define DEVICE_MAX_NUMBER_SUBTRIANGLES (1UL << 12)

// max number of vertices with varying attributes could be rastired
#define DEVICE_VERTICES_SIZE (1UL << 16) // 
#define DEVICE_VARYING_SIZE 2

// ------
// KERNEL CONFIGURATIONS
// ------

// Triangle Setup Configuration
#define DEVICE_SETUP_SUB_GROUPS 2
// Bin Raster Configuration
#define DEVICE_BIN_SUB_GROUPS 16
// Coarse Raster Configuration
#define DEVICE_COARSE_SUB_GROUPS 16

#define DEVICE_FINE_SUB_GROUPS 20

// ------
// FRONTEND CONFIG
// ------

#define HOST_PROGRAMS_SIZE 16
#define HOST_BIN_QUEUES_SIZE 8
#define HOST_FRAMEBUFFER_SIZE 16
#define HOST_IMAGES_REF_SIZE 16
#define HOST_BUFFERS_SIZE 128
#define HOST_RENDERBUFFERS_SIZE 8
#define HOST_TEXTURES_SIZE 8

// -----
// OBJECTS CONFIG
// -----
#define DEVICE_MAX_TEXTURE_SIZE_LOG2 11 // (2048 x 2048) max size  
// TODO: Add mipmapping support
#define DEVICE_MIPMAP_LEVELS

// NVIDIA
#define NVIDIA_SM_VERSION 61


#endif