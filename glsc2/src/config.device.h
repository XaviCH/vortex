#ifndef CONFIG_DEVICE_H
#define CONFIG_DEVICE_H

// -----------
// DEVICE ARCH
// -----------
// Byte size of local memory available for core
#define DEVICE_LOCAL_MEM_SIZE 0xc000u

#define DEVICE_NUM_CORES 36


#define DEVICE_LOCAL_THREADS_LOG2 10

//


// if disable no images nor texture units where used
#ifndef DEVICE_IMAGE_SUPPORT
#define DEVICE_IMAGE_SUPPORT 0
#endif

#define DEVICE_SUB_GROUP_THREADS_LOG2 5

// if disable the threads are executed as each local thread is independant of each other
#ifndef DEVICE_SUB_GROUP_SUPPORT
#define DEVICE_SUB_GROUP_SUPPORT 1
#endif

// if disable the sub groups do not have read after write coherence between them
#ifndef DEVICE_SUB_GROUP_RAW
#define DEVICE_SUB_GROUP_RAW 1
#endif

// if enabled the kernels could use intrinsicts as ballot, all, any, scan or reduce sub group operations. 
#ifndef DEVICE_SUB_GROUP_INTRINSICTS_SUPPORT
#define DEVICE_SUB_GROUP_INTRINSICTS_SUPPORT 0
#endif

// ------
// RENDER CONFIG
// ------
// ------

// KERNEL CONFIG
// ------
// Triangle Setup Configuration
#define DEVICE_SETUP_SUB_GROUPS 2
// Bin Raster Configuration
#define DEVICE_BIN_SUB_GROUPS 16
// Coarse Raster Configuration
#define DEVICE_COARSE_SUB_GROUPS 16

#define CONF_FINE_SUB_GROUPS 16

#endif