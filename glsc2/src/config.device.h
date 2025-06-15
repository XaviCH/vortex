#ifndef CONFIG_DEVICE_H
#define CONFIG_DEVICE_H

// -----------
// DEVICE ARCH
// -----------
// Byte size of local memory available for core
#define DEVICE_LOCAL_MEM_SIZE 0xc000u
 
#define DEVICE_LOCAL_THREADS_LOG2 10
#define DEVICE_IMAGE_SUPPORT 1

#ifndef DEVICE_SUB_GROUP_SUPPORT
#define DEVICE_SUB_GROUP_SUPPORT 1
#endif

#define DEVICE_HAS_SUB_GROUP_INTRINSICTS 1
#define DEVICE_SUBGROUP_THREADS_LOG2 5
#define DEVICE_SUB_GROUP_MEM_COHERENCE 0 // 0 --> explicit coherence, 1 --> implicit coherence
// ------
// RENDER CONFIG
// ------
#define CONF_ENABLE_IMAGE_SUPPORT 1
// ------

// KERNEL CONFIG
// ------
// Triangle Setup Configuration
#define CONF_SETUP_SUB_GROUPS 2
// Bin Raster Configuration
#define CONF_BIN_SUB_GROUPS 16


#endif