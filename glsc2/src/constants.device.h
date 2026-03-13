#ifndef CONSTANTS_DEVICE_H
#define CONSTANTS_DEVICE_H

#include "config.device.h"

// -----------
// DEVICE CONSTANTS
// -----------
#define DEVICE_LOCAL_THREADS (1 << DEVICE_LOCAL_THREADS_LOG2)

#define DEVICE_SUB_GROUP_THREADS (1 << DEVICE_SUB_GROUP_THREADS_LOG2)

#if DEVICE_IMAGE_SUPPORT == 1
    #define DEVICE_IMAGE_ENABLED
    
    #if DEVICE_RW_IMAGE_SUPPORT == 1
        #define DEVICE_RW_IMAGE_ENABLED
    #endif
#endif
    
#if DEVICE_SUB_GROUP_SUPPORT == 1
#define DEVICE_SUB_GROUP_ENABLED
#endif

#ifdef DEVICE_SUB_GROUP_ENABLED

    #if DEVICE_SUB_GROUP_LOCKSTEP == 1 && DEVICE_SUB_GROUP_RAW_COHERENCE == 1
    #define DEVICE_SUB_GROUP_RAW_ENABLED
    #endif

    #if DEVICE_SUB_GROUP_INTRINSICTS_SUPPORT == 1
    #define DEVICE_SUB_GROUP_INTRINSICTS_ENABLED
    #endif

#endif

// ------
// RENDER CONFIG
// ------
#define TRIANGLE_PRIMITIVE_CONFIGS (1 << TRIANGLE_PRIMITIVE_CONFIGS_LOG2)
// ------
// KERNEL CONSTANTS
// ------
// Triangle Setup Configuration
#define CONS_SETUP_LOCAL_REQ_MEM                    (CONF_SETUP_BASE_LOCAL_REQ_MEM + (CONF_ENABLE_SUB_GROUPS_COMM ? CONF_SETUP_COMM_SUB_GROUP_REQ_MEM : CONF_SETUP_COMM_LOCAL_REQ_MEM))
#define CONS_SETUP_MAX_AVAILABLE_LOCAL_THREADS      MIN((DEVICE_LOCAL_MEM_SIZE / CONF_SETUP_LOCAL_REQ_MEM), DEVICE_LOCAL_THREADS)
#define CONS_SETUP_LOCAL_THREADS                    MIN(CONF_SETUP_MAX_AVAILABLE_LOCAL_THREADS, (1 << CONF_SETUP_MAX_LOCAL_THREADS_LOG2))

#define CONF_SETUP_LOCAL_MEM_ENABLED

// Bin Raster Configuration
// #if DEVICE_IMAGE_SUPPORT == 1
// #define CONF_BIN_IMAGE_ENABLED
// #endif

// #define CONF_BIN_LOCAL_MEM_ENABLED

// #if DEVICE_SUB_GROUP_SUPPORT == 1
// #define CONF_BIN_SUB_GROUP_ENABLED
// #endif

// Coarse Raster Configuration
#if DEVICE_IMAGE_SUPPORT == 1
#define CONF_COARSE_IMAGE_ENABLED
#endif

#define CONF_COARSE_LOCAL_MEM_ENABLED

#if DEVICE_SUB_GROUP_SUPPORT == 1
#define CONF_COARSE_SUB_GROUP_ENABLED
#endif

// Fine Raster Configuration
#if DEVICE_IMAGE_SUPPORT == 1
#define CONF_FINE_IMAGE_ENABLED
#endif

#define CONF_FINE_LOCAL_MEM_ENABLED

#if DEVICE_SUB_GROUP_INTRISICTS_SUPPORT == 1
#define CONF_FINE_SUB_GROUP_INTRISICTS_ENABLED
#endif

#if DEVICE_SUB_GROUP_RAW == 1
#define CONF_FINE_SUB_GROUP_RAW_ENABLED
#endif

//

#define FW_U32_MAX          (0xFFFFFFFFu)
#define FW_S32_MIN          (~0x7FFFFFFF)
#define FW_S32_MAX          (0x7FFFFFFF)
#define FW_U64_MAX          ((unsigned long)(long)-1)
#define FW_S64_MIN          ((long)-1 << 63)
#define FW_S64_MAX          (~((long)-1 << 63))
#define FW_F32_MIN          (1.175494351e-38f)
#define FW_F32_MAX          (3.402823466e+38f)
#define FW_F64_MIN          (2.2250738585072014e-308)
#define FW_F64_MAX          (1.7976931348623158e+308)
#define FW_PI               (3.14159265358979323846f)

#define FW_ARRAY_SIZE(X)    ((int)(sizeof(X) / sizeof((X)[0])))

//------------------------------------------------------------------------

// CONFIGURABLE

#define CR_MAXVIEWPORT_LOG2     11      // ViewportSize / PixelSize. Max value for any viewport axis
#define CR_SUBPIXEL_LOG2        4       // PixelSize / SubpixelSize. ??

#define CR_MAXBINS_LOG2         4       // ViewportSize / BinSize. Divide the viewport on x bins
#define CR_BIN_LOG2             4       // BinSize / TileSize.
#define CR_TILE_LOG2            3       // TileSize / PixelSize.

#define CR_COVER8X8_LUT_SIZE    768     // 64-bit entries.
#define CR_FLIPBIT_FLIP_Y       2
#define CR_FLIPBIT_FLIP_X       3
#define CR_FLIPBIT_SWAP_XY      4
#define CR_FLIPBIT_COMPL        5

#define CR_BIN_STREAMS_LOG2     4
#define CR_BIN_SEG_LOG2         9       // 32-bit entries.
#define CR_TILE_SEG_LOG2        5       // 32-bit entries.

#define CR_MAXSUBTRIS_LOG2      24 // 24 reduced to 14     // Triangle structs. Dictated by CoarseRaster.
#define CR_COARSE_QUEUE_LOG2    10      // Triangles.

//------------------------------------------------------------------------

#define CR_MAXVIEWPORT_SIZE     (1 << CR_MAXVIEWPORT_LOG2)
#define CR_SUBPIXEL_SIZE        (1 << CR_SUBPIXEL_LOG2)
#define CR_SUBPIXEL_SQR         (1 << (CR_SUBPIXEL_LOG2 * 2))

#define CR_MAXBINS_SIZE         (1 << CR_MAXBINS_LOG2)
#define CR_MAXBINS_SQR          (1 << (CR_MAXBINS_LOG2 * 2))
#define CR_BIN_SIZE             (1 << CR_BIN_LOG2)
#define CR_BIN_SQR              (1 << (CR_BIN_LOG2 * 2))

#define CR_MAXTILES_LOG2        (CR_MAXBINS_LOG2 + CR_BIN_LOG2)
#define CR_MAXTILES_SIZE        (1 << CR_MAXTILES_LOG2)
#define CR_MAXTILES_SQR         (1 << (CR_MAXTILES_LOG2 * 2))
#define CR_TILE_SIZE            (1 << CR_TILE_LOG2)
#define CR_TILE_SQR             (1 << (CR_TILE_LOG2 * 2))

#define CR_BIN_STREAMS_SIZE     (1 << CR_BIN_STREAMS_LOG2)
#define CR_BIN_SEG_SIZE         (1 << CR_BIN_SEG_LOG2)
#define CR_TILE_SEG_SIZE        (1 << CR_TILE_SEG_LOG2)

#define CR_MAXSUBTRIS_SIZE    (1 << CR_MAXSUBTRIS_LOG2)
#define CR_COARSE_QUEUE_SIZE    (1 << CR_COARSE_QUEUE_LOG2)

// ---

#define CONF_MAX_TRIS       (1 << 12)
#define CONF_MAX_SUBTRIS    (CONF_MAX_TRIS + 4096)
#define CONF_MAX_BIN_SEGS   (CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE + 4096)
#define CONF_MAX_TILE_SEGS   (CR_MAXTILES_SQR + 4096)


//------------------------------------------------------------------------
// When evaluating interpolated Z/W/U/V at pixel centers, we introduce an
// error of (+-CR_LERP_ERROR) ULPs. If it wasn't for integer overflows,
// we could utilize the full U32 range for depth to get maximal
// precision. However, to avoid overflows, we must shrink the range
// slightly from both ends. With W/U/V, another difficulty is that quad
// rendering can cause them to be evaluated outside the triangle, so we
// need to shrink the range even more.

#define CR_LERP_ERROR(SAMPLES_LOG2) (2200u << (SAMPLES_LOG2))
#define CR_DEPTH_MIN                CR_LERP_ERROR(3)
#define CR_DEPTH_MAX                (FW_U32_MAX - CR_LERP_ERROR(3))
#define CR_BARY_MAX                 ((1 << (30 - CR_SUBPIXEL_LOG2)) - 1)

//------------------------------------------------------------------------

// Render flags
#define RENDER_MODE_FLAG_ENABLE_QUADS        (1 << 0)
#define RENDER_MODE_FLAG_ENABLE_DEPTH        (1 << 1)
#define RENDER_MODE_FLAG_ENABLE_LERP         (1 << 2)
#define RENDER_MODE_FLAG_ENABLE_BLENDER      (1 << 3)
#define RENDER_MODE_FLAG_ENABLE_CULL_FRONT   (1 << 4)
#define RENDER_MODE_FLAG_ENABLE_CULL_BACK    (1 << 5)
#define RENDER_MODE_FLAG_ENABLE_STENCIL      (1 << 6)
#define RENDER_MODE_FLAG_TRIANGLE_FAN        (1 << 7)
#define RENDER_MODE_FLAG_TRIANGLE_STRIP      (1 << 8)

// Texture modes
#define TEX_R8                              0
#define TEX_RG8                             1
#define TEX_RGB8                            2
#define TEX_RGBA8                           3
#define TEX_RGBA4                           4
#define TEX_RGB5_A1                         5
#define TEX_RGB565                          6
#define TEX_STENCIL_INDEX8                  7
#define TEX_DEPTH_COMPONENT16               8

// Depth
#define DEPTH_FUNC_NEVER                            0
#define DEPTH_FUNC_LESS                             1
#define DEPTH_FUNC_EQUAL                            2
#define DEPTH_FUNC_LEQUAL                           3
#define DEPTH_FUNC_GREATER                          4
#define DEPTH_FUNC_NOTEQUAL                         5
#define DEPTH_FUNC_GEQUAL                           6
#define DEPTH_FUNC_ALWAYS                           7

// Stencil

#define STENCIL_FUNC_NEVER                            0
#define STENCIL_FUNC_LESS                             1
#define STENCIL_FUNC_EQUAL                            2
#define STENCIL_FUNC_LEQUAL                           3
#define STENCIL_FUNC_GREATER                          4
#define STENCIL_FUNC_NOTEQUAL                         5
#define STENCIL_FUNC_GEQUAL                           6
#define STENCIL_FUNC_ALWAYS                           7

#define STENCIL_OP_KEEP                           0
#define STENCIL_OP_ZERO                           1
#define STENCIL_OP_REPLACE                        2
#define STENCIL_OP_INCR                           3
#define STENCIL_OP_DECR                           4
#define STENCIL_OP_INVERT                         5
#define STENCIL_OP_INCR_WRAP                      6
#define STENCIL_OP_DECR_WRAP                      7

// Blending
#define BLEND_FUNC_ADD                       0
#define BLEND_FUNC_SUBTRACT                  1
#define BLEND_FUNC_REVERSE_SUBTRACT          2

#define BLEND_ZERO                           0
#define BLEND_ONE                            1
#define BLEND_SRC_COLOR                      2
#define BLEND_ONE_MINUS_SRC_COLOR            3
#define BLEND_SRC_ALPHA                      4
#define BLEND_ONE_MINUS_SRC_ALPHA            5
#define BLEND_DST_ALPHA                      6
#define BLEND_ONE_MINUS_DST_ALPHA            7
#define BLEND_DST_COLOR                      8
#define BLEND_ONE_MINUS_DST_COLOR            9
#define BLEND_SRC_ALPHA_SATURATE             10
#define BLEND_CONSTANT_COLOR                 11
#define BLEND_ONE_MINUS_CONSTANT_COLOR       12
#define BLEND_CONSTANT_ALPHA                 13
#define BLEND_ONE_MINUS_CONSTANT_ALPHA       14

// Render modes


// Mask modes
#define ENABLED_STENCIL_CHANNEL_MASK  (0xFFu << 0)
#define ENABLED_COLOR_CHANNEL_RED     (1 << 8)
#define ENABLED_COLOR_CHANNEL_GREEN   (1 << 9)
#define ENABLED_COLOR_CHANNEL_BLUE    (1 << 10)
#define ENABLED_COLOR_CHANNEL_ALPHA   (1 << 11)
#define ENABLED_DEPTH_CHANNEL         (1 << 12)

#define ENABLED_COLOR_CHANNEL_MASK    \
    ( \
    ENABLED_COLOR_CHANNEL_RED   | \
    ENABLED_COLOR_CHANNEL_GREEN | \
    ENABLED_COLOR_CHANNEL_BLUE  | \
    ENABLED_COLOR_CHANNEL_ALPHA   \
    )

#define ENABLED_MASK \
    ( \
        ENABLED_COLOR_CHANNEL_MASK | \
        ENABLED_DEPTH_CHANNEL | \
        ENABLED_STENCIL_CHANNEL_MASK | \
    )

// Clearing modes
#define CLEAR_ENABLED_STENCIL_CHANNEL_MASK  (0xFFu << 0)
#define CLEAR_ENABLED_COLOR_CHANNEL_RED     (1 << 8)
#define CLEAR_ENABLED_COLOR_CHANNEL_GREEN   (1 << 9)
#define CLEAR_ENABLED_COLOR_CHANNEL_BLUE    (1 << 10)
#define CLEAR_ENABLED_COLOR_CHANNEL_ALPHA   (1 << 11)
#define CLEAR_ENABLED_DEPTH_CHANNEL         (1 << 12)

#define CLEAR_ENABLED_COLOR_CHANNEL_MASK    \
    ( \
    CLEAR_ENABLED_COLOR_CHANNEL_RED   | \
    CLEAR_ENABLED_COLOR_CHANNEL_GREEN | \
    CLEAR_ENABLED_COLOR_CHANNEL_BLUE  | \
    CLEAR_ENABLED_COLOR_CHANNEL_ALPHA   \
    )

#define CLEAR_ENABLED_MASK \
    ( \
        CLEAR_ENABLED_COLOR_CHANNEL_MASK | \
        CLEAR_ENABLED_DEPTH_CHANNEL | \
        CLEAR_ENABLED_STENCIL_CHANNEL_MASK \
    )

// Vertex attributes
#define VERTEX_ATTRIBUTE_TYPE_BYTE              0
#define VERTEX_ATTRIBUTE_TYPE_UNSIGNED_BYTE     1
#define VERTEX_ATTRIBUTE_TYPE_SHORT             2
#define VERTEX_ATTRIBUTE_TYPE_UNSIGNED_SHORT    3
#define VERTEX_ATTRIBUTE_TYPE_FLOAT             4

#define VERTEX_ATTRIBUTE_SIZE_1                 0
#define VERTEX_ATTRIBUTE_SIZE_2                 1
#define VERTEX_ATTRIBUTE_SIZE_3                 2
#define VERTEX_ATTRIBUTE_SIZE_4                 3

#define VERTEX_ATTRIBUTE_TYPE_MASK              (0x7u << 0)
#define VERTEX_ATTRIBUTE_SIZE_MASK              (0x3u << 3)
#define VERTEX_ATTRIBUTE_ACTIVE_POINTER         (0x1u << 5)
#define VERTEX_ATTRIBUTE_NORMALIZE              (0x1u << 6)

// texture wrap parameters
#define TEXTURE_WRAP_CLAMP_TO_EDGE 0
#define TEXTURE_WRAP_REPEAT 1
#define TEXTURE_WRAP_MIRRORED_REPEAT 2

// texture filter parameters
#define TEXTURE_FILTER_NEAREST 0
#define TEXTURE_FILTER_LINEAR 1
#define TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST 2
#define TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR 3
#define TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST 4
#define TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR 5


#define MIN(_A,_B) (((_A)<(_B))? (_A):(_B))
#define MAX(_A,_B) (((_A)>(_B))? (_A):(_B))
#define IS_POWER_OF_2(a) !(a & 0x1u) 

#endif