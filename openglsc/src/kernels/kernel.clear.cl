#define GL_DEPTH_BUFFER_BIT               0x00000100
#define GL_STENCIL_BUFFER_BIT             0x00000400
#define GL_COLOR_BUFFER_BIT               0x00004000

#define GL_R8                             0x8229
#define GL_RG8                            0x822B
#define GL_RGB8                           0x8051
#define GL_RGBA8                          0x8058

#define GL_RGBA4                          0x8056
#define GL_RGB5_A1                        0x8057
#define GL_RGB565                         0x8D62

kernel void gl_clear(
    // framebuffer data
    global void* colorbuffer,
    global ushort* depthbuffer,
    global uchar* stencilbuffer,
    const uint colorbuffer_type,
    const uint colorbuffer_width,
    // clear data
    const uint mask,
    const float4 color_value,
    const float depth_value,
    const int stencil_value,
    // scissor data
    const uint scissor_enabled,
    const int left,
    const int bottom,
    const uint width,
    const uint height,
    // mask data
    const uint color_red_mask, 
    const uint color_green_mask, 
    const uint color_blue_mask, 
    const uint color_alpha_mask,
    const uint depth_mask, 
    const uint stencil_mask
) {
    int gid = get_global_id(0);

    uint x, y;
    x = gid % colorbuffer_width;
    y = gid / colorbuffer_width;

    if (scissor_enabled && (left > x || x >= left + width || bottom > y || y >= bottom + height)) return;

    if (mask & GL_DEPTH_BUFFER_BIT && depth_mask) 
        depthbuffer[gid] = depth_value*0xFFFFu;
    if (mask & GL_STENCIL_BUFFER_BIT && stencil_mask) 
        stencilbuffer[gid] = (stencilbuffer[gid] & ~stencil_mask) | (stencil_value & stencil_mask);
    if (mask & GL_COLOR_BUFFER_BIT) {
        switch (colorbuffer_type) {
        case GL_R8:
            if (color_red_mask) ((global char*) colorbuffer)[gid] = color_value.x*0xFFu;
            break;
        case GL_RG8:
            {
                global char* color_ptr = (global char*)colorbuffer + gid*2;
                if (color_red_mask) color_ptr[0] = color_value.x*0xFFu;
                if (color_green_mask) color_ptr[1] = color_value.y*0xFFu;
            }
            break;
        case GL_RGB8:
            {
                global char* color_ptr = (global char*)colorbuffer + gid*3;
                if (color_red_mask) color_ptr[0] = color_value.x*0xFFu;
                if (color_green_mask) color_ptr[1] = color_value.y*0xFFu;
                if (color_blue_mask) color_ptr[2] = color_value.z*0xFFu;
            }
            break;
        case GL_RGBA8:
            {
                global char* color_ptr = (global char*)colorbuffer + gid*4;
                if (color_red_mask) color_ptr[0] = color_value.x*0xFFu;
                if (color_green_mask) color_ptr[1] = color_value.y*0xFFu;
                if (color_blue_mask) color_ptr[2] = color_value.z*0xFFu;
                if (color_alpha_mask) color_ptr[3] = color_value.w*0xFFu;
            }
            break;
        case GL_RGBA4:
            {
                ushort color = ((global ushort*) colorbuffer)[gid];
                if (color_red_mask) {
                    ushort bit_mask = 0x000Fu;
                    color = (color & ~bit_mask) | (((ushort)(color_value.x*0xFu) << 0) & bit_mask);
                }
                if (color_green_mask) {
                    ushort bit_mask = 0x00F0u;
                    color = (color & ~bit_mask) | (((ushort)(color_value.y*0xFu) << 4) & bit_mask);
                }
                if (color_blue_mask) {
                    ushort bit_mask = 0x0F00u;
                    color = (color & ~bit_mask) | (((ushort)(color_value.z*0xFu) << 8) & bit_mask);
                }
                if (color_alpha_mask) {
                    ushort bit_mask = 0xF000u;
                    color = (color & ~bit_mask) | (((ushort)(color_value.w*0xFu) << 12) & bit_mask);
                }
                ((global ushort*) colorbuffer)[gid] = color;
            }
            break;
        case GL_RGB5_A1:
            {
                ushort color = ((global ushort*) colorbuffer)[gid];
                if (color_red_mask) {
                    ushort bit_mask = 0x001Fu;
                    color = (color & ~bit_mask) | (((ushort)(color_value.x*0x1Fu) << 0) & bit_mask);
                }
                if (color_green_mask) {
                    ushort bit_mask = 0x03E0u;
                    color = (color & ~bit_mask) | (((ushort)(color_value.y*0x1Fu) << 5) & bit_mask);
                }
                if (color_blue_mask) {
                    ushort bit_mask = 0x7C00u;
                    color = (color & ~bit_mask) | (((ushort)(color_value.z*0x1Fu) << 10) & bit_mask);
                }
                if (color_alpha_mask) {
                    ushort bit_mask = 0x8000u;
                    color = (color & ~bit_mask) | (((ushort)(color_value.w*0x1u) << 15) & bit_mask);
                }
                ((global ushort*) colorbuffer)[gid] = color;
            }
            break;
        case GL_RGB565:
            {
                ushort color = ((global ushort*) colorbuffer)[gid];
                if (color_red_mask) {
                    ushort bit_mask = 0x001Fu;
                    color = (color & ~bit_mask) | (((ushort)(color_value.x*0x1Fu) << 0) & bit_mask);
                }
                if (color_green_mask) {
                    ushort bit_mask = 0x07E0u;
                    color = (color & ~bit_mask) | (((ushort)(color_value.y*0x3Fu) << 5) & bit_mask);
                }
                if (color_blue_mask) {
                    ushort bit_mask = 0xF800u;
                    color = (color & ~bit_mask) | (((ushort)(color_value.z*0x1Fu) << 11) & bit_mask);
                }
                ((global ushort*) colorbuffer)[gid] = color;
            }
            break;
        }

    }

}