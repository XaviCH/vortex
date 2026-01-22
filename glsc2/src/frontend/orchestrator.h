#include <frontend/device.orch.h>

void device_launch_array_triangle_assembly(
    device_context_t* ctx, render_mode_t mode, size_t frag_config_id, 
    size_t vertex_offset, size_t triangle_offset, size_t num_triangles,
    size_t width, size_t height
);

void device_launch_range_triangle_assembly(
    device_context_t* ctx, render_mode_t mode, size_t frag_config_id, 
    size_t vertex_offset, size_t triangle_offset, size_t num_triangles,
    size_t width, size_t height, cl_ushort* ptr
);

void device_launch_bin_dispatch(
    device_context_t* ctx, 
    size_t num_triangles, size_t width, size_t height
);

void device_launch_tile_dispatch(
    device_context_t* ctx,
    uint32_t deferred_clear,
    size_t width, size_t height
);

void device_launch_fragment_shader(
    device_context_t* ctx,
    clear_data_t c_data,
    enabled_data_t c_enabled,
    size_t colorbuffer_id, size_t depthbuffer_id, size_t stencilbuffer_id,
    uint32_t colorbuffer_mode,
    size_t width, size_t height, 
    size_t bin_queue_id
);

void device_launch_clear_framebuffer(
    device_shared_objects_t* device,
    clear_data_t c_data,
    enabled_data_t c_enabled,
    size_t colorbuffer_id, size_t depthbuffer_id, size_t stencilbuffer_id,
    uint32_t colorbuffer_mode,
    size_t width, size_t height, 
    size_t bin_queue_id
);

void device_launch_read_pixels(
    device_shared_objects_t* device,
    size_t colorbuffer_id,
    uint32_t colorbuffer_mode,
    size_t c_width, size_t c_height,
    size_t x, size_t y,
    size_t width, size_t height,
    size_t bin_queue_id,
    uint32_t ptr_format,
    void* ptr
);

void device_move_context_state(
    device_context_t* dst, 
    device_context_t* src
);

void device_wait_bin_queue(
    device_shared_objects_t* device,
    size_t bin_queue_id
);

void device_bind_vertex_fragment_uniform(
    device_context_t* ctx,
    size_t config_id
);

size_t device_create_bin_queue(
    device_shared_objects_t* device
);

void device_load_vertex_attribute_data(
    device_context_t* ctx,
    vertex_attribute_data_t* data
);

void device_load_vertex_attributes(
    device_context_t* ctx, 
    float* vertex_attribs
);

void device_load_vertex_uniform(
    device_context_t* ctx, 
    void* uniform_data
);

void device_bind_vertex_uniform(
    device_context_t* ctx
);

typedef struct {
    size_t shader_id;
    size_t assembled_triangles;
    size_t assembled_vertices;
    size_t pending_vertices;

    size_t last_config_id;
    render_mode_t last_render_mode;
} draw_state_t;

typedef struct {
    clear_data_t data;
    enabled_data_t enabled;
} clear_state_t;

typedef struct {
    uint32_t width, height;
    uint32_t mode;
    size_t image_id;
} orch_rw_image2d_handler_t;

typedef struct {
    draw_state_t draw_state;
    clear_state_t clear_state;
    device_context_t* context;
    size_t colorbuffer_id, depthbuffer_id, stencilbuffer_id; // 0 is the dummy one
    uint32_t colorbuffer_mode;
    uint32_t width, height;
    size_t bin_queue_id; 
} orch_framebuffer_handler_t;

typedef struct {
    device_shared_objects_t* device;

    size_t framebuffer_size;
    orch_framebuffer_handler_t framebuffers[HOST_FRAMEBUFFER_SIZE];

    size_t rw_image2d_size;
    orch_rw_image2d_handler_t rw_image2ds[HOST_IMAGES_REF_SIZE];

    size_t last_context_id = DEVICE_CONTEXT_NUMBER - 1;
    device_context_t contexts[DEVICE_CONTEXT_NUMBER];
    size_t framebuffer_attachments[DEVICE_CONTEXT_NUMBER];
} orch_handler_t;

static size_t orch_create_framebuffer(orch_handler_t* orch)
{
    if (orch->framebuffer_size == HOST_FRAMEBUFFER_SIZE) exit(0);

    orch_framebuffer_handler_t* f_handler = &orch->framebuffers[orch->framebuffer_size];
    
    size_t bin_queue_id = device_create_bin_queue(orch->device);

    *f_handler = (orch_framebuffer_handler_t) {
        .draw_state = {
            .assembled_triangles = 0,
            .assembled_vertices = 0,
            .last_config_id = 0,
            .last_render_mode = 0,
            .pending_vertices = 0,
            .shader_id = 0,
        },
        .clear_state = {
            .data = 0,
            .enabled = 0,
        },
        .context = NULL,
        .colorbuffer_id = 0, 
        .depthbuffer_id = 0, 
        .stencilbuffer_id = 0,
        .colorbuffer_mode = 0,
        .width = 0, 
        .height = 0,
        .bin_queue_id = bin_queue_id,
    };

    size_t framebuffer_id = orch->framebuffer_size;
    orch->framebuffer_size += 1;

    return  framebuffer_id;
}

static size_t orch_create_image2d(orch_handler_t* orch, size_t width, size_t height, uint32_t mode)
{
    if (orch->rw_image2d_size == HOST_IMAGES_REF_SIZE) exit(0);

    orch_rw_image2d_handler_t* i_handler = &orch->rw_image2ds[orch->rw_image2d_size];

    size_t device_image_id = device_create_2d_texture(orch->device, width, height, mode);

    *i_handler = (orch_rw_image2d_handler_t) {
        .width = width,
        .height = height,
        .mode = mode,
        .image_id = device_image_id
    };

    size_t rw_image2d_id = orch->rw_image2d_size;
    orch->rw_image2d_size += 1;

    return rw_image2d_id;
}

static size_t orch_create_renderbuffer(orch_handler_t* orch, size_t width, size_t height, uint32_t mode)
{
    return orch_create_image2d(orch, width, height, mode);
}

static size_t orch_create_texture(orch_handler_t* orch, size_t width, size_t height, uint32_t mode)
{
    return orch_create_image2d(orch, width, height, mode);
}

static size_t orch_create_shader(orch_handler_t* orch, size_t lenght, void* binary)
{
    return device_create_program_from_binary(orch->device, lenght, (const unsigned char*) binary);
}

static size_t orch_create_buffer(orch_handler_t* orch, size_t size)
{
    return device_create_ro_buffer(orch->device, size);
}

static void orch_attach_rw_image2d_colorbuffer(orch_handler_t* orch, size_t framebuffer_id, size_t color_id)
{
    orch_rw_image2d_handler_t* i_handler = &orch->rw_image2ds[color_id];
    orch_framebuffer_handler_t* f_handler = &orch->framebuffers[framebuffer_id];

    f_handler->colorbuffer_id = i_handler->image_id;
    f_handler->width = i_handler->width;
    f_handler->height = i_handler->height;
    f_handler->colorbuffer_mode = i_handler->mode;
}

static void orch_attach_rw_image2d_depthbuffer(orch_handler_t* orch, size_t framebuffer_id, size_t depth_id)
{
    orch_rw_image2d_handler_t* i_handler = &orch->rw_image2ds[depth_id];
    orch_framebuffer_handler_t* f_handler = &orch->framebuffers[framebuffer_id];

    f_handler->depthbuffer_id = i_handler->image_id;
    f_handler->width = i_handler->width;
    f_handler->height = i_handler->height;
}

static void orch_attach_rw_image2d_stencilbuffer(orch_handler_t* orch, size_t framebuffer_id, size_t stencil_id)
{
    orch_rw_image2d_handler_t* i_handler = &orch->rw_image2ds[stencil_id];
    orch_framebuffer_handler_t* f_handler = &orch->framebuffers[framebuffer_id];

    f_handler->stencilbuffer_id = i_handler->image_id;
    f_handler->width = i_handler->width;
    f_handler->height = i_handler->height;
}

static void orch_attach_render_colorbuffer(orch_handler_t* orch, size_t framebuffer_id, size_t color_id)
{
    orch_attach_rw_image2d_colorbuffer(orch, framebuffer_id, color_id);
}

static void orch_attach_render_depthbuffer(orch_handler_t* orch, size_t framebuffer_id, size_t depth_id)
{
    orch_attach_rw_image2d_depthbuffer(orch, framebuffer_id, depth_id);
}

static void orch_attach_render_stencilbuffer(orch_handler_t* orch, size_t framebuffer_id, size_t stencil_id)
{
    orch_attach_rw_image2d_stencilbuffer(orch, framebuffer_id, stencil_id);
}

static void orch_attach_texture_colorbuffer(orch_handler_t* orch, size_t framebuffer_id, size_t color_id)
{
    orch_attach_rw_image2d_colorbuffer(orch, framebuffer_id, color_id);
}

static void orch_attach_texture_depthbuffer(orch_handler_t* orch, size_t framebuffer_id, size_t depth_id)
{
    orch_attach_rw_image2d_depthbuffer(orch, framebuffer_id, depth_id);
}

static void orch_attach_texture_stencilbuffer(orch_handler_t* orch, size_t framebuffer_id, size_t stencil_id)
{
    orch_attach_rw_image2d_stencilbuffer(orch, framebuffer_id, stencil_id);
}

static void orch_init_handler(orch_handler_t* orch)
{
    *orch = {
        .device = NULL,
        .last_context_id = DEVICE_CONTEXT_NUMBER - 1,
        .framebuffer_size = 0,
        .rw_image2d_size = 0,
    };

    device_create_shared_objects(orch->device);
}

static void orch_upload_vertex_attributes(orch_handler_t* orch, size_t framebuffer_id, float* vertex_attribs)
{
    orch_framebuffer_handler_t* f_handler = &orch->framebuffers[framebuffer_id];

    device_load_vertex_attributes(f_handler->context, vertex_attribs);
}

static void orch_upload_vertex_attribute_data(orch_handler_t* orch, size_t framebuffer_id, vertex_attribute_data_t* data)
{
    orch_framebuffer_handler_t* f_handler = &orch->framebuffers[framebuffer_id];

    device_load_vertex_attribute_data(f_handler->context, data);
}

static void orch_attach_vertex_attribute_ptr(orch_handler_t* orch, size_t framebuffer_id, uint32_t attribute, size_t buffer_id)
{
    orch_framebuffer_handler_t* f_handler = &orch->framebuffers[framebuffer_id];

    device_bind_buffer_to_vertex_attribute_pointer(f_handler->context, attribute, buffer_id);
}

static void orch_attach_vertex_attribute_host_ptr(orch_handler_t* orch, size_t framebuffer_id, uint32_t attribute, size_t stride, void* ptr)
{
    orch_framebuffer_handler_t* f_handler = &orch->framebuffers[framebuffer_id];

    device_bind_host_pointer_to_vertex_attribute_pointer(f_handler->context, attribute, stride, ptr);
}

static void orch_attach_vertex_uniform(orch_handler_t* orch, size_t framebuffer_id, void* uniform_data)
{
    orch_framebuffer_handler_t* f_handler = &orch->framebuffers[framebuffer_id];

    device_load_vertex_uniform(f_handler->context, uniform_data);

    device_bind_vertex_uniform(f_handler->context);
}

static void orch_attach_vertex_fragment_uniform(orch_handler_t* orch, size_t framebuffer_id)
{
    orch_framebuffer_handler_t* f_handler = &orch->framebuffers[framebuffer_id];

    device_bind_vertex_fragment_uniform(f_handler->context, f_handler->draw_state.last_config_id);
}

static void orch_upload_fragment_data(orch_handler_t* orch, size_t framebuffer_id, void* uniform_data, rop_config_t config)
{
    orch_framebuffer_handler_t* f_handler = &orch->framebuffers[framebuffer_id];

    device_load_config(f_handler->context, f_handler->draw_state.last_config_id, &config, (cl_uchar*) uniform_data);
}

static size_t get_num_triangles_from_vertices(render_mode_t mode, size_t num_vertices)
{
    if (is_render_mode_flag_triangle_fan(mode) || is_render_mode_flag_triangle_strip(mode))
        return (num_vertices >= 3) ? num_vertices - 2 : 0;

    return num_vertices / 3;
}

void orch_launch_vertex_shader(orch_framebuffer_handler_t* framebuffer, render_mode_t mode, uint32_t init, uint32_t end)
{
    device_launch_vertex_shader(framebuffer->context, init, end, framebuffer->draw_state.assembled_vertices);

    size_t enqueued_vertices = end - init;

    framebuffer->draw_state.last_render_mode = mode;
    framebuffer->draw_state.assembled_vertices += enqueued_vertices;
    framebuffer->draw_state.pending_vertices += enqueued_vertices;
}

void orch_flush_vertices(orch_framebuffer_handler_t* framebuffer)
{
    if (framebuffer->draw_state.pending_vertices == 0) return;

    size_t pending_triangles = get_num_triangles_from_vertices(
        framebuffer->draw_state.last_render_mode, 
        framebuffer->draw_state.pending_vertices
    );

    size_t vertices_offset = framebuffer->draw_state.assembled_vertices - framebuffer->draw_state.pending_vertices;

    device_launch_array_triangle_assembly(
        framebuffer->context,
        framebuffer->draw_state.last_render_mode,
        framebuffer->draw_state.last_config_id,
        vertices_offset,
        framebuffer->draw_state.assembled_triangles,
        pending_triangles,
        framebuffer->width,
        framebuffer->height
    );

    framebuffer->draw_state.pending_vertices = 0;
    framebuffer->draw_state.assembled_triangles += pending_triangles;
}

uint32_t orch_is_deferred_clear(clear_state_t state)
{
    return state.enabled.misc ? 1 : 0;
}

void orch_flush_draw_state(orch_framebuffer_handler_t* framebuffer)
{
    orch_flush_vertices(framebuffer);

    if (framebuffer->draw_state.assembled_triangles == 0) return;
    
    uint32_t deferred_clear = orch_is_deferred_clear(framebuffer->clear_state);

    device_launch_bin_dispatch(framebuffer->context, framebuffer->draw_state.assembled_triangles, framebuffer->width, framebuffer->height);
    device_launch_tile_dispatch(framebuffer->context, deferred_clear, framebuffer->width, framebuffer->height);
    device_launch_fragment_shader(
        framebuffer->context, 
        framebuffer->clear_state.data, 
        framebuffer->clear_state.enabled,
        framebuffer->colorbuffer_id,
        framebuffer->depthbuffer_id,
        framebuffer->stencilbuffer_id,
        framebuffer->colorbuffer_mode,
        framebuffer->width,
        framebuffer->height,
        framebuffer->bin_queue_id
    );

    framebuffer->draw_state.assembled_triangles  = 0;
    framebuffer->draw_state.assembled_vertices   = 0;
    framebuffer->clear_state.enabled = {0};
}

void orch_flush_clear_state(orch_handler_t* orch, orch_framebuffer_handler_t* framebuffer)
{
    if (framebuffer->clear_state.enabled.misc == 0) return;
    
    device_launch_clear_framebuffer(
        orch->device,
        framebuffer->clear_state.data,
        framebuffer->clear_state.enabled,
        framebuffer->colorbuffer_id,
        framebuffer->depthbuffer_id,
        framebuffer->stencilbuffer_id,
        framebuffer->colorbuffer_mode,
        framebuffer->width,
        framebuffer->height,
        framebuffer->bin_queue_id
    );
    
    framebuffer->clear_state.enabled = {0};
}

void orch_flush(orch_handler_t* orch, orch_framebuffer_handler_t* framebuffer)
{
    orch_flush_draw_state(framebuffer);
    orch_flush_clear_state(orch, framebuffer);
}

void orch_finish(orch_handler_t* orch, orch_framebuffer_handler_t* framebuffer)
{
    orch_flush(orch, framebuffer);
    device_wait_bin_queue(orch->device, framebuffer->bin_queue_id);
}

static uint32_t orch_require_flush_context(orch_framebuffer_handler_t* framebuffer, size_t shader_id, render_mode_t mode, size_t num_vertices)
{
    if (framebuffer->draw_state.shader_id != shader_id) return true;

    // TODO: depends also on the varying size
    if (framebuffer->draw_state.assembled_vertices + num_vertices > DEVICE_VERTICES_SIZE) return true;

    size_t pending_triangles = get_num_triangles_from_vertices(
        framebuffer->draw_state.last_render_mode, 
        framebuffer->draw_state.pending_vertices
    );

    size_t requested_triangles = get_num_triangles_from_vertices(mode, num_vertices);

    if (framebuffer->draw_state.assembled_triangles + pending_triangles + requested_triangles > DEVICE_MAX_NUMBER_TRIANGLES) return true;

    // TODO: maybe take account segments available

    return false;
}

static int orch_is_framebuffer_attached(orch_handler_t* orch, size_t context_id)
{
    return orch->framebuffer_attachments[context_id] < DEVICE_CONTEXT_NUMBER;
}

static void orch_deattach_context(orch_handler_t* orch, size_t context_id)
{
    size_t framebuffer_id = orch->framebuffer_attachments[context_id];

    if (framebuffer_id >= DEVICE_CONTEXT_NUMBER) return; // already deattached

    orch_framebuffer_handler_t* f_handler = &orch->framebuffers[framebuffer_id];

    orch_flush(orch, f_handler);

    f_handler->context = NULL;
}

static size_t orch_get_new_context_id(orch_handler_t* orch)
{
    size_t context_id = (orch->last_context_id + 1) % DEVICE_CONTEXT_NUMBER;

    orch_deattach_context(orch, context_id);

    return context_id;
}

static void orch_switch_context(orch_handler_t* orch, orch_framebuffer_handler_t* framebuffer)
{
    orch_flush(orch, framebuffer);

    #if DEVICE_CONTEXT_NUMBER > 1
    {
        // TODO:
        size_t new_context_id = orch_get_new_context_id(orch);
        
        device_context_t* new_context = &orch->contexts[new_context_id];
        
        device_move_context_state(new_context, framebuffer->context);
    }
    #endif
}

static uint8_t orch_require_flush_vertices(orch_framebuffer_handler_t* framebuffer, render_mode_t mode, uint32_t config_id)
{
    if (is_render_mode_flag_triangle_fan(mode) || is_render_mode_flag_triangle_strip(mode)) return 1;

    if (framebuffer->draw_state.last_render_mode.flags != mode.flags) return 1;

    if (framebuffer->draw_state.last_config_id != config_id) return 1;

    return 0;
}

static uint8_t orch_require_flush_clear(orch_framebuffer_handler_t* framebuffer, clear_data_t data, enabled_data_t enabled)
{
    uint32_t red_enabled = get_enabled_red_data(framebuffer->clear_state.enabled) && get_enabled_red_data(enabled);
    uint32_t red_diff = get_rgba8_red(framebuffer->clear_state.data.color) !=  get_rgba8_red(data.color); 
    
    if (red_enabled && red_diff) return 1;

    uint32_t green_enabled = get_enabled_green_data(framebuffer->clear_state.enabled) && get_enabled_green_data(enabled);
    uint32_t green_diff = get_rgba8_green(framebuffer->clear_state.data.color) !=  get_rgba8_green(data.color);
    
    if (green_enabled && green_diff) return 1;

    uint32_t blue_enabled = get_enabled_blue_data(framebuffer->clear_state.enabled) && get_enabled_blue_data(enabled);
    uint32_t blue_diff = get_rgba8_blue(framebuffer->clear_state.data.color) !=  get_rgba8_blue(data.color);
    
    if (blue_enabled && blue_diff) return 1;

    uint32_t alpha_enabled = get_enabled_alpha_data(framebuffer->clear_state.enabled) && get_enabled_alpha_data(enabled);
    uint32_t alpha_diff = get_rgba8_alpha(framebuffer->clear_state.data.color) !=  get_rgba8_alpha(data.color);
    
    if (alpha_enabled && alpha_diff) return 1;

    uint32_t depth_enabled = get_enabled_depth_data(framebuffer->clear_state.enabled) && get_enabled_depth_data(enabled);
    uint32_t depth_diff = framebuffer->clear_state.data.depth.misc !=  framebuffer->clear_state.data.depth.misc;
    
    if (depth_enabled && depth_diff) return 1;

    uint32_t stencil_enabled = get_enabled_stencil_data(framebuffer->clear_state.enabled) && get_enabled_stencil_data(enabled);
    uint32_t stencil_diff = framebuffer->clear_state.data.stencil.misc !=  framebuffer->clear_state.data.stencil.misc;
    
    if (stencil_enabled && stencil_diff) return 1;

    return 0;
}

static void orch_draw_vertices(orch_handler_t* orch, orch_framebuffer_handler_t* framebuffer, size_t shader_id, render_mode_t mode, uint32_t init, uint32_t end)
{
    size_t num_vertices = end - init;
    size_t num_triangles = get_num_triangles_from_vertices(mode, num_vertices);

    if (orch_require_flush_context(framebuffer, shader_id, mode, num_vertices))
    {
        orch_switch_context(orch, framebuffer);
    }

    if (orch_require_flush_vertices(framebuffer, mode, num_vertices))
    {
        orch_flush_vertices(framebuffer);
    }

    orch_launch_vertex_shader(framebuffer, mode, init, end);
}

static void orch_draw_arrays(orch_handler_t* orch, orch_framebuffer_handler_t* framebuffer, size_t shader_id, render_mode_t mode, uint32_t init, uint32_t end)
{
    orch_draw_vertices(orch, framebuffer, shader_id, mode, init, end);
}

static void orch_draw_range(orch_handler_t* orch, orch_framebuffer_handler_t* framebuffer, size_t shader_id, render_mode_t mode, uint32_t init, uint32_t end, uint32_t count, uint16_t* ptr) 
{
    orch_draw_vertices(orch, framebuffer, shader_id, mode, init, end);

    if (framebuffer->draw_state.pending_vertices == 0) return;

    size_t pending_triangles = get_num_triangles_from_vertices(
        framebuffer->draw_state.last_render_mode, 
        count
    );

    size_t vertices_offset = framebuffer->draw_state.assembled_vertices - framebuffer->draw_state.pending_vertices;

    device_launch_range_triangle_assembly(
        framebuffer->context, 
        framebuffer->draw_state.last_render_mode, 
        framebuffer->draw_state.last_config_id,
        vertices_offset,
        framebuffer->draw_state.assembled_triangles,
        pending_triangles,
        framebuffer->width,
        framebuffer->height,
        ptr
    );

    framebuffer->draw_state.pending_vertices = 0;
    framebuffer->draw_state.assembled_triangles += pending_triangles;
}

static void orch_merge_clear_state(orch_framebuffer_handler_t* framebuffer, clear_data_t data, enabled_data_t enabled)
{
    enabled_data_t *enabled_ptr = &framebuffer->clear_state.enabled;

    // color merge
    rgba8_t *color_ptr = &framebuffer->clear_state.data.color;

    if (get_enabled_red_data(enabled)) {
        set_enabled_data_red_enable(enabled_ptr);
        set_rgba8_red(color_ptr, get_rgba8_red(data.color));
    }

    if (get_enabled_green_data(enabled)) {
        set_enabled_data_green_enable(enabled_ptr);
        set_rgba8_green(color_ptr, get_rgba8_green(data.color));
    }

    if (get_enabled_blue_data(enabled)) {
        set_enabled_data_blue_enable(enabled_ptr);
        set_rgba8_blue(color_ptr, get_rgba8_blue(data.color));
    }

    if (get_enabled_alpha_data(enabled)) {
        set_enabled_data_alpha_enable(enabled_ptr);
        set_rgba8_alpha(color_ptr, get_rgba8_alpha(data.color));
    }

    // depth merge
    depth16_t *depth_ptr = &framebuffer->clear_state.data.depth;
    if (get_enabled_depth_data(enabled)) {
        set_enabled_depth_data(enabled_ptr, 1);
        *depth_ptr = data.depth;
    }

    // stencil merge
    // TODO: use an interface
    stencil8_t *stencil_ptr = &framebuffer->clear_state.data.stencil;
    cl_uchar old_stencil_mask = get_enabled_stencil_data(*enabled_ptr);
    cl_uchar new_stencil_mask = get_enabled_stencil_data(enabled);
    set_enabled_stencil_data(enabled_ptr, old_stencil_mask | new_stencil_mask);
    stencil_ptr->misc |= data.stencil.misc & new_stencil_mask;
}

static void orch_clear(orch_handler_t* orch, orch_framebuffer_handler_t* framebuffer, clear_data_t data, enabled_data_t enabled)
{
    orch_flush_draw_state(framebuffer);

    orch_merge_clear_state(framebuffer, data, enabled);
}

static void orch_readnpixels(orch_handler_t* orch, orch_framebuffer_handler_t* framebuffer, size_t x, size_t y, size_t width, size_t height, uint32_t mode, void* ptr)
{
    orch_flush_draw_state(framebuffer);
    orch_flush_clear_state(orch, framebuffer);

    device_launch_read_pixels(
        orch->device, 
        framebuffer->bin_queue_id,
        framebuffer->colorbuffer_id,
        framebuffer->width,
        framebuffer->height,
        framebuffer->colorbuffer_mode,
        mode,
        x, y, width, height, ptr);
}