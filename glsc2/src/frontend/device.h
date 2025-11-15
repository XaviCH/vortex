#ifndef FRONTEND_DEVICE_H
#define FRONTEND_DEVICE_H

#include <CL/opencl.h>
#include <constants.device.h>

typedef struct {
    cl_platform_id          platform_id;
    cl_device_id            device_id;
    cl_context              context;

    cl_program              triangle_setup_program;
    cl_program              bin_raster_program;
    cl_program              coarse_raster_program;
    cl_program              clear_program;

    cl_mem                  mem_vertex_attrib_pointer;  // dummy attrib pointers
    cl_mem                  mem_texture;                // dummy texture
    cl_mem                  mem_depthbuffer;            // dummy depthbuffer
    cl_mem                  mem_stencilbuffer;          // dummy stencilbuffer
    
    size_t                program_size;
    cl_program              programs        [HOST_PROGRAMS_SIZE];
    cl_kernel               kernels         [HOST_PROGRAMS_SIZE][2]; // 0: vertex, 1: fragment
    size_t               buffers_size;
    cl_mem                  buffers         [HOST_BUFFERS_SIZE];
    size_t                renderbuffers_size;
    cl_mem                  renderbuffers   [HOST_RENDERBUFFERS_SIZE];
    size_t                textures_size;
    cl_mem                  textures        [HOST_TEXTURES_SIZE];

} device_shared_objects_t;

typedef struct {
    
    device_shared_objects_t *shared_objects;

    // Kernels
    cl_kernel vertex_shader_kernel;
    cl_kernel triangle_setup_arrays_kernel;
    cl_kernel triangle_setup_range_kernel;
    cl_kernel bin_raster_kernel;
    cl_kernel coarse_raster_kernel;
    cl_kernel clear_kernel;
    cl_kernel fine_raster_kernel;

    // Buffers
    cl_mem g_tri_header;
    cl_mem g_tri_data; 
    cl_mem g_tri_subtris;
    cl_mem g_vertex_buffer;

    cl_mem g_bin_first_seg; 
    cl_mem g_bin_seg_data; 
    cl_mem g_bin_seg_next;
    cl_mem g_bin_seg_count;
    cl_mem g_bin_total;

    cl_mem g_active_tiles;
    cl_mem g_tile_first_seg;
    cl_mem g_tile_seg_data;
    cl_mem g_tile_seg_next;
    cl_mem g_tile_seg_count;

    // Atomics
    cl_mem a_bin_counter;
    cl_mem a_coarse_counter;
    cl_mem a_fine_counter;
    cl_mem a_num_active_tiles;
    cl_mem a_num_bin_segs;
    cl_mem a_num_subtris;
    cl_mem a_num_tile_segs;

    #ifdef DEVICE_IMAGE_ENABLED
    // Images
    cl_mem t_tri_data;
    cl_mem t_tri_header;
    cl_mem t_vertex_buffer;
    #endif

    // Renderbuffers
    cl_mem t_color_buffer;
    cl_mem t_depth_buffer;
    cl_mem t_stencil_buffer;

    cl_uint c_height_tiles;
    cl_uint c_width_tiles;
    cl_uint c_width_bins;
    cl_uint c_height_bins;
    cl_uint c_viewport_height;
    cl_uint c_viewport_width;
    cl_uint c_color_buffer_mode;

    // Clear
    cl_uint c_deferred_clear;
    cl_ulong c_clear_write_values;
    cl_ushort c_clear_enabled_data;

    // Vertex attributes
    cl_mem                  mem_vertex_attribs;             // sizeof(float[DEVICE_VERTEX_ATTRIBUTE_SIZE][4])
    cl_mem                  mem_vertex_attrib_datas;        // sizeof(vertex_attribute_data_t[DEVICE_VERTEX_ATTRIBUTE_SIZE])
    size_t                  vertex_attrib_pointers_size;
    cl_mem                  mem_vertex_attrib_pointers[DEVICE_VERTEX_ATTRIBUTE_SIZE];
    
    // Textures
    cl_mem                  texture_units[DEVICE_TEXTURE_UNITS];    // binded texture ids
    cl_mem                  texture_datas;                          // sizeof(texture_data_t[DEVICE_TEXTURE_UNITS])
    
    // Primitive Configuration
    render_mode_t           render_mode;
    uint32_t                primitive_config;
    cl_mem                  mem_uniform_buffer;                        // sizeof(cl_uchar[TRIANGLE_PRIMITIVE_CONFIGS][DEVICE_UNIFORM_CAPACITY])
    cl_mem                  mem_uniform_subbuffers[TRIANGLE_PRIMITIVE_CONFIGS]; // subbuffers for vertex shader
    cl_mem                  mem_rop_configs;                        // sizeof(rop_config_t[TRIANGLE_PRIMITIVE_CONFIGS])

    // Sync objects
    uint32_t                cummulative_vertices;
    uint32_t                assembled_triangles;
    uint32_t                vertex_command_queue_index;
    cl_command_queue        vertex_command_queues[DEVICE_VERTEX_COMMAND_QUEUE_SIZE];
    cl_command_queue        raster_command_queue;
    cl_event                assembly_wait_event[DEVICE_VERTEX_COMMAND_QUEUE_SIZE];
    cl_event                fine_wait_event;
    cl_event                previous_wait_event;
} device_context_t;

#define CL_UNSUPORTED_MAPING(...) \
{                                                                       \
    printf("OpenCL unsupported mapping at %s:%d. " #__VA_ARGS__ "\n", __FILE__, __LINE__);    \
    exit(-1);                                                      \
}

#define CL_CHECK(...)       \
{                                                                       \
    cl_int __error = __VA_ARGS__;                                             \
    if (__error != CL_SUCCESS) {                                   \
    printf("OpenCL error at %s:%d. " #__VA_ARGS__ " returned %d.\n", __FILE__, __LINE__, (int)__error);    \
    exit(__error);                                                      \
    } \
}

#define CL_ASSIGN_CHECK(_LEFT, ...)                                     \
{                                                                       \
    cl_int error;                                                       \
    _LEFT = __VA_ARGS__;                                                      \
    if (error != CL_SUCCESS) {                                     \
    printf("OpenCL error at %s:%d. " #__VA_ARGS__ " returned %d.\n", __FILE__, __LINE__, (int)error);      \
    exit(error);   \
    }                                                     \
}


#include <backend/pipeline/triangle_setup.o.c>
#include <backend/pipeline/bin_raster.o.c>
#include <backend/pipeline/coarse_raster.o.c>
#include <kernels/force_clear.ocl.c>

/**
 * @param shared Not created device_shared_object_t
 * 
 * @post shared object is created
 */
void device_create_shared_objects(device_shared_objects_t* shared);

/**
 * @param context
 * @param width
 * @param height
 * @param color_buffer_mode
 * 
 * @return id of the renderbuffer created
 * 
 * creates a renderbuffer of the colorbuffer mode
 */
int device_create_colorbuffer(device_context_t* context, size_t width, size_t height, cl_uint color_buffer_mode);

/**
 * @param context Not created device_context_t
 * @param shared Created device_shared_object_t
 * 
 * device context is created
 */
void device_create_context(device_context_t *context, device_shared_objects_t* shared);

/**
 * @param context
 * @param source_size
 * @param source_code
 * 
 * @post creates a program from a binary source file
 * @return id of the program created.
 */
int device_create_program_from_source(device_context_t* context, size_t source_size, const unsigned char* source_code);



/**
 * @post Wait until all launches are done
 */
void device_finish(device_context_t* context);



void device_create_shared_objects(device_shared_objects_t* shared) 
{
    CL_CHECK(clGetPlatformIDs(1, &shared->platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(shared->platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &shared->device_id, NULL));
    CL_ASSIGN_CHECK(shared->context, clCreateContext(NULL, 1, &shared->device_id, NULL, NULL,  &error));

    size_t triangle_setup_size = sizeof(triangle_setup_o);
    const unsigned char *triangle_setup_bin = triangle_setup_o;
    CL_ASSIGN_CHECK(shared->triangle_setup_program, clCreateProgramWithBinary(shared->context, 1, &shared->device_id, &triangle_setup_size, &triangle_setup_bin, NULL, &error));
    CL_CHECK(clBuildProgram(shared->triangle_setup_program, 1, &shared->device_id, NULL, NULL, NULL));

    size_t bin_raster_size = sizeof(bin_raster_o);
    const unsigned char *bin_raster_bin = bin_raster_o;
    CL_ASSIGN_CHECK(shared->bin_raster_program, clCreateProgramWithBinary(shared->context, 1, &shared->device_id, &bin_raster_size, &bin_raster_bin, NULL, &error));
    CL_CHECK(clBuildProgram(shared->bin_raster_program, 1, &shared->device_id, NULL, NULL, NULL));

    size_t coarse_raster_size = sizeof(coarse_raster_o);
    const unsigned char *coarse_raster_bin = coarse_raster_o;
    CL_ASSIGN_CHECK(shared->coarse_raster_program, clCreateProgramWithBinary(shared->context, 1, &shared->device_id, &coarse_raster_size, &coarse_raster_bin, NULL, &error));
    CL_CHECK(clBuildProgram(shared->coarse_raster_program, 1, &shared->device_id, NULL, NULL, NULL));

    size_t force_clear_size = sizeof(force_clear_ocl);
    const unsigned char *force_clear_bin = force_clear_ocl;
    CL_ASSIGN_CHECK(shared->clear_program, clCreateProgramWithBinary(shared->context, 1, &shared->device_id, &force_clear_size, &force_clear_bin, NULL, &error));
    CL_CHECK(clBuildProgram(shared->clear_program, 1, &shared->device_id, NULL, NULL, NULL));

    #ifdef DEVICE_IMAGE_ENABLE
    {
        cl_image_format image_format; 
        
        image_format = (cl_image_format) {
            .image_channel_data_type = CL_UNSIGNED_INT16,
            .image_channel_order = CL_DEPTH,
        };
        CL_ASSIGN_CHECK(shared->mem_depthbuffer, clCreateImage2D(shared->context, CL_MEM_READ_WRITE, image_format, 1, 1, 0, NULL, &error));
        image_format = (cl_image_format) {
            .image_channel_data_type = CL_UNSIGNED_INT8,
            .image_channel_order = CL_A,
        };
        CL_ASSIGN_CHECK(shared->mem_stencilbuffer, clCreateImage2D(shared->context, CL_MEM_READ_WRITE, image_format, 1, 1, 0, NULL, &error));
        image_format = (cl_image_format) {
            .image_channel_data_type = CL_UNSIGNED_INT8,
            .image_channel_order = CL_RGBA,
        };
        CL_ASSIGN_CHECK(shared->mem_texture, clCreateImage2D(shared->context, CL_MEM_READ_WRITE, image_format, 1, 1, 0, NULL, &error));
    }
    #else
    {
        CL_ASSIGN_CHECK(shared->mem_depthbuffer, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &error));
        CL_ASSIGN_CHECK(shared->mem_stencilbuffer, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &error));
        CL_ASSIGN_CHECK(shared->mem_texture, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &error));
    }
    #endif

    CL_ASSIGN_CHECK(shared->mem_vertex_attrib_pointer, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &error));

}

void device_create_context(device_context_t *context, device_shared_objects_t* shared) {
    context->shared_objects = shared;

    // Kernels
    context->vertex_shader_kernel = NULL;
    CL_ASSIGN_CHECK(context->triangle_setup_arrays_kernel, clCreateKernel(shared->triangle_setup_program, "triangle_setup_arrays", &error));
    CL_ASSIGN_CHECK(context->triangle_setup_range_kernel, clCreateKernel(shared->triangle_setup_program, "triangle_setup_range", &error));
    CL_ASSIGN_CHECK(context->bin_raster_kernel, clCreateKernel(shared->bin_raster_program, "bin_raster", &error));
    CL_ASSIGN_CHECK(context->coarse_raster_kernel, clCreateKernel(shared->coarse_raster_program, "coarse_raster", &error));
    CL_ASSIGN_CHECK(context->clear_kernel, clCreateKernel(shared->clear_program, "force_clear", &error));
    context->fine_raster_kernel = NULL;

    // Buffers
    const size_t max_number_subtriangles = DEVICE_MAX_NUMBER_TRIANGLES + DEVICE_MAX_NUMBER_SUBTRIANGLES;
    CL_ASSIGN_CHECK(context->g_tri_subtris, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_uchar[max_number_subtriangles]), NULL, &error));
    
    const size_t triangle_header_size = sizeof(triangle_header_t[max_number_subtriangles]);
    const size_t triangle_data_size = sizeof(triangle_data_t[max_number_subtriangles]);
    const size_t vertex_buffer_size = sizeof(cl_float4[DEVICE_VERTICES_SIZE][DEVICE_VARYING_SIZE + 1]);
    CL_ASSIGN_CHECK(context->g_tri_header, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, triangle_header_size, NULL, &error));
    CL_ASSIGN_CHECK(context->g_tri_data, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, triangle_data_size, NULL, &error));     
    CL_ASSIGN_CHECK(context->g_vertex_buffer, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, vertex_buffer_size, NULL, &error));

    #ifdef DEVICE_IMAGE_ENABLED
    {
        cl_image_desc image_desc;
        cl_image_format image_format;
        
        image_format = {
            .image_channel_order = CL_RGBA,
            .image_channel_data_type = CL_UNSIGNED_INT32,
        };

        image_desc = {
            .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
            .image_row_pitch = 0,
            .image_width = triangle_header_size / sizeof(cl_uint4),
            .mem_object = context->g_tri_header, 
        };
        
        CL_ASSIGN_CHECK(context->t_tri_header, clCreateImage(shared->context, CL_MEM_READ_ONLY, &image_format, &image_desc, NULL, &error));

        image_desc = {
            .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
            .image_width = triangle_data_size / sizeof(cl_uint4),
            .image_row_pitch = 0,
            .mem_object = context->g_tri_data, 
        };

        CL_ASSIGN_CHECK(context->t_tri_data, clCreateImage(shared->context, CL_MEM_READ_ONLY, &image_format, &image_desc, NULL, &error));

        image_format = {
            .image_channel_order = CL_RGBA,
            .image_channel_data_type = CL_FLOAT,
        };
        image_desc = {
            .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
            .image_width = vertex_buffer_size / sizeof(cl_float4),
            .image_row_pitch = 0,
            .mem_object = context->g_vertex_buffer,
        };

        CL_ASSIGN_CHECK(context->t_vertex_buffer, clCreateImage(shared->context, CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &error));
    }
    #endif

    size_t max_number_bin_segments  = CR_MAXBINS_SQR*CR_BIN_STREAMS_SIZE; // At least one segment x bin
    CL_ASSIGN_CHECK(context->g_bin_first_seg, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]), NULL, &error)); 
    CL_ASSIGN_CHECK(context->g_bin_seg_data, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_bin_segments * CR_BIN_SEG_SIZE]), NULL, &error)); 
    CL_ASSIGN_CHECK(context->g_bin_seg_next, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_bin_segments]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_bin_seg_count, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_bin_segments]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_bin_total, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE]), NULL, &error));

    size_t max_number_tile_segments = CR_MAXTILES_SQR; // At least one segment x tile
    CL_ASSIGN_CHECK(context->g_active_tiles, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXTILES_SQR]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_tile_first_seg, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_int[CR_MAXTILES_SQR]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_tile_seg_data, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_tile_segments * CR_TILE_SEG_SIZE]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_tile_seg_next, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_tile_segments]), NULL, &error));
    CL_ASSIGN_CHECK(context->g_tile_seg_count, clCreateBuffer(shared->context, CL_MEM_READ_WRITE, sizeof(cl_int[max_number_tile_segments]), NULL, &error));

    // Atomics
    cl_uint ZERO = 0;
    size_t max_number_triangles = DEVICE_MAX_NUMBER_TRIANGLES;
    CL_ASSIGN_CHECK(context->a_bin_counter, clCreateBuffer(shared->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO, &error));
    CL_ASSIGN_CHECK(context->a_coarse_counter, clCreateBuffer(shared->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO, &error));
    CL_ASSIGN_CHECK(context->a_fine_counter, clCreateBuffer(shared->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO, &error));
    CL_ASSIGN_CHECK(context->a_num_active_tiles, clCreateBuffer(shared->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO, &error));
    CL_ASSIGN_CHECK(context->a_num_bin_segs, clCreateBuffer(shared->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO, &error));
    CL_ASSIGN_CHECK(context->a_num_subtris, clCreateBuffer(shared->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &max_number_triangles, &error));
    CL_ASSIGN_CHECK(context->a_num_tile_segs, clCreateBuffer(shared->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &ZERO, &error));

    CL_ASSIGN_CHECK(context->mem_vertex_attribs, clCreateBuffer(shared->context, CL_MEM_READ_ONLY, sizeof(cl_float4[DEVICE_VERTEX_ATTRIBUTE_SIZE]), NULL, &error));
    CL_ASSIGN_CHECK(context->mem_vertex_attrib_datas, clCreateBuffer(shared->context, CL_MEM_READ_ONLY, sizeof(vertex_attribute_data_t[DEVICE_VERTEX_ATTRIBUTE_SIZE]), NULL, &error));

    CL_ASSIGN_CHECK(context->texture_datas, clCreateBuffer(shared->context, CL_MEM_READ_ONLY, sizeof(gl_texture_data_t[DEVICE_TEXTURE_UNITS]), NULL, &error));

    context->primitive_config = 0;
    CL_ASSIGN_CHECK(context->mem_uniform_buffer, clCreateBuffer(shared->context, CL_MEM_READ_ONLY, sizeof(cl_uchar[TRIANGLE_PRIMITIVE_CONFIGS][DEVICE_UNIFORM_CAPACITY]), NULL, &error));
    for (int i = 0; i < TRIANGLE_PRIMITIVE_CONFIGS; ++i) {
        cl_buffer_region buffer_region = {
                .origin = DEVICE_UNIFORM_CAPACITY * i,
                .size = DEVICE_UNIFORM_CAPACITY
        };
        CL_ASSIGN_CHECK(context->mem_uniform_subbuffers[i], clCreateSubBuffer(
            context->mem_uniform_buffer, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
            &buffer_region, &error
        ));
    }
    CL_ASSIGN_CHECK(context->mem_rop_configs, clCreateBuffer(shared->context, CL_MEM_READ_ONLY, sizeof(rop_config_t[TRIANGLE_PRIMITIVE_CONFIGS]), NULL, &error));

    context->vertex_command_queue_index = 0;
    for (int i=0; i < DEVICE_VERTEX_COMMAND_QUEUE_SIZE; ++i) {
        CL_ASSIGN_CHECK(context->vertex_command_queues[i], clCreateCommandQueue(shared->context, shared->device_id, 0, &error));
    }
    CL_ASSIGN_CHECK(context->raster_command_queue, clCreateCommandQueue(shared->context, shared->device_id, 0, &error));

    // assign constant args
    cl_kernel kernel;
    cl_uint counter;

    const cl_uint c_max_subtris = DEVICE_MAX_NUMBER_TRIANGLES + DEVICE_MAX_NUMBER_SUBTRIANGLES;
    const cl_uint c_max_bin_segs  = CR_MAXBINS_SQR*CR_BIN_STREAMS_SIZE; // At least one segment x bin
    const cl_uint c_max_tile_segs = CR_MAXTILES_SQR; // At least one segment x tile
    const cl_uint c_samples_log2 = 0; // TODO: not impl
    const cl_uint c_bin_batch_sz = DEVICE_BIN_SUB_GROUPS*DEVICE_SUB_GROUP_THREADS; // at least use all threads available, maybe can be fine tuned for larger executions

    kernel = context->triangle_setup_range_kernel; 
    counter = 0;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),       &context->a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),        &context->g_tri_header));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),          &context->g_tri_data));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),       &context->g_tri_subtris));
    #ifdef DEVICE_IMAGE_ENABLED
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),     &context->t_vertex_buffer)); 
    #else
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),     &context->g_vertex_buffer));
    #endif
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(c_max_subtris),       &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(c_samples_log2),      &c_samples_log2));

    kernel = context->triangle_setup_arrays_kernel; 
    counter = 0;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),       &context->a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),        &context->g_tri_header));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),          &context->g_tri_data));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),       &context->g_tri_subtris));
    #ifdef DEVICE_IMAGE_ENABLED
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),     &context->t_vertex_buffer)); 
    #else
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),     &context->g_vertex_buffer));
    #endif
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(c_max_subtris),       &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(c_samples_log2),      &c_samples_log2));

    kernel = context->bin_raster_kernel;
    counter = 0;
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->a_bin_counter));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->a_num_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_bin_first_seg));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_bin_seg_count));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_bin_seg_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_bin_seg_next));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_bin_total));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_tri_header));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_tri_subtris));
    #ifdef DEVICE_IMAGE_ENABLED
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->t_tri_header));
    #endif
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_bin_batch_sz),      &c_bin_batch_sz));
    CL_CHECK(clSetKernelArg(kernel, counter + 1, sizeof(c_max_bin_segs),      &c_max_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter + 2, sizeof(c_max_subtris),       &c_max_subtris));

    kernel = context->coarse_raster_kernel;
    counter = 0;
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),    &context->a_coarse_counter));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),  &context->a_num_active_tiles));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),      &context->a_num_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),     &context->a_num_tile_segs));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),       &context->a_num_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),      &context->g_active_tiles));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),     &context->g_bin_first_seg));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),     &context->g_bin_seg_count));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),      &context->g_bin_seg_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),      &context->g_bin_seg_next));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),         &context->g_bin_total));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),    &context->g_tile_first_seg));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),    &context->g_tile_seg_count));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),     &context->g_tile_seg_data));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),     &context->g_tile_seg_next));
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),        &context->g_tri_header));
    #ifdef DEVICE_IMAGE_ENABLED
    CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),        &context->t_tri_header));
    #endif
    CL_CHECK(clSetKernelArg(kernel, counter + 2, sizeof(c_max_bin_segs),      &c_max_bin_segs));
    CL_CHECK(clSetKernelArg(kernel, counter + 3, sizeof(c_max_subtris),       &c_max_subtris));
    CL_CHECK(clSetKernelArg(kernel, counter + 4, sizeof(c_max_tile_segs),     &c_max_tile_segs));

    // default values
    context->cummulative_vertices = 0;
    context->assembled_triangles = 0;
    context->vertex_command_queue_index = 0;
    context->primitive_config = 0;
    context->previous_wait_event = NULL;
}

void device_finish(device_context_t* context) {
    CL_CHECK(clFinish(context->raster_command_queue));
}

int device_create_program_from_source(device_context_t* context, size_t source_size, const unsigned char* source_code) 
{
    device_shared_objects_t* shared = context->shared_objects;
    size_t program_id = shared->program_size;

    CL_ASSIGN_CHECK(shared->programs[program_id], clCreateProgramWithBinary(shared->context, 1, &shared->device_id, &source_size, &source_code, NULL, &error));
    CL_CHECK(clBuildProgram(shared->programs[program_id], 1, &shared->device_id, NULL, NULL, NULL));
    CL_ASSIGN_CHECK(shared->kernels[program_id][0], clCreateKernel(shared->programs[program_id], "gl_vertex_shader", &error));
    CL_ASSIGN_CHECK(shared->kernels[program_id][1], clCreateKernel(shared->programs[program_id], "fine_raster_single_sample", &error));

    shared->program_size += 1;

    return program_id;
}

static size_t get_bytes_from_colorbuffer_mode(cl_uint color_buffer_mode) 
{
    switch (color_buffer_mode) {
        case TEX_R8:
            return 1;
        case TEX_RG8:
        case TEX_RGBA4:
        case TEX_RGB5_A1:
        case TEX_RGB565:
            return 2;
        case TEX_RGB8:
            return 3;
        default:
        case TEX_RGBA8:
            return 4;
    }
}

static cl_image_format get_image_format_from_colorbuffer_mode(cl_uint color_buffer_mode) 
{
    cl_image_format image_format;

    switch (color_buffer_mode) {
        case TEX_R8:
            image_format = {
                .image_channel_order = CL_R,
                .image_channel_data_type = CL_UNSIGNED_INT8,
            };
            break;
        case TEX_RG8:
            image_format = {
                .image_channel_order = CL_RG,
                .image_channel_data_type = CL_UNSIGNED_INT8,
            };
            break;
        case TEX_RGB8:
            image_format = {
                .image_channel_order = CL_RGB,
                .image_channel_data_type = CL_UNSIGNED_INT8,
            };
            break;
        case TEX_RGB565:
            image_format = {
                .image_channel_order = CL_RGB,
                .image_channel_data_type = CL_UNORM_SHORT_565,
            };
            break;
        case TEX_RGBA4:
            CL_UNSUPORTED_MAPING("TEX_RGBA4");
            break;
        case TEX_RGB5_A1:
            image_format = {
                .image_channel_order = CL_RGBA,
                .image_channel_data_type = CL_UNORM_SHORT_555,
            };
            break;
        default:
        case TEX_RGBA8:
            image_format = {
                .image_channel_order = CL_RGBA,
                .image_channel_data_type = CL_UNSIGNED_INT8,
            };
            break;
    }

    return image_format;
}

int device_create_colorbuffer(device_context_t* context, size_t width, size_t height, cl_uint color_buffer_mode) 
{
    device_shared_objects_t* shared = context->shared_objects;
    size_t renderbuffer_id = shared->renderbuffers_size;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        cl_image_format image_format = get_image_format_from_colorbuffer_mode(color_buffer_mode);
        CL_ASSIGN_CHECK(shared->renderbuffers[renderbuffer_id], clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, width, height, 0, NULL, &error));
    }
    #else
    {
        size_t buffer_size = width * height * get_bytes_from_colorbuffer_mode(color_buffer_mode);
        CL_ASSIGN_CHECK(shared->renderbuffers[renderbuffer_id], clCreateBuffer(shared->context, CL_MEM_READ_WRITE, buffer_size, NULL, &error));
    }
    #endif

    shared->renderbuffers_size += 1;
    return renderbuffer_id;
}

int device_create_depthbuffer(device_context_t* context, size_t width, size_t height) 
{
    device_shared_objects_t* shared = context->shared_objects;
    size_t renderbuffer_id = shared->renderbuffers_size;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        cl_image_format image_format = {
            .image_channel_order = CL_DEPTH, // this may be not supported for OpenCL 1.2 check cl_khr_depth_images extension
            .image_channel_data_type = CL_UNSIGNED_INT16,
        };
        CL_ASSIGN_CHECK(shared->renderbuffers[renderbuffer_id], clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, width, height, 0, NULL, &error));
    }
    #else
    {
        size_t buffer_size = width * height * sizeof(cl_ushort);
        CL_ASSIGN_CHECK(shared->renderbuffers[renderbuffer_id], clCreateBuffer(shared->context, CL_MEM_READ_WRITE, buffer_size, NULL, &error));
    }
    #endif
    
    shared->renderbuffers_size += 1;
    return renderbuffer_id;
}

int device_create_stencilbuffer(device_context_t* context, size_t width, size_t height) 
{
    device_shared_objects_t* shared = context->shared_objects;
    size_t renderbuffer_id = shared->renderbuffers_size;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        cl_image_format image_format = {
            .image_channel_order = CL_A,
            .image_channel_data_type = CL_UNSIGNED_INT8,
        };
        CL_ASSIGN_CHECK(shared->renderbuffers[renderbuffer_id], clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format, width, height, 0, NULL, &error));
    }
    #else
    {
        size_t buffer_size = width * height * sizeof(cl_uchar);
        CL_ASSIGN_CHECK(shared->renderbuffers[renderbuffer_id], clCreateBuffer(shared->context, CL_MEM_READ_WRITE, buffer_size, NULL, &error));
    }
    #endif
    
    shared->renderbuffers_size += 1;
    return renderbuffer_id;
}

int device_create_ro_buffer(device_context_t* context, size_t size) 
{
    device_shared_objects_t* shared = context->shared_objects;
    size_t buffer_id = shared->buffers_size;

    CL_ASSIGN_CHECK(shared->buffers[buffer_id], clCreateBuffer(shared->context, CL_MEM_READ_ONLY, size, NULL, &error));

    shared->buffers_size += 1;
    return buffer_id;
}

int device_create_2d_texture(device_context_t* context, size_t width, size_t height, cl_uint texture_mode) 
{
    device_shared_objects_t* shared = context->shared_objects;
    size_t texture_id = shared->textures_size;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        cl_image_format image_format = get_image_format_from_colorbuffer_mode(texture_mode);
        CL_ASSIGN_CHECK(shared->textures[texture_id], clCreateImage2D(shared->context, CL_MEM_READ_WRITE, &image_format, width, height, 0, NULL, &error));
    }
    #else
    {
        size_t pixel_size = get_bytes_from_colorbuffer_mode(texture_mode);
        size_t buffer_size = width * height * pixel_size;
        CL_ASSIGN_CHECK(shared->textures[texture_id], clCreateBuffer(shared->context, CL_MEM_READ_WRITE, buffer_size, NULL, &error));
    }
    #endif

    shared->textures_size += 1;
    return texture_id;
}

void device_write_2d_texture(device_context_t* context, size_t texture_id, size_t width, size_t height, int texture_mode, const void* data) 
{
    cl_command_queue queue = context->vertex_command_queues[context->vertex_command_queue_index];
    device_shared_objects_t* shared = context->shared_objects;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {width, height, 1};
        CL_CHECK(clEnqueueWriteImage(
            queue,
            shared->textures[texture_id],
            CL_TRUE,
            origin,
            region,
            0,
            0,
            data,
            0,
            NULL,
            NULL
        ));
    }
    #else
    {
        size_t pixel_size = get_bytes_from_colorbuffer_mode(texture_mode);
        size_t buffer_size = width * height * pixel_size;
        CL_CHECK(clEnqueueWriteBuffer(
            queue,
            shared->textures[texture_id],
            CL_TRUE,
            0,
            buffer_size,
            data,
            0,
            NULL,
            NULL
        ));
    }
    #endif
}

void device_write_on_buffer(device_context_t* context, size_t buffer_id, size_t offset, size_t size, const void* data) 
{
    cl_command_queue queue = context->vertex_command_queues[context->vertex_command_queue_index];
    device_shared_objects_t* shared = context->shared_objects;

    CL_CHECK(clEnqueueWriteBuffer(
        queue,
        shared->buffers[buffer_id],
        CL_TRUE,
        offset,
        size,
        data,
        0,
        NULL,
        NULL
    ));
}

void device_bind_texture_unit(device_context_t* context, size_t unit_index, size_t texture_id) 
{
    context->texture_units[unit_index] = context->shared_objects->textures[texture_id];
}

void device_load_texture_datas(device_context_t* context, const gl_texture_data_t* texture_datas) 
{
    cl_command_queue queue = context->vertex_command_queues[context->vertex_command_queue_index];

    CL_CHECK(clEnqueueWriteBuffer(
        queue,
        context->texture_datas,
        CL_TRUE,
        0,
        sizeof(gl_texture_data_t[DEVICE_TEXTURE_UNITS]),
        texture_datas,
        0,
        NULL,
        NULL
    ));
}

void device_bind_buffer_to_vertex_attribute_pointer(device_context_t* context, size_t attrib_index, size_t buffer_id) 
{
    cl_mem mem_attribute = context->shared_objects->buffers[buffer_id];

    context->mem_vertex_attrib_pointers[attrib_index] = mem_attribute;
    CL_CHECK(clSetKernelArg(context->vertex_shader_kernel, 3 + attrib_index, sizeof(mem_attribute), &mem_attribute));
}

void device_reset_context(device_context_t *context) 
{
    context->cummulative_vertices = 0;
    context->assembled_triangles = 0;
    context->vertex_command_queue_index = 0;
    context->primitive_config = 0;
    // clean atomics
    {
        cl_int ZERO = 0;
        cl_uint max_number_triangles = DEVICE_MAX_NUMBER_TRIANGLES;
        CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_bin_counter, CL_FALSE, 0, sizeof(cl_int), &ZERO, 0, NULL, NULL));
        CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_coarse_counter, CL_FALSE, 0, sizeof(cl_int), &ZERO, 0, NULL, NULL));
        CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_fine_counter, CL_FALSE, 0, sizeof(cl_int), &ZERO, 0, NULL, NULL));
        CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_num_active_tiles, CL_FALSE, 0, sizeof(cl_int), &ZERO, 0, NULL, NULL));
        CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_num_subtris, CL_FALSE, 0, sizeof(cl_int), &max_number_triangles, 0, NULL, NULL));
        CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_num_bin_segs, CL_FALSE, 0, sizeof(cl_int), &ZERO, 0, NULL, NULL));
        CL_CHECK(clEnqueueWriteBuffer(context->raster_command_queue, context->a_num_tile_segs, CL_FALSE, 0, sizeof(cl_int), &ZERO, 0, NULL, NULL));
        CL_CHECK(clFinish(context->raster_command_queue));
    }
}

cl_uint get_kernel_fine_raster_offset() 
{
    cl_uint offset = DEVICE_TEXTURE_UNITS;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        offset += 1;
    }
    #endif

    return offset;
}

void device_load_program(device_context_t *context, int program_id) 
{
    context->vertex_shader_kernel   = context->shared_objects->kernels[program_id][0];
    context->fine_raster_kernel     = context->shared_objects->kernels[program_id][1];

    cl_kernel kernel_varying, kernel_attributes;
    CL_ASSIGN_CHECK(kernel_varying, clCreateKernel(context->shared_objects->programs[program_id], "gl_varying_data", &error));
    CL_ASSIGN_CHECK(kernel_attributes, clCreateKernel(context->shared_objects->programs[program_id], "gl_attribute_data", &error));

    cl_uint kernel_varying_num_args;
    clGetKernelInfo(kernel_varying, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &kernel_varying_num_args, NULL);
    cl_uint c_vertex_size = sizeof(cl_float4[kernel_varying_num_args]);

    cl_uint kernel_attributes_num_args;
    clGetKernelInfo(kernel_attributes, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &kernel_attributes_num_args, NULL);
    context->vertex_attrib_pointers_size = kernel_attributes_num_args - 1;

    CL_CHECK(clSetKernelArg(context->triangle_setup_arrays_kernel, 9, sizeof(c_vertex_size),       &c_vertex_size));
    CL_CHECK(clSetKernelArg(context->triangle_setup_range_kernel, 10, sizeof(c_vertex_size),       &c_vertex_size));

    cl_kernel kernel = context->fine_raster_kernel;
    cl_uint offset = get_kernel_fine_raster_offset();

    CL_CHECK(clSetKernelArg(kernel, 21 + offset, sizeof(context->c_color_buffer_mode), &context->c_color_buffer_mode));
    CL_CHECK(clSetKernelArg(kernel, 25 + offset, sizeof(context->c_viewport_height),   &context->c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 26 + offset, sizeof(context->c_viewport_width),    &context->c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, 27 + offset, sizeof(context->c_width_tiles),       &context->c_width_tiles));
}

void device_load_framebuffer(device_context_t *context, size_t colorbuffer_id, size_t depthbuffer_id, size_t stencilbuffer_id, size_t framebuffer_width, size_t framebuffer_height, cl_uint color_buffer_mode) 
{

    context->t_color_buffer     = context->shared_objects->renderbuffers[colorbuffer_id];
    context->t_depth_buffer     = context->shared_objects->renderbuffers[depthbuffer_id];
    context->t_stencil_buffer   = context->shared_objects->renderbuffers[stencilbuffer_id];
    context->c_viewport_height  = framebuffer_height;
    context->c_viewport_width   = framebuffer_width;
    context->c_height_tiles      = ((framebuffer_height-1) / CR_TILE_SIZE) + 1;
    context->c_width_tiles      = ((framebuffer_width-1) / CR_TILE_SIZE) + 1;
    context->c_height_bins       = ((framebuffer_height-1) / (CR_TILE_SIZE * CR_BIN_SIZE)) + 1;
    context->c_width_bins       = ((framebuffer_width-1) / (CR_TILE_SIZE * CR_BIN_SIZE)) + 1;
    context->c_color_buffer_mode = color_buffer_mode;
    
    
    cl_uint c_viewport_height = framebuffer_height;
    cl_uint c_viewport_width  = framebuffer_width;
    cl_uint c_height_bins      = context->c_height_bins;
    cl_uint c_width_bins      = context->c_width_bins;
    cl_uint c_height_tiles    = context->c_height_tiles;
    cl_uint c_width_tiles      = context->c_width_tiles;
    cl_uint c_num_bins      = c_height_bins * c_width_bins;

    cl_uint c_color_buffer_mode = context->c_color_buffer_mode;

    cl_kernel kernel;

    kernel = context->triangle_setup_arrays_kernel;
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(c_viewport_width),    &c_viewport_width));

    kernel = context->triangle_setup_range_kernel;
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(c_viewport_width),    &c_viewport_width));

    kernel = context->bin_raster_kernel;
    cl_uint extra = 0;
    #ifdef DEVICE_IMAGE_ENABLED
        extra = 1;
    #endif
    CL_CHECK(clSetKernelArg(kernel, 11 + extra, sizeof(c_height_bins),        &c_height_bins));
    CL_CHECK(clSetKernelArg(kernel, 14 + extra, sizeof(c_num_bins),   &c_num_bins));
    CL_CHECK(clSetKernelArg(kernel, 16 + extra, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 17 + extra, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, 18 + extra, sizeof(c_width_bins),        &c_width_bins));

    kernel = context->coarse_raster_kernel;
    CL_CHECK(clSetKernelArg(kernel, 17 + extra, sizeof(c_height_tiles),      &c_height_tiles));
    CL_CHECK(clSetKernelArg(kernel, 21 + extra, sizeof(c_num_bins),          &c_num_bins));
    CL_CHECK(clSetKernelArg(kernel, 22 + extra, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 23 + extra, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, 24 + extra, sizeof(c_width_bins),        &c_width_bins));
    CL_CHECK(clSetKernelArg(kernel, 25 + extra, sizeof(c_width_tiles),       &c_width_tiles));

    // if there is a fine raster kernel loaded, update its args too
    if (context->fine_raster_kernel == NULL) return;

    kernel = context->fine_raster_kernel;
    cl_uint offset = get_kernel_fine_raster_offset();

    CL_CHECK(clSetKernelArg(kernel, 21 + offset, sizeof(c_color_buffer_mode), &c_color_buffer_mode));
    CL_CHECK(clSetKernelArg(kernel, 25 + offset, sizeof(c_viewport_height),   &c_viewport_height));
    CL_CHECK(clSetKernelArg(kernel, 26 + offset, sizeof(c_viewport_width),    &c_viewport_width));
    CL_CHECK(clSetKernelArg(kernel, 27 + offset, sizeof(c_width_tiles),       &c_width_tiles));
    
}

void device_clear_framebuffer(device_context_t *context, cl_ulong clear_write_values, cl_ushort clear_enabled_data, cl_uint deferred_clear) 
{
    context->c_clear_write_values = clear_write_values;
    context->c_clear_enabled_data = clear_enabled_data;
    context->c_deferred_clear = deferred_clear;

    cl_uint extra = 0;
    #ifdef DEVICE_IMAGE_ENABLED
        extra = 1;
    #endif
    CL_CHECK(clSetKernelArg(context->coarse_raster_kernel, 16 + extra, sizeof(context->c_deferred_clear), &context->c_deferred_clear));
}

void device_load_vertex_attributes(device_context_t *context, const float attribs[DEVICE_VERTEX_ATTRIBUTE_SIZE][4], const vertex_attribute_data_t* attrib_datas) 
{
    cl_command_queue queue = context->vertex_command_queues[context->vertex_command_queue_index];
    cl_event write_wait_event[2];

    CL_CHECK(clEnqueueWriteBuffer(
        queue,
        context->mem_vertex_attribs,
        CL_FALSE,
        0,
        sizeof(cl_float[DEVICE_VERTEX_ATTRIBUTE_SIZE][4]),
        attribs,
        0,
        NULL,
        &write_wait_event[0]
    ));

    CL_CHECK(clEnqueueWriteBuffer(
        queue,
        context->mem_vertex_attrib_datas,
        CL_FALSE,
        0,
        sizeof(vertex_attribute_data_t[DEVICE_VERTEX_ATTRIBUTE_SIZE]),
        attrib_datas,
        0,
        NULL,
        &write_wait_event[1]
    ));

    CL_CHECK(clWaitForEvents(2, write_wait_event));
}

void device_load_config(device_context_t *context, size_t config_id, rop_config_t* rop_config, cl_uchar* uniform_data) 
{
    context->render_mode = rop_config->render_mode;
    
    cl_event write_wait_event[2];

    CL_CHECK(clEnqueueWriteBuffer(
        context->vertex_command_queues[config_id],
        context->mem_rop_configs,
        CL_FALSE,
        sizeof(rop_config_t) * config_id,
        sizeof(rop_config_t),
        rop_config,
        0,
        NULL,
        &write_wait_event[0]
    ));

    CL_CHECK(clEnqueueWriteBuffer(
        context->vertex_command_queues[config_id],
        context->mem_uniform_subbuffers[config_id],
        CL_FALSE,
        0,
        DEVICE_UNIFORM_CAPACITY,
        uniform_data,
        0,
        NULL,
        &write_wait_event[1]
    ));
    
    CL_CHECK(clWaitForEvents(2, write_wait_event));
}

void device_destroy_context(device_context_t* context) {

}

void device_load_raster_parameters(device_context_t *context) {
   
}

/*
*/
void device_launch_range_triangle_assembly(device_context_t *context, size_t num_vertices, size_t size, const unsigned short *data, size_t config) 
{
    cl_command_queue queue = context->vertex_command_queues[context->vertex_command_queue_index];

    // vertex shader
    {
        size_t gwo = context->cummulative_vertices;
        size_t gws = num_vertices;

        cl_kernel kernel = context->vertex_shader_kernel;
        cl_uint counter = 0;
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->mem_vertex_attribs));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->mem_vertex_attrib_datas));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->mem_uniform_subbuffers[config]));
        for (cl_uint ap = 0; ap < context->vertex_attrib_pointers_size; ++ap)
        {
            CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->mem_vertex_attrib_pointers[ap]));
        }
        #ifdef DEVICE_IMAGE_ENABLED
        {
            CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->t_vertex_buffer)); 
        }
        #else
        {
            CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_vertex_buffer));
        }
        #endif

        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, &gwo, &gws, NULL, 0, NULL, NULL));
    }

    // triangle setup
    const cl_uint index_buffer_arg = 1;
    const cl_uint vertex_offset_arg = 6;
    const cl_uint render_mode_arg = 8; 
    const cl_uint primitive_config_arg = 13;
    const cl_uint primitive_config = config;

    {
        size_t gwo = context->assembled_triangles;
        size_t gws = size/3; // for GL_TRIANGLES mode
        cl_uint vertex_offset = context->cummulative_vertices;
        
        cl_mem range_buffer;
        CL_ASSIGN_CHECK(range_buffer, clCreateBuffer(context->shared_objects->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_ushort[size]), (void*) data, &error));

        CL_CHECK(clSetKernelArg(context->triangle_setup_range_kernel, index_buffer_arg, sizeof(range_buffer), &range_buffer));
        CL_CHECK(clSetKernelArg(context->triangle_setup_range_kernel, vertex_offset_arg, sizeof(vertex_offset), &vertex_offset));
        CL_CHECK(clSetKernelArg(context->triangle_setup_range_kernel, render_mode_arg, sizeof(context->render_mode), &context->render_mode));
        CL_CHECK(clSetKernelArg(context->triangle_setup_range_kernel, primitive_config_arg, sizeof(primitive_config), &primitive_config));
        
        CL_CHECK(clEnqueueNDRangeKernel(queue, context->triangle_setup_range_kernel, 1, &gwo, &gws, NULL, 0, NULL, &context->assembly_wait_event[context->vertex_command_queue_index]));

        CL_CHECK(clReleaseMemObject(range_buffer));
    }

    // update data
    context->cummulative_vertices += num_vertices;
    context->assembled_triangles += size/3;
    context->vertex_command_queue_index = (context->vertex_command_queue_index + 1) % DEVICE_VERTEX_COMMAND_QUEUE_SIZE;
}

static void device_launch_arrays_triangle_assembly(device_context_t *context, size_t num_vertices, size_t size)
{
    cl_command_queue queue = context->vertex_command_queues[context->vertex_command_queue_index];

    // vertex shader
    {
        size_t gwo = context->cummulative_vertices;
        size_t gws = num_vertices;

        cl_kernel kernel = context->vertex_shader_kernel;
        cl_uint counter = 0;
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->mem_vertex_attribs));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->mem_vertex_attrib_datas));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->mem_uniform_subbuffers[context->primitive_config-1]));
        for (cl_uint ap = 0; ap < DEVICE_VERTEX_ATTRIBUTE_SIZE; ++ap)
        {
            CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->mem_vertex_attrib_pointers[ap]));
        }
        #ifdef DEVICE_IMAGE_ENABLED
        {
            CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->t_vertex_buffer)); 
        }
        #else
        {
            CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_vertex_buffer));
        }
        #endif

        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, &gwo, &gws, NULL, 0, NULL, NULL));
    }

    // triangle setup
    static cl_uint vertex_offset_arg = 5;
    static cl_uint primitive_config_arg = vertex_offset_arg+6;

    {
        size_t gwo = context->assembled_triangles;
        size_t gws = size/3; // for GL_TRIANGLES mode
        cl_uint vertex_offset = context->cummulative_vertices;
        

        CL_CHECK(clSetKernelArg(context->triangle_setup_range_kernel, vertex_offset_arg, sizeof(vertex_offset), &vertex_offset));
        CL_CHECK(clSetKernelArg(context->triangle_setup_range_kernel, primitive_config_arg, sizeof(context->primitive_config), &context->primitive_config));
        
        CL_CHECK(clEnqueueNDRangeKernel(queue, context->triangle_setup_range_kernel, 1, &gwo, &gws, NULL, 0, NULL, &context->assembly_wait_event[context->vertex_command_queue_index]));

    }

    // update data
    context->cummulative_vertices += num_vertices;
    context->assembled_triangles += size/3;
    context->vertex_command_queue_index = (context->vertex_command_queue_index + 1) % DEVICE_VERTEX_COMMAND_QUEUE_SIZE;
}

static void device_launch_triangle_rasterization(device_context_t *context) {
    size_t global_work_size[2], local_work_size[2];

    {
        cl_uint c_num_tris_arg = 15;
        #ifdef CONF_BIN_IMAGE_ENABLED
        ++c_num_tris_arg;
        #endif
        cl_uint c_num_tris = context->assembled_triangles;
        CL_CHECK(clSetKernelArg(context->bin_raster_kernel, c_num_tris_arg, sizeof(c_num_tris), &c_num_tris));
        
        local_work_size[0] = DEVICE_SUB_GROUP_THREADS;
        local_work_size[1] = DEVICE_BIN_SUB_GROUPS;
        global_work_size[0] = local_work_size[0] * CR_BIN_STREAMS_SIZE;
        global_work_size[1] = local_work_size[1];
        CL_CHECK(clEnqueueNDRangeKernel(context->raster_command_queue, context->bin_raster_kernel, 2, NULL, global_work_size, local_work_size, context->vertex_command_queue_index, context->assembly_wait_event, NULL));
    }

    {
        local_work_size[0] = DEVICE_SUB_GROUP_THREADS;
        local_work_size[1] = DEVICE_COARSE_SUB_GROUPS;
        global_work_size[0] = local_work_size[0] * DEVICE_NUM_CORES;
        global_work_size[1] = local_work_size[1];
        CL_CHECK(clEnqueueNDRangeKernel(context->raster_command_queue, context->coarse_raster_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    }

    {
        // program could be share so we need to set args again
        cl_kernel kernel = context->fine_raster_kernel;
        cl_uint counter = 0;
        const cl_uint c_max_bin_segs  = CR_MAXBINS_SQR*CR_BIN_STREAMS_SIZE; // At least one segment x bin
        const cl_uint c_max_tile_segs = CR_MAXTILES_SQR; // At least one segment x tile
        const cl_uint c_max_subtris = DEVICE_MAX_NUMBER_TRIANGLES + DEVICE_MAX_NUMBER_SUBTRIANGLES;
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->mem_uniform_buffer));
        for(int i=0; i<DEVICE_TEXTURE_UNITS; ++i) 
        {
            CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->texture_units[i]));
        }
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),     &context->texture_datas));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->a_fine_counter));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->a_num_active_tiles));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->a_num_bin_segs));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->a_num_subtris));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->a_num_tile_segs));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_active_tiles));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_tile_first_seg));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_tile_seg_count));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_tile_seg_data));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_tile_seg_next));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_tri_header));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->t_color_buffer));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->t_depth_buffer));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->t_stencil_buffer));
        #ifdef DEVICE_IMAGE_ENABLED
        {
            CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->t_tri_data));
            CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->t_tri_header));
            CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->t_vertex_buffer));
        }
        #else
        {
            CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_tri_data));
            CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem), &context->g_vertex_buffer));
        }
        #endif
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(cl_mem),      &context->mem_rop_configs));

        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(context->c_clear_write_values),   &context->c_clear_write_values));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(context->c_clear_enabled_data),   &context->c_clear_enabled_data));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(context->c_color_buffer_mode),    &context->c_color_buffer_mode));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_bin_segs),                  &c_max_bin_segs));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_subtris),                   &c_max_subtris));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(c_max_tile_segs),                 &c_max_tile_segs));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(context->c_viewport_height),      &context->c_viewport_height));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(context->c_viewport_width),       &context->c_viewport_width));
        CL_CHECK(clSetKernelArg(kernel, counter++, sizeof(context->c_width_tiles),          &context->c_width_tiles));

        cl_uint wait_events = context->previous_wait_event == NULL ? 0 : 1;

        local_work_size[0] = DEVICE_SUB_GROUP_THREADS;
        local_work_size[1] = DEVICE_FINE_SUB_GROUPS;
        global_work_size[0] = local_work_size[0] * DEVICE_NUM_CORES;
        global_work_size[1] = local_work_size[1];
        CL_CHECK(clEnqueueNDRangeKernel(context->raster_command_queue, context->fine_raster_kernel, 2, NULL, global_work_size, local_work_size, wait_events, context->previous_wait_event == NULL ? NULL : &context->previous_wait_event, &context->fine_wait_event));
        CL_CHECK(clFinish(context->raster_command_queue));
    }

    // reset atomics
    {

    }

}

void device_get_pixels(device_context_t *context, size_t x, size_t y, size_t width, size_t height, void* pixels) 
{
    cl_command_queue queue = context->raster_command_queue;

    #ifdef DEVICE_IMAGE_ENABLED
    {
        size_t origin[3] = {x, y, 0};
        size_t region[3] = {width, height, 1};
        CL_CHECK(clEnqueueReadImage(
            queue,
            context->t_color_buffer,
            CL_TRUE,
            origin,
            region,
            0,
            0,
            pixels,
            0,
            NULL,
            NULL
        ));
    }
    #else
    {
        size_t pixel_size = get_bytes_from_colorbuffer_mode(context->c_color_buffer_mode);
        size_t buffer_size = width * height * pixel_size;
        size_t buffer_offset = (y * context->c_viewport_width + x) * pixel_size;
        CL_CHECK(clEnqueueReadBuffer(
            queue,
            context->t_color_buffer,
            CL_TRUE,
            buffer_offset,
            buffer_size,
            pixels,
            0,
            NULL,
            NULL
        ));
    }
    #endif
}


#endif // FRONTEND_DEVICE_H
