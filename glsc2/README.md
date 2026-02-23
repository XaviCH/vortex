# OpenGL Installation with OpenCL drivers

Generate OpenGL SC 2.0 lib and a custom OpenCL compiler.

```bash
make -C glsc2 opencl
```

Use libGLSCv2.opencl.so as OpenGL SC 2.0 driver.
Use glslcompiler as an offline compiler for glsl.

# GLSL to OpenCL

For compiling binaries.

```bash
./glsc2/glslcompiler <src> <dst> <flags...>
```

# TODO


## Add support for low level Nvidia CUDA API
For non OpenCL supporter devices like Nvidia Xavier or Nvidia Orin add CUDA API calls instead of OpenCL.

## Build compiler for GLSL

## Add support for sub_group_size != 32

## Add non polygon rendering to the pipeline

## Optimize kernels depending on the hardware
Fetch caracteristics of the hardware before compiling and then compile it.

## Optimize kernels depeding on OpenGL state

## Optimize device work enqueuing 
The driver always tries to enqueue the work at API calls, this ends up on a lot of work waste and a lot of parallelization waste.
The idea is to compact work, until it cannot be compacted more and then launch executions.

## Add runtime checker to avoid runtime crash from OpenCL
There are some calls that generates permanent errors to the OpenCL context.

## Add multisampling

## Add 

## Checkout how tensor cores can be use

## TODO
refactor:
- bin/                      # binaries generated
- lib/                      # libs generated
- src/
    - backend/
        - pipeline/         # cl kernels related with pipeline
        - shaders/          # cl kernels/functions related with shaders
            - glsl/         # cl functions related with glsl language
            - gl/           # cl functions related with gl operations
            - fine_raster.cl
            - vertex_shader.cl
        - utils/            # cl reusable functions 
        - extensions/       # cl functions related with adding unsupported functionalities
    - compiler/
        - 
    - frontend/
        - glsc2.c           # implementation for GLSC2 API 
        - device.h          # c wrapper for calling device driver
        - utils.h           # c functions
        - context.h         # functions to update the context
        - manager.h         # functions to orchestrate the device calls
    - utils/
        - clcompiler.c      # code to compile cl files to binary files
    - config.h              # common configuration
    - constants.h           # common constants
    - types.h               # common types

## TODO refactor pipeline
i = [0,N]; // state of vertex attributes, unique each use
j = [0,N]; // state of fragment attributes
k = [0,N]; // shader used

vE[N]; // wait_event for vertex shader
aE[M]; // wait_event for assembly

vshader(v, i)           // compute v vertices using i vstate 
assembly(n, p, j, e)    // assembles n primitives using p method for j fstate, awaiting e vshader events
bin(e)                  // splits the primitives in bins, awaiting e assembly events
tile(b, e)              // tile the pritives in b bin, awaiting e bin event
fshader(b, et, ef?)     // rop the primitives in tiles from b bin, awaiting et tile and maybe ef fshader       

draw_{i,j,k}      -> vE[i] = vshader_{i}
draw_{i+1,j,k}    -> vE[i+1] = vshader_{i+1}
draw_{i+2,j+1,k}  -> aE[j] = assembly_{j,k,vE[i:i+1]} -> vE[i+2] = vshader_{i+2}
draw_{i+3,j+1,k+1}-> assembly_{j+1} -> bin_{k,e_j} -> tile_{k,e_j} -> fshader_{k} -> vshader_{i+3}  
