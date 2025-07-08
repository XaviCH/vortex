# OpenGL Installation with OpenCL drivers

Generate OpenGL SC 2.0 lib and a custom OpenCL compiler.

```bash
make -C openglsc opencl
```

Use libGLSCv2.opencl.so as OpenGL SC 2.0 driver.
Use glslcompiler as an offline compiler for glsl.

# GLSL to OpenCL

For compiling binaries.

```bash
./openglsc/glslcompiler <src> <dst> <flags...>
```

# TODO

## Enqueue device work smarter
The driver always tries to enqueue the work at API calls, this ends up on a lot of work waste and a lot of parallelization waste.
The idea is to compact work, until it cannot be compacted more and then launch executions.
## Add support for low level Nvidia CUDA API
For non OpenCL supporter devices like Nvidia Xavier or Nvidia Orin add CUDA API calls instead of OpenCL.
## Build compiler for GLSL

## Add non polygon rendering to the pipeline

## Optimize kernels depending on the hardware
Fetch caracteristics of the hardware before compiling and then compile it.