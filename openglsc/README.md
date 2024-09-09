# OpenGL Installation with OpenCL drivers

Generate OpenGLSC2 lib and a custom OpenCL compiler.

```bash
make -C openglsc opencl
```

The makefile generates two binaries: clcompiler, to compile OpenCL files into binaries; and libGLSCv2.opencl.so, the OpenGL SC 2.0 driver.
For compiling binaries.

```bash
./openglsc/clcompiler <src> <dst> <flags...>
```

# GLSL to OpenCL

The vertex shader and fragment shader requires to be in the same file. The entry point has to be main_vs and main_fs, respectively.
All vertex attributes are extended to float4.
```c
attribute float scale;
attribute vec2 coord;
attribute vec3 position;
attribute vec4 color;
void main() {
    ...
}
```
```c
kernel void main_vs(
    const float4 scale,
    const float4 coord,
    const float4 position,
    const float4 color,
) {
    ...
}
```
Vertex Attributes Pointers requires a global pointer.
```c
attribute vec3 position;
attribute vec4 color;
void main() {
    ...
}
```
```c
kernel void main_vs(
    ...
    global const float4* position,
    global const float4* color,
    ...
) {
    int gid = get_global_id(0);

    float4 _position = position[gid]; 
    float4 _color = color[gid]; 

    ...
}
```

For varying vertex attributes it is require to create a parameter in vs and fs with the same name as in the example.
```c
kernel void main_vs(
    ...
    global const float4* in_attrib,
    ...
) {
    ...
}

kernel void main_fs(
    ...
    global float4* out_attrib,
    ...
) {
    ...
}
```
Uniforms are declared as constant pointers.
```c
kernel void main_vs(
    ...
    constant float4* uniform,
    ...
) {
    ...
}
```
Vertex shader requires an explicit declaration of gl_Position.
```c
kernel void main_vs(
    ...
    global float4* gl_Position,
    ...
) {
    int gid = get_global_id(0);

    ...

    gl_Position[gid] = (float4) {1,1,1,1};
}
```
Fragment shader requires an explicit declaration of gl_FragColor.
```c
kernel void main_fs(
    ...
    global float4* gl_FragColor,
) {
    int gid = get_global_id(0);
    
    ...

    gl_FragColor[gid] = (float4) {1,1,1,1};
}
```

Shaders have to be compiled with the -cl-kernel-arg-info flag. 

Also in tests/opengl/* there are some snippets on how to create shaders. 

