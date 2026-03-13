# ROAD MAP

The direction of the project is: Creating a reliable, compatible, fast and complete software renderering stack.

For these reasons multiple goals have to be acomplish before publishing a stable version. Those are explained in this document.

---

## Evaluation of Application and Speed Up Opportunities

Application is based on an Nvidia publication of a software pipeline. GPUs and general computing device, did evolve during
the years and will continue evolving. The backend has to be able to be constantly evaluated in multiple ways, where developers can decide what is need to be done and users can configure the application on their needs.

- Create an stack to evaluate the application on multiple devices.
- Create an stack to simulate the application for multiple configurations.

Once the stack is built now we can research on:

- Evaluate software opportunities for speed up.
- Evaluate hardware opportunities for speed up.

---

## Backend Extensions

Application was firstly created for triangle rasterization based on OpenGL Safety Critical 2.0 pipeline. This is far from what
hardware pipelines nor software pipelines are offering right now. A modern backend has to be abble to handle a high variety of
rendering types, for these reasons work conducted has to be done:

- Include other primitives base rendering types as points, lines and quads.
- Include other pipeline stages not supported originaly, e.g. geometry shader.
- Include a user configurable and developer friendly way to construct a pipelines, so more extensions can be added in the future.

---

## Static/Dynamic Memory Managment

Application uses static memory and dynamic memory without any management. This creates a big footprint of memory during the
program/kernels executions because all memory is required independently the size of the pipeline. Also renders that do not fit
in the pipeline cannot be executed. 

- Create a runtime memory management model for application.
- Enable pipeline to render scenes that do not fit in memory.
- Evaluate memory footprint before/after.
- Evaluate performance impact.

---

## Offline Compiler

Application runs using a basic parsing between GLSL and OpenCL, this creates a multiple problems:

- There is not direct mapping for multiple GLSL features to OpenCL. Those cannot be mapped directly.
- Modifying rendering capabilities or creating new ones could easily break the compiler. 
- Application assumes that compiler binaries are always compatible, this could leat to unexpected behaviours 
during execution without application acknowledging. 

For these reasons there is a need to create a solid compiler where also application can retrieve metadata to 
ensure correctness.

- Design an entry point for pipeline and compiler, so pipeline and compiler can be modified without requiring the
acknowledge of the other.
- Build a complete mapping between OpenCL and GLSL to enable running any shaders.
- Entry point can also be generalize to support SPIRV or HSLS.

---

## Software Performance Update

Application was not design to be a competent one, but more of a fundation, so trying to squeeze maximum throuput was not a goal. Also premature optimizations are not optimal for a good development. Once the tools and the proyect reach an stable fase,
it may be feaseble to start optimizations.

- Implement an optimized version of the software, this could lead with the adoption of multiple extensions, added entrypoints for asm instructions, memory optimizations, algorithm optimizations,...
- Evaluate the speedup with other solutions

---

## Hardware Performance Update

Application is mainly target is to be run by GPGPUs. We are competing againts ASICs designs meant to run rasterization and ROP functions at transitor level. One advantage is that designers do not do fully ASICs, but more of a mix of general porpuse cores and some other ASICs. So there is posibilities that with an increased set of instructions to accelerate our software could close the gap between hardware rasterization and software rasterization. For these reasons there is required to do this following work.

- Simulate program using VortexGPGPU or any other RTL simulator.
- Implement a hardware solution to increase throughput.
- Evaluate solution. 

---

## Integration with MESA Drivers

Building an entire all in one framework is a costly work and at the end using another opensource proyects, as did with OpenCL, to build on top, could help to improve the open source community, and also increase the usability of the application. MESA has included some software rasterization backend in their drivers, so proposing to MESA to merge our backend with their driver can be way to increase availability and reach of the proyect.

- Make entry points for MESA drivers.
- Evaluate performance of the multiple frontends.