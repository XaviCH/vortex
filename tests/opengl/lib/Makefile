PROJECT = lib

SRCS = main.cc

LD_PROJECT_FLAGS ?= 

OPTS ?=

KERNELS = $(wildcard GLSC2/kernel.*.cl)

include ../common.mk

GLSC2: glsc2.c.so glsc2-gpu.c.so

kernel.%.pocl: GLSC2/kernel.%.cl
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_VORTEX_CFLAGS="$(K_CFLAGS)" POCL_VORTEX_LDFLAGS="$(K_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o GLSC2/$@ $<
	
kernel.%.c: kernel.%.pocl
	xxd -i GLSC2/$< > GLSC2/$@

glsc2.c.o: kernel.color.c kernel.dithering.c kernel.clear.c kernel.perspective_division.c kernel.readnpixels.c kernel.rasterization.triangle.c kernel.viewport_division.c kernel.fill.c kernel.depth.c kernel.strided_write.c GLSC2/glsc2.c
	$(CC) $(CXXFLAGS) -o GLSC2/glsc2.c.o -c GLSC2/glsc2.c -I./../include -fPIC

glsc2.c.so: glsc2.c.o
	$(CC) GLSC2/glsc2.c.o -shared -o GLSC2/glsc2.c.so -L$(VORTEX_RT_PATH)/stub -lvortex $(POCL_RT_PATH)/lib/libOpenCL.so

glsc2-gpu.c.o:
	$(CC) $(CXXFLAGS) -o GLSC2/glsc2-gpu.c.o -c GLSC2/glsc2.c -I./../include -fPIC -DHOSTGPU

glsc2-gpu.c.so: glsc2-gpu.c.o
	$(CC) GLSC2/glsc2-gpu.c.o -shared -o GLSC2/glsc2-gpu.c.so -lOpenCL -DHOSTGPU

clean:
	rm GLSC2/kernel.*.pocl GLSC2/kernel.*.c
