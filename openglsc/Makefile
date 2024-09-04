TOOLDIR = /opt
POCL_RT_PATH ?= $(TOOLDIR)/pocl/runtime
VORTEX_RT_PATH ?= ../runtime

# Compiler 
CC = gcc

# Config
LIB_NAME = libGLSCv2

.PHONY: vortex opencl libGLSCv2.opencl.so libGLSCv2.vortex.so

clcompiler: src/clcompiler.c
	$(CC) -o $@ $< -lOpenCL -Iinclude

libGLSCv2.opencl.so: src/glsc2.opencl.o
	$(CC) $< -shared -o $(LIB_NAME).opencl.so -lOpenCL

libGLSCv2.vortex.so: src/glsc2.vortex.o
	$(CC) $< -shared -o $(LIB_NAME).vortex.so -L$(VORTEX_RT_PATH)/stub -lvortex $(POCL_RT_PATH)/lib/libOpenCL.so

opencl: clcompiler
	make -C src opencl
	make libGLSCv2.opencl.so

vortex:
	make -C src vortex
	make libGLSCv2.vortex.so

clean-all:
	rm *.so clcompiler
