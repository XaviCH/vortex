TOOLDIR = /opt
POCL_RT_PATH ?= $(TOOLDIR)/pocl/runtime
VORTEX_RT_PATH ?= ../runtime

# Compiler 
CC = gcc

# Config
LIB_NAME = libGLSCv2.so

.PHONY: vortex opencl libGLSCv2.so.opencl libGLSCv2.so.vortex

clcompiler: src/clcompiler.c
	$(CC) -o $@ $< -lOpenCL -Iinclude

libGLSCv2.so.opencl: src/glsc2.opencl.o
	$(CC) $< -shared -o $(LIB_NAME) -lOpenCL

libGLSCv2.so.vortex: src/glsc2.vortex.o
	$(CC) $< -shared -o $(LIB_NAME) -L$(VORTEX_RT_PATH)/stub -lvortex $(POCL_RT_PATH)/lib/libOpenCL.so

opencl: clcompiler
	make -C src opencl
	make libGLSCv2.so.opencl

vortex:
	make -C src vortex
	make libGLSCv2.so.vortex

clean-all:
	rm *.so clcompiler
