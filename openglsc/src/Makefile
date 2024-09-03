VORTEX_PATH = ../..

include ./common.mk

KERNELS_C = $(wildcard kernels/*.c)
KERNELS_CL = $(wildcard kernels/*.cl)

KERNELS_POCL_C = $(wildcard kernels/*.pocl.c)
KERNELS_CL_C = $(wildcard kernels/*.cl.c)

INCLUDES = -I../include

LIB_PATH = ../lib

.PHONY: all clean vortex opencl kernels

all: kernels glsc2.c.so

kernels: $(KERNELS_CL)
	make -C kernels

glsc2.c.o: glsc2.c config.h debug.h types.h kernel.c $(KERNELS_C)
	$(CC) $(CXXFLAGS) -o $@ -c $< $(INCLUDES) -fPIC

glsc2.c.so: glsc2.c.o
	$(CC) glsc2.c.o -shared -o $(LIB_PATH)/glsc2.c.so -L$(VORTEX_RT_PATH)/stub -lvortex $(POCL_RT_PATH)/lib/libOpenCL.so

glsc2.opencl.o: glsc2.c config.h debug.h types.h kernel.c $(KERNELS_CL_C)
	$(CC) $(CXXFLAGS) -o $@ -c glsc2.c $(INCLUDES) -fPIC -DC_OPENCL_HOSTDRIVER

glsc2.vortex.o: glsc2.c config.h debug.h types.h kernel.c $(KERNELS_POCL_C)
	$(CC) $(CXXFLAGS) -o $@ -c $< $(INCLUDES) -fPIC -DC_OPENCL_VORTEX

vortex:
	make -C kernels vortex
	make glsc2.vortex.o

opencl:
	make -C kernels opencl
	make glsc2.opencl.o

clean:
	rm *.o

clean-all: clean
	rm *.so