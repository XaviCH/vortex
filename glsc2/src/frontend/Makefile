VORTEX_PATH = ../..

SRC_PATH = ../
INCLUDE_PATH = ../../include/
# include ./common.mk

KERNELS_C = $(wildcard kernels/*.c)
KERNELS_CL = $(wildcard kernels/*.cl)

KERNELS_POCL_C = $(wildcard kernels/*.pocl.c)
KERNELS_CL_C = $(wildcard kernels/*.ocl.c)

INCLUDES = -I$(SRC_PATH) -I$(INCLUDE_PATH)
LIBS = -lOpenCL

LIB_PATH = ../lib

.PHONY: all clean vortex opencl kernels

all: kernels glsc2.c.so

kernels: $(KERNELS_CL)
	make -C kernels

HEADER_FILES = $(wildcard *.h)

%.o: %.c $(HEADER_FILES)
	$(MAKE) -C $(SRC_PATH)/backend/pipeline
	$(CC) $(CXXFLAGS) -o $@ -c $< $(INCLUDES) -fPIC

libGLSC2.so: glsc2.o
	$(CC) -o $@ $< -shared $(LIBS)

glsc2.c.o: glsc2_tmp.c config.h debug.h types.h kernel.c $(KERNELS_C)
	$(CC) $(CXXFLAGS) -o $@ -c $< $(INCLUDES) -fPIC

glsc2.c.so: glsc2.c.o
	$(CC) glsc2.c.o -shared -o $(LIB_PATH)/glsc2.c.so -L$(VORTEX_RT_PATH)/stub -lvortex $(POCL_RT_PATH)/lib/libOpenCL.so

glsc2.opencl.o: glsc2_tmp.c config.h debug.h types.h kernel.c $(KERNELS_CL_C)
	$(CC) $(CXXFLAGS) -o $@ -c glsc2_tmp.c $(INCLUDES) -fPIC -DC_OPENCL_HOST

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