VORTEX_PATH = ../..

include ./common.mk

INCLUDES = -I../include

LIB_PATH = ../lib

.PHONY: kernels

all: kernels glsc2.c.so

kernels:
	make -C kernels

glsc2.c.o: glsc2.c config.h
	$(CC) $(CXXFLAGS) -o $@ -c $< $(INCLUDES) -fPIC

glsc2.c.so: glsc2.c.o
	$(CC) glsc2.c.o -shared -o $(LIB_PATH)/glsc2.c.so -L$(VORTEX_RT_PATH)/stub -lvortex $(POCL_RT_PATH)/lib/libOpenCL.so

glsc2-gpu.c.o: glsc2.c config.h
	$(CC) $(CXXFLAGS) -o glsc2-gpu.c.o -c glsc2.c -I./../../include -fPIC -DHOSTGPU

glsc2-gpu.c.so: glsc2-gpu.c.o
	$(CC) glsc2-gpu.c.o -shared -o glsc2-gpu.c.so -lOpenCL -DHOSTGPU

clean:
	rm *.o

clean-all: clean
	rm *.so