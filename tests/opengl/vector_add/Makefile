# Straight forward Makefile to compile all examples in a row

# If cross-compiling, you may wish to set the following environment variable to the root location of 
# your 'sdk'
# SDKSTAGE=/home/foo/raspberrypi
include ../common.mk

LIBS=-lGLESv2 -lEGL -lm -lX11

VORTEX_LIBS=$(VORTEX_OPENGLSC_PATH)/lib/glsc2.c.so $(VORTEX_EGL_PATH)/lib/egl.so

CFLAGS+=-Ofast -ftree-vectorize -funroll-all-loops -g

COMMONSRC=./Common/esShader.c    \
          ./Common/esTransform.c \
          ./Common/esShapes.c    \
          ./Common/esUtil.c
COMMONHRD=esUtil.h

VEC_ADD=./vector_add.c

default: all

all: vector_add

clean:
	find . -name "CH??_*" | xargs rm -f

./kernel.pocl: ./kernel.cl
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_VORTEX_CFLAGS="$(K_CFLAGS)" POCL_VORTEX_LDFLAGS="$(K_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o $@ $<

./vector_add: ${VEC_ADD} ./kernel.pocl
	$(CC) $(CFLAGS) ${VEC_ADD} -o $@ $(CXXFLAGS) ${LDFLAGS} -lm

run: kernel.pocl vector_add
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/simx:$(LD_LIBRARY_PATH) ./vector_add $(OPTS)
