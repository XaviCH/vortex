VORTEX_PATH = ../../..

include ../common.mk

KERNELS_CL = $(wildcard *.cl)
KERNELS_C = $(KERNELS_CL:.cl=.c)

all: $(KERNELS_C)

%.pocl: %.cl
	echo $(VORTEX_PATH)
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_VORTEX_CFLAGS="$(K_CFLAGS)" POCL_VORTEX_LDFLAGS="$(K_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o $@ $<
	
%.c: %.pocl
	xxd -i $< > $@

clean:
	rm *.pocl *.dump