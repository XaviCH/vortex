VORTEX_PATH = ../../..

include ../common.mk

KERNELS_CL = $(wildcard *.cl)
KERNELS_POCL_C = $(KERNELS_CL:.cl=.pocl.c)
KERNELS_CL_C = $(KERNELS_CL:.cl=.ocl.c)


AMM = amm.c

.PHONY: all clean vortex opencl

all: vortex opencl

%.pocl: %.cl
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_VORTEX_CFLAGS="$(K_CFLAGS)" POCL_VORTEX_LDFLAGS="$(K_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o $@ $<

%.ocl: %.cl
	$(VORTEX_GLSC_PATH)/clcompiler $< $@ -DC_OPENCL_HOSTDRIVER

%.pocl.c: %.pocl
	xxd -i $< > $@

%.ocl.c: %.ocl
	xxd -i $< > $@

vortex: $(KERNELS_POCL_C)

opencl: $(KERNELS_CL_C)

clean:
	rm *.pocl *.ocl *.dump

clean-all:
	rm  *.c *.pocl *.ocl *.dump