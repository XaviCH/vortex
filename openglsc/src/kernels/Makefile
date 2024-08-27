include ../common.mk

all: kernel.stencil_test.c kernel.blending.c kernel.scissor_test.c kernel.color.c kernel.dither.c kernel.clear.c kernel.perspective_division.c kernel.readnpixels.c kernel.rasterization.c kernel.rasterization.triangle_strip.c kernel.rasterization.triangle_fan.c kernel.viewport_division.c kernel.depth.c kernel.strided_write.c

%.pocl: %.cl
	echo $(VORTEX_PATH)
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_VORTEX_CFLAGS="$(K_CFLAGS)" POCL_VORTEX_LDFLAGS="$(K_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o $@ $<
	
%.c: %.pocl
	xxd -i $< > $@

clean:
	rm *.pocl *.dump