VORTEX_PATH = $(realpath ../../../..)
GLSC_PATH = $(VORTEX_PATH)/glsc2
SRC_PATH = $(GLSC_PATH)/src 

UTILS_CL = $(wildcard ../utils/*.cl)
KERNELS_CL = $(wildcard *.cl)
KERNELS_O_C = $(KERNELS_CL:.cl=.o.c)

CL_FLAGS += -Werror -D__COMPILER_RELATIVE_PATH__ -I$(SRC_PATH)
# OPT_FLAGS = -D'DEVICE_SUB_GROUP_SUPPORT=0' -D'DEVICE_IMAGE_SUPPORT=0'

.PHONY: all clean clean-all

%.o: %.cl $(UTILS_CL)
	$(GLSC_PATH)/clcompiler $< $@ -DC_OPENCL_HOST $(CL_FLAGS) $(OPT_FLAGS)

%.o.c: %.o
	xxd -i $< > $@

all: $(KERNELS_O_C)

clean:
	rm *.o

clean-all: clean
	rm  *.o.c
