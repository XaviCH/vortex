XLEN ?= 32
TOOLDIR ?= /opt

TARGET ?= opaesim

XRT_SYN_DIR  ?= ../../../hw/syn/xilinx/xrt
XRT_DEVICE_INDEX ?= 0

ifeq ($(XLEN),64)
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv64-gnu-toolchain
VX_CFLAGS += -march=rv64imafd -mabi=lp64d
K_CFLAGS += -march=rv64imafd -mabi=ilp64d
STARTUP_ADDR ?= 0x180000000
else
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
K_CFLAGS += -march=rv32imaf -mabi=ilp32f
STARTUP_ADDR ?= 0x80000000
endif

VORTEX_PATH = $(shell realpath ~/repositories/vortex)

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf
RISCV_SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/$(RISCV_PREFIX)

POCL_CC_PATH ?= $(TOOLDIR)/pocl/compiler
POCL_RT_PATH ?= $(TOOLDIR)/pocl/runtime
OPENGLSC_PATH ?= $(realpath ../..)

VORTEX_RT_PATH ?= $(VORTEX_PATH)/runtime
VORTEX_KN_PATH ?= $(VORTEX_PATH)/kernel

FPGA_BIN_DIR ?= $(VORTEX_RT_PATH)/opae

LLVM_VORTEX ?= $(TOOLDIR)/llvm-vortex
LLVM_POCL ?= $(TOOLDIR)/llvm-vortex

K_CFLAGS   += -v -O3 --sysroot=$(RISCV_SYSROOT) --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH) -Xclang -target-feature -Xclang +vortex
K_CFLAGS   += -fno-rtti -fno-exceptions -nostartfiles -fdata-sections -ffunction-sections
K_CFLAGS   += -I$(VORTEX_KN_PATH)/include -DNDEBUG -DLLVM_VOTEX
K_LDFLAGS  += -Wl,-Bstatic,--gc-sections,-T$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(VORTEX_KN_PATH)/libvortexrt.a -lm

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -Wno-deprecated-declarations -Wno-unused-parameter -Wno-narrowing
CXXFLAGS += -pthread
CXXFLAGS += -I$(POCL_RT_PATH)/include
CXXFLAGS += -I$(OPENGLSC_PATH)/include

ifdef HOSTDRIVER
	CXXFLAGS += -DHOSTDRIVER
	LDFLAGS += -lGLESv2 -lEGL
else ifdef HOSTGPU 
	LDFLAGS += -lOpenCL ../lib/GLSC2/glsc2-gpu.c.so
else
	LDFLAGS += -L$(VORTEX_RT_PATH)/stub -lvortex $(POCL_RT_PATH)/lib/libOpenCL.so ../lib/GLSC2/glsc2.c.so
endif

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

ifeq ($(TARGET), fpga)
	OPAE_DRV_PATHS ?= libopae-c.so
else
ifeq ($(TARGET), asesim)
	OPAE_DRV_PATHS ?= libopae-c-ase.so
else
ifeq ($(TARGET), opaesim)
	OPAE_DRV_PATHS ?= libopae-c-sim.so
endif	
endif
endif

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
