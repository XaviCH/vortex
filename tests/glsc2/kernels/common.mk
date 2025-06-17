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

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf
RISCV_SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/$(RISCV_PREFIX)

POCL_CC_PATH ?= $(TOOLDIR)/pocl/compiler
POCL_RT_PATH ?= $(TOOLDIR)/pocl/runtime
OPENGLSC_PATH ?= $(realpath ..)

VORTEX_RT_PATH ?= $(VORTEX_PATH)/runtime
VORTEX_KN_PATH ?= $(VORTEX_PATH)/kernel
VORTEX_GLSC_PATH ?= $(VORTEX_PATH)/glsc2
VORTEX_EGL_PATH ?= $(VORTEX_PATH)/egl

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
CXXFLAGS += -I$(VORTEX_GLSC_PATH)/include
CXXFLAGS += -I$(VORTEX_EGL_PATH)/include
CXXFLAGS += -I$(VORTEX_GLSC_PATH)/src
CXXFLAGS += -I$(VORTEX_PATH)

ifdef HOSTGPU 
	CXXFLAGS += -DC_OPENCL_HOST
	LDFLAGS += -lOpenCL
else
	CXXFLAGS += -DC_OPENCL_VORTEX
	LDFLAGS += -L$(VORTEX_RT_PATH)/stub -lvortex $(POCL_RT_PATH)/lib/libOpenCL.so $(VORTEX_GLSC_PATH)/libGLSCv2.vortex.so $(VORTEX_EGL_PATH)/lib/egl.so
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

OBJS := $(addsuffix .o, $(notdir $(SRCS)))

all: $(PROJECT)

%.cc.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.c.o: %.c
	$(CC) $(CXXFLAGS) -c $< -o $@

$(PROJECT): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) $(LD_PROJECT_FLAGS) -o $@

run-hostgpu: $(PROJECT)
	./$(PROJECT) $(OPTS)

run-simx: $(PROJECT)
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/simx:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT)
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-opae: $(PROJECT)
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/opae:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-xrt: $(PROJECT)
ifeq ($(TARGET), hw)
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XRT_INI_PATH=$(XRT_SYN_DIR)/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/xrt:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
else
	XCL_EMULATION_MODE=$(TARGET) XRT_INI_PATH=$(XRT_SYN_DIR)/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/xrt:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)	
endif

display:
	feh -Z -F --force-aliasing -Y output.ppm

ifdef HOSTGPU
run:
	$(MAKE) clean
	$(MAKE) run-hostgpu
	$(MAKE) display
else 
run: 
	$(MAKE) clean
	$(MAKE) run-simx
	$(MAKE) display
endif

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend

clean-all: clean
	rm -rf *.dump *.pocl *.ocl

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
