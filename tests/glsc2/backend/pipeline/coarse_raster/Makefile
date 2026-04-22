VORTEX_PATH ?= $(realpath ../../../../..)

GLSC_PATH = $(VORTEX_PATH)/glsc2
TESTS_PATH = $(VORTEX_PATH)/tests
SRC_PATH = $(GLSC_PATH)/src

PROJECT = main

SRCS = main.cpp

LDFLAGS = -lOpenCL

LD_PROJECT_FLAGS ?=

OPTS ?= 

PIPELINE_PATH = $(SRC_PATH)/backend/pipeline

KERNEL_OBJ_FILES = $(wildcard $(PIPELINE_PATH)/*.o.c)

OBJS := $(addsuffix .o, $(notdir $(SRCS)))

INCLUDES = -I$(VORTEX_PATH) -I$(GLSC_PATH)/src -I$(GLSC_PATH)/include

%.o: %.cl
	$(GLSC_PATH)/src/compiler/clc $< $@ $(INCLUDES) -D__COMPILER_RELATIVE_PATH__ $(CL_FLAGS) $(OPT_FLAGS)

%.o.c: %.o
	xxd -i $< > $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(PROJECT): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ $(KERNEL_OBJ_FILES) $(LDFLAGS) $(LD_PROJECT_FLAGS) -o $@

run: $(PROJECT)
	./$(PROJECT)

clean:
	rm *.o