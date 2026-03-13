GLSC2_PATH = ../..

SRC_PATH = $(GLSC2_PATH)/src

INCLUDES = -I$(SRC_PATH) -I$(GLSC2_PATH)/include

LIBS = -lOpenCL

PIPELINE_PATH = $(SRC_PATH)/backend/pipeline

HEADER_FILES = $(wildcard *.h) $(wildcard $(SRC_PATH)/*.h)

OBJECT_FILES = $(wildcard *.o)

KERNEL_OBJ_FILES = $(wildcard $(PIPELINE_PATH)/*.o.c)

%.o: %.c $(HEADER_FILES)
	$(CC) $(CXXFLAGS) -o $@ -c $< $(INCLUDES) -fPIC -O3

libGLSC2.so: backend $(OBJECT_FILES)
	$(CC) -shared -o $@ $(KERNEL_OBJ_FILES) $(OBJECT_FILES) $(LIBS)

.PHONY: all backend

backend:
	$(MAKE) -C $(SRC_PATH)/backend

all: $(OBJECT_FILES)

clean:
	rm *.o

clean-all: clean
	rm *.so