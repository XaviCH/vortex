GLSC2_PATH = ../..

SRC_PATH = $(GLSC2_PATH)/src

INCLUDES = -I$(SRC_PATH) -I$(GLSC2_PATH)/include

LIBS = -lOpenCL

PIPELINE_PATH = $(SRC_PATH)/backend/pipeline

HEADER_FILES = $(wildcard *.h) $(wildcard $(SRC_PATH)/*.h)

C_FILES = $(wildcard *.c)
OBJECT_FILES = $(C_FILES:.c=.o)

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
	rm -f *.o

clean-all: clean
	rm -f *.so