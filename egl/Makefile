
CC = gcc
VORTEX_PATH = ..
LIBS = -lX11 -lm

.PHONEY: all

all: libEGL.so

# mkdir -p lib

libEGL.so: FORCE
	$(MAKE) -C src
	$(CC) -shared -o libEGL.so src/egl.o $(LIBS)

FORCE: ;