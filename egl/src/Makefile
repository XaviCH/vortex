CC = gcc
VORTEX_PATH = ../..
INCLUDE = -I$(VORTEX_PATH)/glsc2/include -I$(VORTEX_PATH)/egl/include

all: egl.o

egl.o: egl.c
	$(CC) -o $@ -c $< -fPIC $(INCLUDE)
