

.PHONEY: all egl

./lib/egl.so: src/EGL/egl.c
	gcc -o lib/EGL/egl.so src/EGL/egl.c -shared -I./include

all: egl

egl:
	make 
