

.PHONEY: all egl

./lib/egl.so: src/egl.c
	gcc -o lib/egl.so src/egl.c -shared -I./include

all: egl

egl:
	mkdir -p lib
	make ./lib/egl.so
