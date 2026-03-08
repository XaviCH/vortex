LIBS = -lOpenCL

OBJ_FILES = $(wildcard frontend/*.o)
KERNEL_OBJ_FILES = $(wildcard backend/pipeline/*.o.c)

.PHONY: all compiler backend frontend clean clean-all

all: compiler backend frontend

libGLSC2.so: $(KERNEL_OBJ_FILES) $(OBJ_FILES)
	$(CC) -shared -o $@ $^ $(LIBS)

clean:
	$(MAKE) -C ./compiler clean
	$(MAKE) -C ./backend clean
	$(MAKE) -C ./frontend clean

clean-all: clean
	$(MAKE) -C ./compiler clean-all
	$(MAKE) -C ./backend clean-all
	$(MAKE) -C ./frontend clean-all
	rm *.so