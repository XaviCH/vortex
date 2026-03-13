LIBS = -lOpenCL

OBJ_FILES = $(wildcard frontend/*.o)
KERNEL_OBJ_FILES = $(wildcard backend/pipeline/*.o.c)

.PHONY: all compiler backend frontend clean clean-all

all: libGLSC2.so

compiler:
	$(MAKE) -C compiler

backend: compiler
	$(MAKE) -C backend

frontend:
	$(MAKE) -C frontend

libGLSC2.so: backend frontend
	$(CC) -shared -o $@ $(KERNEL_OBJ_FILES) $(OBJ_FILES) $(LIBS)

# $(MAKE) -C ./compiler clean
clean:
	$(MAKE) -C ./backend clean
	$(MAKE) -C ./frontend clean

# $(MAKE) -C ./compiler clean-all
clean-all: clean
	$(MAKE) -C ./backend clean-all
	$(MAKE) -C ./frontend clean-all
	rm -f *.so