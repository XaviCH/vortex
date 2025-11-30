CC = gcc

ABS_PATH = $(realpath .)

SRC_PATH = ../
INCLUDE_PATH = ../../include/

INSTALL_PATH = $(ABS_PATH)/../../..

HEADER_FILES = $(wildcard *.h)

INCLUDES += -I$(SRC_PATH) -I$(INCLUDE_PATH)
LIBS += -lOpenCL


clc: clc.c $(HEADER_FILES)
	$(CC) -o $@ $< $(INCLUDES) $(LIBS)

glslc: glslc.c $(HEADER_FILES)
	$(CC) -o $@ $< $(INCLUDES) $(LIBS) -DINSTALL_PATH="$(INSTALL_PATH)"