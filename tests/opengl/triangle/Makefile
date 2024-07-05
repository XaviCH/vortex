PROJECT = triangle

SRCS = main.cc

LD_PROJECT_FLAGS ?= 

OPTS ?=

include ../common.mk

ifdef HOSTGPU
run: clean run-hostgpu
	feh image.ppm
else ifdef HOSTDRIVER
run: clean run-hostdriver
	feh image.ppm
else
run: clean run-simx
	feh image.ppm
endif