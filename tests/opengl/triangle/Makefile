PROJECT = triangle

SRCS = main.cc

LD_PROJECT_FLAGS ?= 

OPTS ?=

include ../common.mk

ifdef HOSTGPU
run: clean
	$(MAKE) run-hostgpu
	feh -Z -F --force-aliasing -Y image.ppm
else ifdef HOSTDRIVER
run: clean run-hostdriver
	feh -Z -F --force-aliasing -Y image.ppm
else
run: clean run-rtlsim
	feh -Z -F --force-aliasing -Y image.ppm
endif

display-gpu:
	$(MAKE) clean
	$(MAKE) run-hostgpu
	feh -Z -F --force-aliasing -Y image.ppm