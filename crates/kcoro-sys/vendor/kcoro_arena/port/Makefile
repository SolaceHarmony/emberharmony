.DEFAULT_GOAL := all
include ../mk/common.mk

BUILDDIR := $(KC_BUILD_ROOT)
OBJ := $(BUILDDIR)/posix.o
LIB := $(BUILDDIR)/libkcoro_port_posix.a
CFLAGS += -I../include

.PHONY: all clean
all: $(LIB)

$(LIB): $(OBJ)
	ar rcs $@ $<

$(OBJ): posix.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR):
	@mkdir -p $@

clean:
	rm -rf build
