# moonstt: build with the Buildroot cross toolchain (CC=...) or natively.
CC ?= cc
CFLAGS ?= -O2
CFLAGS += -Wall -Wextra -std=gnu99
DATADIR ?= /usr/share/moonstt
OBJS = moonstt.o main.o

UNAME := $(shell uname -s)
PREFIX ?= /usr
DESTDIR ?=

# moonstt-dictate needs Linux input/fb headers.
all: moonstt $(if $(filter Linux,$(UNAME)),moonstt-dictate)

moonstt: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) -lm

moonstt-dictate: dictate.c font8x16.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ dictate.c

%.o: %.c moonstt.h
	$(CC) $(CFLAGS) -DMOONSTT_DATADIR='"$(DATADIR)"' -c -o $@ $<

clean:
	rm -f moonstt moonstt-dictate $(OBJS)

install: all
	install -D -m 0755 moonstt $(DESTDIR)$(PREFIX)/bin/moonstt
	install -D -m 0755 moonstt-ptt $(DESTDIR)$(PREFIX)/bin/moonstt-ptt
	[ ! -f moonstt-dictate ] || install -D -m 0755 moonstt-dictate $(DESTDIR)$(PREFIX)/bin/moonstt-dictate

.PHONY: all clean install
