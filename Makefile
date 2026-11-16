# Top-level build for standalone use.  See README.md.
VERSION ?= 0.1.0
CROSS   ?= arm-linux-gnueabi-
IMAGE   ?= moonstt-build
ARMCFLAGS = -O2 -static -march=armv5te -mcpu=arm926ej-s -Wall -Wextra -std=gnu99

.PHONY: all host model docker-image deb test clean

# Static armel binaries for the i.MX28 (needs an arm-linux-gnueabi toolchain).
all:
	$(MAKE) -C src CC=$(CROSS)gcc CFLAGS="$(ARMCFLAGS)" LDFLAGS="-static"

# Native engine build, for regression tests on a PC.
host:
	$(MAKE) -C src moonstt CFLAGS="-O2 -Wall -Wextra -std=gnu99"

model:
	tools/build-model.sh model

docker-image:
	docker build -t $(IMAGE) -f packaging/Dockerfile.build packaging

# moonstt_<ver>_armel.deb and moonstt-model_<ver>_all.deb in ./dist
deb: docker-image
	docker run --rm -v $(CURDIR):/pkg -w /pkg $(IMAGE) sh -c \
		'make all VERSION=$(VERSION) && tools/build-model.sh model && packaging/deb/build.sh $(VERSION) src model dist'

# Compare against Hugging Face reference transcripts: see tests/README.md.
test: host model
	python3 tests/regress.py src/moonstt model tests/clips

clean:
	$(MAKE) -C src clean
	rm -rf dist
