################################################################################
#
# moonstt
#
################################################################################

MOONSTT_VERSION = 0.1.0
MOONSTT_SITE = $(BR2_EXTERNAL_MOONSTT_PATH)
MOONSTT_SITE_METHOD = local
MOONSTT_LICENSE = MIT
MOONSTT_LICENSE_FILES = LICENSE THIRD-PARTY.md

# Integer kernels are the hot path.
define MOONSTT_BUILD_CMDS
	$(MAKE) $(TARGET_CONFIGURE_OPTS) CFLAGS="$(TARGET_CFLAGS) -O2" \
		DATADIR=/usr/share/moonstt -C $(@D)/src
endef

# Model files come from tools/build-model.sh (see README.md); point
# BR2_PACKAGE_MOONSTT_MODEL_DIR at the directory holding model.bin and tok.bin.
define MOONSTT_INSTALL_TARGET_CMDS
	$(INSTALL) -D -m 0755 $(@D)/src/moonstt $(TARGET_DIR)/usr/bin/moonstt
	$(INSTALL) -D -m 0755 $(@D)/src/moonstt-ptt $(TARGET_DIR)/usr/bin/moonstt-ptt
	$(INSTALL) -D -m 0755 $(@D)/src/moonstt-dictate $(TARGET_DIR)/usr/bin/moonstt-dictate
	$(INSTALL) -D -m 0755 $(@D)/init/S96moonstt-dictate $(TARGET_DIR)/etc/init.d/S96moonstt-dictate
	$(INSTALL) -D -m 0644 $(@D)/init/moonstt-dictate.default $(TARGET_DIR)/etc/default/moonstt-dictate
	$(if $(call qstrip,$(BR2_PACKAGE_MOONSTT_MODEL_DIR)),\
		$(INSTALL) -D -m 0644 $(call qstrip,$(BR2_PACKAGE_MOONSTT_MODEL_DIR))/model.bin $(TARGET_DIR)/usr/share/moonstt/model.bin && \
		$(INSTALL) -D -m 0644 $(call qstrip,$(BR2_PACKAGE_MOONSTT_MODEL_DIR))/tok.bin $(TARGET_DIR)/usr/share/moonstt/tok.bin,\
		@echo "moonstt: BR2_PACKAGE_MOONSTT_MODEL_DIR not set; model files not installed")
endef

$(eval $(generic-package))
