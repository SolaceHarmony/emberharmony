ROOT := ..
CXX ?= clang++
AUDIO_OUT ?=
KCORO_OUT ?=
ARCH := $(shell uname -m)

ifeq ($(strip $(AUDIO_OUT)),)
$(error AUDIO_OUT must name the current liquid-audio native archive directory)
endif
ifeq ($(strip $(KCORO_OUT)),)
$(error KCORO_OUT must name the current kcoro-sys native archive directory)
endif

TARGET := build/native_spec_replay_probe
SOURCE := native_spec_replay_probe.cpp
INCLUDES := \
	-I$(ROOT)/crates/liquid-audio/native/include \
	-I$(ROOT)/crates/liquid-audio/native/src/detokenizer \
	-I$(ROOT)/crates/liquid-audio/native/src/model \
	-I$(ROOT)/crates/liquid-audio/native/src/runtime \
	-I$(ROOT)/crates/kcoro-sys/vendor/kcoro_arena/include
CXXFLAGS := -std=c++23 -O3 -Wall -Wextra -Werror -ffp-contract=off -pthread
AUDIO_LIBS := \
	$(AUDIO_OUT)/liblfm_voice_session.a \
	$(AUDIO_OUT)/liblfm_flashkern_engine.a \
	$(AUDIO_OUT)/liblfm_detokenizer.a \
	$(AUDIO_OUT)/liblfm_conformer.a \
	$(AUDIO_OUT)/liblfm_frontend.a \
	$(AUDIO_OUT)/liblfm_flashkern_prng.a \
	$(AUDIO_OUT)/liblfm_safetensors.a \
	$(AUDIO_OUT)/liblfm_sesame_detector.a \
	$(AUDIO_OUT)/liblfm_capture_format.a \
	$(AUDIO_OUT)/liblfm_voice_protocol_c.a \
	$(AUDIO_OUT)/liblfm_mimi.a
ifeq ($(ARCH),arm64)
AUDIO_LIBS += $(AUDIO_OUT)/liblfm_flashkern_neon.a
else
AUDIO_LIBS += $(AUDIO_OUT)/liblfm_flashkern_x86.a
endif
KCORO_LIBS := \
	$(KCORO_OUT)/libkcoro_arena_core.a \
	$(KCORO_OUT)/libkcoro_arena_port_posix.a
FORCE_LIBS := $(foreach lib,$(AUDIO_LIBS) $(KCORO_LIBS),-Wl,-force_load,$(lib))
FRAMEWORKS := \
	-framework Accelerate \
	-framework Security \
	-framework AudioUnit \
	-framework AudioToolbox \
	-framework CoreAudio \
	-framework CoreFoundation

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE) $(AUDIO_LIBS) $(KCORO_LIBS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCE) $(FORCE_LIBS) $(FRAMEWORKS) -o $@

clean:
	rm -f $(TARGET)
