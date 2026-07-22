ROOT := ..
KCORO := ../../../kcoro-sys/vendor/kcoro_arena
BUILD ?= build
CXX ?= clang++
CC ?= clang
CXXFLAGS ?= -O3 -std=c++23 -Wall -Wextra -Werror -pthread
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -pthread -D_GNU_SOURCE
CPPFLAGS := -I$(ROOT)/include -I$(ROOT)/src/io -I$(ROOT)/src/model -I$(ROOT)/vendor -I$(KCORO)/include
PRODUCT_CPPFLAGS := $(CPPFLAGS) -I$(ROOT)/src/detokenizer -I$(ROOT)/src/runtime -I$(ROOT)/src/mimi
PRODUCT_CXXFLAGS := -O3 -std=c++23 -ffp-contract=off -pthread -fvisibility=hidden
PRODUCT_CFLAGS := -O2 -std=c11 -pthread -D_GNU_SOURCE -fvisibility=hidden
DEPFLAGS := -MMD -MP

SYSTEM := $(shell uname -s)
MACHINE := $(shell uname -m)
ifeq ($(MACHINE),arm64)
KERNEL_ARCH := aarch64
KERNEL_CPP := kernels/aarch64/flashkern_neon.cpp
KERNEL_FLAGS := -march=armv8.3-a+bf16+i8mm
else ifeq ($(MACHINE),x86_64)
KERNEL_ARCH := x86_64
KERNEL_CPP := kernels/x86_64/flashkern_x86.cpp
KERNEL_FLAGS :=
else
$(error unsupported native architecture $(MACHINE))
endif

KCORO_CORE := kcoro_stackless kc_runtime kc_service kc_doorbell kc_team kc_fixed_scope kc_deadline
KCORO_OBJECTS := $(addprefix $(BUILD)/,$(addsuffix .o,$(KCORO_CORE))) $(BUILD)/kc_port_posix.o
HOST_OBJECTS := $(BUILD)/host_mailbox.o $(BUILD)/safetensors.o $(KCORO_OBJECTS)
OBJECTS := $(BUILD)/weight_segment_host.o $(HOST_OBJECTS)
HOST_GATE_OBJECTS := $(BUILD)/native_host_mailbox.o $(HOST_OBJECTS)

PRODUCT_CXX_SOURCES := \
	src/runtime/voice_session.cpp \
	src/runtime/lfm_platform_audio.cpp \
	src/engine/flashkern_engine.cpp \
	src/model/lfm_model.cpp \
	src/model/lfm_tokenizer.cpp \
	src/detokenizer/lfm_detokenizer.cpp \
	src/runtime/kernel_bridge.cpp \
	src/frontend/lfm_frontend.cpp \
	src/runtime/lfm_sesame_detector.cpp \
	src/runtime/lfm_capture_format.cpp \
	src/model/lfm_conformer.cpp \
	src/engine/flashkern_prng.cpp \
	src/mimi/mimi_quant.cpp \
	src/mimi/mimi_conv.cpp \
	src/mimi/mimi_seanet.cpp \
	src/mimi/mimi_transformer.cpp \
	src/mimi/mimi_decode.cpp \
	tests/native_speech_to_speech.cpp \
	$(KERNEL_CPP)
PRODUCT_C_SOURCES := \
	src/runtime/voice_protocol_c.c \
	src/runtime/kernel_protocol_c.c
KERNEL_ASM_SOURCES := \
	kernels/$(KERNEL_ARCH)/flashkern_prng.S \
	kernels/$(KERNEL_ARCH)/flashkern_rope.S \
	kernels/$(KERNEL_ARCH)/flashkern_math.S \
	kernels/$(KERNEL_ARCH)/flashkern_sampler.S \
	kernels/$(KERNEL_ARCH)/flashkern_frontend.S \
	kernels/$(KERNEL_ARCH)/flashkern_sesame.S \
	kernels/$(KERNEL_ARCH)/flashkern_capture_format.S \
	kernels/$(KERNEL_ARCH)/flashkern_detokenizer.S \
	kernels/$(KERNEL_ARCH)/flashkern_conformer.S
PRODUCT_CXX_OBJECTS := $(patsubst %.cpp,$(BUILD)/product/%.o,$(PRODUCT_CXX_SOURCES))
PRODUCT_C_OBJECTS := $(patsubst %.c,$(BUILD)/product/%.o,$(PRODUCT_C_SOURCES))
KERNEL_ASM_OBJECTS := $(patsubst %.S,$(BUILD)/product/%.o,$(KERNEL_ASM_SOURCES))
SPEECH_OBJECTS := $(BUILD)/native_speech_gate_main.o $(BUILD)/safetensors.o \
	$(PRODUCT_CXX_OBJECTS) $(PRODUCT_C_OBJECTS) $(KERNEL_ASM_OBJECTS) \
	$(KCORO_OBJECTS)

ifeq ($(SYSTEM),Darwin)
SPEECH_LIBS := -framework Accelerate -framework Security -framework AudioUnit \
	-framework AudioToolbox -framework CoreAudio -framework CoreFoundation
HOST_LIBS := -lbsm
else
SPEECH_LIBS := -lm
HOST_LIBS :=
endif

.PHONY: all clean
all: $(BUILD)/lfm-weight-segment $(BUILD)/lfm-host-mailbox-gate \
	$(BUILD)/lfm-native-speech-gate

$(BUILD):
	mkdir -p $@

$(BUILD)/weight_segment_host.o: weight_segment_host.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/native_speech_gate_main.o: native_speech_gate_main.cpp | $(BUILD)
	$(CXX) $(PRODUCT_CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/safetensors.o: $(ROOT)/src/io/safetensors.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/host_mailbox.o: $(ROOT)/src/io/lfm_host_mailbox.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/native_host_mailbox.o: $(ROOT)/tests/native_host_mailbox.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/%.o: $(KCORO)/core/src/%.c | $(BUILD)
	$(CC) $(CPPFLAGS) -I$(KCORO)/core/src $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/kc_port_posix.o: $(KCORO)/port/posix.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/product/%.o: $(ROOT)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(PRODUCT_CPPFLAGS) $(PRODUCT_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/product/$(KERNEL_CPP:.cpp=.o): $(ROOT)/$(KERNEL_CPP)
	@mkdir -p $(dir $@)
	$(CXX) $(PRODUCT_CPPFLAGS) $(PRODUCT_CXXFLAGS) $(KERNEL_FLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/product/%.o: $(ROOT)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(PRODUCT_CPPFLAGS) $(PRODUCT_CFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/product/%.o: $(ROOT)/%.S
	@mkdir -p $(dir $@)
	$(CC) $(PRODUCT_CPPFLAGS) $(KERNEL_FLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/lfm-weight-segment: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ $(HOST_LIBS) -o $@

$(BUILD)/lfm-host-mailbox-gate: $(HOST_GATE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ $(HOST_LIBS) -o $@

$(BUILD)/lfm-native-speech-gate: $(SPEECH_OBJECTS)
	$(CXX) $(PRODUCT_CXXFLAGS) $^ $(SPEECH_LIBS) -o $@

clean:
	rm -rf $(BUILD)

-include $(OBJECTS:.o=.d) $(HOST_GATE_OBJECTS:.o=.d) \
	$(SPEECH_OBJECTS:.o=.d)
