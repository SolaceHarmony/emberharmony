cmake_minimum_required(VERSION 3.21)

project(liquid_audio_native LANGUAGES C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(
    ../../kcoro-sys/vendor/kcoro_arena
    ${CMAKE_CURRENT_BINARY_DIR}/kcoro
)

set(LFM_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detokenizer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mimi
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
)

add_library(lfm_weights STATIC
    src/io/safetensors.cpp
)
target_include_directories(lfm_weights PUBLIC ${LFM_INCLUDE_DIRS})
target_compile_options(lfm_weights PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
)
target_link_libraries(lfm_weights PUBLIC kcoro)

set(LFM_VOICE_SOURCES
    src/runtime/voice_session.cpp
    src/runtime/lfm_platform_audio.cpp
    src/engine/flashkern_engine.cpp
    src/model/lfm_model.cpp
    src/model/lfm_tokenizer.cpp
    src/detokenizer/lfm_detokenizer.cpp
    src/frontend/lfm_frontend.cpp
    src/runtime/lfm_sesame_detector.cpp
    src/runtime/lfm_capture_format.cpp
    src/model/lfm_conformer.cpp
    src/engine/flashkern_prng.cpp
)

set(LFM_MIMI_SOURCES
    src/mimi/mimi_quant.cpp
    src/mimi/mimi_conv.cpp
    src/mimi/mimi_seanet.cpp
    src/mimi/mimi_transformer.cpp
    src/mimi/mimi_decode.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(LFM_KERNEL_DIR aarch64)
    set(LFM_KERNEL_CPP kernels/aarch64/flashkern_neon.cpp)
    set(LFM_KERNEL_OPTIONS -march=armv8.3-a+bf16+i8mm)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(LFM_KERNEL_DIR x86_64)
    set(LFM_KERNEL_CPP kernels/x86_64/flashkern_x86.cpp)
    set(LFM_KERNEL_OPTIONS)
else()
    message(FATAL_ERROR
        "liquid-audio supports arm64 and x86_64/Rosetta only")
endif()
list(APPEND LFM_VOICE_SOURCES ${LFM_KERNEL_CPP})

set(LFM_ASM_SOURCES
    kernels/${LFM_KERNEL_DIR}/flashkern_prng.S
    kernels/${LFM_KERNEL_DIR}/flashkern_rope.S
    kernels/${LFM_KERNEL_DIR}/flashkern_math.S
    kernels/${LFM_KERNEL_DIR}/flashkern_sampler.S
    kernels/${LFM_KERNEL_DIR}/flashkern_frontend.S
    kernels/${LFM_KERNEL_DIR}/flashkern_sesame.S
    kernels/${LFM_KERNEL_DIR}/flashkern_capture_format.S
    kernels/${LFM_KERNEL_DIR}/flashkern_detokenizer.S
    kernels/${LFM_KERNEL_DIR}/flashkern_conformer.S
)
list(APPEND LFM_VOICE_SOURCES ${LFM_ASM_SOURCES})

set_source_files_properties(
    ${LFM_KERNEL_CPP}
    ${LFM_ASM_SOURCES}
    PROPERTIES COMPILE_OPTIONS "${LFM_KERNEL_OPTIONS}"
)

add_library(lfm_voice STATIC ${LFM_VOICE_SOURCES})
target_include_directories(lfm_voice PUBLIC ${LFM_INCLUDE_DIRS})
target_compile_options(lfm_voice PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-ffp-contract=off;-fvisibility=hidden>
    $<$<COMPILE_LANGUAGE:C>:-O2;-fvisibility=hidden>
)
target_link_libraries(lfm_voice PUBLIC lfm_weights kcoro)

if(APPLE)
    target_link_libraries(lfm_voice PUBLIC
        "-framework Accelerate"
        "-framework Security"
        "-framework AudioUnit"
        "-framework AudioToolbox"
        "-framework CoreAudio"
        "-framework CoreFoundation"
    )

    # Mimi is retained for the future native Moshi tranche. LFM2.5 uses its
    # own detokenizer and must not link this dormant decoder.
    add_library(lfm_mimi STATIC EXCLUDE_FROM_ALL ${LFM_MIMI_SOURCES})
    target_include_directories(lfm_mimi PUBLIC ${LFM_INCLUDE_DIRS})
    target_compile_options(lfm_mimi PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-ffp-contract=off;-fvisibility=hidden;-Wall;-Wextra;-Werror>
    )
    target_link_libraries(lfm_mimi PUBLIC
        lfm_weights
        "-framework Accelerate"
    )
endif()

add_executable(lfm_weight_segment_host
    tools/weight_segment_host.cpp
    src/io/lfm_host_mailbox.cpp
)
target_include_directories(lfm_weight_segment_host PRIVATE ${LFM_INCLUDE_DIRS})
target_compile_options(lfm_weight_segment_host PRIVATE
    -O3 -Wall -Wextra -Werror
)
target_link_libraries(lfm_weight_segment_host PRIVATE lfm_weights kcoro)
if(APPLE)
    target_link_libraries(lfm_weight_segment_host PRIVATE bsm)
endif()

add_executable(lfm_weight_test tests/native_weight_segment.cpp)
target_include_directories(lfm_weight_test PRIVATE ${LFM_INCLUDE_DIRS})
target_compile_options(lfm_weight_test PRIVATE
    -O3 -Wall -Wextra -Werror
)
target_link_libraries(lfm_weight_test PRIVATE lfm_weights kcoro)
if(APPLE)
    target_link_libraries(lfm_weight_test PRIVATE bsm)
endif()

add_executable(lfm_host_mailbox_test
    tests/native_host_mailbox.cpp
    src/io/lfm_host_mailbox.cpp
)
target_include_directories(lfm_host_mailbox_test PRIVATE ${LFM_INCLUDE_DIRS})
target_compile_options(lfm_host_mailbox_test PRIVATE
    -O3 -Wall -Wextra -Werror
)
target_link_libraries(lfm_host_mailbox_test PRIVATE lfm_weights kcoro)
if(APPLE)
    target_link_libraries(lfm_host_mailbox_test PRIVATE bsm)
endif()

add_executable(lfm_speech_test tests/native_speech_to_speech.cpp)
target_include_directories(lfm_speech_test PRIVATE ${LFM_INCLUDE_DIRS})
target_compile_options(lfm_speech_test PRIVATE
    -O3 -ffp-contract=off -fvisibility=hidden
)
target_link_libraries(lfm_speech_test PRIVATE lfm_voice)

enable_testing()
add_test(NAME lfm_weight COMMAND lfm_weight_test)
add_test(NAME lfm_host_mailbox COMMAND lfm_host_mailbox_test)
