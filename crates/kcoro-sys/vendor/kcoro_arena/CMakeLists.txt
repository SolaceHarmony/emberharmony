cmake_minimum_required(VERSION 3.21)

project(kcoro LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(kcoro STATIC
    core/src/kc_deadline.c
    core/src/kc_doorbell.c
    core/src/kc_fixed_scope.c
    core/src/kc_runtime.c
    core/src/kc_service.c
    core/src/kc_team.c
    core/src/kcoro_stackless.c
    port/posix.c
)

target_include_directories(kcoro
    PUBLIC include
    PRIVATE core/src
)

target_link_libraries(kcoro PUBLIC Threads::Threads)

if(NOT APPLE)
    target_compile_definitions(kcoro PRIVATE _GNU_SOURCE)
endif()

enable_testing()

add_executable(kcoro_stackless_test tests/stackless_runtime.cpp)
target_link_libraries(kcoro_stackless_test PRIVATE kcoro)

add_test(NAME kcoro_stackless COMMAND kcoro_stackless_test)
