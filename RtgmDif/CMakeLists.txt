# ---------------------------------------------------------------------------------------
# RtgmDif - RTGMC deinterlacer as a standalone shared library.
# Cross-platform: Windows/MSVC and Linux/gcc.
#
# Build (Windows, x64, from this directory):
#     cmake -S . -B build-cmake -G "Visual Studio 16 2019" -A x64
#     cmake --build build-cmake --config Release
#
# Build (Linux, from this directory; CUDA 11.5 toolkit + gcc 10):
#     cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release \
#           -DCMAKE_C_COMPILER=gcc-10 -DCMAKE_CXX_COMPILER=g++-10 \
#           -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10
#     cmake --build build-cmake -j
#
# gcc 10 rather than the system default: CUDA 11.5's front end cannot parse libstdc++ 11's
# <functional>, and -allow-unsupported-compiler does not get past a syntax-level mismatch.
#
# The output lands in ../Output/, because RtgmDifTest links against the module there and
# loads it from the same place.
#
# Requires: CUDA toolkit (11.x, nvcc on PATH or CUDAToolkit_ROOT), a C++17 compiler
# (MSVC 2019+ or gcc 10), and the vendored NVEnc tree in ./NVEnc.
# ---------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18)

project(RtgmDif LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")

set(CMAKE_CUDA_ARCHITECTURES 50-real 61-real 75-real 86-real 50-virtual)

if(WIN32)
    set(RTGMDIF_MODULE_FILE "RtgmDif.dll"      CACHE INTERNAL "module file (Windows)")
    set(RTGMDIF_IMPLIB_FILE "RtgmDif.lib"      CACHE INTERNAL "import library (Windows)")
else()
    set(RTGMDIF_MODULE_FILE "libRtgmDif.so"    CACHE INTERNAL "module file (Linux)")
    set(RTGMDIF_IMPLIB_FILE "libRtgmDif.so"    CACHE INTERNAL "import library (Linux)")
endif()

foreach(kind RUNTIME LIBRARY ARCHIVE)
    set(CMAKE_${kind}_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../Output")
endforeach()
if(MSVC)
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/compile-pdb")
    set(CMAKE_PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../Output")
endif()

file(GLOB_RECURSE RTGMDIF_CXX_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE RTGMDIF_CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

if(NOT RTGMDIF_CXX_SOURCES OR NOT RTGMDIF_CUDA_SOURCES)
    message(FATAL_ERROR "Cannot find the vendored sources under ${CMAKE_CURRENT_SOURCE_DIR}/NVEnc - is the tree complete?")
endif()

add_library(RtgmDif SHARED
    ${RTGMDIF_CXX_SOURCES}
    ${RTGMDIF_CUDA_SOURCES}
)

set_target_properties(RtgmDif PROPERTIES
    OUTPUT_NAME "RtgmDif"
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
)

if(WIN32)
    target_sources(RtgmDif PRIVATE RtgmDif.def)
else()
    target_link_options(RtgmDif PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/RtgmDif.map")
endif()

if(WIN32)
    target_compile_definitions(RtgmDif PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
        WIN32
        _UNICODE
        UNICODE
    )
endif()

target_include_directories(RtgmDif PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/NVEnc"
    "${CMAKE_CURRENT_SOURCE_DIR}/NVEnc/NVEncCore"
    "${CMAKE_CURRENT_SOURCE_DIR}/NVEnc/NVEncSDK"
    "${CMAKE_CURRENT_SOURCE_DIR}/NVEnc/NVEncSDK/Common"
    "${CMAKE_CURRENT_SOURCE_DIR}/NVEnc/NVEncSDK/Common/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/NVEnc/NVEncSDK/Core"
    "${CMAKE_CURRENT_SOURCE_DIR}/NVEnc/NVEncSDK/Core/include"
)

if(MSVC)
    target_compile_options(RtgmDif PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/FI${CMAKE_CURRENT_SOURCE_DIR}/NVEnc/rtgmc_config.h>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4819>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4505>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4127>
    )
else()
    target_compile_options(RtgmDif PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-include ${CMAKE_CURRENT_SOURCE_DIR}/NVEnc/rtgmc_config.h>
    )
endif()

set(RTGMDIF_CUDAFE_OPTIONS "--display_error_number --diag_suppress=signed_one_bit_field --diag_suppress=63 --diag_suppress=177 --diag_suppress=expr_has_no_effect")

target_compile_options(RtgmDif PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-include ${CMAKE_CURRENT_SOURCE_DIR}/NVEnc/rtgmc_config.h>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe ${RTGMDIF_CUDAFE_OPTIONS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)
if(MSVC)
    target_compile_options(RtgmDif PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use-local-env>
    )
endif()

if(WIN32)
    target_link_libraries(RtgmDif PRIVATE
        cuda
    )
else()
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(RtgmDif PRIVATE
        CUDA::cuda_driver
    )
endif()

add_custom_command(TARGET RtgmDif POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/NVEnc/resource/nnedi3_weights.bin"
        "$<TARGET_FILE_DIR:RtgmDif>"
    COMMENT "Copying nnedi3_weights.bin next to the RtgmDif module"
)
