# ---------------------------------------------------------------------------------------
# Rtgmc - the two RtgmDif projects, moved out of the NVEnc tree and built together.
#
#   RtgmDif      the RTGMC deinterlacer as a shared library, with the NVEnc sources it
#                needs vendored inside it (RtgmDif/NVEnc/...) - nothing outside this
#                directory is read.
#   RtgmDifTest  the raw YUV driver that loads that library.
#
# Build (Windows, x64, from this directory):
#     cmake -S . -B build-cmake -A x64
#     cmake --build build-cmake --config Release
#
# Build (Linux; CUDA 11.5 toolkit + gcc 10):
#     cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
#     cmake --build build-cmake -j
#
# Each project keeps its own CMakeLists.txt and can still be built on its own (see the
# header comment in either of them); this file only says which one comes first and lets
# RtgmDifTest link RtgmDif as a target instead of hunting for the import library.
#
# Outputs (both platforms): the RtgmDif module and its weights under Output/, plus
# RtgmDifTest.exe / RtgmDifTest next to them, so the driver finds the module and
# nnedi3_weights.bin beside itself.
#
# Requires: CUDA toolkit 11.x, and a C++17 compiler (MSVC 2019+ or gcc 10).
# No NVEnc checkout is needed.
# ---------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18)

project(Rtgmc LANGUAGES CXX CUDA)

# Same set as RtgmDif/CMakeLists.txt: every cubin nvcc 11.5 can emit (sm_50/61/75/86) plus the
# compute_50 PTX fallback, so any card from Maxwell onwards runs - natively where a cubin (or
# a same-major higher-minor match, e.g. sm_86 -> sm_89) exists, via JIT everywhere else
# (sm_60, Volta, sm_80, Hopper, Blackwell sm_120). Set here as well so the value is in place
# before either subdirectory creates its targets.
set(CMAKE_CUDA_ARCHITECTURES 50-real 61-real 75-real 86-real 50-virtual)

# RtgmDif first: RtgmDifTest links it and loads its DLL at run time.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/RtgmDif/CMakeLists.txt")
    add_subdirectory(RtgmDif)
else()
    message(STATUS "RtgmDif is not in this tree - skipping it.")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/RtgmDifTest/CMakeLists.txt")
    add_subdirectory(RtgmDifTest)
else()
    message(STATUS
        "RtgmDifTest is not in this tree yet - skipping it. It links RtgmDif, so once it is "
        "here a configure/build of this directory is enough.")
endif()
