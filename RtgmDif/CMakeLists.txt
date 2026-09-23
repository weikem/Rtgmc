# ---------------------------------------------------------------------------------------
# RtgmDif - RTGMC deinterlacer as a standalone shared library (mirrors RtgmDif.vcxproj).
#
# Cross-platform: Windows/MSVC and Linux/gcc.
#
# Build (Windows, x64, from this directory):
#     cmake -S . -B build-cmake -G "Visual Studio 16 2019" -A x64
#     cmake --build build-cmake --config Release
#
# Build (Linux, from this directory; CUDA 11.5 toolkit + gcc 11 required):
#     cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
#     cmake --build build-cmake -j
#
# The output deliberately lands in ../Output/ (RUNTIME and ARCHIVE), because RtgmDifTest
# links against the module there and loads it from the same place. On Linux that is
# libRtgmDif.so, on Windows RtgmDif.dll (+ its import library); the platform-specific
# file names are centralised in the RTGMDIF_* variables below.
#
# Requires: CUDA toolkit (11.x, nvcc on PATH or CUDAToolkit_ROOT), a C++17 compiler
# (MSVC 2019+ or gcc 11), and the vendored NVEnc tree in ./NVEnc (the two Directories
# can be moved out of the NVEnc repository as a pair - only NVEnc\* is needed here).
# On Windows the additional d3d9/dxgi/d3d11 link dependencies of the vcxproj are kept
# as-is; on Linux they do not exist and are not needed (no vendored source includes them).
# ---------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18)

project(RtgmDif LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RtgmDif.vcxproj: <RuntimeLibrary>MultiThreaded</RuntimeLibrary> for both C/C++ and CUDA.
# The static CRT is what the reference driver and the test program are built with as well,
# and mixing them across the DLL boundary is not worth the risk. On Linux only the CUDA
# runtime has a static/dynamic choice, and it stays static for the same reason.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")

# Every cubin the CUDA 11.5 nvcc can emit, so that each arch gets native SASS:
#   sm_50  Maxwell  GTX 9xx, Tesla M40, Quadro M series
#   sm_61  Pascal   P2000 (the card this build is validated on), GTX 10xx
#   sm_75  Turing   RTX 20xx, Tesla T4, Quadro RTX
#   sm_86  Ampere   A2000, RTX 3050/3060, A40
# Each cubin also runs natively on a *higher* minor of the same major, per CUDA's binary
# compatibility rule (a cubin for X.y loads on any X.z with z >= y): the sm_50 cubin covers
# sm_52/53, the sm_61 cubin covers sm_62, and the sm_86 cubin covers sm_87 (Jetson Orin) and
# sm_89 (Ada - RTX 4050/4060) with no JIT. The sm_75 cubin covers only sm_75 (nothing sits
# between it and sm_80). Cubins never cross majors and never load on a *lower* minor, so
# sm_60 (P100), sm_70/72 (Volta), sm_80 (A100/A30), sm_90 (Hopper) and sm_100/120 (Blackwell,
# RTX 50xx) stay uncovered - that is what the PTX below is for.
# + compute_50 PTX as the JIT fallback. PTX is forward-compatible across majors, so this one
# PTX covers exactly those leftovers, including sm_120 which nvcc 11.5 cannot target at all
# (its --list-gpu-arch stops at compute_87). Without it those cards abort with "no kernel
# image is available for execution on the device". Do not drop it.
#
# Cost, if size ever matters again: the PTX dominates, not the cubins - the 4 cubins are
# ~136 MB of the ~307 MB DLL, the single compute_50 PTX is ~171 MB (PTX is a textual IR,
# much larger than SASS). Dropping it needs a toolkit that can emit sm_89/sm_120 (CUDA
# 11.8 / 12.8+), otherwise it is the only route to the newest cards.
set(CMAKE_CUDA_ARCHITECTURES 50-real 61-real 75-real 86-real 50-virtual)

# The platform-specific names of what this project produces, in one place so that the
# sibling RtgmDifTest project can agree with it (it links the import library / .so and
# copies the module next to its executable).
if(WIN32)
    set(RTGMDIF_MODULE_FILE "RtgmDif.dll"      CACHE INTERNAL "module file (Windows)")
    set(RTGMDIF_IMPLIB_FILE "RtgmDif.lib"      CACHE INTERNAL "import library (Windows)")
else()
    set(RTGMDIF_MODULE_FILE "libRtgmDif.so"    CACHE INTERNAL "module file (Linux)")
    set(RTGMDIF_IMPLIB_FILE "libRtgmDif.so"    CACHE INTERNAL "import library (Linux)")
endif()

# All artefacts (module, import library, pdb) land under <repo>/Output. The Visual Studio
# multi-config generator appends <CONFIG>/ (e.g. Output\RelWithDebInfo\RtgmDif.dll); the
# single-config Linux generators use the directory as given.
foreach(kind RUNTIME LIBRARY ARCHIVE)
    set(CMAKE_${kind}_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../Output")
endforeach()
if(MSVC)
    # Only the linker's PDB (RtgmDif.pdb) is a deliverable. The compiler's /Fd PDB is an
    # intermediate by-product, so it stays inside the build tree instead of Output/ - otherwise
    # it lands as Output/<CONFIG>.pdb, which is where the module's .pdb name would be dumped.
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

# Four SIMD translation units used to live under NVEncCore and were excluded from the glob
# here; they have since been deleted from the tree:
#   rgy_memmem_avx2.cpp / rgy_memmem_avx512bw.cpp          AVX2 / AVX512BW search kernels
#   rgy_bitstream_avx2.cpp / rgy_bitstream_avx512bw.cpp    AVX2 / AVX512BW NAL parsers
#
# They existed to speed up codec parsing - scanning a bitstream, locating NAL boundaries -
# which is the NVEncC front end's job, not this module's: RtgmDif takes device frames in and
# hands device frames out, and never looks at a bitstream. Their dispatch entry points
# (get_memmem_func, get_parse_nal_unit_h264_func / _hevc_func, get_find_header_func) have no
# caller in this tree, and the dispatch sites that used to return the SIMD variants now return
# the scalar ones (rgy_memmem.cpp, rgy_bitstream.cpp) - which is what made deleting the files
# safe rather than just excluding them.
#
# Deleting them is also what removed the last /arch:AVX2 and /arch:AVX512 from the build: those
# options existed only for these four, which call AVX intrinsics directly. Every other
# translation unit, CUDA included, is compiled for the baseline ISA.
#
# The declarations in rgy_memmem.h / rgy_bitstream.h are untouched: they are upstream headers,
# they are unreferenced now, and rgy_memmem.h's #if defined(RGY_MEMMEM_AVX2) blocks stay dormant
# because the only file that ever defined that macro was rgy_memmem_avx2.cpp.

add_library(RtgmDif SHARED
    ${RTGMDIF_CXX_SOURCES}
    ${RTGMDIF_CUDA_SOURCES}
)

set_target_properties(RtgmDif PROPERTIES
    OUTPUT_NAME "RtgmDif"
    # The .def (Windows) / version script (Linux) supplies the exports.
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
)

if(WIN32)
    # RtgmDif.vcxproj: <ModuleDefinitionFile>RtgmDif.def</ModuleDefinitionFile>. Nothing in
    # the source is marked __declspec(dllexport) - this file is what makes the two entry
    # points visible, and the linker fails the build if either does not exist.
    target_sources(RtgmDif PRIVATE RtgmDif.def)
else()
    # The same two names as the .def, as an anonymous linker version script: the NVEncCore
    # symbols stay local to the .so, and rtgmdif_create/rtgmdif_destroy remain unversioned so
    # dlopen()/dlsym() and plain linking see them exactly as Windows does.
    target_link_options(RtgmDif PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/RtgmDif.map")
endif()

# RtgmDif.vcxproj PreprocessorDefinitions (C/C++), plus the two UNICODE/_WINDLL defines that
# the CUDA side of the same project carries.
#
# Three DLL name macros used to be defined here - NVENC_NPPI_DLL_NAME_TSTR,
# NVENC_NVRTC_DLL_NAME_TSTR and NVENC_NVRTC_BUILTIN_DLL_NAME_TSTR - feeding
# RtgmDifDllNames.cpp, which no longer exists either. NPPI had a live reader
# (check_if_nppi_dll_available()); the two NVRTC names were read only by functions that
# ENABLE_NVRTC compiles out, and NVRTC is unreachable in this module, because
# ENABLE_VPP_FILTER_COLORSPACE is defined as (ENABLE_NVRTC) and RTGMC does not use the
# colorspace filter. NVEncFilter.h records what to restore if that ever changes - and note that
# those names carry toolkit version numbers, which is the other reason not to keep them here
# on the off chance: a stale one would silently stop matching.
#
# Linux needs no extra definitions of its own, so only the Windows block is conditional.
if(WIN32)
    # Four of the vcxproj's nine definitions are dropped here; the rest are load-bearing. What
    # makes this worth spelling out is that MSVC predefines only _WIN32 and _WIN64 - a probe
    # with #pragma message confirms the bare WIN32 is not among them - so the platform macros
    # below have to be supplied by the build.
    #
    #   UNICODE / _UNICODE      rgy_tchar.h includes <tchar.h> on Windows, so these decide
    #                           whether TCHAR is char or wchar_t, i.e. the module's whole string
    #                           ABI. rgy_log.cpp also carries #ifdef UNICODE branches.
    #   WIN32                   genuinely used in three places that matter: rgy_version.h's
    #                           Windows-vs-Linux split, NVEncFilterDenoiseNLMeans.h and
    #                           helper_functions.h - the latter two test the bare WIN32 with no
    #                           _WIN32 fallback, so dropping it changes what gets compiled.
    #   _CRT_SECURE_NO_WARNINGS the usual C4996 suppression: the vendored code calls fopen,
    #                           strcpy and sscanf throughout.
    #   _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
    #                           no source reads it, but MSVC's STL does, and nvcc's host pass is
    #                           exactly the case it exists for.
    #
    # Dropped: _WINDLL, _CONSOLE and _LIB (no source reads any of them - a grep over the whole
    # vendored tree turns up only incidental substring matches - and this is a DLL, not a
    # console application or a static library), plus NDEBUG, which was the one real defect in
    # the list: hard-coding it put NDEBUG into every configuration, so even a Debug build would
    # have had its asserts compiled out. CMake adds -DNDEBUG to Release/RelWithDebInfo itself.
    target_compile_definitions(RtgmDif PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
        WIN32
        _UNICODE
        UNICODE
    )
endif()

# RtgmDif.vcxproj AdditionalIncludeDirectories. The Windows SDK and CUDA paths come from the
# toolchain itself. The vcxproj also appended $DXSDK_DIR/include for the DirectX parts of
# NVEncCore; that is gone with the d3d9/dxgi/d3d11 link entries above, for the same reason -
# nothing in the vendored tree includes a DirectX header.
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

# Force-include rtgmc_config.h on every host-code translation unit, so the optional-feature
# switches (ENABLE_LIBHDR10PLUS, ENABLE_LIBDOVI, ENABLE_AVSW_READER, ...) are turned off
# before any of the rgy_* headers read them. Without this, the .cpp files that include
# rgy_hdr10plus.cpp / rgy_libdovi.h / rgy_avutil.h see the upstream-default 1 and try to
# pull in libhdr10plus-rs/libdovi/libav* headers that are not vendored.
# /FI is the MSVC spelling, -include the gcc/clang one (the CUDA options use -include on
# both platforms, since nvcc is a gcc front end everywhere but Windows).
if(MSVC)
    target_compile_options(RtgmDif PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/FI${CMAKE_CURRENT_SOURCE_DIR}/NVEnc/rtgmc_config.h>
        # /Zc:__cplusplus and the /wd pragmas are MSVC-only.
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

# RtgmDif.vcxproj CudaCompile AdditionalOptions. -use_fast_math is what the CUDA build
# customization passes in addition to these, and it is part of the picture the
# equivalence harness was validated with, so it is set explicitly here. These flags are
# identical on both platforms except --use-local-env, which only exists in the Windows
# nvcc.
# -Xcudafe forwards everything after it to cudafe++, nvcc's C++ front end (EDG based), so the
# numbers below are EDG diagnostic ids - not MSVC's C4xxx, not gcc's -W names. One -Xcudafe,
# one argv: the sub-options are space-separated inside the quoted string, and SHELL: keeps
# CMake from splitting it at those spaces. The list sits in a variable because a literal "..."
# butted straight against the closing '>' of the generator expression makes CMake warn about a
# token that is not separated by whitespace.
#
# The list is cut down to what this project's sources actually trigger - and the two platforms do
# not trigger the same set, so it had to be checked on both:
#
#   signed_one_bit_field   Linux: 666 x #108-D "signed bit field of length 1", all of them out of
#                          NVIDIA's own nvEncodeAPI.h (lines 1785-1809), a public header we
#                          cannot change. Not emitted under MSVC.
#   63                     Linux: 23 x #63-D "shift count is too large", from NVEncFilterCrop.cu
#                          and convert_csp.h. Not emitted under MSVC.
#   177                    Linux: 2 x #177-D "variable ... was declared but never referenced",
#                          from NVEncFilterRtgmcEdi.cu. Not emitted under MSVC.
#   expr_has_no_effect     Windows: 656 x #174-D "expression has no effect", from rgy_prm.h
#                          (1013, 1049, ...). Not emitted under gcc - which is why a Linux-only
#                          sweep misses it: the two platforms compile different code, because
#                          UNICODE / _WIN32 and the compiler itself select different branches.
#
# Five entries the vcxproj also carried are gone because neither platform triggers them:
# negative_shift_count, 186, 815, 1444 and 2465. They are presumably live elsewhere in the full
# NVEncC tree, whose single suppression list had to cover all of its translation units, while
# this project vendors a subset of it.
#
# If more of NVEncCore is ever vendored in, re-check on both sides: on Linux compile one .cu with
# -Xcudafe removed; on Windows build normally with a trimmed list and read the diagnostic ids
# straight out of the build log.
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

# RtgmDif.vcxproj AdditionalDependencies. cuda.lib is the driver API; on Linux that is
# libcuda.so.1 (shipped with the driver, dev symlink next to it), plus the pthread/dl shims
# of rgy_osdep.h. cudart is linked statically per CMAKE_CUDA_RUNTIME_LIBRARY above.
#
# The vcxproj also listed d3d9.lib, dxgi.lib and d3d11.lib. They are gone: no vendored source
# includes a DirectX header, and dumpbin /dependents on the built DLL shows that none of the
# three ever reached its import table - they came along from the upstream NVEncC project,
# where the D3D11 interop path genuinely uses them. Removing them is a no-op at link time.
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

# RtgmDif.vcxproj post-build step: the weights have to sit next to the module because that
# is where the EDI loads them from at runtime (on both platforms this resolves through the
# executable's directory, and the module and the test executable share ../Output).
add_custom_command(TARGET RtgmDif POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/NVEnc/resource/nnedi3_weights.bin"
        "$<TARGET_FILE_DIR:RtgmDif>"
    COMMENT "Copying nnedi3_weights.bin next to the RtgmDif module"
)
