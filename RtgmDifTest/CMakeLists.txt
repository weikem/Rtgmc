# ---------------------------------------------------------------------------------------
# RtgmDifTest - raw YUV driver for the RtgmDif module.
# Cross-platform: Windows/MSVC and Linux/gcc.
#
# Build (Windows, x64, from this directory), after RtgmDif has been built once:
#     cmake -S . -B build-cmake -G "Visual Studio 16 2019" -A x64
#     cmake --build build-cmake --config Release
#
# Build (Linux, from this directory):
#     cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
#     cmake --build build-cmake -j
#
# This project builds no NVEncCore source of its own: it links the module next door and
# reaches outside its own directory for nothing but that project. The module and
# nnedi3_weights.bin are copied next to the executable, because the module loads the
# weights from its own directory.
#
# The sibling layout is assumed:
#     <parent>/RtgmDif/Output/<RtgmDif.lib | libRtgmDif.so>
#     <parent>/RtgmDif/Output/<RtgmDif.dll | libRtgmDif.so>
# Override with -DRTGMDIF_MODULE_DIR=<path to RtgmDif> if the two are not siblings.
# ---------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18)

project(RtgmDifTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The same static CRT the module uses.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(RTGMDIF_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../RtgmDif"
    CACHE PATH "Directory holding the RtgmDif project (source and build).")

# The module's platform-specific file names, matching RtgmDif/CMakeLists.txt. The module's
# own CMakeLists sets these as CACHE variables, so in the top-level build they are already
# in place; the fallback here covers the standalone build.
if(NOT DEFINED RTGMDIF_IMPLIB_FILE)
    if(WIN32)
        set(RTGMDIF_IMPLIB_FILE "RtgmDif.lib")
        set(RTGMDIF_MODULE_FILE "RtgmDif.dll")
    else()
        set(RTGMDIF_IMPLIB_FILE "libRtgmDif.so")
        set(RTGMDIF_MODULE_FILE "libRtgmDif.so")
    endif()
endif()

# Standalone use needs the module built first; in the top-level build RtgmDif is a CMake
# target and the ordering is CMake's business, so the check is skipped there. The
# multi-config generators append <CONFIG>/ to the Output directory, the single-config
# ones do not (CMAKE_CFG_INTDIR is "."), which is why it goes through a variable.
if(NOT TARGET RtgmDif AND NOT EXISTS "${RTGMDIF_MODULE_DIR}/../Output/${CMAKE_CFG_INTDIR}/${RTGMDIF_IMPLIB_FILE}")
    message(FATAL_ERROR
        "Cannot find ${RTGMDIF_MODULE_DIR}/../Output/${RTGMDIF_IMPLIB_FILE}.\n"
        "Build the module first (see ${RTGMDIF_MODULE_DIR}/CMakeLists.txt), or point "
        "-DRTGMDIF_MODULE_DIR= at it.")
endif()

add_executable(RtgmDifTest RtgmDifTest.cpp)

# The executable lands next to the module copy, which is where the module is loaded from
# at runtime.
set_target_properties(RtgmDifTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../Output"
)
if(MSVC)
    set_target_properties(RtgmDifTest PROPERTIES
        PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../Output"
    )
endif()

# NDEBUG comes from the build configuration on both platforms (Release/RelWithDebInfo).
if(WIN32)
    target_compile_definitions(RtgmDifTest PRIVATE
        _CRT_SECURE_NO_WARNINGS
        WIN32
        _CONSOLE
    )
endif()

# The module's public headers, i.e. RtgmDif.h next door.
target_include_directories(RtgmDifTest PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${RTGMDIF_MODULE_DIR}"
)

# The module links its CUDA runtime statically, so the driver does the same to keep one
# runtime in the process.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(RtgmDifTest PRIVATE
    $<IF:$<TARGET_EXISTS:RtgmDif>,RtgmDif,"${RTGMDIF_MODULE_DIR}/../Output/${CMAKE_CFG_INTDIR}/${RTGMDIF_IMPLIB_FILE}">
    CUDA::cudart_static
)
if(TARGET RtgmDif)
    # Top-level build: let CMake order the two targets instead of relying on the module
    # having been built by hand beforehand.
    add_dependencies(RtgmDifTest RtgmDif)
endif()

# The module and the EDI weights next to the exe. Prefer the RtgmDif target when present
# (CMake picks up its real RUNTIME_OUTPUT_DIRECTORY through $<TARGET_FILE:...>); otherwise
# fall back to the sibling Output/<CONFIG>/ tree.
add_custom_command(TARGET RtgmDifTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<IF:$<TARGET_EXISTS:RtgmDif>,$<TARGET_FILE:RtgmDif>,${RTGMDIF_MODULE_DIR}/../Output/${CMAKE_CFG_INTDIR}/${RTGMDIF_MODULE_FILE}>"
        "$<TARGET_FILE_DIR:RtgmDifTest>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${RTGMDIF_MODULE_DIR}/NVEnc/resource/nnedi3_weights.bin"
        "$<TARGET_FILE_DIR:RtgmDifTest>"
    COMMENT "Copying the RtgmDif module and nnedi3_weights.bin next to the test executable"
)
