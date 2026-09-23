// -----------------------------------------------------------------------------------------
// RTGMC demo - forced-include configuration
// -----------------------------------------------------------------------------------------
//
// rgy_version.h defines the optional-feature switches (ENABLE_NVVFX, ENABLE_NVRTC, ...)
// unconditionally, so they cannot be overridden with /D on the command line: the later
// #define inside rgy_version.h always wins.
//
// This header is force-included first (/FI) on every translation unit of the demo build.
// It pulls rgy_version.h in once (it is #pragma once), then #undef/#define the switches
// that would otherwise require SDKs which a pure RTGMC demo does not need:
//
//   ENABLE_NVVFX        -> NVIDIA MAXINE Video Effects SDK  (nvCVStatus.h / nvVideoEffects.h)
//   ENABLE_NVSDKNGX     -> NVIDIA NGX SDK                   (nvsdk_ngx.h)
//   ENABLE_NVOFFRUC     -> NVIDIA Optical Flow SDK          (NvOFFRUC interface)
//   ENABLE_ONNXRUNTIME  -> ONNX Runtime                     (onnxruntime_c_api.h)
//   ENABLE_LIBVSHIP     -> libvship
//   ENABLE_LIBPLACEBO   -> libplacebo (+ D3D11/Vulkan interop)
//   ENABLE_NVRTC        -> NVRTC runtime compilation
//   ENABLE_NVML         -> NVML
//   ENABLE_VMAF         -> libvmaf
//
// Everything else (convert_csp, CUDA kernels, NPP, D3D11 headers from the Windows SDK) is
// left at its upstream value, so the RTGMC pipeline itself is bit-identical to a normal
// NVEncCore build.
//
// NOTE: this header must be applied to EVERY translation unit of the build (including the
// .cu files via nvcc -include / MSBuild AdditionalOptions -include), otherwise the class
// layouts guarded by these macros would differ between objects.
// -----------------------------------------------------------------------------------------

#pragma once

#include "rgy_version.h"

#undef ENABLE_NVVFX
#define ENABLE_NVVFX        0

#undef ENABLE_NVSDKNGX
#define ENABLE_NVSDKNGX     0

#undef ENABLE_NVOFFRUC
#define ENABLE_NVOFFRUC     0

#undef ENABLE_ONNXRUNTIME
#define ENABLE_ONNXRUNTIME  0

#undef ENABLE_LIBVSHIP
#define ENABLE_LIBVSHIP     0

#undef ENABLE_LIBPLACEBO
#define ENABLE_LIBPLACEBO   0

#undef ENABLE_NVRTC
#define ENABLE_NVRTC        0

#undef ENABLE_NVML
#define ENABLE_NVML         0

#undef ENABLE_VMAF
#define ENABLE_VMAF         0

// RtgmDif addition. RtgmDif is the RTGMC deinterlacer and nothing else: it takes frames as
// device pointers from its caller, so it has no file readers and no business pulling in
// ffmpeg. rgy_avutil.h is the one header that reaches for libav*, and every libav* include and
// every `#pragma comment(lib, "av*.lib")` inside it sits behind ENABLE_AVSW_READER - so
// turning that off is what removes the ffmpeg dependency completely (headers, import
// libraries and the whole ffmpeg\include tree) while leaving the RTGMC path untouched.
//
// NOTE: like the ENABLE_* switches above, this belongs to the set that can shift class
// layouts, which is why rtgmc_config.h has to be force-included into every translation unit,
// the .cu files included.
#undef ENABLE_AVSW_READER
#define ENABLE_AVSW_READER  0

#undef ENABLE_LIBAVDEVICE
#define ENABLE_LIBAVDEVICE  0

// Dolby Vision and HDR10+ metadata. RTGMC never looks at either, and both come with a large
// third-party import library (dovi.lib, hdr10plus-rs.lib - 27 MB together) that would have to
// be vendored for nothing.
#undef ENABLE_LIBDOVI
#define ENABLE_LIBDOVI      0

#undef ENABLE_LIBHDR10PLUS
#define ENABLE_LIBHDR10PLUS 0
