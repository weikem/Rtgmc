// -----------------------------------------------------------------------------------------
// RTGMC demo - Linux platform configuration
// -----------------------------------------------------------------------------------------
//
// rgy_version.h #includes "rgy_config.h" in its non-Windows branch only, which is why the
// Windows-only vendored tree never carried the file: no Windows build needed it. It exists
// in the upstream NVEnc repository (NVEncCore/rgy_config.h) and carries the platform-side
// feature switches that the Windows branch of rgy_version.h defines inline.
//
// This is the RTGMC demo's counterpart for the Linux build: everything that would need a
// third-party SDK not vendored here is off, matching what rtgmc_config.h (force-included on
// every translation unit) already turns off on Windows. The values below are the Linux
// equivalents of rgy_version.h's Windows block (lines 78-121 there):
//
//   ENABLE_CPP_REGEX                       1  (std::regex is used by the vendored sources)
//   ENABLE_DTL                             0  (Distributed Task Layer - not vendored)
//   ENABLE_PERF_COUNTER                    0  (pulls NVML / Windows-only counters)
//   ENABLE_LIBASS_SUBBURN                   0  (libass - not vendored)
//   ENABLE_VMAF                            0  (libvmaf - not vendored)
//   ENABLE_LIBVSHIP                        0  (libvship - not vendored)
//   ENABLE_ONNXRUNTIME                     0  (onnxruntime - not vendored)
//   AV_CHANNEL_LAYOUT_STRUCT_AVAIL etc.    1  (libavutil feature detection; moot because
//                                              rtgmc_config.h sets ENABLE_AVSW_READER 0,
//                                              kept at 1 to keep the avutil headers
//                                              self-consistent if they are ever reached)
// -----------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_CONFIG_H__
#define __RGY_CONFIG_H__

#define ENABLE_CPP_REGEX 1
#define ENABLE_DTL 0
#define ENABLE_PERF_COUNTER 0

#define ENABLE_LIBASS_SUBBURN 0
#define ENABLE_VMAF 0
#define ENABLE_LIBVSHIP 0
#define ENABLE_ONNXRUNTIME 0

#define AV_CHANNEL_LAYOUT_STRUCT_AVAIL 1
#define AV_FRAME_DURATION_AVAIL 1
#define AVCODEC_PAR_CODED_SIDE_DATA_AVAIL 1

#endif //__RGY_CONFIG_H__
