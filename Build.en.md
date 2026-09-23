# Building

[Readme](Readme.en.md) | **Build** | [中文](Build.cn.md)

## Requirements

| item | requirement |
|---|---|
| CUDA Toolkit | **11.5** (`nvcc` on PATH, or point at it with `-DCUDAToolkit_ROOT=<path>`) |
| C++ compiler | Windows: MSVC 2019 or newer; Linux: **gcc/g++ 10** (see below) |
| CMake | 3.18 or newer |
| NVIDIA driver | not needed to build; needed at run time, when the module reaches the GPU through `libcuda.so` / `nvcuda.dll` |

No NVEnc checkout is needed and neither is the Video Codec SDK: the headers the build wants are vendored under `RtgmDif/NVEnc/`.

### Linux: gcc-10 is required

The system default gcc-11 (as on Ubuntu 22.04) does not work: **CUDA 11.5's nvcc front end cannot parse libstdc++ 11's `<functional>` and friends**, and 11.8 is the first toolkit that supports gcc 11. The failure is a syntax-level error (`parameter packs not expanded with '...'`), which `-allow-unsupported-compiler` does **not** get past - that flag only relaxes the version check.

Install it with:

```bash
sudo apt-get install -y g++-10
```

## Windows

```bat
cmake -S . -B build -A x64
cmake --build build --config RelWithDebInfo
```

`Release` works too; `RelWithDebInfo` keeps the PDBs, which makes debugging easier (the outputs land in `Output\Release\` or `Output\RelWithDebInfo\` respectively).

## Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc-10 -DCMAKE_CXX_COMPILER=g++-10 \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10
cmake --build build -j$(nproc)
```

All three compiler options are needed: with only `CMAKE_CXX_COMPILER` set, `nvcc` still goes looking for the default `g++` (11, in this case) and fails the same way.

### Building one project on its own

The two `CMakeLists.txt` files are independent. `RtgmDifTest` expects the module at the sibling path `../RtgmDif` by default, so no extra argument is needed when the source directories sit next to each other:

```bash
cd RtgmDif && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release ... && cmake --build build -j
```

Module only, skipping the reference driver:

```bash
cmake --build build --target RtgmDif
```

## Output

Both platforms write to `Output/`:

| file | Windows | Linux |
|---|---|---|
| module | `RtgmDif.dll` + import library `RtgmDif.lib` | `libRtgmDif.so` |
| reference driver | `RtgmDifTest.exe` | `RtgmDifTest` |
| EDI weights | `nnedi3_weights.bin` (copied next to the module by the build) | same |

Measured sizes, with all five CUDA architectures compiled in:

| file | size |
|---|---|
| module (Windows) | about 307 MB |
| module (Linux) | about 267 MB |
| `nnedi3_weights.bin` | 13.5 MB |

The module exports exactly two symbols, `rtgmdif_create` and `rtgmdif_destroy` - controlled by `RtgmDif.def` on Windows and by the `RtgmDif.map` version script on Linux - and keeps everything else internal.

## CUDA architecture coverage

```cmake
set(CMAKE_CUDA_ARCHITECTURES 50-real 61-real 75-real 86-real 50-virtual)
```

Two rules decide what is covered:

- **cubin (SASS) is forward-compatible within a major version**: a cubin built for X.y runs on X.z where z >= y, but never on a lower minor and never across majors.
- **PTX is forward-compatible across majors**: one `compute_50` PTX runs on any card of compute capability 5.0 or above, JIT-compiled by the driver.

| architecture | representative cards | route |
|---|---|---|
| sm_50 / 52 / 53 | GTX 9xx, Tesla M40, Quadro M series | native (`50-real`) |
| sm_60 | Tesla P100 | PTX JIT |
| sm_61 / 62 | **Quadro P2000**, GTX 10xx, Tesla P40, Jetson TX2 | native (`61-real`) |
| sm_70 / 72 | V100, Titan V, Jetson Xavier | PTX JIT |
| sm_75 | RTX 20xx, Tesla T4, Quadro RTX | native (`75-real`) |
| sm_80 | A100, A30 | PTX JIT |
| sm_86 / 87 / 89 | **Quadro A2000**, **RTX 3050/3060**, A40, Jetson Orin, **RTX 4050/4060** (Ada) | native (`86-real`) |
| sm_90 | H100, H200 | PTX JIT |
| sm_100 / 101 / 103 / 120 | B100/B200, **RTX 5060** and the rest of RTX 50 | PTX JIT |

Cards in bold are the ones this project has been tested on.

Two things that are easy to get wrong:

1. **RTX 40 (sm_89) runs the `86-real` cubin natively, not through JIT** - same-major forward compatibility. It is fully functional, just without Ada-specific instruction tuning, and the performance difference is usually small.
2. **RTX 50 (sm_120) has to go through PTX**: CUDA 11.5's `nvcc --list-gpu-arch` stops at `compute_87`, so no cubin can be built for it. Dropping `50-virtual` makes those cards abort with `no kernel image is available for execution on the device`.

### Where the size goes, and how to cut it

The size is dominated by the PTX, not the cubins: the four cubins are about 136 MB together, the single `compute_50` PTX about 171 MB (PTX is a textual IR, much larger than SASS).

| goal | setting | module size |
|---|---|---|
| everything (default) | `50-real 61-real 75-real 86-real 50-virtual` | about 307 MB (Win) / 267 MB (Linux) |
| no RTX 50 | `50-real 61-real 75-real 86-real` | about 136 MB |
| Pascal + Ampere/Ada only | `61-real 86-real` | about 68 MB |
| CUDA 12.8+, all native | `61 86 89 120` | about 136 MB, no PTX at all |

The last row is the long-term answer: once the toolkit can emit `sm_89` / `sm_120` cubins directly, the PTX fallback can go away entirely and Ada and Blackwell each get their own native tuning.

## Resources

- **Disk**: `Output/` is about 300 MB; the intermediates in `build/` need a few GB (74 `.cu` files times five architectures worth of object files).
- **Memory**: each extra `-gencode` raises the peak `nvcc`/`cicc` memory noticeably, and RTGMC's translation units are large, so keep `-j` at or below the physical core count (a `-j8` build on 8 cores is known to work).
- **Time**: a full build on 8 cores with `-j8` takes minutes; an incremental build after touching one `.cu` takes about a minute, a good part of which is linking the 300 MB module.

## Verifying a build

`RtgmDifTest` is a raw YUV reference driver and can be run right after the build:

```bash
./RtgmDifTest -i input_1080i50.yuv -o output_1080p50.yuv \
              --width 1920 --height 1080 --in-csp yuv422p \
              --fps 25 --flow slow_both --low-latency
```

Expect two output frames per input frame, so a 4-frame input produces 8 frames and an output file of input frames x 2 x frame bytes. The program prints the in/out ledger and the device memory it uses:

```
test   : flow slow_both, yuv422p 1920x1080, input frame 4147200 bytes, output frame 4147200 bytes
test   : input file 16588800 bytes (4 frame(s) at that size)
mem    : init   after     0 in      0.0 MiB on the card      +0.0 MiB since start
```

If `init` fails with `RGY_ERR_INVALID_PARAM`, check for `nnedi3_weights.bin` next to the module first.

### Command line

Input and output:

| option | meaning |
|---|---|
| `-i, --input <file>` | input raw YUV file (required) |
| `-o, --output <file>` | output file, same csp as the input. Optional - without it the whole chain still runs but nothing is written (a timing run) |
| `--width <n>` | frame width, default 1920 |
| `--height <n>` | frame height, default 1080 |
| `--in-csp <name>` | `yuv422p` / `yuv420p` / `yuv444p`, default `yuv444p`; `yuv422p` is what it is normally run on |
| `--fps <n>` | input frame rate, default 25 |
| `--bff` | input is bottom field first (the default is top field first) |
| `--frames <n>` | stop after n frames read, counted from the first frame read (default: the whole file). The frames `--skip` discards count towards it, so `--frames` has to be larger than `--skip`: `--skip 400 --frames 430` runs 430 frames and keeps the last 30 |
| `--skip <n>` | feed the first n input frames through the chain without writing or timing them, to jump into a hard part of the clip; the chain is warmed up where the timed part starts |
| `--stats-every <n>` | print the in/out ledger every n input frames (default 16). `1` is every frame, which is how the pipeline delay reads out |
| `--device <n>` | CUDA device index, default 0 |
| `--log-level <n>` | 0 quiet .. 5 debug, default 3 |
| `--own-stream` | let the module create and own its own stream instead of handing it one (slower: it then has to synchronise the output copies, because the caller has no handle to order them against) |

Flows:

| option | meaning |
|---|---|
| `--flow <name>` | `fast_deint` / `fast_both` (default) / `fast_opt` / `faster_nn1` / `slow` / `slow_both` / `slower`, described in the [Readme](Readme.en.md#flows) |

First pass (the deinterlace):

| option | meaning |
|---|---|
| `--deint-pel <n>` | 1 = half-pel, 2 = quarter-pel (the preset value), -1 = preset |
| `--deint-pel-search <n>` | sub-pel search steps, -1 = preset |
| `--deint-search-refine <n>` | refinements after the first match, -1 = preset |
| `--deint-search-param <n>` | search window size, -1 = preset |
| `--deint-analyze-delta <n>` | 1 = compute vectors for one frame distance instead of two (half the search), -1 = preset |
| `--deint-overlap <n>` | block overlap; the step is blockSize - overlap, so smaller means fewer blocks and a coarser motion field, -1 = preset |
| `--deint-tr0 <n>` | the first pass's search radius, -1 = preset (2 at slow, 1 at faster) |
| `--deint-tr1-delta <n>` | TR1 temporal radius, 0 = stage off, -1 = preset |
| `--deint-tr2-delta <n>` | TR2 temporal radius, 0 = stage off (a whole stage costs far more delay than its radius does), -1 = preset |
| `--deint-retouch-smode <n>` | the first pass's re-sharpen mode, 0 = off, -1 = preset (0 is what the clean pass of a two-pass flow runs with) |
| `--low-latency` | the low-latency variant as one switch, see the [Readme](Readme.en.md#the-knobs-that-matter) |
| `--spatial-early-sad <n>` | skip the spatial refine when a block's base vector SAD is below n: -1 = off (the default), 0 = only on exact matches |

Second pass (the clean chain; two-pass flows only):

| option | meaning |
|---|---|
| `--clean-preset <n>` | build the second pass from this preset (0 placebo, 1 veryslow, 2 slower, 3 slow, 4 medium, 5 fast ... 10 draft; -1 = the same preset as the first pass) |
| `--clean-analyze-delta <n>` | the clean chain's analyze delta (1 is about half the search), -1 = preset, 0 = let TR1/TR2 decide |
| `--clean-tr1-delta <n>` | clean chain TR1 delta, 0 = off, -1 = preset |
| `--clean-tr2-delta <n>` | clean chain TR2 delta, 0 = off, -1 = preset |
| `--clean-rep2-thin <n>` | clean chain REP2 `repairThin`, 0 = off, -1 = preset |

Split frame:

| option | meaning |
|---|---|
| `--split-y <n>` | run the chain on one band of rows and a cheap deinterlacer on the rest. n is the boundary row and must be even; 0 (the default) means no split |
| `--seldif <name>` | the cheap deinterlacer for the other band: `yadif` (default) or `bwdif`. A `:top` / `:bottom` suffix sets the side too, so `--seldif bwdif:bottom` is both choices at once |
| `--seldif-side <name>` | which band that one takes: `top` (default) means rows `[0,n)` are the cheap one and `[n,height)` the chain |

Other:

| option | meaning |
|---|---|
| `-h, --help` | print this message |

## Deployment

- **`nnedi3_weights.bin` must sit next to the module.** The EDI stage looks for it in the current directory, then the module's directory, then the executable's directory. When it is missing, `init` fails with `RGY_ERR_INVALID_PARAM` rather than degrading.
- **The CUDA runtime is linked statically** (`CMAKE_CUDA_RUNTIME_LIBRARY = Static`), so the target machine needs a working NVIDIA driver but no CUDA Toolkit.
- **Keep the user-space and kernel-space driver versions in step.** When they differ, `nvidia-smi` reports `Driver/library version mismatch`; CUDA usually still runs, but a reboot to realign them is the cleaner state.
- **NVRTC, NPP, NVVFX, NGX, NVML, ONNX and the rest are all switched off** (see `RtgmDif/NVEnc/rtgmc_config.h`), so none of their libraries are needed at run time.

## FAQ

**Does the target machine need the CUDA Toolkit?**
Only to build. At run time the module needs the driver alone; the CUDA runtime is linked inside it.

**Can it be built with CUDA 12.x?**
Yes, but change the architecture list at the same time: `61 86 89 120` gives Ada and Blackwell native cubins, after which `50-virtual` can be dropped. That brings the module down to about 136 MB and drops Maxwell, Volta, sm_80 and Hopper - list those too if you need them.

**What is `-allow-unsupported-compiler` for?**
`nvcc`'s host compiler version check is conservative; this relaxes it. It is applied on both platforms - a safety net when building with gcc-10 on Linux, and what lets newer MSVC versions through on Windows.

**Why both `RtgmDif.map` and `RtgmDif.def`?**
`.def` is the Windows module definition file, `.map` is the Linux version script, and both list the same thing: `rtgmdif_create` and `rtgmdif_destroy`. Without `RtgmDif.map` the `.so` would export all ~2000 NVEncCore symbols.

**What is `rgy_config.h`, and why did Windows builds never need it?**
It is the platform configuration header that `rgy_version.h` includes on **non-Windows** builds only. Upstream carries it; the vendored subset did not, because Windows never reached that include. This repository adds a minimal version.

**How high can `-j` go?**
Keep it at or below the physical core count. Every `-gencode` raises the peak `nvcc`/`cicc` memory, RTGMC's `.cu` files are large, and overshooting `-j` ends in an out-of-memory failure - which is exactly why the original project file compiled a single architecture.
