# Rtgmc

RTGMC deinterlacing (50i/60i -> 50p/60p) as a standalone shared library.

`RtgmDif` takes the RTGMC filter out of [rigaya's NVEnc](https://github.com/rigaya/NVEnc) and packages it as a module you can call from your own program: **device frames in, device frames out**, with all work enqueued on a CUDA stream you own. It is the filter itself, not a front end - it does not read files, decode, encode, or touch host memory on your behalf.

Chinese version: [Readme.cn.md](Readme.cn.md). Build instructions: [Build.en.md](Build.en.md).

## Layout

    RtgmDif/          the module: a shared library with a C API; the NVEnc sources it needs
                      are vendored under RtgmDif/NVEnc/
    RtgmDifTest/      a deliberately simple reference driver (raw YUV in, raw YUV out)
    Output/           build output, same on both platforms
    RtgmDif/RtgmDif.h the full interface documentation, and the document to read first

## Interface

Two exported functions; everything else goes through the object they return.

```cpp
#include "RtgmDif.h"

RtgmDif *dif = rtgmdif_create();          // nullptr if the module could not start
if (!dif) { /* the module failed to start */ }

RtgmDifConfig cfg;                        // member initialisers, defaults are the 1080i50 case
cfg.flow   = RTGMDIF_FLOW_SLOW_BOTH;      // see the flows below
cfg.width  = 1920;
cfg.height = 1080;
cfg.csp    = RTGMDIF_CSP_YUV422P;          // the format this is normally run on
dif->configure(cfg);                      // only valid before init()

dif->init(stream);                        // your stream, or nullptr to let the module own one

dif->handdif(pDeviceIn);                  // hand it one 50i frame (device memory)
for (int i = 0; i < dif->getoutputsize(); i++)
    dif->getoutputbuf(i, pDeviceOut);     // read this batch's 50p frames, in order

dif->handdif(nullptr);                    // flush: read the tail the same way
rtgmdif_destroy(dif);
```

Worth knowing:

- **Two outputs per input.** Every 50i frame handed in produces two 50p frames. A few frames of priming delay at the start, and the same amount of tail released by the flush.
- **Batch semantics.** The frames one `handdif()` releases form the current batch, read with `getoutputsize()` / `getoutputbuf()`. The next call replaces it, so read it before calling again.
- **Your stream is never blocked.** The input copy, every kernel of the chain and the output copies all go on the stream passed to `init()`, and nothing is synchronised behind your back - upload, RTGMC and download can be queued back to back. The exception is `init(nullptr)`, where the module creates and owns a blocking stream and `getoutputbuf()` therefore has to wait.
- **Buffer layout.** Tightly packed planar, exactly the raw YUV file layout (plane Y, then U, then V, no row padding), so a frame can be handed over without any repacking.
- **The input frame is yours again immediately.** `handdif()` copies it into the module's own ring, so `pIn` can be reused as soon as the call returns.
- **`outputPoolFrames`** (32 by default) has to cover the temporal tail the flush releases in one go. Measured, that tail is 26 frames, so 32 leaves a little room. The module fails loudly if it is ever too small rather than dropping frames.

`RtgmDif.h` documents every config field, its range, and the measurements behind it.

## Flows

Seven preset flows, from single pass to two pass:

| flow | preset | structure | notes |
|---|---|---|---|
| `fast_deint` | fast | single pass | the plain deinterlace |
| `fast_both` | fast | two pass | plus the clean pass (QTGMC's second stage) |
| `fast_opt` | fast | two pass | stronger nnedi3 window (nnsize 3) with Slower's TR structure (TR1 2 / TR2 1). Measured on 1080i: residual field-parity alternation in static areas 0.0525 -> 0.0422, for about +0.1% cost |
| `faster_nn1` | faster | single pass | the cheapest flow, added for the scrolling-text case (`faster` alone is weaker than `fast`, but at nnsize 1 its EDI is the strong one) |
| `slow` | slow | single pass | lines up with QTGMC(preset="Slow") |
| `slower` | slower | single pass | lines up with QTGMC(preset="Slower") |
| `slow_both` | slow | two pass | the two-stage setup QTGMC itself describes: `QTGMC(50i, Slow)` then `QTGMC(50p, InputType=1, Sharpness=0)` |

A single-pass flow is not the finished picture: the clean pass is what removes the residual field-parity shimmer. When the single-pass result still shows some, use the matching two-pass flow.

## Input format

- Colour space: **`yuv422p` is the format this is normally run on**; `yuv420p` and `yuv444p` are supported as well. The filter runs on the input colour space as it is (**no conversion**); the output has the same csp and bit depth as the input.
- Bit depth: **8 only**.
- Width and height must be even.

## The knobs that matter

| field | what it does |
|---|---|
| `lowLatency` | The low-latency variant as one switch: search radius 1, analyze vectors at distance 1 only, the TR stages no longer waiting for a search they do not run, the scene-change readback no longer pipelined, TR2 off. Measured at 1080p 422p: single-pass `slow` 8.5 -> 2.5 input frames (26 -> 18 ms/frame), two-pass `slow_both` 15.5 -> 4.0 (40 -> 24 ms/frame). A two-pass flow stays close to its own picture (intra-pair 0.693 against 0.682, indistinguishable by eye); a single-pass flow cannot be tuned into a two-pass one (0.917 at best against 0.693) because the deinterlace pass works in the **field** domain, where the vertical neighbourhood is a different instant and the vectors are least accurate, while the clean pass re-runs the same chain in the **progressive** domain |
| `splitY` / `splitDif` / `splitDifSide` | Run the chain over one band of rows only and let a cheap YADIF/BWDIF do the rest, trading a possibly visible seam for time. What decides the cost is the number of rows the chain gets, not which side they are on: about 0.046 ms per chain row plus 4 ms of fixed cost at 1080p 422p. `splitY = 0` (the default) means no split and is byte-for-byte the old behaviour |
| `deintPel` / `deintPelSearch` / `deintSearchRefine` / `deintSearchParam` | The first pass's motion search, where a frame's time goes: at 1080p 422p `slow` costs 26.4 ms, with `fast`'s search settings (pel 1, pelSearch 1, refine 2) 13.6 ms, and 7.7 ms with a 32-wide block on top |
| `deintTr0` / `deintTr1Delta` / `deintTr2Delta` | The first pass's temporal radii - the **latency** levers. What sets the backlog is the number of active temporal stages times what each holds, not the radius alone: `fast` and `slow` have the same radii (tr0 2, tr1 1) yet `fast` is 2.5 input frames shorter because its TR2 is off |
| `cleanAnalyzeDelta` / `cleanPreset` / `cleanTr1Delta` / `cleanTr2Delta` / `cleanRep2Thin` | The second pass (clean chain) only. `cleanAnalyzeDelta 1` is the one knob so far that visibly moves the clock: `slow_both` 43.2 -> 36.7 ms (-15%). `cleanPreset` builds the second pass from a different preset (the usual Slow + Medium pairing); in this tree slow and medium differ only in `pel`/`pelSearch`, 2 -> 1 |

## Threads and device memory

- **No host memory.** The caller does the file I/O and the host <-> device copies; the module only ever touches device memory.
- **Everything runs on the caller's stream** (unless `init(nullptr)`), and the module creates no threads of its own.
- **Device memory**: the clean chain of a two-pass flow costs about 1.2 GiB extra at 1080p 444p; the output ring holds `outputPoolFrames` (32 by default) frames.

## Platforms and GPUs

One build covers every NVIDIA card from Maxwell onwards: every cubin CUDA 11.5 can emit, plus a `compute_50` PTX fallback. See [Build.en.md](Build.en.md#cuda-architecture-coverage) for the full table.

Verified on:

| platform | environment | GPU |
|---|---|---|
| Windows | MSVC 2019 + CUDA 11.5 | RTX 3080 |
| Linux | Ubuntu 22.04 + gcc-10 + CUDA 11.5.119 | RTX 5000 Ada |

## Building

One command per platform; requirements, outputs, architecture notes and how to verify a build are in [Build.en.md](Build.en.md).

## Licence and provenance

MIT - see [LICENSE](LICENSE).

**This project is a derivative work of [rigaya's NVEnc](https://github.com/rigaya/NVEnc).** Everything under `RtgmDif/NVEnc/` comes from that project, the RTGMC filter at the heart of this one included, and the upstream MIT notice is kept verbatim at the top of each of those files. What this project contributes is the packaging - the C API, the device-in/device-out interface, the CMake build for both platforms, the reference driver - and the modifications needed to make the filter stand alone.

A few files are not covered by this project's MIT grant (NVIDIA SDK headers, the nnedi3 weights). [NOTICE](NOTICE) lists them and gives the full list of changes relative to upstream - read it before you redistribute.
