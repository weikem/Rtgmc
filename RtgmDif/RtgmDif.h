// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------
//
// RtgmDif : the RTGMC deinterlacer as a library, device-in / device-out.
//
// It owns the whole filter chain (the integrated NVEncFilterRtgmc plus, for the two-pass
// flows, the hand-chained single-rate clean stage) and exposes one call per input frame:
//
//     RtgmDif *dif = rtgmdif_create();
//     dif->configure(cfg);
//     dif->init();
//     for (;;) {
//         dif->handdif(pDeviceIn);                    // one 50i frame, device memory
//         for (i = 0; i < dif->getoutputsize(); i++)
//             dif->getoutputbuf(i, pDeviceOut);       // 50p frames, device memory
//     }
//     dif->handdif(nullptr);                          // flush: read the tail the same way
//     rtgmdif_destroy(dif);
//
// The caller does the file I/O and the host<->device copies and nothing else; every kernel,
// every intermediate frame buffer and every CUDA stream belongs to this module.
//
// Buffer layout: tightly packed planar, exactly the raw YUV layout of the file - plane Y,
// then U, then V, one row after another, no padding. That is also what a raw YUV reader
// produces, so a frame can be handed over without any repacking.
//
// Stream ordering: everything this module enqueues - the input copy, every kernel of the
// chain and the output copies - goes on the stream passed to init(). That makes the module a
// normal citizen of an existing pipeline: upload, RTGMC and download can be queued back to
// back on one stream and run without a host round trip in between, and the module never
// blocks that stream.
//
// The exception is init() with a null stream, where the module creates (and then owns) a
// blocking stream of its own. Then the caller has no handle to order anything against, so
// getoutputbuf() has to wait, and that is the only place the module blocks.
//
// ------------------------------------------------------------------------------------------
//
// How the module is exported, and why there is no macro here.
//
// There is no __declspec(dllexport) either, not even on the definitions: the two exported names
// are listed in RtgmDif.def (Windows) or the RtgmDif.map version script (Linux) and the linker
// resolves them there. A .def cannot disagree with the declaration in the way an attribute can
// (that mismatch is exactly what C2375 is), and it says in one readable place what the module
// offers.
//
// The whole exported surface is two plain C functions, declared at the bottom of this file and
// named in that export list:
//
//     RtgmDif *rtgmdif_create();          // or nullptr if the module could not start
//     void     rtgmdif_destroy(RtgmDif *);
//
// RtgmDif itself is a pure interface: no constructors, no data members, no code except the
// out-of-line destructor that anchors its vtable inside the DLL. A consumer never allocates,
// destroys or copies one - it only calls through the pointer the factory returns, and a call
// through a vtable needs no imported symbol at all. So this header needs no conditional, no
// RTGMDIF_EXPORTS to define on one side and forget on the other, no dllimport that quietly
// turns into a link error when the import library is missing, and no RTGMDIF_STATIC third
// case for a configuration that does not exist.
//
// The shape is the same one the NVENC SDK uses on this very machine
// (NvEncodeAPICreateInstance): one factory that is the ABI, everything behind it reached
// through an interface. It also keeps three things on the module's side of the wall - the C++
// name mangling, the class layout, and the allocator that has to free what it allocated.
//
// `dumpbin /exports RtgmDif.dll` (Windows) or `nm -D --defined-only libRtgmDif.so` (Linux)
// prints exactly two names, both undecorated.
//
// ------------------------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

// The flows:
//
//   fast_deint : preset fast, deinterlace only
//                one pass, 50i -> 50p. The plain deinterlace.
//
//   fast_both  : preset fast, deinterlace + clean pass
//                the same pass 1, then the single-rate clean pass with the preset's own
//                values (QTGMC's second pass: TR1 delta 1, TR2 off for this preset since
//                `fast` maps to TR2=0, Rep2 thin 4, Sharpness=0).
//
//   fast_opt   : preset fast, clean TR1 2 / TR2 1, nnsize 3
//                pass 1 with the stronger nnedi3 window, and the clean pass brought up to
//                QTGMC's Slower TR structure (TR1 delta 2 then TR2 delta 1). Measured on
//                1080i: the residual field-parity alternation in static areas drops to about
//                0.0422 from 0.0525 (mean |Y(n) - Y(n+1)| over the top 216 rows) for +0.1%
//                of the `fast` cost, and the extra EDI quality raises the vertical detail
//                measure from 1.1383 to 1.1439 - slightly above `slower`.
//
//   faster_nn1 : preset faster, deinterlace only, nnsize 1
//                the cheapest flow, and the one that came out of the scrolling-text case. On
//                interlaced text the `fast` family left visible residue and trails, which
//                turned out to be the nnedi3 window rather than the port: the upstream
//                implementation agrees byte for byte with RtgmDif there, at every preset tested.
//                `faster` alone is weaker than `fast` (TR0 1 instead of 2, Rep0 thin 0 instead
//                of 3), but with nnsize 1 its EDI is the strong one, and on that content it
//                was the only fast-side combination that looked right. One pass, so it is
//                also cheaper and lighter than fast_both / fast_opt.
//
//   slow       : preset slow, deinterlace only
//                QTGMC's Slow preset, one pass. Same parameter set as `slower` except for the
//                two entries whose threshold is Slower in apply_vpp_rtgmc_preset(): Sbb is 0
//                here against 1 there, and chroma motion search is off here against on there.
//                QTGMC's Slow has Sbb = 0, so this is the one that lines up with
//                QTGMC(preset="Slow") - `slower` lines up with QTGMC(preset="Slower").
//
//   slower     : preset slower, deinterlace only
//                QTGMC's Slower preset, one pass: TR0/TR1/TR2 = 2/2/1, Rep0/Rep2 = 4/4, NNEDI3
//                nnsize 1 with nneurons 1, SMode/SLMode/SLRad = 2/2/1, Sbb 1, chroma motion
//                search on. Also accepted on the text case; roughly 1.4x the cost of `fast`
//                in the upstream timing.
//
//   slow_both  : preset slow, deinterlace + clean pass
//                `slow` plus the single-rate clean pass, i.e. QTGMC's two stages:
//                QTGMC(50i, preset="Slow") then QTGMC(50p, InputType=1, Sharpness=0). The clean
//                pass inherits the preset's TR1/TR2 deltas and rep2 thinning and only forces
//                SMode to 0, which is what the two-stage setup does as well. This is the one
//                to reach for when the single-rate result still shows a little of the residual
//                field-parity alternation - it is what the second stage exists for. Costs the
//                clean chain's buffers (about 1.2 GiB at 1080p in 444p) on top of `slow`.
//
// All flows are 50i -> 50p, i.e. two output frames per input frame, with a priming delay of a
// few frames at the start and the same amount again released by the flush.
enum RtgmDifFlow {
    RTGMDIF_FLOW_FAST_DEINT = 0,
    RTGMDIF_FLOW_FAST_BOTH  = 1,
    RTGMDIF_FLOW_FAST_OPT   = 2,
    RTGMDIF_FLOW_FASTER_NN1 = 3,
    RTGMDIF_FLOW_SLOWER     = 4,
    RTGMDIF_FLOW_SLOW       = 5,
    RTGMDIF_FLOW_SLOW_BOTH  = 6,
};

// Colours the module accepts. The filter runs on the input csp as it is (no conversion), and
// the output has the same csp and bit depth as the input.
//
// yuv422p is the format this is normally run on.
enum RtgmDifCsp {
    RTGMDIF_CSP_YUV420P = 0,
    RTGMDIF_CSP_YUV422P = 1,
    RTGMDIF_CSP_YUV444P = 2,
};

// The split (see RtgmDifConfig::splitY) runs the chain on one band of rows and one of these on
// the other. Both are single-pass spatial-temporal deinterlacers that emit two frames per input
// frame in bob mode, so they are interchangeable here and cost a few milliseconds either way -
// nothing next to what the chain costs on its band. BWDIF is the better of the two on paper (it
// weights its spatial and temporal estimates against the local difference instead of switching
// between them), so it is worth a look, but it is still a single pass: it does not do the
// temporal smoothing the chain does, and the band it handles will show the field-parity shimmer
// that the chain exists to remove.
enum RtgmDifSplitDif {
    RTGMDIF_SPLITDIF_YADIF = 0,
    RTGMDIF_SPLITDIF_BWDIF = 1,
};

// Which band of the split that deinterlacer takes. The chain gets the other one.
enum RtgmDifSplitSide {
    RTGMDIF_SPLITSIDE_TOP    = 0,   // rows [0, splitY) are the cheap one's
    RTGMDIF_SPLITSIDE_BOTTOM = 1,   // rows [splitY, height) are the cheap one's
};

// Filled in by the caller and handed to configure(). The defaults are the 1080p 50i case.
//
// They are member initialisers and not a constructor on purpose: a constructor would have to
// live in the DLL and be exported for the caller's `RtgmDifConfig cfg;` to work, which is
// exactly the kind of glue this header avoids. As it is, the struct is an aggregate, the
// defaults are compiled into the caller, and nothing about it crosses the boundary.
struct RtgmDifConfig {
    int flow       = RTGMDIF_FLOW_FAST_BOTH;   // RtgmDifFlow
    int width      = 1920;      // frame width  (must be even)
    int height     = 1080;      // frame height (must be even)
    int csp        = RTGMDIF_CSP_YUV444P;      // RtgmDifCsp
    int bitdepth   = 8;         // 8 only
    int fpsNum     = 25;        // input frame rate, fields counted as one frame (25 for 50i)
    int fpsDen     = 1;
    int tff        = 1;         // 1 = top field first, 0 = bottom field first
    int deviceId   = 0;         // CUDA device index
    int logLevel   = 3;         // 0 quiet, 1 error, 2 warn, 3 info, 4 more, 5 debug
    // How many finished frames the output ring can hold at once. The steady state only ever
    // needs two (one input frame produces two output frames), but the flush releases the whole
    // temporal tail in one go, so it has to cover that too. Measured on the three flows, that
    // tail is 26 frames, so 32 covers it with a little room; it costs 32 * frame bytes of device
    // memory, and the module fails loudly (it does not silently drop) if it is ever too small.
    int outputPoolFrames = 32;

    // The first pass's motion search, on its own - the four fields below. See spatialEarlySad
    // for why the search and not the temporal stages is where a frame's time goes: on a detailed
    // frame almost every candidate has to be measured, so those frames cost half again what a
    // flat frame costs, and the search is the only part that behaves that way.
    //
    //   deintPel            2 = quarter-pel (the presets at slow and above), 1 = half-pel.
    //                       The biggest cut of the four and the one to try first.
    //   deintPelSearch      2 or 1: how far the sub-pel search walks at each step.
    //   deintSearchRefine   3 (the preset's value down to slow) to 0: how many refinements
    //                       follow a block's first match.
    //   deintSearchParam    2 or 1: the size of the search window.
    //
    // Measured at 1080p 422p on the first pass alone, my test clip, which is lighter than the
    // 1080i text case: slow 26.4 ms; with fast's search settings (pel 1, pelSearch 1,
    // searchRefine 2) 13.6 ms; with a 32-wide block on top of that 7.7 ms. So the four fields
    // are worth about what the preset step is worth, and the block size is the fifth lever if
    // they are not enough.
    //
    // They are applied to both passes; cleanPreset and the clean-side fields come after them, so
    // they still win on the second pass where the two overlap. -1 leaves the preset's value, and
    // all four change the picture.
    int deintPel = -1;
    int deintPelSearch = -1;
    int deintSearchRefine = -1;
    int deintSearchParam = -1;

    // Two more ways to bound what a frame can cost, and neither of them touches sub-pel
    // accuracy - which is what made deintPel unacceptable: that one lowers the precision of
    // every block, static ones included, while these two only take work away.
    //
    // deintAnalyzeDelta   1 = compute the vectors for one frame distance instead of two, which
    //                     halves the search. The clean chain's equivalent (cleanAnalyzeDelta)
    //                     was pure waste - nothing consumed delta 2 - and measured -15%. The
    //                     first pass may not be the same: the integrated filter also has its EDI
    //                     matching and, for input_type 2/3, the progSADMask blend on that same
    //                     motion field, so watch the picture after setting it.
    // deintOverlap        The block step is blockSize - overlap, so overlap decides how many
    //                     blocks a frame is cut into: 8 (the preset) is a step of 8 and 240
    //                     blocks across 1920, 4 is a step of 12 and 160. The search runs per
    //                     block, so the saving is proportional - and so is the coarsening of the
    //                     motion field. 0 gives the coarsest, non-overlapping grid.
    //
    // -1 = the preset's value.
    int deintAnalyzeDelta = -1;
    int deintOverlap = -1;

    // The first pass's temporal radii - the latency levers. The chain holds a fixed backlog of
    // frames, and the ledger (`in X -> out Y`, 2 outputs per input) reads it out: 17 output
    // frames (8.5 input frames) on `slow`, 8 (4) on `faster`, 31 (15.5) on `slow_both`.
    //
    // What sets that backlog is the number of active temporal stages times what each holds, not
    // the radius alone: `fast` and `slow` have the same radii (tr0 2, tr1 1) and `fast` is 2.5
    // input frames shorter simply because its TR2 is off. Measured, 1080p 422p:
    //
    //     slower 9.0   slow 8.5   slow + analyze.delta 1: 8.0   fast 6.0   faster 4.0
    //
    //   deintTr0        2 at slow and fast, 1 from faster up: the search's temporal radius.
    //   deintTr1Delta   2 up to slower, 1 from slow up: the first temporal denoise's radius.
    //   deintTr2Delta   3 placebo, 2 veryslow, 1 up to medium, 0 from fast up (stage off).
    //                   0 turns that stage off, which removes its whole holding - that is where
    //                   the delay goes, far more than the radius does.
    //
    // -1 = the preset's value, which is what every flow did before these fields existed. All
    // three change the picture, and the second pass inherits them through the clean copy.
    int deintTr0 = -1;
    int deintTr1Delta = -1;
    int deintTr2Delta = -1;

    // The first pass's retouch (re-sharpen) mode. 0 leaves the block a pass-through, which is
    // what the clean pass of a two-pass flow runs with: it is the one setting every single-pass
    // flow keeps at its preset value (2 at slow, 1 at faster), and therefore the last untested
    // candidate for the residual field-parity shimmer that the two-pass flow removes. Applied
    // before the clean copy, so a two-pass flow gets the same value on both sides.
    int deintRetouchSmode = -1;

    // The first pass's post-TR2 shimmer repair, the same stage the clean pass exposes as
    // cleanRep2Thin - and it reaches both, because the deint fields are applied before the clean
    // copy: setting this to 0 leaves the clean pass' repair off as well unless cleanRep2Thin puts
    // it back. Measured, on the window where the single pass leaves its field-parity shimmer
    // (60 input frames, 1080p 422p, two-pass flow with the low-latency switch):
    //
    //   deint 4 / clean 4 (presets)      intra-pair 0.694 whole, 0.081 in the window
    //   deint 4 / clean 0                intra-pair 0.671 whole, 0.013 in the window
    //   deint 0 / clean 0                identical to the line above
    //
    // So it is the clean side's copy that does the work and the first pass' own repair is neither
    // helping nor hurting there; this knob is how the first pass' side is reached when the clean
    // one is put back explicitly (`--deint-rep2-thin 0 --clean-rep2-thin 4`). It is a spatial
    // stage, so no value of it changes a temporal window or the pipeline delay. -1 = the preset's
    // value (4 at slow). rep1 is off in every preset. A rep2Thin knob for the first pass was
    // measured and reverted: it bought nothing (the clean side's copy already carries the effect)
    // and it moved the 422p equivalence hashes, so the field is gone.

    // The low-latency variant, as one switch. It shortens every wait the chain has and drops the
    // work nothing reads:
    //
    //   - the search radius at 1 (deintTr0);
    //   - the analyze stage's vectors at distance 1: nothing here reads a distance-2 vector (TR1
    //     asks for 1, source_match is off), so half the search work goes away as well as a wait;
    //   - the degrain's TR stages stop waiting for the frames their own search would have needed -
    //     the vectors come from the analyze stage;
    //   - the scene-change readback is resolved in the call that submits it instead of being
    //     pipelined;
    //   - TR2 off, in every flow. This is the one part of the switch that is a picture change
    //     rather than a wait: it is what the last frame of the budget costs.
    //
    // Measured at 1080p 422p, both flows:
    //
    //     single-pass `slow`      8.5 -> 2.5 input frames (340 -> 100 ms), 26.3 -> 18 ms/frame
    //     two-pass   `slow_both` 15.5 -> 4.0 input frames (620 -> 160 ms), 40.1 -> 24.4 ms/frame
    //
    // The two passes split that 4.0 frames and ~25 ms roughly 5:3 in both: the deinterlace pass
    // holds 2.5 frames and takes about 16 ms (the motion search is in it, which is what makes it
    // the content-dependent half), and the clean chain holds 1.5 frames and takes about 8.5 ms.
    //
    // Getting to 4.0 on `slow_both` was a choice. Turning the clean chain's TR2 off instead of the
    // deinterlace pass's lands at 4.5 frames (intra-pair 0.689 against 0.693, on a flow whose own
    // reference is 0.682), and side by side the two could not be told apart, so the switch takes
    // the cheaper one. Against the unswitched flow the picture is a little worse - visible next to
    // it, close enough to use - and the mean per-frame difference is 0.030 gray levels. Two-pass
    // flows are what this switch is for.
    //
    // A single-pass flow is a different matter: its picture is not the finished one whatever the
    // switch does, because the clean pass is what removes the residual shimmer. Measured on one
    // 422p window, same frames, intra-pair: 0.917 on `slow` itself, 0.975 switched with TR2 off
    // (2.5 frames), 0.931 switched with TR2 back at delta 1 (3.0), 0.950 with TR1 at delta 2
    // (3.5), and 0.918 with TR1 2, TR2 1 and tr0 2 - every temporal knob that flow has - for 5.0
    // frames. The two-pass flow beats all of that from 4.0 frames (0.693 against 0.682
    // unswitched). The reason is where the blending happens: the deinterlace pass works in the
    // field domain, where a field's vertical neighbourhood is a different instant and its vectors
    // are the least accurate, while the clean pass re-runs the same chain in the progressive
    // domain where they are not. So a single-pass flow cannot be tuned into a two-pass one, and
    // `deintTr1Delta`/`deintTr2Delta`/`deintTr0` on top of the switch only buy back what the
    // switch took away.
    //
    // Three things are not part of the switch, all of them measured. Dropping the analyze stage's
    // own tr0 term reaches 3.0 input frames - one frame per pass - but the analysis luma of a frame
    // is built from its +/-2 neighbours, so an analysis that runs a frame early searches a luma
    // whose future neighbours have not arrived and are clamped: two clamped instead of one.
    // Intra-pair shimmer on the same window went 0.693 -> 0.893, which is what a single-pass flow
    // measures, so half a frame costs the whole picture. (Done naively in a two-pass flow it also
    // breaks outright - the clean analyze finds no luma slot at all and the degrain fails with
    // "invalid call sequence" at input frame 3; the cache has to be prefetched to the newest frame
    // before it even runs, and the picture is what pays either way.) And `deintTr0 0` measures 1.5
    // input frames (60 ms) and 14.3 ms per frame, but taking the search's temporal radius away
    // moves the motion field itself (~1.0 gray levels); it looked unusable on interlaced drama.
    // Both stay knobs to set on purpose.
    //
    // What is left after the switch is algorithmic rather than bookkeeping: the prefilter holds two
    // frames because a +/-1 temporal prefilter needs the next frame to produce this one, the
    // analyze holds two (its delta plus the tr0 that the clean pass's own analyze needs), and each
    // TR stage in each pass holds one.
    //
    // Off by default: 0 = upstream behaviour, 1 = on. Explicit deintTr0/deintTr2Delta still win
    // over what the switch sets.
    int lowLatency = 0;

    // Spatial-refine early exit, in SAD units: after a block's base vector has been found, the
    // spatial refinement of it is skipped when that vector's SAD is already good enough.
    //
    // This is the only search early-exit the vendored NVEncCore actually implements. The preset
    // table's search_early_sad is written, range-checked and printed, but no kernel reads it, so
    // a knob for that one would silently do nothing (measured: 0, 16 and 32 gave byte-identical
    // output and timing within noise).
    //
    // -1 (the default, and what the preset tables leave this at) = off, so -1 is byte-identical
    // to how every flow ran before the field existed. 0 skips refine only on an exact zero-SAD
    // candidate; 1..65535 skip it when the SAD is below the threshold.
    int spatialEarlySad = -1;

    // The second pass on its own - the clean chain. Both passes start from the same preset, so
    // these exist to lighten only the clean side. They are applied after the copy from the
    // deinterlace side and after the flow's own clean overrides, so -1 everywhere is exactly
    // the old behaviour. 0 turns that stage off; an off stage is a pass-through and is not even
    // built, and rep2 off can also release the reference ring.
    //
    // -1 = the preset's value. TR1 delta 0..5, TR2 delta 0..5, REP2 repairThin 0..7 (0 = off).
    //
    // Measured at 1080p 422p: the clean chain is 12.6 ms of fast_both's 26.2 ms and 16.7 ms of
    // slow_both's 43.1 ms, so this is where a frame's second half goes.
    // The clean chain's analyze stage, which is the expensive part of that pass. It runs its own
    // motion analysis - pass 1's MV/SAD never leave the integrated filter - so the second pass
    // is a second full search per frame. The analyze delta is how many frame distances it
    // computes vectors for, so 1 is roughly half the work of the preset's 2. CleanChain::init
    // raises it to cover TR1 and TR2 if they ask for more (and says so in the log), so lowering
    // this on its own does nothing while a clean TR delta above it is still at the preset value.
    //
    // -1 = the preset's value. 0 lets TR1/TR2 decide (i.e. the raise above), 1..5 set it.
    //
    // Measured at 1080p 422p, and it is the only knob so far that moves the clock: the clean
    // chain costs 16.7 ms of slow_both's 43.1 ms, and 1 takes that to 10.3 - slow_both 43.2 ->
    // 36.7 ms, -15%. Turning the clean TR2 off, its rep2 off and halving TR1's delta together
    // bought 1.4 ms, which is what says the time is in this search and not in the mixing.
    // The output does change, so the result has to be looked at after setting it.
    int cleanAnalyzeDelta = -1;

    // The preset the second pass is built from, in NVEncCore's VppRtgmcPreset order:
    // 0 placebo, 1 veryslow, 2 slower, 3 slow, 4 medium, 5 fast, 6 faster, 7 veryfast,
    // 8 superfast, 9 ultrafast, 10 draft. -1 = the same preset as the first pass, which is what
    // every flow did before this field existed.
    //
    // This is the equivalent of the second QTGMC call in a two-pass setup using a different
    // preset (the usual `QTGMC(preset="Slow")` followed by `QTGMC(clip, InputType=1,
    // preset="Medium")`). In this tree the difference between slow and medium is only `pel` and
    // `pelSearch`, 2 -> 1, i.e. quarter-pel search becomes half-pel: the EDI size, the deltas,
    // rep2 and the search radius are all identical between those two. Going further down the
    // list does drop more (faster takes subpel 1, rep0 3 and a 32-wide block).
    //
    // It changes the picture, so set it and look at the result. The clean-side fields below are
    // applied after this, so they still win.
    //
    // Measured at 1080p 422p on slow_both (one session, 40 frames, 32 timed): baseline 38.8 ms,
    // 4 (medium, i.e. pel 2 -> 1) 34.3 ms, 4 together with cleanAnalyzeDelta 1 31.3 ms. Going on
    // to 5 (fast) was not faster than 4. The baseline itself swings about 10% between sessions,
    // so only differences of a few milliseconds upward mean anything here.
    int cleanPreset = -1;

    int cleanTr1Delta = -1;
    int cleanTr2Delta = -1;
    int cleanRep2Thin = -1;

    // The split: instead of running the whole chain over the whole frame, run it over one band
    // of rows and let a cheap deinterlacer (YADIF or BWDIF, `splitDif`) do the rest. `splitY` is
    // the boundary row and `splitDifSide` says which side of it the cheap one takes:
    //
    //   splitDifSide = TOP    : rows [0, splitY) = the cheap one, [splitY, height) = the chain
    //   splitDifSide = BOTTOM : rows [0, splitY) = the chain,     [splitY, height) = the cheap one
    //
    // The test program's --seldif and --seldif-side set the last two.
    //
    // splitY = 0 leaves the cheap one's band empty, so that is the default and it means "no split
    // at all": the whole frame goes through exactly as before, byte for byte. splitY = height is
    // the other end - all cheap - which is only useful as a test.
    //
    // What it buys is time, and it only buys it because the chain really does run on band sized
    // frames: a 380-row band has 35% of the blocks and 35% of the EDI, so the deinterlace and
    // the clean pass shrink with it. What does not shrink is the fixed cost of a frame - the
    // kernel launches, the synchronises, the priming - so the saving is less than the area
    // suggests.
    //
    // What decides the time is the rows the chain gets, not which side they are: measured at
    // 1080p 422p a frame costs about 0.046 ms per chain row plus 4 ms, so a boundary at 540
    // gives the chain 556 rows whichever side it is on and the two cost the same, while a
    // boundary at 750 leaves it 346 rows from below and 766 from above - a factor of two.
    //
    // What it costs is a seam: two deinterlacers on one picture do not agree about sub-pel
    // detail, so the boundary can be visible. Each side is therefore processed with a margin of
    // rows (RTGMDIF_SPLIT_MARGIN) that is computed but not used, so that the boundary row is an
    // interior row of both filters rather than an edge row.
    //
    // The two sides are put together by output frame number, not by timestamp: both paths emit
    // two frames per input frame, in order, so the k-th frame of one belongs with the k-th frame
    // of the other whatever their internal delays are. Whichever runs ahead is held in a queue
    // until its counterpart arrives.
    int splitY = 0;                                    // 0 .. height
    int splitDif = RTGMDIF_SPLITDIF_YADIF;             // RtgmDifSplitDif
    int splitDifSide = RTGMDIF_SPLITSIDE_TOP;          // RtgmDifSplitSide
};

// The interface. Obtained from rtgmdif_create() and never constructed, copied or deleted by the
// caller - use rtgmdif_destroy(), or hold it in a unique_ptr with that as the deleter.
class RtgmDif {
public:
    // Declared here, defined in RtgmDif.cpp. That out-of-line definition is what puts the
    // vtable in the DLL rather than in every consumer that includes this header.
    virtual ~RtgmDif();

    RtgmDif(const RtgmDif &) = delete;
    RtgmDif &operator=(const RtgmDif &) = delete;

    // Sets the configuration. Only valid before init(); returns false if it is inconsistent.
    virtual bool configure(const RtgmDifConfig &cfg) = 0;

    // Builds the chain. Has to be called before the first handdif().
    //
    // `stream` is the CUDA stream every kernel and every copy of this module is enqueued on.
    // Pass the stream of the surrounding pipeline to keep the work ordered with it, or pass
    // nullptr (or just call init()) and the module creates and owns a blocking stream.
    //
    // A stream supplied by the caller is never destroyed here, and outside of close() it is
    // never synchronised either - see getoutputbuf() for what that means for the caller.
    virtual bool init(cudaStream_t stream = nullptr) = 0;

    // Hands one 50i input frame to the chain.
    //
    //   pIn != nullptr : pIn points at a tightly packed planar frame in device memory. The
    //                    frame is copied into the module's own ring, so the caller may reuse
    //                    pIn as soon as this returns.
    //   pIn == nullptr : flush. Runs the whole pipeline out so that the frames the temporal
    //                    filter is still holding are released into the last batch. Safe to
    //                    call more than once; after the first call there is nothing left.
    //
    // Frames released by this call form the current batch; read them with getoutputsize() and
    // getoutputbuf(). The batch is replaced by the next call, so read it before calling again.
    virtual bool handdif(void *pIn) = 0;

    // Number of frames in the current batch.
    virtual int getoutputsize() = 0;

    // Copies frame idx of the current batch into pOut (tightly packed planar, device memory).
    // Returns false if idx is out of range or the copy failed.
    //
    // The copy is queued on the module's stream. When the caller supplied that stream - the
    // normal case - nothing is waited for here: order the download after it by using the same
    // stream, and the module never blocks the pipeline. When the module owns the stream
    // (init() with a null stream) the copy is synchronised before returning, so pOut is
    // complete as soon as this returns.
    virtual bool getoutputbuf(int idx, void *pOut) = 0;

    // Frame sizes of the tightly packed planar layout, in bytes.
    virtual int inputFrameBytes() const = 0;
    virtual int outputFrameBytes() const = 0;

    // Counters, for a caller that wants to check the frame relationship.
    virtual int framesIn() const = 0;
    virtual int framesOut() const = 0;

    virtual bool inited() const = 0;

    // Releases everything. Called by the destructor as well.
    virtual void close() = 0;

protected:
    // Protected, so that only a derived class can make one, and the only derived class is the
    // one inside this module that rtgmdif_create() constructs. It has to be declared because the
    // deleted copy constructor above suppresses the implicit default constructor, and a derived
    // class's constructor needs a base constructor to call.
    RtgmDif() = default;
};

// The module's entire exported surface. Declared without any decoration: the names are exported
// by name through RtgmDif.def on Windows and the RtgmDif.map version script on Linux, so a
// consumer needs nothing - not even dllimport, which for two calls would buy nothing but a way
// to get it wrong.
extern "C" {
    RtgmDif *rtgmdif_create();
    void rtgmdif_destroy(RtgmDif *dif);
}
