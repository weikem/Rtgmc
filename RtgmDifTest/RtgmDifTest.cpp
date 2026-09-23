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
// RtgmDifTest : a deliberately dumb driver for RtgmDif.
//
// This program knows how to do exactly four things with the GPU, and nothing else:
//
//     fread        -> host memory        (read a raw YUV frame from the file)
//     cudaMemcpy   -> device memory      (upload it)
//     cudaMemcpy   -> host memory        (download the result)
//     fwrite       -> the output file    (write it out)
//
// There is no .cu file in this project, no kernel, no filter, no knowledge of what RTGMC
// does: every kernel lives in RtgmDif.dll. All this program knows is the four entry points
// of RtgmDif and the fact that frames are copied back and forth in the raw YUV layout.
//
// The copies are the synchronous form of cudaMemcpy on purpose. That makes the ordering
// obvious - the upload is complete before handdif() is called and the download starts after
// getoutputbuf() has already synchronised its own stream - so the test can be about the
// interface rather than about stream bookkeeping.
//
// ------------------------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "RtgmDif.h"

// The two MSVC spellings of the 64-bit stdio seek/tell, used when sizing the input file.
// On Linux off_t is already 64-bit, so fseeko/ftello are the exact equivalents - and the
// same names the module's own rgy_osdep.h maps them to.
#if !defined(_WIN32) && !defined(_WIN64)
#ifndef _fseeki64
#define _fseeki64 fseeko
#endif
#ifndef _ftelli64
#define _ftelli64 ftello
#endif
#endif

struct TestConfig {
    std::string inPath;
    std::string outPath;
    int         flow;
    std::string flowName;
    int         width;
    int         height;
    int         csp;
    std::string cspName;
    int         frames;      // 0 = until the end of the file
    int         skip;        // frames fed through the chain but neither written nor timed
    int         statsEvery;  // ledger print cadence, in input frames (1 = every frame)
    int         fps;
    int         tff;
    int         device;
    int         logLevel;
    int         ownStream;   // 1 = let the module create its own stream instead of passing one
    int         deintPel;          // -1 = the preset's value
    int         deintPelSearch;    // -1 = the preset's value
    int         deintSearchRefine; // -1 = the preset's value
    int         deintSearchParam;  // -1 = the preset's value
    int         deintAnalyzeDelta; // -1 = the preset's value
    int         deintOverlap;      // -1 = the preset's value
    int         deintTr0;          // -1 = the preset's value: the search's temporal radius
    int         deintTr1Delta;     // -1 = the preset's value
    int         deintTr2Delta;     // -1 = the preset's value; 0 turns the TR2 stage off
    int         deintRetouchSmode; // -1 = the preset's value; 0 = the resharpen is a pass-through
    int         lowLatency;        // 1 = the low-latency variant of the first pass
    int         spatialEarlySad;   // -1 = off (the preset's own value)
    int         cleanPreset;       // -1 = the first pass's preset
    int         cleanAnalyzeDelta; // -1 = the preset's value
    int         cleanTr1Delta;     // -1 = the preset's value
    int         cleanTr2Delta;     // -1 = the preset's value
    int         cleanRep2Thin;     // -1 = the preset's value
    int         splitY;            // 0 = no split; otherwise the boundary row (even)
    int         splitDif;          // RTGMDIF_SPLITDIF_*: the deinterlacer on the other band
    int         splitDifSide;      // RTGMDIF_SPLITSIDE_*: which band that one takes

    TestConfig()
        : inPath()
        , outPath()
        , flow(RTGMDIF_FLOW_FAST_BOTH)
        , flowName("fast_both")
        , width(1920)
        , height(1080)
        , csp(RTGMDIF_CSP_YUV444P)
        , cspName("yuv444p")
        , frames(0)
        , skip(0)
        , statsEvery(16)
        , fps(25)
        , tff(1)
        , device(0)
        , logLevel(3)
        , ownStream(0)
        , deintPel(-1)
        , deintPelSearch(-1)
        , deintSearchRefine(-1)
        , deintSearchParam(-1)
        , deintAnalyzeDelta(-1)
        , deintOverlap(-1)
        , deintTr0(-1)
        , deintTr1Delta(-1)
        , deintTr2Delta(-1)
        , deintRetouchSmode(-1)
        , lowLatency(0)
        , spatialEarlySad(-1)
        , cleanPreset(-1)
        , cleanAnalyzeDelta(-1)
        , cleanTr1Delta(-1)
        , cleanTr2Delta(-1)
        , cleanRep2Thin(-1)
        , splitY(0)
        , splitDif(RTGMDIF_SPLITDIF_YADIF)
        , splitDifSide(RTGMDIF_SPLITSIDE_TOP) {
    }
};

static void print_usage(const char *exe) {
    printf("RtgmDifTest : raw YUV 50i -> 50p through RtgmDif\n\n");
    printf("usage: %s -i <input.yuv> -o <output.yuv> [options]\n\n", exe);
    printf("  -i, --input   <file>   input raw YUV file (required)\n");
    printf("  -o, --output  <file>   output file, same csp as the input. Optional: without it the\n");
    printf("                         whole chain still runs but nothing is written (timing runs)\n");
    printf("      --flow <name>      fast_deint / fast_both (default) / fast_opt /\n");
    printf("                         faster_nn1 / slow / slow_both / slower\n");
    printf("                           fast_deint : preset fast, deinterlace only\n");
    printf("                           fast_both  : preset fast, deinterlace + clean pass\n");
    printf("                           fast_opt   : fast_both with TR1=2 TR2=1 nnsize=3\n");
    printf("                           faster_nn1 : preset faster, deinterlace only, nnsize=1\n");
    printf("                           slow       : preset slow, deinterlace only (= QTGMC Slow)\n");
    printf("                           slow_both  : slow + clean pass (QTGMC's two stages)\n");
    printf("                           slower     : preset slower, deinterlace only\n");
    printf("      --width  <n>       frame width   (default: 1920)\n");
    printf("      --height <n>       frame height  (default: 1080)\n");
    printf("      --in-csp <name>    yuv420p / yuv422p / yuv444p (default: yuv444p)\n");
    printf("      --deint-pel <n>          1 = half-pel, 2 = quarter-pel (preset), -1 = preset\n");
    printf("      --deint-pel-search <n>   sub-pel search steps, -1 = preset\n");
    printf("      --deint-search-refine <n>  refinements after the first match, -1 = preset\n");
    printf("      --deint-search-param <n> search window size, -1 = preset\n");
    printf("      --deint-analyze-delta <n>  1 = one frame distance instead of two (half the\n");
    printf("                         search), -1 = preset\n");
    printf("      --deint-overlap <n>      block overlap; the step is blockSize - overlap, so\n");
    printf("                         smaller means fewer blocks, -1 = preset\n");
    printf("      --deint-tr0 <n>          the first pass's search radius; it is one of the\n");
    printf("                         latency levers, -1 = preset (2 at slow, 1 at faster)\n");
    printf("      --deint-tr1-delta <n>    TR1 temporal radius, -1 = preset, 0 = stage off\n");
    printf("      --deint-tr2-delta <n>    TR2 temporal radius, -1 = preset, 0 = stage off\n");
    printf("                         (a whole stage costs far more delay than its radius does)\n");
    printf("      --deint-retouch-smode <n>  the first pass's re-sharpen mode; 0 = off, -1 = preset\n");
    printf("                         (0 is what the clean pass of a two-pass flow runs with)\n");
    printf("      --low-latency            the low-latency variant as one switch: search\n");
    printf("                         radius 1, analyze vectors at distance 1, the TR stages\n");
    printf("                         not waiting for a search they do not run, no scene-change\n");
    printf("                         readback pipeline, and TR2 off. Measured at 1080p 422p:\n");
    printf("                         slow 8.5 -> 2.5 input frames (26 -> 18 ms/frame), slow_both\n");
    printf("                         15.5 -> 4.0 (40 -> 24 ms/frame). On slow_both the picture\n");
    printf("                         is close to the flow's own - a little worse by eye, usable\n");
    printf("                         (intra-pair 0.693 against 0.682). The 4.5 frame variant\n");
    printf("                         (clean-tr2-delta 0, 0.689) looks the same, so the cheaper\n");
    printf("                         one is in the switch. A single-pass flow is not the finished\n");
    printf("                         picture either way - the clean pass is what removes the\n");
    printf("                         shimmer (0.68 with it, 0.97 without). --deint-tr0 0 goes to\n");
    printf("                         1.5 frames but was rejected by eye\n");
    printf("      --spatial-early-sad <n>  skip the spatial refine when a block's base vector\n");
    printf("                         SAD is below n: -1 = off (default), 0 = only exact matches\n");
    printf("      --clean-preset <n>       clean chain only: build that pass from this preset\n");
    printf("                         (0 placebo .. 4 medium, 5 fast, 10 draft; -1 = as pass 1)\n");
    printf("      --clean-analyze-delta <n>  clean chain only: analyze delta (1 = about half\n");
    printf("                         the search), -1 = preset, 0 = let TR1/TR2 decide\n");
    printf("      --clean-tr1-delta <n>    clean chain only: TR1 delta, 0 = off, -1 = preset\n");
    printf("      --clean-tr2-delta <n>    clean chain only: TR2 delta, 0 = off, -1 = preset\n");
    printf("      --clean-rep2-thin <n>    clean chain only: REP2 repairThin, 0 = off, -1 = preset\n");
    printf("      --split-y <n>            run the chain on one band of rows and a cheap\n");
    printf("                         deinterlacer on the rest. n is the boundary row and must be\n");
    printf("                         even. 0 (default) = no split: the whole frame is the chain\n");
    printf("      --seldif <name>          the cheap deinterlacer for the other band: yadif\n");
    printf("                         (default) or bwdif. A ':top'/':bottom' suffix sets the side\n");
    printf("                         too, so --seldif bwdif:bottom is both choices at once\n");
    printf("      --seldif-side <name>     which band that one takes: top (default) or bottom.\n");
    printf("                         top = rows [0,n) are the cheap one and [n,height) the chain\n");
    printf("      --frames <n>       stop after n input frames, counted from the first frame\n");
    printf("                         read (default: whole file). The frames --skip discards are\n");
    printf("                         part of this count, so --frames has to be larger than\n");
    printf("                         --skip: --skip 400 --frames 430 runs 430 frames and\n");
    printf("                         keeps the last 30. With --frames <= --skip there is\n");
    printf("                         nothing to keep and only the flush tail reaches the file\n");
    printf("      --skip <n>         feed the first n input frames through the chain without\n");
    printf("                         writing or timing them: use it to jump into a hard part\n");
    printf("                         of the clip. The chain is warmed up where the timed part\n");
    printf("                         starts\n");
    printf("      --stats-every <n>  print the in/out ledger every n input frames (default 16).\n");
    printf("                         1 = every frame, which is how the pipeline delay reads out\n");
    printf("      --fps    <n>       input frame rate (default: 25)\n");
    printf("      --bff              input is bottom field first (default: tff)\n");
    printf("      --device <n>       CUDA device index (default: 0)\n");
    printf("      --log-level <n>    0 quiet .. 5 debug (default: 3)\n");
    printf("      --own-stream       let the module create and own its own stream instead of\n");
    printf("                         handing it one (slower: it then has to synchronise the\n");
    printf("                         output copies, because the caller has no handle on it)\n");
    printf("  -h, --help             show this message\n");
}

static bool parse_int(const char *str, int *out) {
    char *end = nullptr;
    const long v = strtol(str, &end, 10);
    if (end == str || *end != '\0') {
        return false;
    }
    *out = (int)v;
    return true;
}

static bool parse_flow(const std::string &name, int *flow, std::string *flowName) {
    static const struct { const char *name; int value; } table[] = {
        { "fast_deint", RTGMDIF_FLOW_FAST_DEINT },
        { "fast_both",  RTGMDIF_FLOW_FAST_BOTH  },
        { "fast_opt",   RTGMDIF_FLOW_FAST_OPT   },
        { "faster_nn1", RTGMDIF_FLOW_FASTER_NN1 },
        { "slower",     RTGMDIF_FLOW_SLOWER     },
        { "slow",       RTGMDIF_FLOW_SLOW       },
        { "slow_both",  RTGMDIF_FLOW_SLOW_BOTH  },
    };
    for (const auto &entry : table) {
        if (name == entry.name) {
            *flow = entry.value;
            *flowName = name;
            return true;
        }
    }
    return false;
}

static bool parse_csp(const std::string &name, int *csp, std::string *cspName) {
    static const struct { const char *name; int value; } table[] = {
        { "yuv420p", RTGMDIF_CSP_YUV420P },
        { "yuv422p", RTGMDIF_CSP_YUV422P },
        { "yuv444p", RTGMDIF_CSP_YUV444P },
    };
    for (const auto &entry : table) {
        if (name == entry.name) {
            *csp = entry.value;
            *cspName = name;
            return true;
        }
    }
    return false;
}

// Case-folded comparison, so the option values can be typed in either case.
static std::string lower_ascii(const std::string &s) {
    std::string r = s;
    for (char &c : r) {
        if (c >= 'A' && c <= 'Z') {
            c = (char)(c - 'A' + 'a');
        }
    }
    return r;
}

static bool parse_seldif_side(const std::string &name, int *side) {
    static const struct { const char *name; int value; } table[] = {
        { "top",    RTGMDIF_SPLITSIDE_TOP    },
        { "bottom", RTGMDIF_SPLITSIDE_BOTTOM },
    };
    for (const auto &entry : table) {
        if (lower_ascii(name) == entry.name) {
            *side = entry.value;
            return true;
        }
    }
    return false;
}

// --seldif names the deinterlacer that takes the band the chain does not. The side can be given
// in the same argument as a suffix ("--seldif bwdif:bottom"), which is the same thing as setting
// --seldif-side separately.
static bool parse_seldif(const std::string &value, TestConfig *cfg) {
    std::string name = value;
    const size_t colon = value.find(':');
    if (colon != std::string::npos) {
        name = value.substr(0, colon);
        if (!parse_seldif_side(value.substr(colon + 1), &cfg->splitDifSide)) {
            return false;
        }
    }
    static const struct { const char *name; int value; } table[] = {
        { "yadif", RTGMDIF_SPLITDIF_YADIF },
        { "bwdif", RTGMDIF_SPLITDIF_BWDIF },
    };
    for (const auto &entry : table) {
        if (lower_ascii(name) == entry.name) {
            cfg->splitDif = entry.value;
            return true;
        }
    }
    return false;
}

static bool parse_args(const int argc, char **argv, TestConfig *cfg) {
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&](std::string *out) -> bool {
            if (i + 1 >= argc) {
                return false;
            }
            *out = argv[++i];
            return true;
        };
        auto want_int = [&](int *out) -> bool {
            std::string v;
            return next(&v) && parse_int(v.c_str(), out);
        };
        std::string v;
        if (arg == "-h" || arg == "--help") {
            return false;
        } else if (arg == "-i" || arg == "--input") {
            if (!next(&cfg->inPath)) return false;
        } else if (arg == "-o" || arg == "--output") {
            if (!next(&cfg->outPath)) return false;
        } else if (arg == "--flow") {
            if (!next(&v) || !parse_flow(v, &cfg->flow, &cfg->flowName)) return false;
        } else if (arg == "--width") {
            if (!want_int(&cfg->width)) return false;
        } else if (arg == "--height") {
            if (!want_int(&cfg->height)) return false;
        } else if (arg == "--in-csp") {
            if (!next(&v) || !parse_csp(v, &cfg->csp, &cfg->cspName)) return false;
        } else if (arg == "--deint-pel") {
            if (!want_int(&cfg->deintPel)) return false;
        } else if (arg == "--deint-pel-search") {
            if (!want_int(&cfg->deintPelSearch)) return false;
        } else if (arg == "--deint-search-refine") {
            if (!want_int(&cfg->deintSearchRefine)) return false;
        } else if (arg == "--deint-search-param") {
            if (!want_int(&cfg->deintSearchParam)) return false;
        } else if (arg == "--deint-analyze-delta") {
            if (!want_int(&cfg->deintAnalyzeDelta)) return false;
        } else if (arg == "--deint-overlap") {
            if (!want_int(&cfg->deintOverlap)) return false;
        } else if (arg == "--deint-tr0") {
            if (!want_int(&cfg->deintTr0)) return false;
        } else if (arg == "--deint-tr1-delta") {
            if (!want_int(&cfg->deintTr1Delta)) return false;
        } else if (arg == "--deint-tr2-delta") {
            if (!want_int(&cfg->deintTr2Delta)) return false;
        } else if (arg == "--deint-retouch-smode") {
            if (!want_int(&cfg->deintRetouchSmode)) return false;
        } else if (arg == "--low-latency") {
            cfg->lowLatency = 1;
        } else if (arg == "--spatial-early-sad") {
            if (!want_int(&cfg->spatialEarlySad)) return false;
        } else if (arg == "--clean-preset") {
            if (!want_int(&cfg->cleanPreset)) return false;
        } else if (arg == "--clean-analyze-delta") {
            if (!want_int(&cfg->cleanAnalyzeDelta)) return false;
        } else if (arg == "--clean-tr1-delta") {
            if (!want_int(&cfg->cleanTr1Delta)) return false;
        } else if (arg == "--clean-tr2-delta") {
            if (!want_int(&cfg->cleanTr2Delta)) return false;
        } else if (arg == "--clean-rep2-thin") {
            if (!want_int(&cfg->cleanRep2Thin)) return false;
        } else if (arg == "--split-y") {
            if (!want_int(&cfg->splitY)) return false;
        } else if (arg == "--seldif") {
            if (!next(&v) || !parse_seldif(v, cfg)) return false;
        } else if (arg == "--seldif-side") {
            if (!next(&v) || !parse_seldif_side(v, &cfg->splitDifSide)) return false;
        } else if (arg == "--frames") {
            if (!want_int(&cfg->frames)) return false;
        } else if (arg == "--skip") {
            if (!want_int(&cfg->skip)) return false;
        } else if (arg == "--stats-every") {
            if (!want_int(&cfg->statsEvery)) return false;
        } else if (arg == "--fps") {
            if (!want_int(&cfg->fps)) return false;
        } else if (arg == "--bff") {
            cfg->tff = 0;
        } else if (arg == "--device") {
            if (!want_int(&cfg->device)) return false;
        } else if (arg == "--log-level") {
            if (!want_int(&cfg->logLevel)) return false;
        } else if (arg == "--own-stream") {
            cfg->ownStream = 1;
        } else {
            printf("unknown option: %s\n", arg.c_str());
            return false;
        }
    }
    if (cfg->inPath.empty()) {
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    // The module prints its own progress through the same stdout, and with the two of them
    // buffered differently the lines would come out interleaved out of order when the output is
    // redirected to a file. Unbuffered output keeps the log readable in the order it happened.
    setvbuf(stdout, nullptr, _IONBF, 0);

    TestConfig cfg;
    if (!parse_args(argc, argv, &cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    FILE *fpIn = fopen(cfg.inPath.c_str(), "rb");
    if (fpIn == nullptr) {
        printf("[error] failed to open input file: %s\n", cfg.inPath.c_str());
        return 1;
    }
    // Without -o this is a timing / working-set run: everything below still executes, including
    // the getoutputbuf() that releases each frame back into the module's output pool (the pool
    // is fixed at 32 frames and the module fails loudly rather than dropping, so the releases
    // are not optional), but the frames are left on the device and nothing is written.
    FILE *fpOut = nullptr;
    if (!cfg.outPath.empty()) {
        fpOut = fopen(cfg.outPath.c_str(), "wb");
        if (fpOut == nullptr) {
            printf("[error] failed to open output file: %s\n", cfg.outPath.c_str());
            fclose(fpIn);
            return 1;
        }
    }

    // --- the module ---------------------------------------------------------------------
    RtgmDifConfig difCfg;
    difCfg.flow      = cfg.flow;
    difCfg.width     = cfg.width;
    difCfg.height    = cfg.height;
    difCfg.csp       = cfg.csp;
    difCfg.bitdepth  = 8;
    difCfg.fpsNum    = cfg.fps;
    difCfg.fpsDen    = 1;
    difCfg.tff       = cfg.tff;
    difCfg.deviceId  = cfg.device;
    difCfg.logLevel  = cfg.logLevel;
    difCfg.deintPel          = cfg.deintPel;
    difCfg.deintPelSearch    = cfg.deintPelSearch;
    difCfg.deintSearchRefine = cfg.deintSearchRefine;
    difCfg.deintSearchParam  = cfg.deintSearchParam;
    difCfg.deintAnalyzeDelta = cfg.deintAnalyzeDelta;
    difCfg.deintOverlap      = cfg.deintOverlap;
    difCfg.deintTr0          = cfg.deintTr0;
    difCfg.deintTr1Delta     = cfg.deintTr1Delta;
    difCfg.deintTr2Delta     = cfg.deintTr2Delta;
    difCfg.deintRetouchSmode = cfg.deintRetouchSmode;
    difCfg.lowLatency        = cfg.lowLatency;
    difCfg.spatialEarlySad = cfg.spatialEarlySad;
    difCfg.cleanPreset      = cfg.cleanPreset;
    difCfg.cleanAnalyzeDelta = cfg.cleanAnalyzeDelta;
    difCfg.cleanTr1Delta   = cfg.cleanTr1Delta;
    difCfg.cleanTr2Delta   = cfg.cleanTr2Delta;
    difCfg.cleanRep2Thin   = cfg.cleanRep2Thin;
    difCfg.splitY          = cfg.splitY;
    difCfg.splitDif        = cfg.splitDif;
    difCfg.splitDifSide    = cfg.splitDifSide;

    // One factory call is this module's whole exported surface; everything after it is a call
    // through the interface. The deleter is the module's own, so the object is freed by the CRT
    // that allocated it, and holding it in a unique_ptr makes every early return below leak-free
    // without a single explicit destroy.
    std::unique_ptr<RtgmDif, void (*)(RtgmDif *)> dif(rtgmdif_create(), rtgmdif_destroy);
    if (!dif) {
        printf("[error] rtgmdif_create() failed.\n");
        fclose(fpIn);
        if (fpOut != nullptr) fclose(fpOut);
        return 1;
    }
    if (!dif->configure(difCfg)) {
        printf("[error] RtgmDif::configure() failed.\n");
        fclose(fpIn);
        if (fpOut != nullptr) fclose(fpOut);
        return 1;
    }

    _fseeki64(fpIn, 0, SEEK_END);
    const int64_t fileBytes = _ftelli64(fpIn);
    _fseeki64(fpIn, 0, SEEK_SET);

    // The stream belongs to this program and is handed to the module, which enqueues all of its
    // work on it. Upload, RTGMC and download therefore run back to back on one stream without
    // the module ever blocking or synchronising it - the only synchronise here is this program
    // waiting for its own download before it reads the host buffer.
    // --own-stream leaves `stream` null, which makes the module create (and own) one of its
    // own. The copies below then run on the legacy default stream; that is a legal fallback
    // because the module synchronises the output copies itself in that case.
    // --- device memory probe ------------------------------------------------------------
    //
    // cudaMemGetInfo reports the card's free and total memory, so the number that means anything
    // is the change since before init(): that is this process's own footprint, with the desktop
    // and everything else on the card cancelled out. Printed after init(), every 16 frames, and
    // after the flush, which is what separates a leak (a column that keeps climbing) from a
    // working set (one that levels off). It is a query, not a kernel - no .cu, no CUDA work.
    size_t memTotalBytes = 0;
    size_t memFreeBytes = 0;
    double memBaselineMiB = 0.0;
    if (cudaMemGetInfo(&memFreeBytes, &memTotalBytes) == cudaSuccess) {
        memBaselineMiB = (double)(memTotalBytes - memFreeBytes) / 1048576.0;
    }
    auto probeDeviceMemory = [&](const char *where, int frames) {
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
            return;
        }
        const double usedMiB = (double)(totalBytes - freeBytes) / 1048576.0;
        printf("mem    : %-6s after %5d in  %7.1f MiB on the card  %+8.1f MiB since start\n",
            where, frames, usedMiB, usedMiB - memBaselineMiB);
    };

    cudaStream_t stream = nullptr;
    if (!cfg.ownStream) {
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            printf("[error] cudaStreamCreate failed.\n");
            fclose(fpIn);
            if (fpOut != nullptr) fclose(fpOut);
            return 1;
        }
    }

    if (!dif->init(stream)) {
        printf("[error] RtgmDif::init() failed.\n");
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        fclose(fpIn);
        if (fpOut != nullptr) fclose(fpOut);
        return 1;
    }

    const int inBytes  = dif->inputFrameBytes();
    const int outBytes = dif->outputFrameBytes();
    printf("test   : flow %s, %s %dx%d, input frame %d bytes, output frame %d bytes\n",
        cfg.flowName.c_str(), cfg.cspName.c_str(), cfg.width, cfg.height, inBytes, outBytes);
    printf("test   : input file %lld bytes (%lld frame(s) at that size)\n",
        (long long)fileBytes, (long long)(fileBytes / (inBytes > 0 ? inBytes : 1)));
    printf("test   : output %s\n",
        fpOut != nullptr ? cfg.outPath.c_str() : "<none, nothing is written>");
    probeDeviceMemory("init", 0);

    // --- host and device frame buffers --------------------------------------------------
    //
    // Three device frames and no more, allocated once and reused for the whole run: one for the
    // input the module is handed, two for the output it hands back. One input frame produces two
    // output frames, so the pair of output buffers covers the steady state exactly - both frames
    // of a batch are downloaded back to back and the program waits for the pair once, instead of
    // waiting after each frame and draining the pipeline in between. The flush releases the
    // whole temporal tail at once; that longer batch is written in the same groups of two, over
    // the same two buffers.
    static const int OUT_BUFS = 2;
    std::vector<uint8_t> hostIn((size_t)inBytes);
    std::vector<uint8_t> hostOut[OUT_BUFS];
    void *devIn  = nullptr;
    void *devOut[OUT_BUFS] = { nullptr, nullptr };
    for (int i = 0; i < OUT_BUFS; i++) {
        hostOut[i].resize((size_t)outBytes);
    }
    bool allocated = (cudaMalloc(&devIn, inBytes) == cudaSuccess);
    for (int i = 0; i < OUT_BUFS && allocated; i++) {
        allocated = (cudaMalloc(&devOut[i], outBytes) == cudaSuccess);
    }
    if (!allocated) {
        printf("[error] cudaMalloc failed.\n");
        if (devIn != nullptr) {
            cudaFree(devIn);
        }
        for (int i = 0; i < OUT_BUFS; i++) {
            if (devOut[i] != nullptr) {
                cudaFree(devOut[i]);
            }
        }
        dif->close();
        fclose(fpIn);
        if (fpOut != nullptr) fclose(fpOut);
        return 1;
    }

    int inRead = 0;
    int outWritten = 0;   // frames the module released (the ledger's "out")
    int outStored = 0;    // frames actually written to the output file
    bool aborted = false;

    // --- per-frame timing ---------------------------------------------------------------
    //
    // Wall clock rather than cudaEventElapsedTime: what is being measured is how long one frame
    // holds the caller, and that is the wall time between the two synchronises below. The one
    // before t0 makes sure the previous frame's work is finished - without it the measurement
    // would only be how long handdif() took to enqueue - and the one after it waits for this
    // frame's own work. That serialises the pipeline, which is the point: this is a measurement
    // run, not the throughput path.
    //
    // The first frames are dominated by kernel JIT (the filters ship as PTX and are compiled on
    // first use), so they are printed but left out of the summary. Read the summary numbers.
    static const int TIMING_WARMUP_FRAMES = 8;
    // With --own-stream the module owns its stream and this program has no handle to order
    // against, so the wait has to be device-wide; otherwise it is the stream the module runs on.
    auto cudasync = [&]() {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        } else {
            cudaDeviceSynchronize();
        }
    };
    auto gettime = []() {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
    };
    double t0 = 0.0;
    double t1 = 0.0;
    double timingTotal = 0.0;
    double timingMin = 0.0;
    double timingMax = 0.0;
    int    timingCount = 0;

    // Writes out whatever the last handdif() released.
    //
    // Download a group of at most two frames, wait once for the group, then write the group. In
    // the steady state that is one group of two per input frame, so this program synchronises
    // once per input frame rather than once per output frame - each synchronise empties the
    // pipeline, and the filters are the only thing running on it.
    // Set by the loop below before each call: true while the frames being released are ones the
    // run is skipping (--skip) - fed through the chain to warm it up, but not kept.
    bool discardFrames = false;
    auto writeBatch = [&]() -> bool {
        const bool writing = !discardFrames && fpOut != nullptr;
        const int num = dif->getoutputsize();
        for (int k = 0; k < num; k += OUT_BUFS) {
            const int n = std::min(OUT_BUFS, num - k);
            for (int j = 0; j < n; j++) {
                if (!dif->getoutputbuf(k + j, devOut[j])) {
                    printf("[error] getoutputbuf(%d) failed.\n", k + j);
                    return false;
                }
                if (!writing) {
                    continue;   // nothing is written: the release above is all this run needs
                }
                // getoutputbuf() queued its copy on `stream`, so this download lands behind it
                // without any synchronisation in between - and behind the other frame's
                // download as well, since both are queued before the wait below.
                if (cudaMemcpyAsync(hostOut[j].data(), devOut[j], (size_t)outBytes,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
                    printf("[error] download failed at output frame %d.\n", outWritten + j);
                    return false;
                }
            }
            if (!writing) {
                // Those device copies are queued and unwaited, which is deliberate: the wait
                // this program does before starting the clock covers them, so the timed span
                // stays free of them exactly as it does when a file is being written.
                outWritten += n;
                continue;
            }
            // The only wait in this program: fwrite() is about to read the host buffers, and
            // the next fread() is about to overwrite hostIn.
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                printf("[error] stream synchronise failed at output frame %d.\n", outWritten);
                return false;
            }
            for (int j = 0; j < n; j++) {
                if (fwrite(hostOut[j].data(), 1, (size_t)outBytes, fpOut) != (size_t)outBytes) {
                    printf("[error] failed to write output frame %d.\n", outStored + j);
                    return false;
                }
            }
            outWritten += n;
            outStored += n;
        }
        return true;
    };

    // --- read / upload / run / download / write ----------------------------------------
    for (int i = 0; cfg.frames <= 0 || i < cfg.frames; i++) {
        if (fread(hostIn.data(), 1, (size_t)inBytes, fpIn) != (size_t)inBytes) {
            break;
        }
        inRead++;

        // Queued on the same stream the module runs on, so handdif() reads a buffer that is
        // already uploaded - no synchronisation needed between the two.
        if (cudaMemcpyAsync(devIn, hostIn.data(), inBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            printf("[error] upload failed at input frame %d.\n", i);
            aborted = true;
            break;
        }

        cudasync();
        t0 = gettime();
        if (!dif->handdif(devIn)) {
            printf("[error] handdif() failed at input frame %d.\n", i);
            aborted = true;
            break;
        }
        cudasync();
        t1 = gettime();
        const double frameMs = (t1 - t0) * 1000.0;
        // --skip: the frames before the skip are fed through the chain so that its temporal
        // history is full when the timed part starts, but their outputs are released unwritten
        // and they stay out of the summary. With no skip this is exactly the old behaviour.
        const bool timedFrame = (i >= cfg.skip + TIMING_WARMUP_FRAMES);
        printf("handdif: frame %5d  %9.3f ms%s\n", i, frameMs,
            timedFrame ? "" : "   (warm-up, excluded from the summary)");
        if (timedFrame) {
            if (timingCount == 0 || frameMs < timingMin) {
                timingMin = frameMs;
            }
            if (timingCount == 0 || frameMs > timingMax) {
                timingMax = frameMs;
            }
            timingTotal += frameMs;
            timingCount++;
        }

        discardFrames = (i < cfg.skip);
        if (!writeBatch()) {
            aborted = true;
            break;
        }
        // The ledger is what tells the pipeline's delay apart from its throughput: after every n
        // input frames, how much has come out. --stats-every 1 gives it frame by frame. The
        // memory probe keeps its own slower cadence - it is the expensive part of this print.
        if (cfg.statsEvery > 0 && (i % cfg.statsEvery) == 0) {
            printf("in  %5d  ->  out %5d\n", inRead, outWritten);
            fflush(stdout);
        }
        if ((i % 16) == 0) {
            probeDeviceMemory("frame", inRead);
        }
    }

    // --- flush --------------------------------------------------------------------------
    // The temporal filter still holds the tail of the stream; handdif(nullptr) runs it out.
    if (!aborted) {
        printf("\rflushing ...%*s", 24, "");
        fflush(stdout);
        discardFrames = false;   // the flush releases frames the run means to keep
        if (!dif->handdif(nullptr)) {
            printf("\n[error] handdif(nullptr) (flush) failed.\n");
            aborted = true;
        } else if (!writeBatch()) {
            aborted = true;
        } else {
            printf("\rflushed     ->  out %5d%*s", outWritten, 24, "");
            printf("\n");
            probeDeviceMemory("flush", inRead);
        }
    }

    printf("\n");
    if (fpOut != nullptr) {
        printf("test   : %d input frame(s) read, %d output frame(s) released, %d written\n",
            inRead, outWritten, outStored);
    } else {
        printf("test   : %d input frame(s) read, %d output frame(s) released (no output file)\n",
            inRead, outWritten);
    }
    // --skip discards the first frames of the run and --frames counts them, so --skip >= --frames
    // means every frame was fed through the chain and none was kept: the file then holds only what
    // the flush releases from the pipeline's tail, which looks like a truncated file.
    if (fpOut != nullptr && cfg.skip > 0 && cfg.frames > 0 && cfg.skip >= cfg.frames) {
        printf("[warn] nothing was kept: all %d frame(s) of this run are inside --skip %d, so the "
               "file holds only the %d frame(s) the flush released. --frames counts the skipped "
               "frames, so it has to be larger than --skip (--skip 400 --frames 430 keeps 30).\n",
            cfg.frames, cfg.skip, outStored);
    }

    // The flush is not timed: it releases the whole temporal tail in one go, so its "per frame"
    // cost says nothing about the steady state.
    if (timingCount > 0) {
        const double avgMs = timingTotal / timingCount;
        printf("handdif: %d frame(s) timed (first %d excluded, kernel JIT): avg %.3f ms, min %.3f ms, max %.3f ms\n",
            timingCount, TIMING_WARMUP_FRAMES, avgMs, timingMin, timingMax);
        printf("handdif: %.2f input frame(s)/s, %.2f output frame(s)/s\n",
            1000.0 / avgMs, 2000.0 / avgMs);
    }
    if (inRead > 0 && outWritten != inRead * 2) {
        printf("[warn] expected %d output frames from %d interlaced input frames.\n",
            inRead * 2, inRead);
    }

    cudaFree(devIn);
    for (int i = 0; i < OUT_BUFS; i++) {
        cudaFree(devOut[i]);
    }
    // close() synchronises the stream (the filters must not be torn down while their work is
    // still queued) but it does not destroy a stream that was handed to it.
    dif->close();
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    fclose(fpIn);
    if (fpOut != nullptr) fclose(fpOut);

    if (aborted) {
        printf("[error] aborted due to an error above.\n");
        return 1;
    }
    return 0;
}
