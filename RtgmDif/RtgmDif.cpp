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
// RtgmDif implementation. See RtgmDif.h for the interface.
//
// The chain itself is the same code the integrated filter drives, kept in the same shape on
// purpose so that the two can be compared frame by frame:
//
//   stage 0 : NVEncFilterRtgmc, the integrated filter. It builds its own 17 stage pipeline
//             (bob -> search prefilter -> analyze -> EDI -> TR1 -> rep1 -> retouch -> TR2 ->
//             rep2 -> ...) and doubles the frame rate, 50i -> 50p.
//
//   stage 1 : the clean chain, i.e. QTGMC's single-rate second pass
//             (QTGMC(clip, InputType=1, Sharpness=0)). It is not one filter but six:
//
//                 analyze -> TR1 -> rep1 -> retouch -> TR2 -> rep2
//
//             and they are driven by hand, because the single-rate pass needs two things the
//             generic pipeline cannot give it:
//
//               * TR1/TR2 must be told the MV/SAD result of the analyze stage, which the
//                 integrated filter does through setDirectAnalyzeResultSet() before calling
//                 them (NVEncFilterRtgmc.cu, runNestedFilter).
//               * The shimmer repair stages need a reference frame. The generic overload
//                 passes the input as its own reference, which makes the kernel an exact
//                 no-op (input == reference gives diff == rangeHalf). At single rate the
//                 reference is the unfiltered clean input - which is what QTGMC uses too,
//                 since with InputType=1 its `edi` clip *is* the input.
//
//             The sharpLimit1 envelope follows the same rule: QTGMC builds it from `edi`
//             warped onto the current time from both sides, so two compensate passes over the
//             clean input produce those two frames.
//
// Everything that only existed to serve the command line - argument parsing, the file reader
// and writer, the benchmark clock, the timing report - is gone; what is left here is the
// filters and the frame plumbing.
//
// ------------------------------------------------------------------------------------------

#include "RtgmDif.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "rgy_log.h"
#include "rgy_frame.h"
#include "rgy_prm.h"
#include "rgy_util.h"
#include "rgy_cuda_util.h"
#include "NVEncFilterRtgmc.h"
#include "NVEncFilterDegrain.h"
#include "NVEncFilterRtgmcRetouch.h"
#include "NVEncFilterRtgmcShimmerRepair.h"
#include "NVEncFilterBwdif.h"
#include "NVEncFilterYadif.h"

// ---------------------------------------------------------------------------------------
// constants / small helpers
// ---------------------------------------------------------------------------------------

// NVEncFilterRtgmc::AllocFrameBuf() reserves RGY_RTGMC_MAX_OUT_FRAMES (= 32) output slots, and
// the clean chain's stages are built on NVEncFilter with the same limit.
static const int RTGMDIF_MAX_OUT_FRAMES = 32;
// Round-robin pool for the input frames. Sized generously so that a frame which is still
// referenced by the filters' internal caches is never overwritten in place.
static const int RTGMDIF_IN_BUF_RING = 8;
static const int RTGMDIF_PLANES = 3;
// Rows of margin on each side of a split band. The boundary row is then an interior row of both
// filters instead of an edge row, so what the filters do not have is vertical context they can
// see, not vertical context that exists. Must be even so that the bands stay chroma aligned.
static const int RTGMDIF_SPLIT_MARGIN = 16;

struct PlaneGeom {
    int width;
    int height;
};

// Plane geometry of the raw layout the caller uses: tightly packed, so the pitch of a plane is
// its row width. YUV422P: U/V are half width, full height. YUV420P: U/V are half width and
// half height.
static void get_plane_geom(const RGY_CSP csp, const int width, const int height,
    PlaneGeom geom[RTGMDIF_PLANES]) {
    geom[0].width  = width;
    geom[0].height = height;
    for (int i = 1; i < RTGMDIF_PLANES; i++) {
        geom[i].width  = width;
        geom[i].height = height;
    }
    switch (RGY_CSP_CHROMA_FORMAT[csp]) {
    case RGY_CHROMAFMT_YUV422:
        for (int i = 1; i < RTGMDIF_PLANES; i++) {
            geom[i].width  = width / 2;
            geom[i].height = height;
        }
        break;
    case RGY_CHROMAFMT_YUV420:
        for (int i = 1; i < RTGMDIF_PLANES; i++) {
            geom[i].width  = width / 2;
            geom[i].height = height / 2;
        }
        break;
    default:
        break;
    }
}

static int64_t plane_total_bytes(const PlaneGeom geom[RTGMDIF_PLANES], const int bytesPerPix) {
    int64_t total = 0;
    for (int i = 0; i < RTGMDIF_PLANES; i++) {
        total += (int64_t)geom[i].width * geom[i].height * bytesPerPix;
    }
    return total;
}

// One TR stage of the clean chain.
//
// The base is always the matching stage of the preset-applied VppRtgmc, so the clean stage
// reuses exactly the motion search settings the deinterlacer used: blksize / overlap / pel /
// search / searchParam / pelSearch / searchRefine / searchEarlySad / lambda / tr0 / rep0 are
// all propagated into analyze/tr1/tr2 by apply_vpp_rtgmc_preset() (rgy_prm.cpp:2784).
static VppDegrain make_degrain_stage(const VppDegrain &base, const VppDegrainMode mode,
    const VppDegrainStage stage, const int delta) {
    VppDegrain d = base;
    d.enable = true;
    d.mode   = mode;
    d.stage  = stage;
    d.delta  = delta;
    return d;
}

// ---------------------------------------------------------------------------------------
// clean stage : the single-rate second pass (QTGMC InputType=1)
// ---------------------------------------------------------------------------------------
//
//     analyze -> TR1 -> rep1 -> retouch -> TR2 -> rep2
//
// The order is QTGMC's, and is the same one the integrated filter uses internally
// (NVEncFilterRtgmc.cu, enum RtgmcFilterIndex) and the same one the upstream component
// pipeline builds (NVEncCore.cpp, InitFiltersCreateVppList).
//
// A stage whose parameter is switched off (tr2.delta == 0, repair-thin == 0) degrades to a
// plain pass-through, so the shape of the chain never changes.

enum CleanStageIndex : int {
    CLEAN_STAGE_ANALYZE = 0,
    CLEAN_STAGE_TR1,
    CLEAN_STAGE_REP1,
    CLEAN_STAGE_RETOUCH,
    CLEAN_STAGE_TR2,
    CLEAN_STAGE_REP2,
};

static const int CLEAN_STAGE_NUM = 6;
static const int CLEAN_MAX_OUT    = 32;   // == RGY_RTGMC_MAX_OUT_FRAMES
// Device frames kept for the rep reference ring. This is the measured live-reference peak with
// headroom: the ring reports its own peak at the end of a run, and the three flows need 0 (no
// clean stage), 10 and 16 frames respectively, so 24 covers the widest of them by half again.
// It used to be 32, which was a guess.
static const int CLEAN_REF_POOL   = 24;
// Device frames per motion-compensated neighbour ring (the sharpLimit1 envelope). The envelope
// only ever reaches back to the TR1 output of the neighbouring frames - slmode=2 with slrad=1 in
// the presets, so a couple of frames each way - and the ring warns and drops the oldest entry if
// it is ever more than this, which is what the run log and the output hash would show.
static const int CLEAN_COMP_POOL  = 6;

struct CleanRefEntry {
    int     poolIndex;
    int     inputFrameId;
    int64_t timestamp;

    CleanRefEntry() : poolIndex(0), inputFrameId(-1), timestamp(0) {}
    CleanRefEntry(const int pool, const int id, const int64_t ts)
        : poolIndex(pool), inputFrameId(id), timestamp(ts) {}
};

// A FIFO ring of device frames keyed by timestamp. Everything the clean stage needs to look up
// "the frame with this timestamp" later goes through one of these: the shimmer repair reference
// (= QTGMC's edi) and the two motion-compensated neighbours of QTGMC's sharpLimit1 envelope.
struct CleanFrameRing {
    std::vector<std::unique_ptr<CUFrameBuf>> pool;
    std::deque<CleanRefEntry>                entries;
    bool                                     overflowWarned;
    int                                      peak;

    CleanFrameRing() : pool(), entries(), overflowWarned(false), peak(0) {}

    RGY_ERR alloc(const int count, const RGYFrameInfo &frameInfo) {
        for (int i = 0; i < count; i++) {
            // From the whole RGYFrameInfo, not just width/height/csp: the (w,h,csp) constructor
            // leaves bitdepth at its default, and NVEncFilterRtgmcRetouch::isFrameCompatible()
            // compares bitdepth as well - a ring frame with bitdepth 0 is rejected.
            auto buf = std::make_unique<CUFrameBuf>(frameInfo);
            const auto sts = buf->alloc();
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            pool.push_back(std::move(buf));
        }
        return RGY_ERR_NONE;
    }

    RGY_ERR push(const RGYFrameInfo *src, cudaStream_t stream) {
        if (pool.empty() || src == nullptr) {
            return RGY_ERR_NONE;
        }
        while (entries.size() >= pool.size()) {
            if (!overflowWarned) {
                printf("\n[warn] a clean frame ring is full (%d frames); the oldest entry is "
                       "being dropped early.\n", (int)pool.size());
                overflowWarned = true;
            }
            entries.pop_front();
        }
        // Entries are pushed and dropped in the same order over a fixed ring of buffers, so
        // the slot after the newest live entry is always free.
        const int index = entries.empty() ? 0 : (entries.back().poolIndex + 1) % (int)pool.size();
        const auto sts = pool[index]->copyFrameAsync(src, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        entries.push_back(CleanRefEntry(index, src->inputFrameId, src->timestamp));
        peak = std::max(peak, (int)entries.size());
        return RGY_ERR_NONE;
    }

    void dropUpTo(const int64_t timestamp) {
        while (!entries.empty() && entries.front().timestamp <= timestamp) {
            entries.pop_front();
        }
    }

    const RGYFrameInfo *find(const int64_t timestamp) const {
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
            if (it->timestamp == timestamp) {
                return &pool[it->poolIndex]->frame;
            }
        }
        return nullptr;
    }

    void clear() {
        entries.clear();
        pool.clear();
    }
};

struct CleanStage {
    std::string                   name;      // for the log
    std::unique_ptr<NVEncFilter>  filter;
    // Shimmer repair is the only stage that takes an explicit reference frame, and the only
    // one whose multi-argument overload must be called instead of NVEncFilter::filter().
    NVEncFilterRtgmcShimmerRepair *repair;
    bool                          enabled;   // false == pass-through

    CleanStage() : name(), filter(), repair(nullptr), enabled(false) {}
};

class CleanChain {
public:
    CleanChain() :
        m_stages(),
        m_sink(),
        m_refRing(),
        m_backRing(),
        m_forwRing(),
        m_analyze(nullptr),
        m_compBack(),
        m_compForw(),
        m_stream(nullptr),
        m_tr1Delta(0),
        m_tr2Delta(0),
        m_needRef(false),
        m_limitWanted(false),
        m_limitMissing(0),
        m_missingRefWarned(false),
        m_guardMax(1 << 20) {
    }

    // `rtgmc` must already have the preset and all clean-stage overrides applied.
    RGY_ERR init(const VppRtgmc &rtgmc, const RGYFrameInfo &frameIn, const rgy_rational<int> &baseFps,
        const std::shared_ptr<RGYLog> &log, const cudaStream_t stream);

    // Sink for finished output frames. Set before pushing anything.
    void setSink(std::function<bool(RGYFrameInfo *)> sink) { m_sink = std::move(sink); }

    // Hands one frame to the chain, first retaining it as a rep reference.
    bool push(RGYFrameInfo *f) {
        if (f != nullptr) {
            const auto sts = m_refRing.push(f, m_stream);
            if (sts != RGY_ERR_NONE) {
                printf("\n[error] failed to retain clean reference frame: %s\n",
                    tchar_to_string(get_err_mes(sts)).c_str());
                return false;
            }
        }
        return runStage(CLEAN_STAGE_ANALYZE, f) >= 0;
    }

    // Flushes the chain front to back: stage 0 first, so whatever it still holds reaches
    // stage 1 before stage 1 is flushed in turn.
    bool drain() {
        for (int idx = 0; idx < CLEAN_STAGE_NUM; idx++) {
            int guard = 0;
            for (;;) {
                const int n = runStage(idx, nullptr);
                if (n < 0) {
                    return false;
                }
                if (n == 0) {
                    break;
                }
                if (++guard > m_guardMax) {
                    printf("\n[error] clean stage %d drain did not finish within %d iterations.\n",
                        idx, m_guardMax);
                    return false;
                }
            }
            // The compensate passes run alongside the chain rather than in it, so they are
            // flushed right after the stage that feeds them.
            if (idx == CLEAN_STAGE_ANALYZE && !drainCompensate()) {
                return false;
            }
        }
        return true;
    }

    int refPeak() const { return m_refRing.peak; }
    int limitMissing() const { return m_limitMissing; }
    bool limitWanted() const { return m_limitWanted; }

    // Releases every stage and every ring. Has to run before the CUDA stream is destroyed,
    // because the stages and the rings own device memory and events.
    void clear() {
        for (auto &stage : m_stages) {
            stage.repair = nullptr;
            stage.filter.reset();
            stage.enabled = false;
        }
        m_analyze = nullptr;
        m_compBack.reset();
        m_compForw.reset();
        m_refRing.clear();
        m_backRing.clear();
        m_forwRing.clear();
        m_sink = nullptr;
    }

private:
    // Feeds both compensate passes with the frame the analyze stage just released. They need
    // the same frames as the chain because the motion field travels with them: the analyze
    // stage attaches RGYFrameDataDegrain to every frame it passes through, and that is what
    // the compensate path binds (NVEncFilterDegrain.cpp).
    bool feedCompensate(const RGYFrameInfo *f) {
        if (!m_limitWanted || f == nullptr) {
            return true;
        }
        struct Helper {
            NVEncFilterDegrain *filter;
            CleanFrameRing    *ring;
        };
        const Helper helpers[2] = { { m_compBack.get(), &m_backRing }, { m_compForw.get(), &m_forwRing } };
        for (const auto &helper : helpers) {
            if (helper.filter == nullptr) {
                continue;
            }
            RGYFrameInfo *out[CLEAN_MAX_OUT] = { nullptr };
            int outNum = 0;
            const auto sts = helper.filter->filter(const_cast<RGYFrameInfo *>(f), out, &outNum, m_stream);
            if (sts != RGY_ERR_NONE) {
                printf("\n[error] clean compensate pass failed: %s\n",
                    tchar_to_string(get_err_mes(sts)).c_str());
                return false;
            }
            for (int j = 0; j < outNum; j++) {
                const auto err = helper.ring->push(out[j], m_stream);
                if (err != RGY_ERR_NONE) {
                    printf("\n[error] failed to retain clean limit frame: %s\n",
                        tchar_to_string(get_err_mes(err)).c_str());
                    return false;
                }
            }
        }
        return true;
    }

    // Flushes the two compensate passes so the envelopes for the tail frames exist before the
    // retouch is drained. Their own delay is one frame.
    bool drainCompensate() {
        if (!m_limitWanted) {
            return true;
        }
        struct Helper {
            NVEncFilterDegrain *filter;
            CleanFrameRing    *ring;
        };
        const Helper helpers[2] = { { m_compBack.get(), &m_backRing }, { m_compForw.get(), &m_forwRing } };
        for (const auto &helper : helpers) {
            if (helper.filter == nullptr) {
                continue;
            }
            for (int guard = 0; guard < 8; guard++) {
                RGYFrameInfo *out[CLEAN_MAX_OUT] = { nullptr };
                int outNum = 0;
                const auto sts = helper.filter->filter(nullptr, out, &outNum, m_stream);
                if (sts != RGY_ERR_NONE) {
                    printf("\n[error] clean compensate drain failed: %s\n",
                        tchar_to_string(get_err_mes(sts)).c_str());
                    return false;
                }
                for (int j = 0; j < outNum; j++) {
                    if (helper.ring->push(out[j], m_stream) != RGY_ERR_NONE) {
                        return false;
                    }
                }
                if (outNum <= 0) {
                    break;
                }
            }
        }
        return true;
    }

    // QTGMC's sharpLimit1: `backBlend1.mt_clamp(tMax, tMin, Sovs, Sovs)` with
    // tMax/tMin = max/min over { edi, edi warped from T-1, edi warped from T+1 }. The kernel
    // takes those three frames directly (NVEncFilterRtgmcRetouch.cu,
    // rtgmc_retouch_temporal_detail_guard_value), so hand them over before running the stage.
    int applyLimitFrames(const RGYFrameInfo *f) {
        auto *stage = &m_stages[CLEAN_STAGE_RETOUCH];
        auto *retouch = dynamic_cast<NVEncFilterRtgmcRetouch *>(stage->filter.get());
        if (retouch == nullptr) {
            return 0;
        }
        if (!m_limitWanted || f == nullptr) {
            return 0;
        }
        RGYRtgmcRetouchTemporalLimitFrames limit;
        limit.ref        = m_refRing.find(f->timestamp);
        limit.motionBack = m_backRing.find(f->timestamp);
        limit.motionForw = m_forwRing.find(f->timestamp);
        if (!limit.valid()) {
            // Without the envelope the kernel silently falls back to a spatial clamp against
            // the frame itself, which is a no-op. Say so instead of pretending it ran.
            retouch->clearTemporalLimitFrames();
            m_limitMissing++;
            return 0;
        }
        retouch->setTemporalLimitFrames(limit);
        return 1;
    }

    // Runs stage `idx` on `f` (nullptr == drain tick) and pushes everything it releases into
    // the next stage. Returns the number of frames this stage itself emitted, -1 on error.
    int runStage(const int idx, RGYFrameInfo *f);

    std::array<CleanStage, CLEAN_STAGE_NUM> m_stages;
    std::function<bool(RGYFrameInfo *)>     m_sink;
    CleanFrameRing                          m_refRing;
    CleanFrameRing                          m_backRing;
    CleanFrameRing                          m_forwRing;
    NVEncFilterDegrain                     *m_analyze;
    std::unique_ptr<NVEncFilterDegrain>     m_compBack;
    std::unique_ptr<NVEncFilterDegrain>     m_compForw;
    cudaStream_t                            m_stream;
    int                                     m_tr1Delta;
    int                                     m_tr2Delta;
    bool                                    m_needRef;
    bool                                    m_limitWanted;
    int                                     m_limitMissing;
    bool                                    m_missingRefWarned;
    int                                     m_guardMax;
};

RGY_ERR CleanChain::init(const VppRtgmc &rtgmc, const RGYFrameInfo &frameIn,
    const rgy_rational<int> &baseFps, const std::shared_ptr<RGYLog> &log, const cudaStream_t stream) {
    m_stream = stream;

    RGYFrameInfo frame = frameIn;
    frame.picstruct = RGY_PICSTRUCT_FRAME;

    auto makeParam = [&](std::shared_ptr<NVEncFilterParam> param) {
        param->frameIn       = frame;
        param->frameOut      = frame;
        param->baseFps       = baseFps;
        param->bOutOverwrite = false;
        return param;
    };

    // The analyze stage produces the MV/SAD data every TR stage consumes, so its delta has
    // to cover the largest delta used anywhere in the chain - the same rule the integrated
    // filter applies (NVEncFilterRtgmc.cu, rtgmcNestedAnalyzeDelta).
    const int nestedDelta = std::max({ 1,
        std::min(rtgmc.analyze.delta, RGY_DEGRAIN_MAX_DELTA),
        std::min(rtgmc.tr1.delta,     RGY_DEGRAIN_MAX_DELTA),
        std::min(rtgmc.tr2.delta,     RGY_DEGRAIN_MAX_DELTA) });
    if (rtgmc.analyze.delta != nestedDelta) {
        printf("[note] clean analyze delta=%d is raised to %d to cover TR1/TR2.\n",
            rtgmc.analyze.delta, nestedDelta);
    }

    // --- 0: analyze ---------------------------------------------------------------------
    {
        auto filter = std::make_unique<NVEncFilterDegrain>();
        auto param  = std::make_shared<NVEncFilterParamDegrain>();
        param->degrain = make_degrain_stage(rtgmc.analyze, VppDegrainMode::Analyze,
            VppDegrainStage::TR1, nestedDelta);
        param->attachAnalysisData = true;
        param->zeroCopyCache      = true;
        auto sts = filter->init(makeParam(param), log);
        if (sts != RGY_ERR_NONE) {
            printf("[error] clean analyze init failed: %s\n", tchar_to_string(get_err_mes(sts)).c_str());
            return sts;
        }
        m_analyze = filter.get();
        m_stages[CLEAN_STAGE_ANALYZE].name    = "analyze";
        m_stages[CLEAN_STAGE_ANALYZE].filter  = std::move(filter);
        m_stages[CLEAN_STAGE_ANALYZE].enabled = true;
        printf("clean0 : %s\n", tchar_to_string(param->print()).c_str());
    }

    // --- 1: TR1 -------------------------------------------------------------------------
    // A stage that is off is not built at all. runStage() returns early for a disabled stage and
    // never calls its filter, so building one could only ever cost memory: an NVEncFilterDegrain
    // allocates a DEGRAIN_CACHE_SIZE (16) frame input cache plus a 16-plane analysis luma cache in
    // init(), and an NVEncFilterRtgmcShimmerRepair allocates eight output slots. The fast preset
    // has tr2 off, so that one degrain instance was 16 device frames and 16 luma planes that no
    // run would ever touch; rep1 is off in all three flows and had the same problem.
    //
    // VppDegrain::delta of 0 means "this stage is off" for the preset tables, but
    // NVEncFilterDegrain::checkParam() rejects 0 outright - it wants 1..5 - so a legal delta is
    // passed below whatever the table says. The parameter object is still filled in and printed
    // either way, which is what keeps the log and the batch relationship identical.
    m_tr1Delta = std::max(0, rtgmc.tr1.delta);
    {
        auto param  = std::make_shared<NVEncFilterParamDegrain>();
        param->degrain = make_degrain_stage(rtgmc.tr1, VppDegrainMode::Degrain,
            VppDegrainStage::TR1, std::max(1, m_tr1Delta));
        param->zeroCopyCache = true;
        makeParam(param);
        auto &stage = m_stages[CLEAN_STAGE_TR1];
        stage.name    = "tr1";
        stage.enabled = (m_tr1Delta > 0);
        if (stage.enabled) {
            auto filter = std::make_unique<NVEncFilterDegrain>();
            auto sts = filter->init(param, log);
            if (sts != RGY_ERR_NONE) {
                printf("[error] clean TR1 init failed: %s\n", tchar_to_string(get_err_mes(sts)).c_str());
                return sts;
            }
            stage.filter = std::move(filter);
        }
        printf("clean1 : %s%s\n", tchar_to_string(param->print()).c_str(),
            stage.enabled ? "" : "   [off in this preset -> pass-through]");
    }

    // --- 2: rep1 ------------------------------------------------------------------------
    {
        const int thin = std::max(0, std::min(rtgmc.rep1.repThin, 7));
        auto param  = std::make_shared<NVEncFilterParamRtgmcShimmerRepair>();
        param->stage        = RGYRtgmcShimmerRepairStage::PreRetouch;
        param->repairThin   = thin;
        param->repairPad    = rtgmc.rep1.repPad;
        param->processChroma = rtgmc.rep1.repChroma;
        makeParam(param);
        auto &stage = m_stages[CLEAN_STAGE_REP1];
        stage.name    = "rep1";
        stage.enabled = (thin > 0);
        if (stage.enabled) {
            auto filter = std::make_unique<NVEncFilterRtgmcShimmerRepair>();
            auto sts = filter->init(param, log);
            if (sts != RGY_ERR_NONE) {
                printf("[error] clean rep1 init failed: %s\n", tchar_to_string(get_err_mes(sts)).c_str());
                return sts;
            }
            stage.repair  = filter.get();
            stage.filter  = std::move(filter);
        }
        m_needRef = m_needRef || stage.enabled;
        printf("clean2 : %s%s\n", tchar_to_string(param->print()).c_str(),
            stage.enabled ? "" : "   [disabled, repair-thin=0 -> pass-through]");
    }

    // --- 3: retouch ---------------------------------------------------------------------
    {
        auto filter = std::make_unique<NVEncFilterRtgmcRetouch>();
        auto param  = std::make_shared<NVEncFilterParamRtgmcRetouch>();
        param->rtgmc_retouch = rtgmc.retouch;
        param->rtgmc_retouch.tr1 = m_tr1Delta;
        param->rtgmc_retouch.tr2 = std::max(0, rtgmc.tr2.delta);
        // Same as inside the integrated filter: post-TR2 limit modes are handled by a
        // separate stage there, so this instance must not try to do them itself.
        param->skipPostTR2LimitModes = true;
        auto sts = filter->init(makeParam(param), log);
        if (sts != RGY_ERR_NONE) {
            printf("[error] clean retouch init failed: %s\n", tchar_to_string(get_err_mes(sts)).c_str());
            return sts;
        }
        auto &stage = m_stages[CLEAN_STAGE_RETOUCH];
        stage.name    = "retouch";
        stage.filter  = std::move(filter);
        stage.enabled = true;
        printf("clean3 : %s\n", tchar_to_string(param->print()).c_str());
        // QTGMC with Sharpness=0 sets SMode=0, which turns the resharpen and the back blend into
        // pass-throughs but leaves the limit stage (SLMode) running. Keeping those two apart is
        // what NVEncFilterRtgmcRetouch::checkParam() got wrong; the temporal clamp is armed
        // separately below, from the two compensate passes.
        const auto &rt = param->rtgmc_retouch;
        if (rt.smode == 0 || rt.sharpness <= 0.0f) {
            printf("[note] clean retouch: SMode=0 (QTGMC's Sharpness=0), so resharpen and back\n"
                   "       blend are pass-through; the SLMode clamp is unaffected and still runs.\n");
        }
    }

    // --- 4: TR2 -------------------------------------------------------------------------
    // See the note on TR1 above, both for delta 0 meaning "off" and for why an off stage is not
    // built at all.
    m_tr2Delta = std::max(0, rtgmc.tr2.delta);
    {
        auto param  = std::make_shared<NVEncFilterParamDegrain>();
        param->degrain = make_degrain_stage(rtgmc.tr2, VppDegrainMode::Degrain,
            VppDegrainStage::TR2, std::max(1, m_tr2Delta));
        param->zeroCopyCache = true;
        makeParam(param);
        auto &stage = m_stages[CLEAN_STAGE_TR2];
        stage.name    = "tr2";
        stage.enabled = (m_tr2Delta > 0);
        if (stage.enabled) {
            auto filter = std::make_unique<NVEncFilterDegrain>();
            auto sts = filter->init(param, log);
            if (sts != RGY_ERR_NONE) {
                printf("[error] clean TR2 init failed: %s\n", tchar_to_string(get_err_mes(sts)).c_str());
                return sts;
            }
            stage.filter = std::move(filter);
        }
        printf("clean4 : %s%s\n", tchar_to_string(param->print()).c_str(),
            stage.enabled ? "" : "   [off in this preset -> pass-through]");
    }

    // --- 5: rep2 ------------------------------------------------------------------------
    {
        const int thin = std::max(0, std::min(rtgmc.rep2.repThin, 7));
        auto param  = std::make_shared<NVEncFilterParamRtgmcShimmerRepair>();
        param->stage        = RGYRtgmcShimmerRepairStage::PostTR2;
        param->repairThin   = thin;
        param->repairPad    = rtgmc.rep2.repPad;
        param->processChroma = rtgmc.rep2.repChroma;
        makeParam(param);
        auto &stage = m_stages[CLEAN_STAGE_REP2];
        stage.name    = "rep2";
        stage.enabled = (thin > 0);
        if (stage.enabled) {
            auto filter = std::make_unique<NVEncFilterRtgmcShimmerRepair>();
            auto sts = filter->init(param, log);
            if (sts != RGY_ERR_NONE) {
                printf("[error] clean rep2 init failed: %s\n", tchar_to_string(get_err_mes(sts)).c_str());
                return sts;
            }
            stage.repair  = filter.get();
            stage.filter  = std::move(filter);
        }
        m_needRef = m_needRef || stage.enabled;
        printf("clean5 : %s%s\n", tchar_to_string(param->print()).c_str(),
            stage.enabled ? "" : "   [disabled, repair-thin=0 -> pass-through]");
    }

    // --- reference ring -----------------------------------------------------------------
    if (m_needRef) {
        const auto sts = m_refRing.alloc(CLEAN_REF_POOL, frame);
        if (sts != RGY_ERR_NONE) {
            printf("[error] failed to allocate the clean reference ring: %s\n",
                tchar_to_string(get_err_mes(sts)).c_str());
            return sts;
        }
        printf("clean  : rep reference ring = %d frames, reference = the unfiltered clean input\n",
            (int)m_refRing.pool.size());
    } else {
        printf("clean  : both shimmer repair stages are off, no reference ring needed\n");
    }

    // --- QTGMC's sharpLimit1 ------------------------------------------------------------
    //
    //     spatialSL = (SLMode == 1 || SLMode == 3)
    //     temporalSL = (SLMode == 2 || SLMode == 4)
    //     tMax = (temporalSL) ? edi.mt_logic(fComp1,"max").mt_logic(bComp1,"max") : NOP()
    //     tMin = (temporalSL) ? edi.mt_logic(fComp1,"min").mt_logic(bComp1,"min") : NOP()
    //     sharpLimit1 = (SLMode == 2) ? backBlend1.mt_clamp(tMax, tMin, Sovs, Sovs) : ...
    //
    // so the temporal clamp needs edi warped onto the current time from both sides. Two
    // compensate passes over the clean input produce exactly those two frames.
    m_limitWanted = (rtgmc.retouch.slmode == 2 || rtgmc.retouch.slmode == 4) && rtgmc.retouch.slrad >= 1;
    if (m_limitWanted) {
        VppDegrain compParam = rtgmc.tr1;
        compParam.enable = true;
        compParam.stage  = VppDegrainStage::TR1;
        compParam.delta  = 1;          // QTGMC uses bVec1/fVec1, i.e. the delta-1 vectors
        // The motion field is shared, so the search side of these two only matters if the
        // shared result is unavailable - and skipping it also skips the analysis luma cache.
        compParam.tr0 = 0;
        compParam.rep0 = 0;
        compParam.searchRefine = 0;
        compParam.tvRange = false;

        const struct { const char *name; VppDegrainMode mode; std::unique_ptr<NVEncFilterDegrain> *slot; } defs[2] = {
            { "compb", VppDegrainMode::MotionBack, &m_compBack },
            { "compf", VppDegrainMode::MotionForw, &m_compForw },
        };
        for (const auto &def : defs) {
            auto filter = std::make_unique<NVEncFilterDegrain>();
            auto param  = std::make_shared<NVEncFilterParamDegrain>();
            compParam.mode = def.mode;
            param->degrain = compParam;
            param->attachAnalysisData = false;
            param->frameIn       = frame;
            param->frameOut      = frame;
            param->baseFps       = baseFps;
            param->bOutOverwrite = false;
            auto sts = filter->init(param, log);
            if (sts != RGY_ERR_NONE) {
                printf("[error] clean %s init failed: %s\n", def.name,
                    tchar_to_string(get_err_mes(sts)).c_str());
                return sts;
            }
            printf("clean  : %s (sharpLimit1 envelope) %s\n", def.name,
                tchar_to_string(param->print()).c_str());
            *def.slot = std::move(filter);
        }
        for (auto *ring : { &m_backRing, &m_forwRing }) {
            const auto sts = ring->alloc(CLEAN_COMP_POOL, frame);
            if (sts != RGY_ERR_NONE) {
                printf("[error] failed to allocate a clean limit ring: %s\n",
                    tchar_to_string(get_err_mes(sts)).c_str());
                return sts;
            }
        }
        printf("clean  : sharpLimit1 temporal clamp on (slmode=%d slrad=%d)\n",
            rtgmc.retouch.slmode, rtgmc.retouch.slrad);
    } else if (rtgmc.retouch.slmode == 1 || rtgmc.retouch.slmode == 3) {
        printf("clean  : sharpLimit1 is spatial (slmode=%d), no temporal envelope needed\n",
            rtgmc.retouch.slmode);
    }
    return RGY_ERR_NONE;
}

int CleanChain::runStage(const int idx, RGYFrameInfo *f) {
    if (idx >= CLEAN_STAGE_NUM) {
        // Past the last stage: this is a finished output frame, and everything up to and
        // including its timestamp has now left the chain, so those references are spent.
        if (!m_sink(f)) {
            return -1;
        }
        m_refRing.dropUpTo(f->timestamp);
        return 1;
    }

    CleanStage &stage = m_stages[idx];
    if (!stage.enabled) {
        // A pass-through holds nothing, so it has nothing to give back on a drain tick.
        return (f == nullptr) ? 0 : runStage(idx + 1, f);
    }

    if (stage.repair != nullptr) {
        // The reference is mandatory: without it the filter emits nothing, and its generic
        // overload would silently use the input as its own reference (an exact no-op).
        if (f == nullptr) {
            return 0;   // a shimmer repair holds no frames of its own
        }
        const RGYFrameInfo *ref = m_refRing.find(f->timestamp);
        if (ref == nullptr) {
            if (!m_missingRefWarned) {
                printf("\n[warn] clean %s: no reference frame for inputFrameId=%d ts=%lld; "
                       "passing the frame through unchanged.\n",
                    stage.name.c_str(), f->inputFrameId, (long long)f->timestamp);
                m_missingRefWarned = true;
            }
            return runStage(idx + 1, f);
        }
        RGYFrameInfo *out[CLEAN_MAX_OUT] = { nullptr };
        int outNum = 0;
        RGYCudaEvent ev;
        const auto sts = stage.repair->run_filter(f, ref, out, &outNum, m_stream, {}, &ev);
        if (sts != RGY_ERR_NONE) {
            printf("\n[error] clean %s failed: %s\n", stage.name.c_str(),
                tchar_to_string(get_err_mes(sts)).c_str());
            return -1;
        }
        for (int j = 0; j < outNum; j++) {
            if (runStage(idx + 1, out[j]) < 0) {
                return -1;
            }
        }
        return outNum;
    }

    // TR1/TR2 need the MV/SAD result of the analyze stage. The integrated filter does exactly
    // this before calling them (NVEncFilterRtgmc.cu, runNestedFilter), and likewise tolerates
    // a result set that is not ready yet.
    if (f != nullptr && (idx == CLEAN_STAGE_TR1 || idx == CLEAN_STAGE_TR2) && m_analyze != nullptr) {
        auto tr = dynamic_cast<NVEncFilterDegrain *>(stage.filter.get());
        if (tr != nullptr && !tr->setDirectAnalyzeResultSet(m_analyze->analyzeResultSet())) {
            printf("\r[note] clean %s: no direct MV/SAD result yet, using the fallback path.\n",
                stage.name.c_str());
        }
    }

    // QTGMC's sharpLimit1 temporal clamp has to be armed before the retouch runs on this
    // frame, with the envelope of the clean input warped onto this frame's time.
    if (idx == CLEAN_STAGE_RETOUCH && f != nullptr) {
        applyLimitFrames(f);
    }

    RGYFrameInfo *out[CLEAN_MAX_OUT] = { nullptr };
    int outNum = 0;
    const auto sts = stage.filter->filter(f, out, &outNum, m_stream);
    if (sts != RGY_ERR_NONE) {
        printf("\n[error] clean %s failed: %s\n", stage.name.c_str(),
            tchar_to_string(get_err_mes(sts)).c_str());
        return -1;
    }

    // The analyze stage passes every frame through with the MV/SAD data attached, and that is
    // what the two compensate passes bind, so they are fed from here rather than from the raw
    // input. Their outputs are the frames the clamp envelope is built from.
    if (idx == CLEAN_STAGE_ANALYZE) {
        for (int j = 0; j < outNum; j++) {
            if (!feedCompensate(out[j])) {
                return -1;
            }
        }
    }

    // The limit rings are only needed from the frame the retouch is on up to the newest frame
    // the compensate passes have reached, so once the retouch has run for this timestamp the
    // envelope entries up to it are spent.
    if (idx == CLEAN_STAGE_RETOUCH && f != nullptr) {
        m_backRing.dropUpTo(f->timestamp);
        m_forwRing.dropUpTo(f->timestamp);
    }

    for (int j = 0; j < outNum; j++) {
        if (runStage(idx + 1, out[j]) < 0) {
            return -1;
        }
    }
    return outNum;
}

// ---------------------------------------------------------------------------------------
// the three flows
// ---------------------------------------------------------------------------------------

static const char *flow_name(const int flow) {
    switch (flow) {
    case RTGMDIF_FLOW_FAST_DEINT: return "fast_deint";
    case RTGMDIF_FLOW_FAST_BOTH:  return "fast_both";
    case RTGMDIF_FLOW_FAST_OPT:   return "fast_opt";
    case RTGMDIF_FLOW_FASTER_NN1: return "faster_nn1";
    case RTGMDIF_FLOW_SLOWER:     return "slower";
    case RTGMDIF_FLOW_SLOW:       return "slow";
    case RTGMDIF_FLOW_SLOW_BOTH:  return "slow_both";
    default:                      return "?";
    }
}

static bool flow_wants_deint(const int flow) {
    return flow == RTGMDIF_FLOW_FAST_DEINT
        || flow == RTGMDIF_FLOW_FAST_BOTH
        || flow == RTGMDIF_FLOW_FAST_OPT
        || flow == RTGMDIF_FLOW_FASTER_NN1
        || flow == RTGMDIF_FLOW_SLOWER
        || flow == RTGMDIF_FLOW_SLOW
        || flow == RTGMDIF_FLOW_SLOW_BOTH;
}

static bool flow_wants_clean(const int flow) {
    return flow == RTGMDIF_FLOW_FAST_BOTH
        || flow == RTGMDIF_FLOW_FAST_OPT
        || flow == RTGMDIF_FLOW_SLOW_BOTH;
}

// Builds the two VppRtgmc parameter sets a flow needs. Both passes come from the same preset so
// they cannot drift apart, and only what the flow changes on top is overridden.
//
// The clean side is QTGMC(clip, InputType=1, Sharpness=0): SMode = 0, which turns the
// resharpen block into a pass-through and forces Sbb = 0 (QTGMC: `Sbb = (SMode == 0) ? 0 :
// Sbb`), while the SLMode clamp keeps running.
static void build_flow_params(const RtgmDifConfig &cfg, VppRtgmc *deint, VppRtgmc *clean) {
    const int flow = cfg.flow;
    const auto preset = (flow == RTGMDIF_FLOW_FASTER_NN1) ? VppRtgmcPreset::Faster
                      : (flow == RTGMDIF_FLOW_SLOWER)     ? VppRtgmcPreset::Slower
                      : (flow == RTGMDIF_FLOW_SLOW)       ? VppRtgmcPreset::Slow
                      : (flow == RTGMDIF_FLOW_SLOW_BOTH)  ? VppRtgmcPreset::Slow
                                                          : VppRtgmcPreset::Fast;
    apply_vpp_rtgmc_preset(*deint, preset, VppRtgmcTuning::None);

    // Search early exit. Only the spatial one is implemented in this tree: the search kernels
    // take a spatialEarlySad threshold and skip the refinement of a block whose base vector is
    // already good enough, while the preset table's search_early_sad is written, range-checked
    // and printed but read by no kernel (measured: 0, 16 and 32 gave byte-identical output and
    // the same time). It goes on the deinterlace side before the clean copy below, so both
    // passes end up with the same value.
    if (cfg.spatialEarlySad >= 0) {
        deint->analyze.spatialEarlySad = cfg.spatialEarlySad;
        deint->tr1.spatialEarlySad     = cfg.spatialEarlySad;
        deint->tr2.spatialEarlySad     = cfg.spatialEarlySad;
    }

    // The low-latency variant, as one switch. Everything below it still wins, so an explicit
    // deintTr0/deintTr2Delta on top of it is honoured. See RtgmDifConfig::lowLatency.
    if (cfg.lowLatency) {
        deint->searchPrefilter.tr0 = 1;
        deint->analyze.tr0         = 1;
        deint->tr1.tr0             = 1;
        deint->tr2.tr0             = 1;
        // The TR stages stop waiting for the frames their own search would have needed; the
        // vectors come from the analyze stage instead.
        deint->tr1.lowLatency = true;
        deint->tr2.lowLatency = true;
        // The analyze stage's vector set shrinks to distance 1: nothing in this variant reads a
        // distance-2 vector (TR1 asks for 1, TR2 is off, source_match is off), so half the search
        // work goes away for the same result. This is RtgmDifConfig::deintAnalyzeDelta's meaning.
        deint->analyze.delta = 1;
        // The analyze stage keeps its tr0 term, and that was measured rather than assumed. With
        // the term dropped and the luma cache prefetched to the newest frame instead - which is
        // what it takes to reach 3.0 input frames - the analysis runs one frame earlier, and
        // because the analysis luma of a frame is built from its +/-2 neighbours it then has two
        // clamped future neighbours instead of one. Intra-pair shimmer on the same window: 0.693
        // with the term, 0.893 without, which is what a single-pass flow measures. Half a frame is
        // not worth that, so the term stays.
        // TR2 off, in every flow. The last frame of the budget was a choice between this and
        // turning the clean chain's TR2 off instead, which lands at 4.5 input frames: both were
        // looked at side by side against the unswitched flow and could not be told apart, so the
        // cheaper one is taken. Measured on slow_both, intra-pair: 0.693 here, 0.689 there, on a
        // flow whose own reference is 0.682.
        deint->tr2.delta = 0;
    }

    // The first pass's search settings, one at a time. -1 leaves the preset's value. These are
    // the fields that decide what a detailed frame costs, because the search is the only stage
    // whose work grows with how hard the picture is - see RtgmDifConfig for the measurements.
    {
        const auto applySearch = [deint](int VppDegrain::*member, const int value) {
            if (value >= 0) {
                deint->analyze.*member = value;
                deint->tr1.*member     = value;
                deint->tr2.*member     = value;
            }
        };
        applySearch(&VppDegrain::pel,          cfg.deintPel);
        applySearch(&VppDegrain::pelSearch,    cfg.deintPelSearch);
        applySearch(&VppDegrain::searchRefine, cfg.deintSearchRefine);
        applySearch(&VppDegrain::searchParam,  cfg.deintSearchParam);
    }

    // Two caps on how much work a frame can cost, rather than a cut in precision: one reference
    // distance instead of two, and a coarser block grid. Both take effect before the clean copy,
    // so the clean-side fields still win on the second pass. See RtgmDifConfig.
    if (cfg.deintAnalyzeDelta >= 0) {
        deint->analyze.delta = std::min(cfg.deintAnalyzeDelta, RGY_DEGRAIN_MAX_DELTA);
    }
    if (cfg.deintOverlap >= 0) {
        deint->analyze.overlap = cfg.deintOverlap;
        deint->tr1.overlap     = cfg.deintOverlap;
        deint->tr2.overlap     = cfg.deintOverlap;
    }

    // The temporal radii - what the pipeline delay is actually made of. tr0 goes into the four
    // places the preset copies it to (the search prefilter and the three stages); the two deltas
    // go straight onto their stage. A zero delta turns that stage off, and removing a stage takes
    // its whole holding with it, which is where the delay goes - see RtgmDifConfig for the
    // measured ledger. They are applied before the clean copy, so the second pass inherits the
    // same shape unless its own fields override.
    if (cfg.deintTr0 >= 0) {
        deint->searchPrefilter.tr0 = cfg.deintTr0;
        deint->analyze.tr0         = cfg.deintTr0;
        deint->tr1.tr0             = cfg.deintTr0;
        deint->tr2.tr0             = cfg.deintTr0;
    }
    if (cfg.deintTr1Delta >= 0) {
        deint->tr1.delta = std::min(cfg.deintTr1Delta, RGY_DEGRAIN_MAX_DELTA);
    }
    if (cfg.deintTr2Delta >= 0) {
        deint->tr2.delta = std::min(cfg.deintTr2Delta, RGY_DEGRAIN_MAX_DELTA);
    }
    if (cfg.deintRetouchSmode >= 0) {
        deint->retouch.smode = cfg.deintRetouchSmode;
    }

    // The EDI window, which is the parameter that decided the scrolling-text case: smaller is
    // stronger. It has to go on the *deinterlace* side, before the clean parameters are copied
    // from it, because that is what the upstream override does - its nnsize override goes
    // through the shared quality override set, so pass 1 picks it up as well and the
    // deinterlaced frames themselves come out sharper. That gain is the measured
    // `detail` 1.1383 -> 1.1439, and the clean stage inherits it through the copy below.
    // `slower` brings nnsize 1 with it from its own table, so only the other two set it here.
    const int nnsize = (flow == RTGMDIF_FLOW_FAST_OPT)   ? 3
                     : (flow == RTGMDIF_FLOW_FASTER_NN1) ? 1
                     : 0;   // 0 = keep whatever the preset table chose
    if (nnsize > 0) {
        deint->edi.nnsize      = nnsize;
        deint->matchEdi.nnsize = nnsize;
    }

    // `placebo` / `veryslow` request noise_process=2, which NVEncFilterRtgmc does not implement
    // yet (checkParam() rejects it). `fast` does not ask for it, but the guard is kept so that
    // a flow that moves to a different preset cannot silently break.
    if (deint->noise.noiseProcess == 2) {
        deint->noise.noiseProcess = 0;
        deint->noise.noiseTR = 0;
        deint->noise.grainRestore = 0.0f;
        deint->noise.noiseRestore = 0.0f;
    }

    *clean = *deint;
    if (!flow_wants_clean(flow)) {
        return;
    }

    // A different preset for the second pass, if one was asked for. The copy above brought the
    // first pass's parameters over, so this rebuilds the clean side from its own preset; the
    // flow's own clean overrides and the clean-side fields further down are applied after it and
    // still win. A preset only decides values, so nothing else has to change.
    if (cfg.cleanPreset >= 0) {
        apply_vpp_rtgmc_preset(*clean,
            (VppRtgmcPreset)std::max(0, std::min(cfg.cleanPreset, (int)VppRtgmcPreset::Draft)),
            VppRtgmcTuning::None);
    }

    // fast_opt also takes the TR structure of QTGMC's Slower preset. It is free: the analyze
    // stage already computes the delta-1 and delta-2 motion vectors, so TR2 delta 1 only adds a
    // mix, and the residual field-parity alternation in static areas drops from 0.0525 to
    // 0.0422 (mean |Y(n) - Y(n+1)| over the top 216 rows of the 1080i test source) for about
    // +0.1% of the `fast` cost.
    if (flow == RTGMDIF_FLOW_FAST_OPT) {
        clean->tr1.delta = 2;
        clean->tr2.delta = 1;
    }

    // The clean-only knobs, applied last so they win over the flow's own clean overrides above.
    // The analyze delta is deliberately left alone: CleanChain::init raises it to cover whatever
    // TR1 and TR2 ask for, so lowering these can never leave the analyze stage short.
    if (cfg.cleanAnalyzeDelta >= 0) {
        clean->analyze.delta = std::min(cfg.cleanAnalyzeDelta, RGY_DEGRAIN_MAX_DELTA);
    }
    if (cfg.cleanTr1Delta >= 0) {
        clean->tr1.delta = std::min(cfg.cleanTr1Delta, RGY_DEGRAIN_MAX_DELTA);
    }
    if (cfg.cleanTr2Delta >= 0) {
        clean->tr2.delta = std::min(cfg.cleanTr2Delta, RGY_DEGRAIN_MAX_DELTA);
    }
    if (cfg.cleanRep2Thin >= 0) {
        clean->rep2.repThin = std::min(cfg.cleanRep2Thin, 7);
    }

    clean->retouch.sharpness = 0.0f;   // QTGMC's second pass runs with Sharpness=0
    clean->retouch.smode     = 0;
    clean->retouch.sbb       = 0;
    // retouch carries its own copy of the TR radii and is checked against them.
    clean->retouch.tr1 = clean->tr1.delta;
    clean->retouch.tr2 = clean->tr2.delta;
}

// ---------------------------------------------------------------------------------------
// impl
// ---------------------------------------------------------------------------------------

// The interface is pure, so this is the class that implements it, and it is never exported:
// the only way to get one is rtgmdif_create() at the bottom of this file.
//
// The defaults of RtgmDifConfig used to live in a constructor here. They are member
// initialisers in the header now, so that a caller's `RtgmDifConfig cfg;` is compiled entirely
// on the caller's side and nothing about the struct has to be exported.
class RtgmDifImpl final : public RtgmDif {
public:
    RtgmDifImpl();
    ~RtgmDifImpl() override;

    bool configure(const RtgmDifConfig &cfg) override;
    bool init(cudaStream_t stream) override;
    bool handdif(void *pIn) override;
    int  getoutputsize() override;
    bool getoutputbuf(int idx, void *pOut) override;
    int  inputFrameBytes() const override;
    int  outputFrameBytes() const override;
    int  framesIn() const override;
    int  framesOut() const override;
    bool inited() const override;
    void close() override;

private:
    struct Impl;
    Impl *m_impl;
};

struct RtgmDifImpl::Impl {
    RtgmDifConfig cfg;
    bool          configured;

    RGY_CSP       csp;
    int           bitdepth;
    int           bytesPerPix;
    PlaneGeom     geom[RTGMDIF_PLANES];
    int64_t       inFrameBytes;
    int64_t       outFrameBytes;
    RGYFrameInfo  inFrameInfo;      // the frame layout handed to the filters
    RGY_PICSTRUCT inPicStruct;

    std::shared_ptr<RGYLog> log;
    cudaStream_t  stream;
    bool          ownsStream;   // false when the stream came from the caller
    bool          inited;
    bool          flushed;

    std::unique_ptr<NVEncFilterRtgmc> rtgmc;
    CleanChain    clean;
    bool          wantDeint;
    bool          wantClean;

    // --- the split (see RtgmDifConfig::splitY) ------------------------------------------
    //
    // wantSplit false leaves every field below untouched and every path exactly as it was.
    // When it is true the chain runs on the rows [chainProcY0, chainProcY1) of each frame and
    // the cheap deinterlacer on [cheapProcY0, cheapProcY1); the rows that are handed to the
    // caller are [chainY0, chainY1) and [cheapY0, cheapY1). The two differ by
    // RTGMDIF_SPLIT_MARGIN on the side that faces the other band, which is computed so that the
    // boundary row is interior but never copied out.
    bool          wantSplit;
    int           chainY0, chainY1;          // rows the chain owns in the output frame
    int           cheapY0, cheapY1;          // rows the cheap deinterlacer owns
    int           chainProcY0, chainProcY1;  // rows actually processed (owned rows + margin)
    int           cheapProcY0, cheapProcY1;
    PlaneGeom     chainGeom[RTGMDIF_PLANES];
    PlaneGeom     cheapGeom[RTGMDIF_PLANES];
    std::vector<std::unique_ptr<CUFrameBuf>> chainInBufs;   // ring, same role as inBufs
    int           chainInPut;
    std::vector<std::unique_ptr<CUFrameBuf>> cheapInBufs;
    int           cheapInPut;
    // The cheap deinterlacer for the far side of the split: YADIF or BWDIF, whichever
    // RtgmDifConfig::splitDif asked for. Both are NVEncFilter and both take the same frames
    // and emit two per input frame in bob mode, so nothing below has to know which it is.
    std::unique_ptr<NVEncFilter> cheapDeint;
    bool          cheapDrained;
    // The cheap one runs ahead of the chain, so its finished band frames wait here, in order, for
    // the chain frame with the same output number. The k-th output of one belongs with the k-th
    // output of the other because both emit two frames per input frame, in order.
    std::deque<std::unique_ptr<CUFrameBuf>> cheapOut;
    std::vector<std::unique_ptr<CUFrameBuf>> cheapOutFree;

    std::vector<std::unique_ptr<CUFrameBuf>> inBufs;
    int           inPut;
    int64_t       inIndex;
    int64_t       inFrameDuration;

    // Finished output frames. The filters hand back pointers into their own internal rings,
    // which are only valid until the next filter() call, so every released frame is copied
    // into a slot of this pool and the batch refers to the slots.
    std::vector<std::unique_ptr<CUFrameBuf>> outPool;
    std::vector<int>  freeSlots;
    std::deque<int>   batch;
    std::vector<char> taken;
    int           droppedUnread;
    int64_t       outTotal;

    Impl()
        : cfg()
        , configured(false)
        , csp(RGY_CSP_YUV444)
        , bitdepth(8)
        , bytesPerPix(1)
        , inFrameBytes(0)
        , outFrameBytes(0)
        , inFrameInfo()
        , inPicStruct(RGY_PICSTRUCT_FRAME_TFF)
        , log()
        , stream(nullptr)
        , ownsStream(false)
        , inited(false)
        , flushed(false)
        , rtgmc()
        , clean()
        , wantDeint(true)
        , wantClean(false)
        , inBufs()
        , inPut(0)
        , inIndex(0)
        , inFrameDuration(2)
        , outPool()
        , freeSlots()
        , batch()
        , taken()
        , droppedUnread(0)
        , outTotal(0) {
        std::memset(geom, 0, sizeof(geom));
        // The split is off until init() finds a band in the configuration, but these have to be
        // defined before anything can look at them.
        wantSplit = false;
        chainY0 = chainY1 = cheapY0 = cheapY1 = 0;
        chainProcY0 = chainProcY1 = cheapProcY0 = cheapProcY1 = 0;
        std::memset(chainGeom, 0, sizeof(chainGeom));
        std::memset(cheapGeom, 0, sizeof(cheapGeom));
        chainInPut = 0;
        cheapInPut = 0;
        cheapDrained = false;
    }

    // --- output batch ------------------------------------------------------------------

    void retireBatch() {
        if (!batch.empty()) {
            int unread = 0;
            for (size_t i = 0; i < batch.size(); i++) {
                if (taken[i] == 0) {
                    unread++;
                }
            }
            if (unread > 0) {
                droppedUnread += unread;
                if (droppedUnread == unread) {
                    printf("[warn] %d output frame(s) were never read before the next handdif(); "
                           "read the whole batch between calls.\n", unread);
                }
            }
            for (const int slot : batch) {
                freeSlots.push_back(slot);
            }
            batch.clear();
            taken.clear();
        }
    }

    // Row number of a plane for a frame row. Only YUV420 halves chroma vertically; YUV422 keeps
    // full height and halves width only.
    int planeRow(const int plane, const int y) const {
        if (plane == 0 || RGY_CSP_CHROMA_FORMAT[csp] != RGY_CHROMAFMT_YUV420) {
            return y;
        }
        return y / 2;
    }

    // Copies rows [srcY0, srcY1) of `src` into `dst` at the same row numbers. `src` is a band
    // whose top row is srcProcY0 of the frame, `dst` is the whole frame. Each side of a split is
    // only copied where it owns the rows: the margin it also processed is thrown away, which is
    // what the margin is for.
    bool copyOwnedRows(const RGYFrameInfo &dst, const RGYFrameInfo &src,
        const int srcProcY0, const int srcY0, const int srcY1) {
        for (int i = 0; i < RTGMDIF_PLANES; i++) {
            const int dy0 = planeRow(i, srcY0);
            const int dy1 = planeRow(i, srcY1);
            const int sy0 = dy0 - planeRow(i, srcProcY0);
            const size_t rowBytes = (size_t)geom[i].width * bytesPerPix;
            const auto cudaerr = cudaMemcpy2DAsync(
                dst.ptr[i] + (int64_t)dy0 * dst.pitch[i], (size_t)dst.pitch[i],
                src.ptr[i] + (int64_t)sy0 * src.pitch[i], (size_t)src.pitch[i],
                rowBytes, dy1 - dy0, cudaMemcpyDeviceToDevice, stream);
            if (cudaerr != cudaSuccess) {
                printf("\n[error] cudaMemcpy2DAsync (band composite) failed: %s\n",
                    cudaGetErrorString(cudaerr));
                return false;
            }
        }
        return true;
    }

    // Copies rows [procY0, procY1) of the whole frame `src` into the top of the band `dst`.
    bool copyBandIn(const RGYFrameInfo &dst, const RGYFrameInfo &src,
        const int procY0, const int procY1) {
        for (int i = 0; i < RTGMDIF_PLANES; i++) {
            const int sy0 = planeRow(i, procY0);
            const int sy1 = planeRow(i, procY1);
            const size_t rowBytes = (size_t)geom[i].width * bytesPerPix;
            const auto cudaerr = cudaMemcpy2DAsync(dst.ptr[i], (size_t)dst.pitch[i],
                src.ptr[i] + (int64_t)sy0 * src.pitch[i], (size_t)src.pitch[i],
                rowBytes, sy1 - sy0, cudaMemcpyDeviceToDevice, stream);
            if (cudaerr != cudaSuccess) {
                printf("\n[error] cudaMemcpy2DAsync (band input copy) failed: %s\n",
                    cudaGetErrorString(cudaerr));
                return false;
            }
        }
        return true;
    }

    // Sink for the filters' output frames: copy each one into a pool slot and add it to the
    // batch. The frame is only guaranteed valid during this call, which is exactly why the
    // copy happens here.
    //
    // With a split, what arrives here is the chain's band rather than a whole frame, so the band
    // goes into the slot at its own row range and the cheap one's band is composited in behind
    // it. That frame is taken from the head of the queue: the k-th frame of one path belongs
    // with the k-th frame of the other because both emit two frames per input frame, in order.
    bool sink(RGYFrameInfo *f) {
        if (f == nullptr) {
            return true;
        }
        if (freeSlots.empty()) {
            printf("\n[error] the output pool (%d frames) is exhausted; read the batch with "
                   "getoutputsize()/getoutputbuf() before handing in the next frame.\n",
                (int)outPool.size());
            return false;
        }
        const int expectH = wantSplit ? (chainProcY1 - chainProcY0) : cfg.height;
        if (f->width != cfg.width || f->height != expectH || f->csp != csp) {
            printf("\n[error] the filter returned a %dx%d %s frame but the module is configured "
                   "for %dx%d %s.\n",
                f->width, f->height, tchar_to_string(RGY_CSP_NAMES[f->csp]).c_str(),
                cfg.width, expectH, tchar_to_string(RGY_CSP_NAMES[csp]).c_str());
            return false;
        }
        const int slot = freeSlots.back();
        freeSlots.pop_back();
        auto &dst = *outPool[slot];
        if (!wantSplit) {
            for (int i = 0; i < RTGMDIF_PLANES; i++) {
                const size_t rowBytes = (size_t)geom[i].width * bytesPerPix;
                const auto cudaerr = cudaMemcpy2DAsync(dst.frame.ptr[i], (size_t)dst.frame.pitch[i],
                    f->ptr[i], (size_t)f->pitch[i], rowBytes, geom[i].height,
                    cudaMemcpyDeviceToDevice, stream);
                if (cudaerr != cudaSuccess) {
                    printf("\n[error] cudaMemcpy2DAsync (output copy) failed: %s\n",
                        cudaGetErrorString(cudaerr));
                    freeSlots.push_back(slot);
                    return false;
                }
            }
        } else {
            if (!copyOwnedRows(dst.frame, *f, chainProcY0, chainY0, chainY1)) {
                freeSlots.push_back(slot);
                return false;
            }
            if (cheapOut.empty()) {
                printf("\n[error] split: no cheap-dif band frame is queued for output frame "
                       "%lld, so the two sides are out of step. The chain emitted more frames "
                       "than the cheap one did.\n", (long long)outTotal);
                freeSlots.push_back(slot);
                return false;
            }
            auto yFrame = std::move(cheapOut.front());
            cheapOut.pop_front();
            const bool ok = copyOwnedRows(dst.frame, yFrame->frame, cheapProcY0, cheapY0, cheapY1);
            cheapOutFree.push_back(std::move(yFrame));
            if (!ok) {
                freeSlots.push_back(slot);
                return false;
            }
        }
        batch.push_back(slot);
        taken.push_back(0);
        outTotal++;
        return true;
    }

    // Takes everything one call of the cheap deinterlacer released and queues it for the
    // composite. The frames are copied because the filter's own frames are only valid until its
    // next call.
    bool queueCheapOut(RGYFrameInfo **out, const int outNum) {
        for (int j = 0; j < outNum; j++) {
            if (out[j] == nullptr) {
                continue;
            }
            std::unique_ptr<CUFrameBuf> buf;
            if (!cheapOutFree.empty()) {
                buf = std::move(cheapOutFree.back());
                cheapOutFree.pop_back();
            } else {
                buf = std::make_unique<CUFrameBuf>(cfg.width, cheapProcY1 - cheapProcY0, csp);
                const auto sts = buf->alloc();
                if (sts != RGY_ERR_NONE) {
                    printf("\n[error] failed to allocate a cheap-dif band frame: %s\n",
                        tchar_to_string(get_err_mes(sts)).c_str());
                    return false;
                }
            }
            for (int i = 0; i < RTGMDIF_PLANES; i++) {
                const size_t rowBytes = (size_t)cheapGeom[i].width * bytesPerPix;
                const auto cudaerr = cudaMemcpy2DAsync(buf->frame.ptr[i],
                    (size_t)buf->frame.pitch[i], out[j]->ptr[i], (size_t)out[j]->pitch[i],
                    rowBytes, cheapGeom[i].height, cudaMemcpyDeviceToDevice, stream);
                if (cudaerr != cudaSuccess) {
                    printf("\n[error] cudaMemcpy2DAsync (cheap-dif band copy) failed: %s\n",
                        cudaGetErrorString(cudaerr));
                    return false;
                }
            }
            cheapOut.push_back(std::move(buf));
        }
        return true;
    }

    // --- driving ---------------------------------------------------------------------

    bool feedStage1(RGYFrameInfo *f) {
        if (!wantClean) {
            return sink(f);
        }
        return clean.push(f);
    }

    bool flush() {
        if (flushed) {
            return true;
        }
        int guard = 0;
        const int guardMax = (int)(inIndex * 4 + 4096);
        // The cheap one goes first: it is the side that runs ahead, so everything the chain is
        // about to release from its tail has to be in the queue before that happens.
        if (wantSplit && !cheapDrained) {
            int yguard = 0;
            // This filter's drain convention is a frame whose plane pointer is null, not a null
            // frame pointer: run_filter() reads pInputFrame->ptr[0] before it does anything else,
            // so a null pointer crashes it. The RTGMC filter does take a null pointer - the two
            // are not interchangeable.
            RGYFrameInfo drainFrame = {};
            for (;;) {
                // The array has to be cleared before every call. This filter checks
                // ppOutputFrames[0] to decide whether it allocates the output frame itself and
                // otherwise takes the "caller supplied it" path - in which pOutFrame stays null
                // and the next line dereferences it. The RTGMC filter does not look at the array,
                // which is why the same pattern is fine in the stage 0 drain above.
                RGYFrameInfo *yOut[RTGMDIF_MAX_OUT_FRAMES] = { nullptr };
                int yOutNum = 0;
                const auto ysts = cheapDeint->filter(&drainFrame, yOut, &yOutNum, stream);
                if (ysts != RGY_ERR_NONE) {
                    printf("\n[error] cheap-dif drain failed: %s\n",
                        tchar_to_string(get_err_mes(ysts)).c_str());
                    return false;
                }
                if (!queueCheapOut(yOut, yOutNum)) {
                    return false;
                }
                if (yOutNum <= 0 || ++yguard >= guardMax) {
                    break;
                }
            }
            cheapDrained = true;
        }
        if (wantDeint) {
            RGYFrameInfo *out[RTGMDIF_MAX_OUT_FRAMES] = { nullptr };
            while (!rtgmc->drainComplete() && guard < guardMax) {
                int outNum = 0;
                const auto sts = rtgmc->filter(nullptr, out, &outNum, stream);
                if (sts != RGY_ERR_NONE) {
                    printf("\n[error] stage0 drain failed: %s\n", tchar_to_string(get_err_mes(sts)).c_str());
                    return false;
                }
                for (int j = 0; j < outNum; j++) {
                    if (!feedStage1(out[j])) {
                        return false;
                    }
                }
                guard++;
            }
            if (guard >= guardMax) {
                printf("\n[warn] stage0 drain did not complete within %d iterations.\n", guardMax);
            }
        }
        if (wantClean) {
            if (!clean.drain()) {
                return false;
            }
        }
        if (wantSplit && !cheapOut.empty()) {
            printf("[warn] split: %d cheap-dif band frame(s) were never used, so the two sides "
                   "did not emit the same number of frames.\n", (int)cheapOut.size());
        }
        cudaStreamSynchronize(stream);
        flushed = true;
        if (wantClean) {
            printf("clean ref : peak %d of %d live references\n",
                clean.refPeak(), CLEAN_REF_POOL);
            if (clean.limitWanted()) {
                printf("clean lim : sharplimit1 envelope, %d frame(s) without a complete envelope\n",
                    clean.limitMissing());
            }
        }
        fflush(stdout);
        return true;
    }
};

// ---------------------------------------------------------------------------------------
// public interface
// ---------------------------------------------------------------------------------------

RtgmDifImpl::RtgmDifImpl() : m_impl(new Impl()) {
}

RtgmDifImpl::~RtgmDifImpl() {
    close();
    delete m_impl;
}

bool RtgmDifImpl::configure(const RtgmDifConfig &cfg) {
    if (m_impl == nullptr || m_impl->inited) {
        printf("[error] RtgmDif::configure() has to be called before init().\n");
        return false;
    }
    m_impl->cfg = cfg;
    m_impl->configured = true;
    return true;
}

bool RtgmDifImpl::init(cudaStream_t stream) {
    Impl &d = *m_impl;
    // The module links the CRT statically, so its stdout is its own FILE and its own buffer, and
    // the caller's setvbuf() does not apply to it. Do NOT be tempted to call
    // setvbuf(stdout, nullptr, _IONBF, 0) here to make the log survive a crash: tried, and the
    // module's output disappeared from the log completely - the DLL's stdio is apparently not in
    // a state where that call is legal. Use fflush(stdout) after the lines that matter instead.
    if (d.inited) {
        printf("[warn] RtgmDif::init() called twice.\n");
        return true;
    }
    const RtgmDifConfig &cfg = d.cfg;

    if (cfg.width <= 0 || cfg.height <= 0
        || (cfg.width % 2) != 0 || (cfg.height % 2) != 0) {
        printf("[error] invalid frame size %dx%d; both dimensions must be even and positive.\n",
            cfg.width, cfg.height);
        return false;
    }
    if (cfg.bitdepth != 8) {
        printf("[error] bitdepth %d is not supported yet; only 8 is implemented.\n", cfg.bitdepth);
        return false;
    }
    switch (cfg.flow) {
    case RTGMDIF_FLOW_FAST_DEINT:
    case RTGMDIF_FLOW_FAST_BOTH:
    case RTGMDIF_FLOW_FAST_OPT:
    case RTGMDIF_FLOW_FASTER_NN1:
    case RTGMDIF_FLOW_SLOWER:
    case RTGMDIF_FLOW_SLOW:
    case RTGMDIF_FLOW_SLOW_BOTH:
        break;
    default:
        printf("[error] unknown flow %d.\n", cfg.flow);
        return false;
    }
    switch (cfg.csp) {
    case RTGMDIF_CSP_YUV420P: d.csp = RGY_CSP_YV12;   break;
    case RTGMDIF_CSP_YUV422P: d.csp = RGY_CSP_YUV422; break;
    case RTGMDIF_CSP_YUV444P: d.csp = RGY_CSP_YUV444; break;
    default:
        printf("[error] unknown csp %d.\n", cfg.csp);
        return false;
    }
    if (cfg.fpsNum <= 0 || cfg.fpsDen <= 0) {
        printf("[error] invalid frame rate %d/%d.\n", cfg.fpsNum, cfg.fpsDen);
        return false;
    }
    if (cfg.outputPoolFrames <= 0) {
        printf("[error] outputPoolFrames must be positive.\n");
        return false;
    }

    d.bitdepth    = cfg.bitdepth;
    d.bytesPerPix = (d.bitdepth + 7) / 8;
    get_plane_geom(d.csp, cfg.width, cfg.height, d.geom);
    d.inFrameBytes  = plane_total_bytes(d.geom, d.bytesPerPix);
    d.outFrameBytes = d.inFrameBytes;
    d.inPicStruct   = cfg.tff ? RGY_PICSTRUCT_FRAME_TFF : RGY_PICSTRUCT_FRAME_BFF;
    d.wantDeint     = flow_wants_deint(cfg.flow);
    d.wantClean     = flow_wants_clean(cfg.flow);
    d.inFrameDuration = d.wantDeint ? 2 : 1;

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        printf("[error] no CUDA device found.\n");
        return false;
    }
    if (cfg.deviceId < 0 || cfg.deviceId >= deviceCount) {
        printf("[error] device %d does not exist (%d device(s) found).\n", cfg.deviceId, deviceCount);
        return false;
    }
    cudaSetDevice(cfg.deviceId);
    cudaDeviceProp prop = {};
    cudaGetDeviceProperties(&prop, cfg.deviceId);

    d.log = std::make_shared<RGYLog>(nullptr, (RGYLogLevel)cfg.logLevel);

    if (stream != nullptr) {
        // The caller owns it and orders its own work against it, so the module never has to
        // block it and never destroys it.
        d.stream = stream;
        d.ownsStream = false;
    } else {
        const auto cudaerr = cudaStreamCreate(&d.stream);
        if (cudaerr != cudaSuccess) {
            printf("[error] cudaStreamCreate failed: %s\n", cudaGetErrorString(cudaerr));
            return false;
        }
        d.ownsStream = true;
    }

    printf("rtgmdif: flow %s, %s %dx%d, %d fps %s\n",
        flow_name(cfg.flow), tchar_to_string(RGY_CSP_NAMES[d.csp]).c_str(),
        cfg.width, cfg.height, cfg.fpsNum / cfg.fpsDen,
        d.wantDeint ? (cfg.tff ? "interlaced (tff)" : "interlaced (bff)") : "progressive");
    printf("rtgmdif: device %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("rtgmdif: stream %s\n", d.ownsStream ? "created by the module (blocking)" : "supplied by the caller");

    // --- the parameter sets ------------------------------------------------------------
    VppRtgmc deintParams;
    VppRtgmc cleanParams;
    build_flow_params(cfg, &deintParams, &cleanParams);

    // --- input ring ---------------------------------------------------------------------
    d.inFrameInfo = RGYFrameInfo(cfg.width, cfg.height, d.csp, d.bitdepth,
        d.wantDeint ? d.inPicStruct : RGY_PICSTRUCT_FRAME, RGY_MEM_TYPE_GPU);
    for (int i = 0; i < RTGMDIF_IN_BUF_RING; i++) {
        auto buf = std::make_unique<CUFrameBuf>(cfg.width, cfg.height, d.csp);
        const auto sts = buf->alloc();
        if (sts != RGY_ERR_NONE) {
            printf("[error] failed to allocate input frame %d on device: %s\n", i,
                tchar_to_string(get_err_mes(sts)).c_str());
            return false;
        }
        d.inBufs.push_back(std::move(buf));
    }

    // --- output pool --------------------------------------------------------------------
    for (int i = 0; i < cfg.outputPoolFrames; i++) {
        auto buf = std::make_unique<CUFrameBuf>(cfg.width, cfg.height, d.csp);
        const auto sts = buf->alloc();
        if (sts != RGY_ERR_NONE) {
            printf("[error] failed to allocate output frame %d on device: %s\n", i,
                tchar_to_string(get_err_mes(sts)).c_str());
            return false;
        }
        d.outPool.push_back(std::move(buf));
    }
    for (int i = (int)d.outPool.size() - 1; i >= 0; i--) {
        d.freeSlots.push_back(i);
    }

    // --- the split (RtgmDifConfig::splitY) ------------------------------------------------
    //
    // Everything below is initialized on the chain's band instead of the whole frame, which is
    // the only reason the split saves time: a shorter frame has fewer blocks and less EDI. With
    // no split asked for, chainFrameH stays cfg.height and every path below is exactly what it
    // was.
    int  chainFrameH = cfg.height;
    int  cheapFrameH = 0;
    const bool cheapOnTop = (cfg.splitDifSide == RTGMDIF_SPLITSIDE_TOP);
    {
        const int splitY = std::max(0, std::min(cfg.splitY, cfg.height));
        d.wantSplit = (splitY > 0 && splitY < cfg.height);
        if (d.wantSplit) {
            if ((splitY % 2) != 0) {
                printf("[error] splitY %d is odd; it has to be even so that the chroma planes of "
                       "the two bands stay aligned.\n", splitY);
                return false;
            }
            if (!d.wantDeint) {
                printf("[error] splitY %d needs a deinterlacer, and this flow has none.\n", splitY);
                return false;
            }
            d.cheapY0 = cheapOnTop ? 0 : splitY;
            d.cheapY1 = cheapOnTop ? splitY : cfg.height;
            d.chainY0 = cheapOnTop ? splitY : 0;
            d.chainY1 = cheapOnTop ? cfg.height : splitY;
            // The margin goes on the side that faces the other band. The outer edge of the frame
            // has no vertical context to gain, so it is not grown.
            d.cheapProcY0 = std::max(0, d.cheapY0 - ((d.cheapY0 > 0) ? RTGMDIF_SPLIT_MARGIN : 0));
            d.cheapProcY1 = std::min(cfg.height, d.cheapY1 + ((d.cheapY1 < cfg.height) ? RTGMDIF_SPLIT_MARGIN : 0));
            d.chainProcY0 = std::max(0, d.chainY0 - ((d.chainY0 > 0) ? RTGMDIF_SPLIT_MARGIN : 0));
            d.chainProcY1 = std::min(cfg.height, d.chainY1 + ((d.chainY1 < cfg.height) ? RTGMDIF_SPLIT_MARGIN : 0));
            chainFrameH = d.chainProcY1 - d.chainProcY0;
            cheapFrameH = d.cheapProcY1 - d.cheapProcY0;
            get_plane_geom(d.csp, cfg.width, chainFrameH, d.chainGeom);
            get_plane_geom(d.csp, cfg.width, cheapFrameH, d.cheapGeom);
            printf("rtgmdif: split at row %d: %s rows [%d,%d) through the chain "
                   "(%d rows processed), %s rows [%d,%d) (%d rows processed)\n",
                splitY, cheapOnTop ? "lower" : "upper",
                d.chainY0, d.chainY1, chainFrameH,
                (cfg.splitDif == RTGMDIF_SPLITDIF_BWDIF) ? "BWDIF" : "YADIF",
                d.cheapY0, d.cheapY1, cheapFrameH);
        }
    }

    // --- the cheap deinterlacer's side ----------------------------------------------------
    //
    // Its own ring of band frames, for the same reason the input ring exists: the filter holds
    // on to the frames it is handed (its source keeps four of them for the bob), so a band
    // buffer cannot be reused until the filter is finished with it.
    if (d.wantSplit) {
        for (int i = 0; i < RTGMDIF_IN_BUF_RING; i++) {
            auto chainBuf = std::make_unique<CUFrameBuf>(cfg.width, chainFrameH, d.csp);
            auto sts = chainBuf->alloc();
            if (sts != RGY_ERR_NONE) {
                printf("[error] failed to allocate chain band input %d: %s\n", i,
                    tchar_to_string(get_err_mes(sts)).c_str());
                return false;
            }
            d.chainInBufs.push_back(std::move(chainBuf));

            auto cheapBuf = std::make_unique<CUFrameBuf>(cfg.width, cheapFrameH, d.csp);
            sts = cheapBuf->alloc();
            if (sts != RGY_ERR_NONE) {
                printf("[error] failed to allocate cheap-dif band input %d: %s\n", i,
                    tchar_to_string(get_err_mes(sts)).c_str());
                return false;
            }
            d.cheapInBufs.push_back(std::move(cheapBuf));
        }

        // Both filters are set up the same way: the band as frameIn and frameOut, bob mode so
        // that they emit two frames per input frame like the chain does, and the same output
        // timebase stage 0 uses (one unit per 50p frame). The field order comes either from the
        // frame's picstruct - BWDIF reads it per frame - or from a mode flag, which is what YADIF
        // has.
        const RGYFrameInfo bandIn(cfg.width, cheapFrameH, d.csp, d.bitdepth, d.inPicStruct,
            RGY_MEM_TYPE_GPU);
        const RGYFrameInfo bandOut(cfg.width, cheapFrameH, d.csp, d.bitdepth,
            RGY_PICSTRUCT_FRAME, RGY_MEM_TYPE_GPU);
        const auto bandTimebase = rgy_rational<int>(cfg.fpsDen, cfg.fpsNum * 2);
        const auto bandBaseFps  = rgy_rational<int>(cfg.fpsNum, cfg.fpsDen);
        const bool useBwdif = (cfg.splitDif == RTGMDIF_SPLITDIF_BWDIF);
        RGY_ERR sts = RGY_ERR_NONE;
        if (useBwdif) {
            auto prm = std::make_shared<NVEncFilterParamBwdif>();
            prm->bwdif.enable = true;
            prm->bwdif.mode   = VppBwdifMode::Bob;
            prm->frameIn  = bandIn;
            prm->frameOut = bandOut;
            prm->timebase = bandTimebase;
            prm->baseFps  = bandBaseFps;
            auto filter = std::make_unique<NVEncFilterBwdif>();
            sts = filter->init(prm, d.log);
            if (sts == RGY_ERR_NONE) {
                d.cheapDeint = std::move(filter);
            }
        } else {
            auto prm = std::make_shared<NVEncFilterParamYadif>();
            prm->yadif.enable = true;
            prm->yadif.log    = false;
            prm->yadif.mode = cfg.tff ? VPP_YADIF_MODE_BOB_TFF : VPP_YADIF_MODE_BOB_BFF;
            prm->frameIn  = bandIn;
            prm->frameOut = bandOut;
            prm->timebase = bandTimebase;
            prm->baseFps  = bandBaseFps;
            auto filter = std::make_unique<NVEncFilterYadif>();
            sts = filter->init(prm, d.log);
            if (sts == RGY_ERR_NONE) {
                d.cheapDeint = std::move(filter);
            }
        }
        if (sts != RGY_ERR_NONE || !d.cheapDeint) {
            printf("[error] %s::init failed: %s\n",
                useBwdif ? "NVEncFilterBwdif" : "NVEncFilterYadif",
                tchar_to_string(get_err_mes(sts)).c_str());
            return false;
        }
        printf("rtgmdif: cheap dif %s, band %d rows, bob %s\n", useBwdif ? "bwdif" : "yadif",
            cheapFrameH, cfg.tff ? "tff" : "bff");
    }

    // --- stage 0 ------------------------------------------------------------------------
    if (d.wantDeint) {
        auto prm = std::make_shared<NVEncFilterParamRtgmc>();
        prm->rtgmc = deintParams;
        prm->rtgmc.enable = true;
        prm->frameIn  = RGYFrameInfo(cfg.width, chainFrameH, d.csp, d.bitdepth, d.inPicStruct,
            RGY_MEM_TYPE_GPU);
        prm->frameOut = prm->frameIn;
        // Stage 0 of the RTGMC chain is RtgmcBob, which doubles baseFps (25 -> 50).
        prm->baseFps  = rgy_rational<int>(cfg.fpsNum, cfg.fpsDen);
        // Output timebase: one unit == one 50p frame. The upstream pipeline passes its output
        // timebase here for exactly the same reason (NVEncCore.cpp, createFilter for
        // VppType::CL_RTGMC).
        prm->timebase = rgy_rational<int>(cfg.fpsDen, cfg.fpsNum * 2);
        prm->bOutOverwrite = false;
        prm->sharedAnalysisMode = false;  // standalone use: the filter runs its own analysis

        d.rtgmc = std::make_unique<NVEncFilterRtgmc>();
        const auto sts = d.rtgmc->init(prm, d.log);
        if (sts != RGY_ERR_NONE) {
            printf("[error] NVEncFilterRtgmc::init failed: %s\n", tchar_to_string(get_err_mes(sts)).c_str());
            return false;
        }
        printf("stage0 : %s\n", tchar_to_string(prm->print()).c_str());
        printf("stage0 : priming %d source frames of temporal delay\n",
            d.rtgmc->requiredPrimingSourceFrames());
    }

    // --- stage 1 ------------------------------------------------------------------------
    if (d.wantClean) {
        const auto cleanBaseFps = rgy_rational<int>(cfg.fpsNum * 2, cfg.fpsDen);
        const auto cleanFrameIn = RGYFrameInfo(cfg.width, chainFrameH, d.csp, d.bitdepth,
            RGY_PICSTRUCT_FRAME, RGY_MEM_TYPE_GPU);
        const auto sts = d.clean.init(cleanParams, cleanFrameIn, cleanBaseFps, d.log, d.stream);
        if (sts != RGY_ERR_NONE) {
            printf("[error] clean chain init failed: %s\n", tchar_to_string(get_err_mes(sts)).c_str());
            return false;
        }
        d.clean.setSink([this](RGYFrameInfo *f) -> bool { return m_impl->sink(f); });
    }

    // The DLL has its own stdout FILE object (each module links the CRT statically), so with
    // the output redirected its startup lines would otherwise sit in a block buffer and only
    // appear after the caller's. One flush here keeps the combined log in the order things
    // actually happened; this is the only place that prints a burst.
    fflush(stdout);

    d.inited = true;
    return true;
}

bool RtgmDifImpl::handdif(void *pIn) {
    Impl &d = *m_impl;
    if (!d.inited) {
        printf("[error] RtgmDif::handdif() called before init().\n");
        return false;
    }

    // Whatever the caller did not read is retired here: the filters' frames are only valid for
    // the duration of the call that produced them, so the batch cannot outlive the next one.
    d.retireBatch();

    if (pIn == nullptr) {
        return d.flush();
    }
    if (d.flushed) {
        printf("[error] handdif() called with a frame after the flush.\n");
        return false;
    }

    // Copy the caller's frame into the ring. The filters keep references to the frames they are
    // given (stage 0 still holds the last few for its temporal lookahead), so the caller's
    // buffer cannot be adopted directly.
    auto &inBuf = *d.inBufs[d.inPut];
    d.inPut = (d.inPut + 1) % (int)d.inBufs.size();
    {
        const uint8_t *src = (const uint8_t *)pIn;
        int64_t offset = 0;
        for (int i = 0; i < RTGMDIF_PLANES; i++) {
            const size_t rowBytes = (size_t)d.geom[i].width * d.bytesPerPix;
            const auto cudaerr = cudaMemcpy2DAsync(inBuf.frame.ptr[i], (size_t)inBuf.frame.pitch[i],
                src + offset, rowBytes, rowBytes, d.geom[i].height,
                cudaMemcpyDeviceToDevice, d.stream);
            if (cudaerr != cudaSuccess) {
                printf("[error] cudaMemcpy2DAsync (input copy) failed: %s\n",
                    cudaGetErrorString(cudaerr));
                return false;
            }
            offset += (int64_t)rowBytes * d.geom[i].height;
        }
    }

    RGYFrameInfo inFrame = inBuf.frame;
    inFrame.picstruct    = d.wantDeint ? d.inPicStruct : RGY_PICSTRUCT_FRAME;
    inFrame.timestamp    = d.inIndex * d.inFrameDuration;
    inFrame.duration     = d.inFrameDuration;
    inFrame.inputFrameId = (int)d.inIndex;
    d.inIndex++;

    // With a split each side gets its own crop of the input, and the cheap one is fed first
    // because it is the side that runs ahead: the composite takes its frames by output number,
    // and the queue absorbs however far ahead it gets.
    RGYFrameInfo chainFrame = inFrame;
    if (d.wantDeint && d.wantSplit) {
        auto &cheapBuf = *d.cheapInBufs[d.cheapInPut];
        d.cheapInPut = (d.cheapInPut + 1) % (int)d.cheapInBufs.size();
        if (!d.copyBandIn(cheapBuf.frame, inFrame, d.cheapProcY0, d.cheapProcY1)) {
            return false;
        }
        cheapBuf.frame.picstruct    = inFrame.picstruct;
        cheapBuf.frame.timestamp    = inFrame.timestamp;
        cheapBuf.frame.duration     = inFrame.duration;
        cheapBuf.frame.inputFrameId = inFrame.inputFrameId;
        RGYFrameInfo *yOut[RTGMDIF_MAX_OUT_FRAMES] = { nullptr };
        int yOutNum = 0;
        const auto ysts = d.cheapDeint->filter(&cheapBuf.frame, yOut, &yOutNum, d.stream);
        if (ysts != RGY_ERR_NONE) {
            printf("[error] cheap-dif failed at input frame %d: %s\n", (int)d.inIndex - 1,
                tchar_to_string(get_err_mes(ysts)).c_str());
            return false;
        }
        if (!d.queueCheapOut(yOut, yOutNum)) {
            return false;
        }

        auto &cBuf = *d.chainInBufs[d.chainInPut];
        d.chainInPut = (d.chainInPut + 1) % (int)d.chainInBufs.size();
        if (!d.copyBandIn(cBuf.frame, inFrame, d.chainProcY0, d.chainProcY1)) {
            return false;
        }
        cBuf.frame.picstruct    = inFrame.picstruct;
        cBuf.frame.timestamp    = inFrame.timestamp;
        cBuf.frame.duration     = inFrame.duration;
        cBuf.frame.inputFrameId = inFrame.inputFrameId;
        chainFrame = cBuf.frame;
    }

    if (d.wantDeint) {
        RGYFrameInfo *out[RTGMDIF_MAX_OUT_FRAMES] = { nullptr };
        int outNum = 0;
        const auto sts = d.rtgmc->filter(&chainFrame, out, &outNum, d.stream);
        if (sts != RGY_ERR_NONE) {
            printf("[error] stage0 failed at input frame %d: %s\n", (int)d.inIndex - 1,
                tchar_to_string(get_err_mes(sts)).c_str());
            return false;
        }
        for (int j = 0; j < outNum; j++) {
            if (!d.feedStage1(out[j])) {
                return false;
            }
        }
    } else if (!d.feedStage1(&inFrame)) {
        return false;
    }
    return true;
}

int RtgmDifImpl::getoutputsize() {
    return (m_impl == nullptr) ? 0 : (int)m_impl->batch.size();
}

bool RtgmDifImpl::getoutputbuf(int idx, void *pOut) {
    Impl &d = *m_impl;
    if (pOut == nullptr) {
        printf("[error] getoutputbuf() called with a null destination.\n");
        return false;
    }
    if (idx < 0 || idx >= (int)d.batch.size()) {
        printf("[error] getoutputbuf(%d) is out of range; the batch holds %d frame(s).\n",
            idx, (int)d.batch.size());
        return false;
    }
    const auto &frame = d.outPool[d.batch[idx]]->frame;
    uint8_t *dst = (uint8_t *)pOut;
    int64_t offset = 0;
    for (int i = 0; i < RTGMDIF_PLANES; i++) {
        const size_t rowBytes = (size_t)d.geom[i].width * d.bytesPerPix;
        const auto cudaerr = cudaMemcpy2DAsync(dst + offset, rowBytes,
            frame.ptr[i], (size_t)frame.pitch[i], rowBytes, d.geom[i].height,
            cudaMemcpyDeviceToDevice, d.stream);
        if (cudaerr != cudaSuccess) {
            printf("[error] cudaMemcpy2DAsync (output download) failed: %s\n",
                cudaGetErrorString(cudaerr));
            return false;
        }
        offset += (int64_t)rowBytes * d.geom[i].height;
    }
    // Only a stream the module owns has to be waited for here: the caller has no handle on it
    // and would otherwise not know when pOut is complete. A caller-supplied stream is left
    // alone, so the caller can queue its download straight behind this copy on that stream.
    if (d.ownsStream) {
        const auto cudaerr = cudaStreamSynchronize(d.stream);
        if (cudaerr != cudaSuccess) {
            printf("[error] cudaStreamSynchronize failed: %s\n", cudaGetErrorString(cudaerr));
            return false;
        }
    }
    d.taken[idx] = 1;
    return true;
}

int RtgmDifImpl::inputFrameBytes() const {
    return (m_impl == nullptr) ? 0 : (int)m_impl->inFrameBytes;
}

int RtgmDifImpl::outputFrameBytes() const {
    return (m_impl == nullptr) ? 0 : (int)m_impl->outFrameBytes;
}

int RtgmDifImpl::framesIn() const {
    return (m_impl == nullptr) ? 0 : (int)m_impl->inIndex;
}

int RtgmDifImpl::framesOut() const {
    return (m_impl == nullptr) ? 0 : (int)m_impl->outTotal;
}

bool RtgmDifImpl::inited() const {
    return m_impl != nullptr && m_impl->inited;
}

void RtgmDifImpl::close() {
    if (m_impl == nullptr || !m_impl->inited) {
        return;
    }
    Impl &d = *m_impl;
    // The chain and the filters own device memory and CUDA events, and tearing them down while
    // their work is still queued would be a use-after-free. So this one synchronise is always
    // needed, including on a caller-supplied stream that is otherwise never blocked.
    if (d.stream != nullptr) {
        cudaStreamSynchronize(d.stream);
    }
    d.clean.clear();
    d.rtgmc.reset();
    d.outPool.clear();
    d.inBufs.clear();
    if (d.ownsStream && d.stream != nullptr) {
        cudaStreamDestroy(d.stream);
    }
    d.stream = nullptr;
    d.inited = false;
    if (d.droppedUnread > 0) {
        printf("[warn] %d output frame(s) were never read during this run.\n", d.droppedUnread);
    }
}

// ---------------------------------------------------------------------------------------
// the exported surface
// ---------------------------------------------------------------------------------------

// The interface's destructor, out of line so that its vtable has a home here - in the module,
// not in every translation unit that includes the header.
RtgmDif::~RtgmDif() = default;

// The two exported symbols, and the only two. Neither carries dllexport: RtgmDif.def names them
// and the linker resolves them from there, so there is no attribute here that could disagree
// with the declaration in the header.
//
// create() catches everything: a C++ exception must not cross this boundary, not even the
// std::bad_alloc from the allocation itself. nullptr is the only failure channel the caller has
// to check, and there is nothing half-constructed to clean up if it is returned.
extern "C" RtgmDif *rtgmdif_create() {
    try {
        return new RtgmDifImpl();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void rtgmdif_destroy(RtgmDif *dif) {
    // Deleted through the interface's virtual destructor, inside this module, so that the
    // allocator that made the object is the allocator that frees it.
    delete dif;
}
