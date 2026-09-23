# Rtgmc

RTGMC deinterlacing (50i/60i → 50p/60p) as a standalone library.

`RtgmDif` takes the RTGMC filter out of [rigaya's
NVEnc](https://github.com/rigaya/NVEnc) and packages it as a module you can
call from your own program: device frames in, device frames out, on a CUDA
stream you own. It is the filter itself, not a front end - it does not read
files, decode, encode or touch host memory on your behalf.

## What is here

    RtgmDif/          the module: a shared library with a C API
    RtgmDifTest/      a deliberately dumb driver that loads it (raw YUV in/out)

## The interface

Two exported functions, everything else goes through the object they return:

```c
RtgmDif *dif = rtgmdif_create();     // nullptr if the module could not start
dif->configure(cfg);                 // RtgmDifConfig: size, csp, flow, device, knobs
dif->init();

for (;;) {
    dif->handdif(pDeviceIn, stream);         // hand it a device frame
    for (;;) {
        void *out;
        if (dif->getoutputbuf(&out) == 0) break;   // drained
        /* out is the next finished frame, in order */
    }
}
dif->flush();                        // drain the tail, then
rtgmdif_destroy(dif);
```

All work is enqueued on the `cudaStream_t` you pass in - the module never
synchronises behind your back. Finished frames come out of a ring buffer
(32 frames), which is what lets a caller read a batch after a flush.

Seven flows, from single-pass to the two-pass ones that run a second,
clean-up pass over the progressive result:

    fast_deint   fast_both   fast_opt   faster_nn1   slow   slower   slow_both

`slow_both` is the two-stage configuration QTGMC itself describes; the
`*_both` flows are the ones that remove the residual field-parity shimmer a
single pass leaves behind. A `lowLatency` switch trades some of the search
work for output latency. See `RtgmDif.h` - it documents every knob, with the
measurements behind them.

Input is device memory in yuv420p, yuv422p or yuv444p. yuv422p works but is
not recommended: NVEncC never feeds 422 to RTGMC (it normalises to 444 first),
and feeding it directly makes static areas shimmer.

## Building

Requires the CUDA 11.5 toolkit and its headers; the module links the CUDA
runtime statically and loads the driver at run time.

**Windows** (MSVC 2019, CMake 3.18+):

    cmake -S . -B build -A x64
    cmake --build build --config RelWithDebInfo

**Linux** (CMake 3.18+, gcc-10):

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_COMPILER=gcc-10 -DCMAKE_CXX_COMPILER=g++-10 \
          -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10
    cmake --build build -j

gcc-10 rather than the system default matters: CUDA 11.5's front end cannot
parse libstdc++ 11's `<functional>` (it stops at gcc 10), and the failure is a
confusing "parameter packs not expanded" error rather than a version warning.

The build produces the module under `Output/` (about 307 MB on Windows,
267 MB on Linux), plus `nnedi3_weights.bin` next to it - the EDI stage loads
that from the module's own directory at run time.

## Licence

MIT - see `LICENSE`. The same licence as the upstream project this is built on.

## Provenance

**This project is a derivative work of [rigaya's
NVEnc](https://github.com/rigaya/NVEnc)**, which is MIT licensed. Everything
under `RtgmDif/NVEnc/` comes from that project, including the RTGMC filter at
the heart of this one; the MIT permission text is kept verbatim at the top of
each of those files, and this project is released under the same terms.

The packaging - the C API, the device-in/device-out interface, the CMake build
for both platforms, the test driver - is this project's contribution, along
with the modifications needed to make the filter stand alone.

Some files here are not covered by this project's MIT grant: NVIDIA's SDK
headers, and the nnedi3 weights. `NOTICE` lists them, along with what was
changed relative to upstream - read it before you redistribute.
