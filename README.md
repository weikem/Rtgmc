# Rtgmc

RTGMC deinterlacing (50i/60i -> 50p/60p) as a standalone shared library.

针对广播电视中的字幕和滚动字幕进行特殊优化。

## 优势对比

### 相比 yadif / bwdif

- 解决字幕（尤其是滚动字幕）边缘波动。
- 消除单趟去隔行固有的场奇偶闪烁。
- 锐度补齐。

### 相比 QTGMC

- **Slow 预设**：优化运动搜索与时域流水线调度。
- **Slow_both 预设**：内置二次抗抖。
- 提供 **split 模式混合滤镜**：上半部分 bwdif，下半部分 Rtgmc。

## 说明

- `RtgmcDif` 输入第 4 帧 50i 时，才输出第 1 帧 50p。
- 1080p 422p、`slow_both` 预设显存占用约 2G。
- Windows 侧 MSVC 2019 + CUDA 11.5 + P2000，编译测试通过。
- Windows 侧 MSVC 2019 + CUDA 11.5 + RTX 3080，编译测试通过。
- Ubuntu 22.04 + g++-10 + CUDA 11.5.119 + RTX 5000 Ada，编译测试通过。

## 文档

| 文档 | 链接 | 说明 |
|---|---|---|
| English | [Readme.en.md](Readme.en.md) | what it is, the API, the flows, the knobs |
| 中文 | [Readme.cn.md](Readme.cn.md) | 项目说明、接口、流程与配置项 |
| Building | [Build.en.md](Build.en.md) / [Build.cn.md](Build.cn.md) | requirements, both platforms, CUDA architecture coverage, verification, deployment |
| Interface reference | [RtgmDif/RtgmDif.h](RtgmDif/RtgmDif.h) | every config field, with the measurements behind it |
| Licence and provenance | [NOTICE](NOTICE) | a derivative work of rigaya's NVEnc; read it before redistributing |

## 性能测试

| 滤镜 | 预设 | 硬件 | 速度（50i 输入，fps） |
|---|---|---:|---:|
| QTGMC | Slow | i9-7920X@2.90GHz | 3 fps/s |
| RtgmcDif | Slow | RTX3080 | 60 fps/s |
| RtgmcDif | Slow | Ada5000 | 125 fps/s |
| RtgmcDif | Slow_both | RTX3080 | 40 fps/s |
| RtgmcDif | Slow_both | Ada5000 | 72 fps/2 |

## 效果对比,上yadif,下Rtgmcdif
yadif
![yadif](/sampleimage/yadif.gif)
Rtgmcdif
![rtgmc](/sampleimage/rtgmc.gif)
yadif
![rtgmc](/sampleimage/2/yadif600.jpg)
Rtgmcdif
![rtgmc](/sampleimage/2/rtgmc600.jpg)
