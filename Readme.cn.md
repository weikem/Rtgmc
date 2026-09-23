# Rtgmc

RTGMC 去隔行（50i/60i -> 50p/60p），以独立共享库的形式提供。

`RtgmDif` 把 [rigaya 的 NVEnc](https://github.com/rigaya/NVEnc) 中的 RTGMC 去隔行滤波器提取出来，封装成可由你自己的程序调用的模块：**设备帧进、设备帧出**，全部工作排在你提供的 CUDA 流上。它是滤波器本身，不是前端——不读文件、不解码、不编码，也不代你搬运主机内存。

英文版见 [Readme.en.md](Readme.en.md)，构建细节见 [Build.cn.md](Build.cn.md)。

## 目录结构

    RtgmDif/          模块本体（共享库 + C API）；NVEnc 源码以 vendored 形式内嵌在 RtgmDif/NVEnc/
    RtgmDifTest/      参考驱动，刻意保持简单（raw YUV 进 / raw YUV 出）
    Output/           构建产物，两个平台相同
    RtgmDif/RtgmDif.h 完整接口文档（本项目的核心文档）

## 接口

导出面只有两个 C 函数，其余全部通过它们返回的对象调用：

```cpp
#include "RtgmDif.h"

RtgmDif *dif = rtgmdif_create();          // 失败返回 nullptr
if (!dif) { /* 模块未能启动 */ }

RtgmDifConfig cfg;                        // 成员初始化器带默认值（1080i50 场景）
cfg.flow   = RTGMDIF_FLOW_SLOW_BOTH;      // 见下方 flows
cfg.width  = 1920;
cfg.height = 1080;
cfg.csp    = RTGMDIF_CSP_YUV422P;         // 主力格式
dif->configure(cfg);                      // 只在 init() 之前有效

dif->init(stream);                        // 传入你自己的流；传 nullptr 则模块自建

dif->handdif(pDeviceIn);                  // 送一帧 50i（设备内存）
for (int i = 0; i < dif->getoutputsize(); i++)
    dif->getoutputbuf(i, pDeviceOut);     // 取出本批的 50p 帧，按序

dif->handdif(nullptr);                    // 冲刷：尾部帧同样按批读取
rtgmdif_destroy(dif);
```

几个要点：

- **一进两出**：每送 1 帧 50i 产出 2 帧 50p。开头有数帧预热延迟，等量的尾部帧由冲刷释放。
- **批语义**：一次 `handdif()` 释放的帧构成"当前批"，用 `getoutputsize()` / `getoutputbuf()` 读取；下一次调用会替换它，所以要在那之前读完。
- **不阻塞调用方的流**：模块把输入拷贝、链条中每个 kernel、输出拷贝全部排进 `init()` 传入的流，从不主动同步，因此可与上下游背靠背排队。唯一例外是传入 `nullptr`（模块自建阻塞流），此时 `getoutputbuf()` 必须等待。
- **内存布局**：紧密排列的 planar，与 raw YUV 文件布局完全一致（Y、U、V 依次，行间无填充），可直接送入，无需重排。
- **输入帧可立即复用**：`handdif()` 会把帧拷进模块自己的环形缓冲，返回后调用方即可重用 `pIn`。
- **输出环容量** `outputPoolFrames`（默认 32）需要覆盖冲刷一次性释放的时域尾部；实测该尾部为 26 帧，32 有少量余量。容量不足时模块会明确报错，不会静默丢帧。

`RtgmDif.h` 记录了每个配置字段的语义、取值范围与背后的实测数据，是使用前值得通读的第一份文档。

## Flows

七种预设流程，从单趟到两趟：

| flow | 预设 | 结构 | 说明 |
|---|---|---|---|
| `fast_deint` | fast | 单趟 | 纯去隔行 |
| `fast_both` | fast | 两趟 | 加 clean pass（QTGMC 的第二段） |
| `fast_opt` | fast | 两趟 | 更强的 nnedi3 窗口（nnsize 3）+ Slower 的 TR 结构（TR1 2 / TR2 1）。1080i 实测：静态区残留场奇偶交替 0.0525 -> 0.0422，代价约 +0.1% |
| `faster_nn1` | faster | 单趟 | 最省的流程，为滚动字幕场景所加（`faster` 本身弱于 `fast`，但 nnsize 1 时 EDI 是强的那一档） |
| `slow` | slow | 单趟 | 对应 QTGMC(preset="Slow") |
| `slower` | slower | 单趟 | 对应 QTGMC(preset="Slower") |
| `slow_both` | slow | 两趟 | QTGMC 描述的完整两段式：`QTGMC(50i, Slow)` 再做一次 `QTGMC(50p, InputType=1, Sharpness=0)` |

单趟流程的画面不是最终成品：clean pass 才是消除残留场奇偶 shimmer 的那一步。单趟结果仍有残留时，用对应的两趟流程。

## 输入格式

- 色彩空间：**`yuv422p` 是主力格式**，`yuv420p` 与 `yuv444p` 同样支持。滤波器在输入的色彩空间上直接运行（**不做转换**），输出与输入 csp、位深一致。
- 位深：**仅支持 8 bit**。
- 宽、高必须为偶数。

## 关键开关

| 字段 | 作用 |
|---|---|
| `lowLatency` | 低延迟变体总开关：搜索半径降到 1、analyze 只算距离 1 的矢量、TR 阶段不再等待它们并不执行的搜索、去掉场景切换回读流水线、TR2 关闭。1080p 422p 实测：单趟 `slow` 8.5 -> 2.5 输入帧（26 -> 18 ms/帧），两趟 `slow_both` 15.5 -> 4.0（40 -> 24 ms/帧）。两趟流程画面接近原流程（intra-pair 0.693 对 0.682，肉眼看不出差别）；单趟流程无论加多少时域旋钮都不是成品画面（0.917 起步，两趟从 0.693 起）——去隔行趟工作在**场**域，那里的垂直邻域是另一个时刻、矢量也最不准，clean 趟在**逐行**域重跑同一条链，所以单趟调不成两趟 |
| `splitY` / `splitDif` / `splitDifSide` | 只对一部分行跑完整链条，其余行交给廉价的 YADIF/BWDIF，用一条可能可见的接缝换时间。时间由链条拿到的**行数**决定，与哪一侧无关：1080p 422p 下约 0.046 ms/行 + 4 ms 固定开销。`splitY = 0`（默认）= 不分割，逐字节等同原行为 |
| `deintPel` / `deintPelSearch` / `deintSearchRefine` / `deintSearchParam` | 第一趟的运动搜索，帧耗时的大头：1080p 422p 上 `slow` 26.4 ms，换成 `fast` 的搜索设置（pel 1、pelSearch 1、refine 2）降到 13.6 ms，再叠加 32 宽块降到 7.7 ms |
| `deintTr0` / `deintTr1Delta` / `deintTr2Delta` | 第一趟的时域半径，即**延迟**旋钮。决定积压的不是半径本身，而是活跃时域**级数**乘以每级持有量：`fast` 与 `slow` 半径相同（tr0 2、tr1 1），`fast` 却短 2.5 输入帧，只因它的 TR2 关着 |
| `cleanAnalyzeDelta` / `cleanPreset` / `cleanTr1Delta` / `cleanTr2Delta` / `cleanRep2Thin` | 只影响第二趟（clean 链）。`cleanAnalyzeDelta 1` 是目前唯一显著改变耗时的旋钮：`slow_both` 43.2 -> 36.7 ms（-15%）。`cleanPreset` 可让第二趟用不同预设（常见的 Slow + Medium 组合），本树中 slow 与 medium 的差别只在 `pel`/`pelSearch`（2 -> 1） |

## 线程与显存

- **不用主机内存**：调用方负责文件 I/O 与主存<->显存拷贝，模块只碰设备内存。
- **一切都在调用方的流上**（除非 `init(nullptr)`），模块不创建额外线程。
- **显存**：两趟流程的 clean 链在 1080p 444p 下额外占用约 1.2 GiB；输出环按 `outputPoolFrames`（默认 32）计，每帧一份。

## 平台与显卡

一次构建覆盖 Maxwell 及以后的全部 NVIDIA 卡：CUDA 11.5 能编出的 cubin 全部编入，另加一份 `compute_50` PTX 兜底。细节与完整对照表见 [Build.cn.md](Build.cn.md#cuda-架构覆盖)。

已验证环境：

| 平台 | 环境 | 显卡 |
|---|---|---|
| Windows | MSVC 2019 + CUDA 11.5 | RTX 3080 |
| Linux | Ubuntu 22.04 + gcc-10 + CUDA 11.5.119 | RTX 5000 Ada |

## 构建

Windows 与 Linux 各一条命令，要求、产物、架构说明与验证方法见 [Build.cn.md](Build.cn.md)。

## 授权与来源

MIT，见 [LICENSE](LICENSE)。

**本项目是 [rigaya 的 NVEnc](https://github.com/rigaya/NVEnc) 的派生作品**：`RtgmDif/NVEnc/` 下的内容全部来自上游，其中 RTGMC 滤波器本身也是 rigaya 的工作，MIT 许可原文逐文件保留在文件头。本项目贡献的是打包部分——C API、设备进/设备出接口、两平台的 CMake 构建与参考驱动——以及为使其独立运行所需的修改。

少数文件不在本项目的 MIT 授权范围内（NVIDIA SDK 头文件、nnedi3 权重数据）。[NOTICE](NOTICE) 逐一列出它们，并给出相对上游的完整改动清单。再分发前请先读它。
