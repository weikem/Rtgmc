# 构建

[Readme](Readme.cn.md) | **Build** | [English](Build.en.md)

## 环境要求

| 项目 | 要求 |
|---|---|
| CUDA Toolkit | **11.5**（`nvcc` 在 PATH 上，或用 `-DCUDAToolkit_ROOT=<path>` 指定） |
| C++ 编译器 | Windows：MSVC 2019 或更新；Linux：**gcc/g++ 10**（见下） |
| CMake | 3.18 或更新 |
| NVIDIA 驱动 | 不参与构建。运行期通过驱动加载程序（`libcuda.so` / `nvcuda.dll`）访问 GPU |

不需要 NVEnc 仓库，也不需要 Video Codec SDK：所需头文件已 vendored 在 `RtgmDif/NVEnc/` 下。

### Linux 上必须用 gcc-10

不能用系统默认的 gcc-11（Ubuntu 22.04 即如此）：**CUDA 11.5 的 nvcc 前端解析不了 libstdc++ 11 的 `<functional>` 等头文件**，官方从 CUDA 11.8 起才支持 gcc 11。报错是 `parameter packs not expanded with '...'` 这种语法级错误，`-allow-unsupported-compiler` **绕不过去**——它只放开版本判定，不解决头文件不兼容。

安装：

```bash
sudo apt-get install -y g++-10
```

## Windows

```bat
cmake -S . -B build -A x64
cmake --build build --config RelWithDebInfo
```

`Release` 同样可用；`RelWithDebInfo` 会保留 PDB，排查问题更方便（产物路径分别为 `Output\Release\` 与 `Output\RelWithDebInfo\`）。

## Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc-10 -DCMAKE_CXX_COMPILER=g++-10 \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10
cmake --build build -j$(nproc)
```

三个编译器选项**都要给**：只设 `CMAKE_CXX_COMPILER` 时，`nvcc` 仍会自己去找默认的 `g++`（也就是 11），然后以同样的方式失败。

### 只构建其中一个项目

两个 `CMakeLists.txt` 相互独立，可以单独配置。`RtgmDifTest` 默认按 `../RtgmDif` 的相对路径找模块，源目录并列时无需额外参数：

```bash
cd RtgmDif && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release ... && cmake --build build -j
```

只构建模块（跳过参考驱动）：

```bash
cmake --build build --target RtgmDif
```

## 产物

两个平台都输出到 `Output/`：

| 文件 | Windows | Linux |
|---|---|---|
| 模块 | `RtgmDif.dll` + 导入库 `RtgmDif.lib` | `libRtgmDif.so` |
| 参考驱动 | `RtgmDifTest.exe` | `RtgmDifTest` |
| EDI 权重 | `nnedi3_weights.bin`（构建后自动拷贝到模块旁） | 同左 |

实测大小（五档 CUDA 架构全部编入）：

| 文件 | 大小 |
|---|---|
| 模块（Windows） | 约 307 MB |
| 模块（Linux） | 约 267 MB |
| `nnedi3_weights.bin` | 13.5 MB |

模块的导出符号只有 `rtgmdif_create` 和 `rtgmdif_destroy` 两个（Windows 由 `RtgmDif.def` 控制，Linux 由 `RtgmDif.map` 这个 version script 控制），其余符号全部内部化。

## CUDA 架构覆盖

```cmake
set(CMAKE_CUDA_ARCHITECTURES 50-real 61-real 75-real 86-real 50-virtual)
```

两条规则决定了覆盖面：

- **cubin（SASS）同 major 内向上兼容**：为 X.y 编的 cubin 可运行在 X.z（z >= y），但**不能**向下兼容，也不能跨 major。
- **PTX 跨 major 向前兼容**：一份 `compute_50` 的 PTX 可在任何算力 >= 5.0 的卡上由驱动 JIT 编译执行。

| 架构 | 代表显卡 | 途径 |
|---|---|---|
| sm_50 / 52 / 53 | GTX 9xx、Tesla M40、Quadro M 系列 | 原生（`50-real`） |
| sm_60 | Tesla P100 | PTX JIT |
| sm_61 / 62 | **Quadro P2000**、GTX 10xx、Tesla P40、Jetson TX2 | 原生（`61-real`） |
| sm_70 / 72 | V100、Titan V、Jetson Xavier | PTX JIT |
| sm_75 | RTX 20xx、Tesla T4、Quadro RTX | 原生（`75-real`） |
| sm_80 | A100、A30 | PTX JIT |
| sm_86 / 87 / 89 | **Quadro A2000**、**RTX 3050/3060**、A40、Jetson Orin、**RTX 4050/4060**（Ada） | 原生（`86-real`） |
| sm_90 | H100、H200 | PTX JIT |
| sm_100 / 101 / 103 / 120 | B100/B200、**RTX 5060** 及 RTX 50 全系 | PTX JIT |

加粗的是本项目实测过的卡。

两点容易误解的地方：

1. **RTX 40 系（sm_89）走的是 `86-real`，是原生 SASS，不是 JIT**——同 major 向上兼容。功能完全正常，只是用不到 Ada 专属指令的调优，性能差异通常很小。
2. **RTX 50 系（sm_120）必须靠 PTX**：CUDA 11.5 的 `nvcc --list-gpu-arch` 只到 `compute_87`，压根编不出它的 cubin。删掉 `50-virtual` 会让这些卡直接报 `no kernel image is available for execution on the device`。

### 体积构成与裁剪

体积几乎全部来自 PTX 而非 SASS：四份 cubin 合计约 136 MB，单份 `compute_50` PTX 就有约 171 MB（PTX 是文本 IR，比机器码大得多）。

| 目标 | 设置 | 模块大小 |
|---|---|---|
| 全系兼容（默认） | `50-real 61-real 75-real 86-real 50-virtual` | 约 307 MB (Win) / 267 MB (Linux) |
| 不要 RTX 50 系 | `50-real 61-real 75-real 86-real` | 约 136 MB |
| 只要 Pascal + Ampere 及 Ada | `61-real 86-real` | 约 68 MB |
| 升级到 CUDA 12.8+ 后全原生 | `61 86 89 120` | 约 136 MB，无 PTX |

最后一行是长期解：换到能直接产出 `sm_89` / `sm_120` cubin 的工具链后，PTX 兜底可以彻底删掉，Ada 与 Blackwell 也能拿到各自的原生调优。

## 资源占用

- **磁盘**：`Output/` 约 300 MB；`build/` 的中间产物需要数 GB 余量（74 个 `.cu` 乘以五档架构的目标文件）。
- **内存**：每多一档 `-gencode` 都会显著抬高 `nvcc`/`cicc` 的峰值内存，RTGMC 的 `.cu` 又普遍很大，因此 `-j` 不建议超过物理核数（8 核机器上 `-j8` 实测正常）。
- **时间**：8 核 `-j8` 全量构建在数分钟量级；改动单个 `.cu` 后的增量构建约一分钟（链接 300 MB 的模块本身就要一小会儿）。

## 验证构建

`RtgmDifTest` 是 raw YUV 的参考驱动，构建完可直接运行：

```bash
./RtgmDifTest -i input_1080i50.yuv -o output_1080p50.yuv \
              --width 1920 --height 1080 --in-csp yuv422p \
              --fps 25 --flow slow_both --low-latency
```

预期：输入 4 帧产出 8 帧，输出文件大小 = 输入帧数 x 2 x 单帧字节数。程序会打印每一批的 in/out 账目与显存占用：

```
test   : flow slow_both, yuv422p 1920x1080, input frame 4147200 bytes, output frame 4147200 bytes
test   : input file 16588800 bytes (4 frame(s) at that size)
mem    : init   after     0 in      0.0 MiB on the card      +0.0 MiB since start
```

若 `init` 失败并报 `RGY_ERR_INVALID_PARAM`，先检查模块旁边是否有 `nnedi3_weights.bin`。

### 命令行参数

输入 / 输出：

| 参数 | 说明 |
|---|---|
| `-i, --input <file>` | 输入 raw YUV 文件（必填） |
| `-o, --output <file>` | 输出文件，csp 与输入相同。可省略——省略时链条照常运行但不写盘（纯计时） |
| `--width <n>` | 帧宽，默认 1920 |
| `--height <n>` | 帧高，默认 1080 |
| `--in-csp <name>` | `yuv422p` / `yuv420p` / `yuv444p`，默认 `yuv444p`；实际使用以 `yuv422p` 为主 |
| `--fps <n>` | 输入帧率，默认 25 |
| `--bff` | 底场优先（默认顶场优先） |
| `--frames <n>` | 读入 n 帧后停止，从读到的第一帧算起（默认整个文件）。`--skip` 丢弃的帧也计入此数，所以 `--frames` 必须大于 `--skip`：`--skip 400 --frames 430` 会跑 430 帧、保留最后 30 帧 |
| `--skip <n>` | 前 n 帧过链条但不写盘也不计时，用于跳到片子的困难段落；计时部分开始时链条已经预热 |
| `--stats-every <n>` | 每 n 输入帧打印一次 in/out 账目，默认 16；`1` 即每帧，是读出流水线延迟的方式 |
| `--device <n>` | CUDA 设备序号，默认 0 |
| `--log-level <n>` | 0 静默 .. 5 debug，默认 3 |
| `--own-stream` | 让模块自建并持有流，而不是传入一个（更慢：此时模块必须同步输出拷贝，因为调用方没有可排序的句柄） |

流程：

| 参数 | 说明 |
|---|---|
| `--flow <name>` | `fast_deint` / `fast_both`（默认）/ `fast_opt` / `faster_nn1` / `slow` / `slow_both` / `slower`，含义见 [Readme](Readme.cn.md#flows) |

第一趟（去隔行）：

| 参数 | 说明 |
|---|---|
| `--deint-pel <n>` | 1 = half-pel，2 = quarter-pel（预设值），-1 = 预设 |
| `--deint-pel-search <n>` | 子像素搜索步长，-1 = 预设 |
| `--deint-search-refine <n>` | 首次匹配后的细化次数，-1 = 预设 |
| `--deint-search-param <n>` | 搜索窗口大小，-1 = 预设 |
| `--deint-analyze-delta <n>` | 1 = 只算一个帧距的矢量（搜索量减半），-1 = 预设 |
| `--deint-overlap <n>` | 块重叠；步长为 blockSize - overlap，越小块越少、运动场越粗，-1 = 预设 |
| `--deint-tr0 <n>` | 第一趟的搜索时域半径，-1 = 预设（slow 为 2，faster 为 1） |
| `--deint-tr1-delta <n>` | TR1 时域半径，0 = 关闭该级，-1 = 预设 |
| `--deint-tr2-delta <n>` | TR2 时域半径，0 = 关闭该级（整级的延迟远大于半径本身带来的延迟），-1 = 预设 |
| `--deint-retouch-smode <n>` | 第一趟的重新锐化模式，0 = 关闭，-1 = 预设（两趟流程的 clean pass 以 0 运行） |
| `--low-latency` | 低延迟变体总开关，见 [Readme](Readme.cn.md#关键开关) |
| `--spatial-early-sad <n>` | 基础矢量 SAD 低于 n 时跳过空间细化：-1 = 关闭（默认），0 = 仅对精确匹配跳过 |

第二趟（clean 链，仅两趟流程有效）：

| 参数 | 说明 |
|---|---|
| `--clean-preset <n>` | 用该预设构建第二趟（0 placebo、1 veryslow、2 slower、3 slow、4 medium、5 fast ... 10 draft；-1 = 与第一趟相同） |
| `--clean-analyze-delta <n>` | clean 链的 analyze 帧距（1 约等于搜索量减半）；-1 = 预设，0 = 由 TR1/TR2 决定 |
| `--clean-tr1-delta <n>` | clean 链 TR1 delta，0 = 关闭，-1 = 预设 |
| `--clean-tr2-delta <n>` | clean 链 TR2 delta，0 = 关闭，-1 = 预设 |
| `--clean-rep2-thin <n>` | clean 链 REP2 `repairThin`，0 = 关闭，-1 = 预设 |

分带处理：

| 参数 | 说明 |
|---|---|
| `--split-y <n>` | 只让完整链条处理一部分行，其余交给廉价去隔行器。n 是分界行（必须为偶数），0（默认）= 不分割 |
| `--seldif <name>` | 另一侧的廉价去隔行器：`yadif`（默认）或 `bwdif`；可加 `:top` / `:bottom` 后缀同时指定侧别，如 `--seldif bwdif:bottom` |
| `--seldif-side <name>` | 该廉价去隔行器占据哪一侧：`top`（默认）表示行 `[0,n)` 归它、`[n,height)` 归链条 |

其他：

| 参数 | 说明 |
|---|---|
| `-h, --help` | 打印帮助 |

## 部署

- **`nnedi3_weights.bin` 必须与模块放在一起**。EDI 阶段会按"当前工作目录 -> 模块所在目录 -> 可执行文件所在目录"的顺序查找它；找不到时 `init` 直接失败（`RGY_ERR_INVALID_PARAM`），不会退化运行。
- **CUDA 运行期是静态链接的**（`CMAKE_CUDA_RUNTIME_LIBRARY = Static`），目标机器不需要安装 CUDA Toolkit，只需要可用的 NVIDIA 驱动。
- **驱动用户态与内核态版本要一致**。不一致时 `nvidia-smi` 会报 `Driver/library version mismatch`，CUDA 通常仍能运行，但重启对齐更稳妥。
- **NVRTC / NPP / NVVFX / NGX / NVML / ONNX 等可选组件全部关闭**（见 `RtgmDif/NVEnc/rtgmc_config.h`），运行时不需要它们对应的库。

## 常见问题

**Q：必须装 CUDA Toolkit 吗？**
构建时必须；运行时不需要，只要驱动。模块把 CUDA 运行期静态链接进去了。

**Q：可以换成 CUDA 12.x 吗？**
可以，但要同步调整架构列表：改用 `61 86 89 120` 让 Ada 与 Blackwell 都拿到原生 cubin，然后删掉 `50-virtual`。这样模块降到约 136 MB，且失去对 Maxwell / Volta / sm_80 / Hopper 的覆盖——需要它们就把对应架构也列进去。

**Q：`-allow-unsupported-compiler` 是干什么的？**
`nvcc` 对宿主编译器版本的判定比较保守，这一项让它接受。两个平台都加着，Linux 上以 gcc-10 构建时是保险，Windows 上用于新版 MSVC。

**Q：为什么 `RtgmDif.map` 和 `RtgmDif.def` 两个导出控制文件？**
Windows 用 `.def`（模块定义文件），Linux 用 `.map`（version script），两者的导出清单一致：只有 `rtgmdif_create` / `rtgmdif_destroy`。不加 `RtgmDif.map` 的话，`.so` 会导出全部约 2000 个 NVEncCore 符号。

**Q：`rgy_config.h` 是什么？为什么 Windows 构建从来没提过它？**
它是 `rgy_version.h` 在**非 Windows 分支**才 include 的平台配置头，上游仓库里有，vendored 子集没有（Windows 构建不需要，所以一直没暴露）。本仓库补了一份最小版本。

**Q：构建能并行到多少？**
`-j` 不要超过物理核数。每档 `-gencode` 都会抬高 `nvcc`/`cicc` 的峰值内存，RTGMC 的 `.cu` 普遍很大，`-j` 开过头会因内存不足失败——历史上正是这个原因让原来的工程文件只编单一架构。
