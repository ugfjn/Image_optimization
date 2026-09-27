# 图片优化工具

将图片批量提升至 **2K / 3K / 4K** 分辨率或按 **×4 / ×3 / ×2** 倍增尺寸的图片放大工具，提供 **PyQt5 图形界面**与**命令行**两种使用方式，内置 4 种放大引擎（经典插值 / GPU AI 超分 / CPU AI 超分）。

## 功能特性

- 三档目标分辨率：2K (2560×1440)、3K (3072×1728)、4K (3840×2160)，一次输入可同时输出多个版本
- 尺寸倍增（仅图形界面）：×4 / ×3 / ×2，宽高各乘以倍数，严格保持原始宽高比，不对齐标准分辨率
- 四种放大引擎可选：
  - `lanczos`（默认）—— Lanczos 高质量重采样 + UnsharpMask 锐化，速度快、零额外依赖
  - `realesrgan` —— Real-ESRGAN GPU 超分（Vulkan 加速，动漫特化模型；超大图自动切条带推理）
  - `edsr` —— AI 超分（画质最好，CPU 较慢）
  - `fsrcnn` —— AI 超分（速度接近实时，画质提升有限）
- 六种输出格式：`auto` / `jpeg` / `png` / `webp` / `tiff` / `bmp`，可强制统一输出格式
- 等比缩放，以目标分辨率为上限，不产生超尺寸图片（`--stretch` 可强制拉伸）
- 自动读取 EXIF 方向标记，手机照片不再旋转错位
- 透明通道安全：RGBA 图片输出 JPEG/BMP 时自动合成到白色背景，输出 PNG/WebP/TIFF 时保留透明
- WebP 保护：单边超过 16383px（libwebp 硬限制）时自动回退为 JPEG
- 支持单文件与目录批量处理；图形界面实时显示进度、处理日志与耗时预估，可随时取消
- AI 模型首次使用自动下载，缓存至本地 `models/` 目录

## 环境要求

- Windows（Real-ESRGAN 引擎与内置 PyQt5 运行库为 Windows 版）
- Python 3.8+

```bash
# 基础依赖（必装）
pip install Pillow PyQt5

# AI 超分引擎（可选，仅 edsr / fsrcnn 引擎需要）
pip install opencv-contrib-python numpy
```

> 项目已内置 `pyqt5_lib/` 完整 PyQt5 运行库：即使 pip 未安装 PyQt5，启动图形界面时也会自动回退加载内置版本。

## 图形界面（推荐）

```bash
python gui.py
```

界面分为四步：

1. **选择图片** —— 点击添加或将图片/文件夹直接拖入列表，支持缩略图预览
2. **处理方式** —— 点选引擎卡片（4 种引擎）、目标分辨率 2K/3K/4K（可多选）或尺寸倍增 ×4/×3/×2
3. **输出设置** —— 输出目录、输出格式（auto/jpeg/png/webp/tiff/bmp）、JPEG 质量、是否强制拉伸
4. **尺寸倍增计算** —— 输入原始宽高，实时计算 ×4/×3/×2 的输出尺寸与各步骤耗时预估

点击「开始处理」后后台线程逐张处理，进度条与日志实时刷新，可随时取消。

## 命令行用法

```bash
# 单文件，默认输出全部三个分辨率到 ./output
python image_optimizer.py photo.jpg

# 批量处理目录，仅输出 4K，指定输出目录
python image_optimizer.py ./photos ./result --targets 4K

# 使用 GPU 动漫特化引擎
python image_optimizer.py photo.jpg out --engine realesrgan

# 使用 AI 画质引擎
python image_optimizer.py photo.jpg out --engine edsr

# 统一输出 WebP + 强制拉伸 + 自定义 JPEG 质量
python image_optimizer.py img.png out --format webp --stretch --quality 90

# 不带参数启动，进入交互式输入
python image_optimizer.py
```

### 命令行参数

```
用法: python image_optimizer.py <输入路径> [输出目录] [选项]
```

| 参数 | 说明 |
|---|---|
| `input` | 输入路径，单个图片文件或包含图片的目录；不填则进入交互式输入 |
| `output` | 输出目录，默认 `./output` |
| `--targets` | 逗号分隔的输出分辨率，如 `2K,3K,4K`（默认全部） |
| `--stretch` | 不保持宽高比，强制拉伸到目标分辨率 |
| `--quality` | JPEG 输出质量 1-100（默认 95，禁用色度抽样） |
| `--engine` | 放大引擎：`lanczos`（默认）/ `realesrgan` / `edsr` / `fsrcnn` |
| `--format` | 输出格式：`auto`（默认，跟随源图）/ `jpeg` / `png` / `webp` / `tiff` / `bmp` |
| `-h, --help` | 查看帮助 |

> 尺寸倍增（×4/×3/×2）目前仅图形界面支持，命令行请使用 `--targets` 分辨率模式。

## 引擎对比

| 引擎 | 类型 | 画质 | 速度 | 额外依赖 |
|---|---|---|---|---|
| `lanczos` | 经典插值 | 好（插值上限） | 毫秒级 | 仅 Pillow |
| `realesrgan` | AI 超分（GPU） | 极佳（动漫/插画特化） | 快（显卡加速） | 需 Vulkan 显卡 + `realesrgan-ncnn-vulkan/` 目录 |
| `edsr` | AI 超分（CPU） | 最佳（照片细节） | 慢（数秒~数分钟/张） | opencv-contrib-python |
| `fsrcnn` | AI 超分（CPU） | 中等 | 接近实时 | opencv-contrib-python |

选型建议：

- 动漫、插画、二次元截图 → `realesrgan`（`realesrgan-x4plus-anime` 动漫特化模型）
- 照片小图追求画质且无合适显卡 → `edsr`
- 大批量快速处理 → `fsrcnn` 或默认 `lanczos`
- 源图本身大于 1080p 时，`lanczos` 与 AI 差距不大，直接用默认引擎

## 支持格式

输入：`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff`

输出：六种格式可选（`--format` / 图形界面输出设置）

| 格式 | 特点 |
|---|---|
| `auto`（默认） | PNG 源图输出 PNG，其余输出 JPEG，兼容原图格式 |
| `jpeg` | 有损压缩，不支持透明，体积小，适合照片与日常分享 |
| `png` | 无损压缩，支持透明，画质最佳，适合截图与设计稿 |
| `webp` | 有损压缩，支持透明，同画质体积比 JPEG 小约 25~35%，单边上限 16383px，超限自动改存 JPEG |
| `tiff` | 无损，支持透明，适合存档与专业印刷 |
| `bmp` | 无压缩位图，不支持透明，兼容性最广 |

## 工作原理

```
打开图片 → EXIF 方向纠偏 → 模式归一化 (RGB/RGBA)
    ├─ lanczos:   Lanczos 缩放到目标尺寸 → UnsharpMask 锐化
    ├─ realesrgan: 子进程调用 realesrgan-ncnn-vulkan（Vulkan GPU, -j 1:1:1 单线程推理）
    │              模型原生 ×4 超分 → 精确对齐目标尺寸
    │              崩溃时自动沿分块阶梯 256→128→64 减小分块重试;
    │              超大图 (>12MP) 切成带重叠的条带逐条推理后像素级拼接
    ├─ AI(CPU):   模型 ×4 整数倍超分（大图自动按 768px 分块 + 多进程并行推理）
    │              → Lanczos 对齐目标尺寸（不锐化，避免噪点放大）
→ 按所选格式保存（RGBA → JPEG/BMP 自动合成白底，WebP 超限自动回退 JPEG）
```

## 项目结构

```
图片优化工具/
├── gui.py                       # 图形界面主程序（推荐入口）
├── image_optimizer.py           # 核心处理库 + 命令行入口（GUI 复用其函数）
├── 图片优化工具.spec             # PyInstaller 打包配置
├── pyqt5_lib/                   # 内置 PyQt5 运行库（pip 未安装时自动回退加载）
├── realesrgan-ncnn-vulkan/      # Real-ESRGAN GPU 引擎（含 exe、模型与 vcomp140.dll）
│   ├── realesrgan-ncnn-vulkan.exe
│   └── models/                  # realesrgan-x4plus-anime 等模型
├── models/                      # EDSR/FSRCNN 模型缓存（首次使用自动下载，不入库）
│   ├── EDSR_x4.pb               # 36.8 MB，画质优先
│   └── FSRCNN_x4.pb             # 41 KB，速度优先
└── README.md
```

## 打包发布

使用 PyInstaller 按 `图片优化工具.spec` 打包：

```bash
pip install pyinstaller
pyinstaller 图片优化工具.spec
```

产物在 `dist/` 目录。构建产物（`build/`、`dist/`）已通过 `.gitignore` 排除。

## 常见问题

**模型下载失败 / GitHub 无法访问？**

EDSR / FSRCNN 模型手动下载后放入 `models\` 目录即可：
- EDSR: `https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb`
- FSRCNN: `https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb`

**提示未找到 Real-ESRGAN 引擎？**

`realesrgan` 引擎通过子进程调用 `realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan.exe`（已随仓库提供）。若缺失，请下载 [realesrgan-ncnn-vulkan Windows 版](https://github.com/xinntao/Real-ESRGAN/releases)并解压到项目目录的 `realesrgan-ncnn-vulkan/` 文件夹。

**realesrgan 引擎报 Vulkan 初始化失败？**

该引擎依赖 **Vulkan 显卡驱动**，需较新的独立/核显显卡并安装官方驱动。老机器请改用 `edsr` / `fsrcnn`。

**realesrgan 引擎推理中途崩溃（`exit=3221225477` / 0xC0000005）？**

多为显存不足或显卡驱动压力过大。工具已内置多重防护：强制单线程推理（`-j 1:1:1`）降低显存峰值；崩溃时自动沿分块阶梯 256→128→64 减小分块重试；超大图（超过 12MP）自动切成带重叠的条带逐条推理后拼接。若全部重试仍失败，请更新显卡驱动、关闭占用显存的程序（如浏览器硬件加速），或改用 `lanczos` / `edsr` 引擎。

**报错 `Can't open "...models\xxx.pb"`？**

OpenCV 在 Windows 上无法读取含中文/非 ASCII 字符的路径。工具已内置处理：会把模型复制到临时目录再加载，正常情况下无需干预。

**EDSR 太慢？**

EDSR 耗时随源图尺寸急剧增长（纯 CPU 推理）。工具已内置 768px 分块 + 多进程并行加速；仍嫌慢可改用 `fsrcnn`（CPU）或 `realesrgan`（GPU）。

**输出文件命名规则？**

`原文件名_标识.扩展名`：分辨率模式为 `photo_4K.jpg`，尺寸倍增模式为 `photo_4x.png`。同目录同名输出会互相覆盖，批量处理时请确保源文件名不重复。
