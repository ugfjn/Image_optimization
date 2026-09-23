# 图片优化工具

将图片批量提升至 **2K / 3K / 4K** 分辨率的命令行工具，支持经典插值与 AI 超分双引擎。

## 功能特性

- 三档目标分辨率：2K (2560×1440)、3K (3072×1728)、4K (3840×2160)，一次输入可同时输出多个版本
- 双引擎可选：
  - `lanczos`（默认）—— Lanczos 高质量重采样 + UnsharpMask 锐化，速度快、零额外依赖
  - `edsr` —— AI 超分（画质最好，CPU 较慢）
  - `fsrcnn` —— AI 超分（速度接近实时，画质提升有限）
- 等比缩放，以目标分辨率为上限，不产生超尺寸图片（`--stretch` 可强制拉伸）
- 自动读取 EXIF 方向标记，手机照片不再旋转错位
- 透明通道安全：RGBA 图片输出 JPEG 时自动合成到白色背景，输出 PNG 时保留透明
- 支持单文件与目录批量处理，逐张打印耗时统计
- AI 模型首次使用自动下载，缓存至本地 `models/` 目录

## 环境要求

- Python 3.8+

```bash
# 基础依赖（必装）
pip install Pillow

# AI 超分引擎（可选，仅 --engine edsr / fsrcnn 需要）
pip install opencv-contrib-python numpy
```

## 快速开始

```bash
# 单文件，默认输出全部三个分辨率到 ./output
python image_optimizer.py photo.jpg

# 批量处理目录，仅输出 4K，指定输出目录
python image_optimizer.py ./photos ./result --targets 4K

# 使用 AI 画质引擎
python image_optimizer.py photo.jpg out --engine edsr

# 强制拉伸 + 自定义 JPEG 质量
python image_optimizer.py img.png out --stretch --quality 90
```

## 命令行参数

```
用法: python image_optimizer.py <输入路径> [输出目录] [选项]
```

| 参数 | 说明 |
|---|---|
| `input` | 输入路径，单个图片文件或包含图片的目录（必需） |
| `output` | 输出目录，默认 `./output` |
| `--targets` | 逗号分隔的输出分辨率，如 `2K,3K,4K`（默认全部） |
| `--stretch` | 不保持宽高比，强制拉伸到目标分辨率 |
| `--quality` | JPEG 输出质量 1-100（默认 95，禁用色度抽样） |
| `--engine` | 放大引擎：`lanczos`（默认）/ `edsr` / `fsrcnn` |
| `-h, --help` | 查看帮助 |

## 引擎对比

| 引擎 | 画质 | 速度 | 模型大小 | 额外依赖 |
|---|---|---|---|---|
| `lanczos` | 好（插值上限） | 毫秒级 | 无 | 仅 Pillow |
| `edsr` | 最佳（AI 超分） | 慢（CPU 数秒~数分钟/张） | 36.8 MB | opencv-contrib-python |
| `fsrcnn` | 中等（AI 超分） | 接近实时 | 41 KB | opencv-contrib-python |

选型建议：

- 源图本身大于 1080p 时，`lanczos` 与 AI 差距不大，直接用默认引擎
- 小图（< 720p）追求画质用 `edsr`，大批量快速处理用 `fsrcnn`

## 支持格式

输入：`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff`

输出：源为 `.png` 时输出 PNG（保留透明），其余格式输出 JPEG（quality 95）

## 工作原理

```
打开图片 → EXIF 方向纠偏 → 模式归一化 (RGB/RGBA)
    ├─ lanczos: Lanczos 缩放到目标尺寸 → UnsharpMask 锐化
    └─ AI:     模型 ×4 整数倍超分 → Lanczos 对齐目标尺寸（不锐化，避免噪点放大）
→ 保存（JPEG 高质量 / PNG 无损）
```

## 项目结构

```
图片优化工具/
├── image_optimizer.py   # 主程序（单文件，约 250 行）
├── models/              # AI 模型缓存（首次使用自动下载）
│   ├── EDSR_x4.pb       # 36.8 MB，画质优先
│   └── FSRCNN_x4.pb     # 41 KB，速度优先
└── README.md
```

## 常见问题

**模型下载失败 / GitHub 无法访问？**

手动下载后放入 `models\` 目录即可：
- EDSR: `https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb`
- FSRCNN: `https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb`

**报错 `Can't open "...models\xxx.pb"`？**

OpenCV 在 Windows 上无法读取含中文/非 ASCII 字符的路径。工具已内置处理：会把模型复制到临时目录再加载，正常情况下无需干预。

**EDSR 太慢？**

EDSR 耗时随源图尺寸急剧增长（纯 CPU 推理）。可改用 `fsrcnn`，或仅对小图使用 AI 引擎。

**输出文件命名规则？**

`原文件名_分辨率.扩展名`，例如 `photo_4K.jpg`。同目录同名输出会互相覆盖，批量处理时请确保源文件名不重复。
