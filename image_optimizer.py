"""
图片优化工具 - 将图片提升至 4K / 3K / 2K 清晰度
依赖: pip install Pillow
AI 超分引擎 (可选): pip install opencv-contrib-python numpy
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps

# 目标分辨率定义 (宽, 高)
RESOLUTIONS = {
    "2K": (2560, 1440),   # QHD
    "3K": (3072, 1728),   # QHD+
    "4K": (3840, 2160),   # UHD
}

# 支持的图片格式
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# AI 超分模型: 名称 -> (模型文件, 放大倍数, 下载地址)
AI_MODELS = {
    "edsr": ("EDSR_x4.pb", 4,
             "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"),
    "fsrcnn": ("FSRCNN_x4.pb", 4,
               "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb"),
}
MODEL_DIR = Path(__file__).parent / "models"


def calculate_size(src_w: int, src_h: int, target_w: int, target_h: int, keep_ratio: bool = True):
    """根据目标分辨率计算实际输出尺寸。"""
    if not keep_ratio:
        return target_w, target_h

    # 保持宽高比: 以目标分辨率为上限, 等比缩放
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    if src_ratio > target_ratio:
        # 以宽度为基准
        new_w = target_w
        new_h = round(target_w / src_ratio)
    else:
        # 以高度为基准
        new_h = target_h
        new_w = round(target_h * src_ratio)

    return new_w, new_h


def enhance_image(img: Image.Image) -> Image.Image:
    """对图片进行锐化和细节增强。"""
    # 轻微锐化以补偿放大带来的模糊
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=130, threshold=3))
    return img


def ensure_model(name: str) -> Path:
    """确保 AI 模型文件存在, 首次使用时自动下载。"""
    filename, _, url = AI_MODELS[name]
    model_path = MODEL_DIR / filename
    if not model_path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"首次使用 AI 引擎, 正在下载模型 {filename} ...")
        try:
            urllib.request.urlretrieve(url, str(model_path) + ".part")
            os.replace(str(model_path) + ".part", model_path)
        except Exception as e:
            print(f"模型下载失败: {e}")
            print(f"请手动下载: {url}")
            print(f"并放置到: {model_path}")
            raise
        print(f"模型已保存: {model_path}")
    return model_path


def _ascii_safe_model_path(model_path: Path) -> str:
    """OpenCV 在 Windows 上读不了含非 ASCII 字符的路径, 必要时复制到临时目录。"""
    try:
        str(model_path).encode("ascii")
        return str(model_path)
    except UnicodeEncodeError:
        tmp_dir = Path(tempfile.gettempdir()) / "image_optimizer_models"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / model_path.name
        if not tmp_path.exists():
            shutil.copy(model_path, tmp_path)
        return str(tmp_path)


def ai_upscale_image(img: Image.Image, target: str, model: str = "edsr",
                     keep_ratio: bool = True) -> Image.Image:
    """AI 超分: 模型先做整数倍放大, 再精确对齐目标分辨率。"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise ImportError("AI 引擎需要 OpenCV, 请先执行: pip install opencv-contrib-python numpy")

    if model not in AI_MODELS:
        raise ValueError(f"未知 AI 模型: {model}, 可选: {list(AI_MODELS)}")

    model_path, scale, _ = AI_MODELS[model]
    model_path = ensure_model(model)

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(_ascii_safe_model_path(model_path))
    sr.setModel(model, scale)

    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    arr = sr.upsample(arr)
    result = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    # AI 输出已含细节, 直接对齐目标尺寸, 不做锐化
    target_w, target_h = RESOLUTIONS[target]
    new_w, new_h = calculate_size(*result.size, target_w, target_h, keep_ratio)
    return result.resize((new_w, new_h), Image.LANCZOS)


def upscale_image(img: Image.Image, target: str, keep_ratio: bool = True,
                  engine: str = "lanczos") -> Image.Image:
    """将图片放大到指定目标分辨率。"""
    if target not in RESOLUTIONS:
        raise ValueError(f"未知目标分辨率: {target}, 可选: {list(RESOLUTIONS.keys())}")

    if engine == "lanczos":
        target_w, target_h = RESOLUTIONS[target]
        src_w, src_h = img.size
        new_w, new_h = calculate_size(src_w, src_h, target_w, target_h, keep_ratio)

        # 使用 Lanczos 高质量重采样
        img = img.resize((new_w, new_h), Image.LANCZOS)
        return enhance_image(img)

    # AI 引擎: 超分后不再锐化, 避免噪点被二次放大
    return ai_upscale_image(img, target, engine, keep_ratio)


def process_file(src_path: str, output_dir: str, targets=None, keep_ratio: bool = True,
                 quality: int = 95, engine: str = "lanczos") -> dict:
    """处理单个图片文件, 输出多个分辨率版本。"""
    if targets is None:
        targets = list(RESOLUTIONS.keys())

    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(f"文件不存在: {src_path}")

    ext = src_path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"不支持的格式: {ext}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = src_path.stem
    # 统一以 PNG 输出以避免 JPEG 压缩损失 (如源为 JPG 也可保留原格式)
    save_ext = ".png" if ext == ".png" else ".jpg"

    results = {}
    with Image.open(src_path) as img:
        # 根据 EXIF 方向标记自动旋转 (手机拍摄的照片常见)
        img = ImageOps.exif_transpose(img)
        # 转为 RGB 以避免模式问题
        mode = img.mode
        if mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        for target in targets:
            t0 = time.time()
            out_img = upscale_image(img, target, keep_ratio=keep_ratio, engine=engine)
            elapsed = time.time() - t0
            out_name = f"{stem}_{target}{save_ext}"
            out_path = out_dir / out_name

            if save_ext == ".jpg":
                # JPEG 不支持透明通道, 将 alpha 合成到白色背景
                if out_img.mode == "RGBA":
                    background = Image.new("RGB", out_img.size, (255, 255, 255))
                    background.paste(out_img, mask=out_img.split()[3])
                    out_img = background
                out_img.save(out_path, "JPEG", quality=quality, subsampling=0)
            else:
                out_img.save(out_path, "PNG", optimize=True)

            results[target] = str(out_path)
            print(f"  [{target}] {out_img.size[0]}x{out_img.size[1]} ({elapsed:.1f}s) -> {out_path}")

    return results


def process_batch(input_path: str, output_dir: str, targets=None, keep_ratio: bool = True,
                  quality: int = 95, engine: str = "lanczos"):
    """批量处理: 输入可为文件或目录。"""
    input_path = Path(input_path)
    if input_path.is_file():
        print(f"处理文件: {input_path}")
        process_file(str(input_path), output_dir, targets, keep_ratio, quality, engine)
    elif input_path.is_dir():
        print(f"批量处理目录: {input_path}")
        count = 0
        for item in sorted(input_path.iterdir()):
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXT:
                print(f"-> {item.name}")
                try:
                    process_file(str(item), output_dir, targets, keep_ratio, quality, engine)
                    count += 1
                except Exception as e:
                    print(f"   失败: {e}")
        print(f"完成, 共处理 {count} 张图片")
    else:
        raise FileNotFoundError(f"路径不存在: {input_path}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="image_optimizer",
        description="图片优化工具 - 将图片提升至 4K / 3K / 2K 清晰度",
        epilog="示例: python image_optimizer.py ./photos ./result --targets 4K --engine edsr")
    parser.add_argument("input", help="单个图片文件或包含图片的目录")
    parser.add_argument("output", nargs="?", default="./output",
                        help="输出目录 (默认 ./output)")
    parser.add_argument("--targets", default="2K,3K,4K",
                        help="逗号分隔的输出分辨率, 如 2K,3K,4K (默认全部)")
    parser.add_argument("--stretch", action="store_true",
                        help="不保持宽高比, 强制拉伸到目标分辨率")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG 输出质量 1-100 (默认 95)")
    parser.add_argument("--engine", default="lanczos",
                        choices=["lanczos", "edsr", "fsrcnn"],
                        help="放大引擎: lanczos=经典插值(默认), "
                             "edsr=AI超分(画质好,较慢), fsrcnn=AI超分(速度快)")
    args = parser.parse_args(argv[1:])

    targets = [t.strip().upper() for t in args.targets.split(",")]
    for t in targets:
        if t not in RESOLUTIONS:
            parser.error(f"未知分辨率 {t}, 可选: {', '.join(RESOLUTIONS)}")

    return {
        "input_path": args.input,
        "output_dir": args.output,
        "targets": targets,
        "keep_ratio": not args.stretch,
        "quality": max(1, min(100, args.quality)),
        "engine": args.engine,
    }


def main():
    args = parse_args(sys.argv)
    try:
        process_batch(**args)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
