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
from collections import deque
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps

# 本工具会生成/读取多轮超分产物 (可达上亿像素), 解除 Pillow 的像素量防护限制
Image.MAX_IMAGE_PIXELS = None

# 目标分辨率定义 (宽, 高)
RESOLUTIONS = {
    "2K": (2560, 1440),   # QHD
    "3K": (3072, 1728),   # QHD+
    "4K": (3840, 2160),   # UHD
}

# 支持的图片格式
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# 输出格式: key -> (显示名, 扩展名(None=跟随源图), 特性说明)
OUTPUT_FORMATS = {
    "auto": ("自动（跟随源图）", None, "PNG 源图输出 PNG，其余输出 JPEG，兼容原图格式"),
    "jpeg": ("JPEG", ".jpg", "有损压缩 · 不支持透明 · 体积小，适合照片与日常分享"),
    "png":  ("PNG", ".png", "无损压缩 · 支持透明 · 画质最佳，适合截图与设计稿，体积较大"),
    "webp": ("WebP", ".webp", "有损压缩 · 支持透明 · 同画质体积比 JPEG 小约 25~35% · 单边上限 16383px, 超限自动改存 JPEG"),
    "tiff": ("TIFF", ".tiff", "无损 · 支持透明 · 适合存档与专业印刷，体积大"),
    "bmp":  ("BMP", ".bmp", "无压缩位图 · 不支持透明 · 兼容性最广，体积最大"),
}

# WebP 格式单边像素上限 (libwebp 硬限制), 超限输出自动回退为 JPEG
WEBP_MAX_SIDE = 16383

# AI 超分模型: 名称 -> (模型文件, 放大倍数, 下载地址)
AI_MODELS = {
    "edsr": ("EDSR_x4.pb", 4,
             "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"),
    "fsrcnn": ("FSRCNN_x4.pb", 4,
               "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb"),
}
def _base_dir() -> Path:
    """资源基目录: PyInstaller 打包后为 exe 所在目录, 源码运行为脚本所在目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


MODEL_DIR = _base_dir() / "models"

# AI 推理内存控制: 单次整图推理的输出像素上限, 超过则自动分块 (防止 DNN 内存耗尽)
SR_MAX_FULL_OUTPUT_PIXELS = 12_000_000
SR_TILE = 768     # 分块推理: 核心 块尺寸 (768*768*4^2 ≈ 9.4MP 输出/块)
SR_OVERLAP = 32   # 分块推理: 块间重叠像素, 消除接缝边界伪影

# 分块推理并行度: DNN 单进程多线程的扩展效率低 (实测 16 线程仅相当于 4 线程的 1.45 倍),
# 拆成多个子进程并行推理更快 (实测 4 进程 x 4 线程比 1 进程 x 16 线程快约 2.8 倍)
SR_WORKERS = max(1, min(4, (os.cpu_count() or 4) // 4))
SR_THREADS_PER_WORKER = max(1, (os.cpu_count() or 4) // SR_WORKERS)

# 实测基准: 单进程 CPU 推理每百万输入像素所需秒数 (用于大图预计耗时提示)
SR_SPEED_S_PER_MP = {"edsr": 225.0, "fsrcnn": 1.0}

# Real-ESRGAN GPU 引擎 (ncnn-vulkan): 独立 exe, 走 Vulkan, 兼容 NVIDIA/AMD/Intel 显卡
REALESRGAN_EXE = _base_dir() / "realesrgan-ncnn-vulkan" / "realesrgan-ncnn-vulkan.exe"
REALESRGAN_MODEL = "realesrgan-x4plus-anime"   # 动漫特化模型 (通用照片可改 realesrgan-x4plus)
REALESRGAN_TILE = 256        # exe 内部分块尺寸, 控制 VRAM 占用 (256 约需 2~3GB 显存)
REALESRGAN_S_PER_MP = 2.5    # 实测基准: RTX 3050 每百万输入像素约 2.5 秒 (含加载开销摊薄)

# 经典 Lanczos 管线 (重采样 + 锐化 + 保存) 的粗略耗时基准, 用于 GUI 预计处理时间
LANCZOS_S_PER_OUT_MP = 0.08   # 每百万输出像素约 0.08 秒
LANCZOS_BASE_S = 0.3          # 每张图片单个输出的固定开销 (解码/编码等)


def estimate_seconds(src_w: int, src_h: int, factor: int = None,
                     target: str = None, engine: str = "lanczos") -> float:
    """估算单张图片单个输出版本的处理耗时(秒)。

    尺寸倍增模式传 factor (如 4 表示 ×4), 分辨率模式传 target (如 "4K")。
    """
    mp_in = src_w * src_h / 1e6
    if engine == "realesrgan":
        # GPU 引擎: 按实测推理速度估算 + 进程/加载固定开销
        return 1.0 + mp_in * REALESRGAN_S_PER_MP
    if engine in SR_SPEED_S_PER_MP:
        # AI 引擎: 按实测推理速度估算 (多进程并行的等效速度)
        return mp_in * SR_SPEED_S_PER_MP[engine] / SR_WORKERS
    if factor is not None:
        mp_out = mp_in * factor * factor
    elif target in RESOLUTIONS:
        mp_out = min(mp_in, RESOLUTIONS[target][0] * RESOLUTIONS[target][1] / 1e6)
    else:
        return 0.0
    return LANCZOS_BASE_S + LANCZOS_S_PER_OUT_MP * mp_out


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


def calculate_scaled_sizes(src_w: int, src_h: int) -> dict:
    """按原始尺寸直接乘以倍数计算缩放尺寸 (不做标准分辨率对齐)。

    返回: {"4x": (w, h), "3x": (w, h), "2x": (w, h)}
    """
    return {
        "4x": (src_w * 4, src_h * 4),
        "3x": (src_w * 3, src_h * 3),
        "2x": (src_w * 2, src_h * 2),
    }


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


# 分块推理子进程内的全局模型实例 (仅子进程内使用)
_TILED_SR = None


def _tiled_worker_init(model_path: str, model_name: str, scale: int, threads: int):
    """分块推理子进程初始化: 独立加载模型, 并限制该进程的 DNN 线程数。"""
    global _TILED_SR
    import cv2
    cv2.setNumThreads(threads)
    _TILED_SR = cv2.dnn_superres.DnnSuperResImpl_create()
    _TILED_SR.readModel(model_path)
    _TILED_SR.setModel(model_name, scale)


def _tiled_worker_task(job, sr=None):
    """推理单个分块, 返回 (y0, x0, 核心区域的放大结果)。"""
    import numpy as np
    y0, y1, x0, x1, yi0, yi1, xi0, xi1, scale, buf = job
    tile = np.frombuffer(buf, np.uint8).reshape(yi1 - yi0, xi1 - xi0, 3)
    up = (sr if sr is not None else _TILED_SR).upsample(tile)
    return (y0, x0,
            up[(y0 - yi0) * scale:(y1 - yi0) * scale,
               (x0 - xi0) * scale:(x1 - xi0) * scale])


def _sr_upsample_tiled(sr, arr, scale: int, model_name: str = "", model_path: str = "",
                       final=None):
    """大图分块推理: 每块内存占用恒定, 避免一次性推理撑爆内存。

    每块核心区域 SR_TILE 像素, 四周向外扩展 SR_OVERLAP 后送入模型,
    输出只取核心部分拼回, 保证接缝与整图推理基本一致。
    多核 CPU 下用多个子进程并行推理 (见 SR_WORKERS 注释)。
    final=(W, H) 时每块核心输出直接降采样到目标尺寸后再拼接,
    避免分配 x4 全尺寸的输出大数组 (亿级像素图可达数 GB)。
    """
    import cv2
    import numpy as np
    h, w = arr.shape[:2]
    step, ov = SR_TILE, SR_OVERLAP
    cols = (w + step - 1) // step
    rows = (h + step - 1) // step
    total = cols * rows

    if final is None:
        out = np.empty((h * scale, w * scale, 3), dtype=np.uint8)
    else:
        out = np.empty((final[1], final[0], 3), dtype=np.uint8)
        fx, fy = final[0] / w, final[1] / h   # 源图坐标 -> 目标坐标 的比例

    # 预计耗时提示 (基于实测基准)
    etc_min = h * w / 1e6 * SR_SPEED_S_PER_MP.get(model_name, 185.0) / SR_WORKERS / 60
    print(f"    大图 AI 推理: {w}x{h} -> 分块 {cols}x{rows} "
          f"(块 {step}px, 重叠 {ov}px, {SR_WORKERS} 进程并行)")
    if etc_min > 2 and model_name == "edsr":
        print(f"    预计 CPU 耗时约 {etc_min:.0f} 分钟, 大图建议改用 fsrcnn 引擎 (快约 180 倍)")

    def jobs():
        for y0 in range(0, h, step):
            y1 = min(y0 + step, h)
            yi0, yi1 = max(0, y0 - ov), min(h, y1 + ov)
            for x0 in range(0, w, step):
                x1 = min(x0 + step, w)
                xi0, xi1 = max(0, x0 - ov), min(w, x1 + ov)
                yield (y0, y1, x0, x1, yi0, yi1, xi0, xi1, scale,
                       arr[yi0:yi1, xi0:xi1].tobytes())

    def put(y0, x0, up):
        if final is None:
            out[y0 * scale:y0 * scale + up.shape[0],
                x0 * scale:x0 * scale + up.shape[1]] = up
        else:
            # 核心区域 [x0, x0+w0)x[y0, y0+h0) 映射到目标画布上的对应矩形, 降采样后拼入
            dx, dy = round(x0 * fx), round(y0 * fy)
            dw = max(1, round((x0 + up.shape[1] // scale) * fx) - dx)
            dh = max(1, round((y0 + up.shape[0] // scale) * fy) - dy)
            out[dy:dy + dh, dx:dx + dw] = cv2.resize(up, (dw, dh),
                                                     interpolation=cv2.INTER_AREA)

    def report(done):
        if done % max(1, total // 20) == 0 or done == total:
            print(f"    推理进度: {done}/{total} 块 ({done * 100 // total}%)")

    if SR_WORKERS <= 1 or total <= 1:
        for done, job in enumerate(jobs(), 1):
            y0, x0, up = _tiled_worker_task(job, sr)
            put(y0, x0, up)
            report(done)
        return out

    # 多进程并行: 有界提交窗口, 控制同时在途的任务数量以限制内存占用
    from concurrent.futures import ProcessPoolExecutor
    done = 0
    with ProcessPoolExecutor(max_workers=SR_WORKERS,
                             initializer=_tiled_worker_init,
                             initargs=(model_path, model_name, scale,
                                       SR_THREADS_PER_WORKER)) as ex:
        window = deque()
        it = jobs()
        for _ in range(SR_WORKERS + 1):
            try:
                window.append(ex.submit(_tiled_worker_task, next(it)))
            except StopIteration:
                break
        while window:
            y0, x0, up = window.popleft().result()
            put(y0, x0, up)
            done += 1
            report(done)
            try:
                window.append(ex.submit(_tiled_worker_task, next(it)))
            except StopIteration:
                pass
    return out


def ai_upscale_image(img: Image.Image, target_size: tuple, model: str = "edsr",
                     keep_ratio: bool = True) -> Image.Image:
    """AI 超分: 模型先做整数倍放大, 再精确对齐目标尺寸 target_size=(宽, 高)。"""
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
    h, w = arr.shape[:2]
    tw, th = target_size
    # 输出像素量超过阈值时自动分块, 否则整图一次推理 (更快)
    tiled_final = None
    if h * w * scale * scale <= SR_MAX_FULL_OUTPUT_PIXELS:
        arr = sr.upsample(arr)
    else:
        # 大图分块: 推理时直接对齐目标尺寸, 避免分配 x4 全尺寸的中间数组 (可达数 GB)
        tiled_final = calculate_size(w, h, tw, th, keep_ratio)
        arr = _sr_upsample_tiled(sr, arr, scale, model,
                                 _ascii_safe_model_path(model_path), final=tiled_final)
    result = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    # AI 输出已含细节, 直接对齐目标尺寸, 不做锐化
    if tiled_final is not None or result.size == (tw, th):
        return result
    new_w, new_h = calculate_size(*result.size, tw, th, keep_ratio)
    return result.resize((new_w, new_h), Image.LANCZOS)


def realesrgan_available() -> bool:
    """GPU 引擎是否就绪 (exe 与模型已下载解压)。"""
    return REALESRGAN_EXE.exists()


def realesrgan_upscale_image(img: Image.Image, target_size: tuple,
                             keep_ratio: bool = True) -> Image.Image:
    """Real-ESRGAN GPU 超分: 模型原生 ×4 放大, 再精确对齐目标尺寸 target_size=(宽, 高)。

    通过子进程调用 realesrgan-ncnn-vulkan (Vulkan 图形接口),
    exe 内部按 REALESRGAN_TILE 分块推理控制显存, 大图无需在 Python 侧分块。
    """
    if not realesrgan_available():
        raise FileNotFoundError(
            f"未找到 Real-ESRGAN 引擎: {REALESRGAN_EXE}\n"
            "请下载 realesrgan-ncnn-vulkan Windows 版并解压到项目目录的 "
            "realesrgan-ncnn-vulkan/ 文件夹")

    import subprocess
    tw, th = target_size
    with tempfile.TemporaryDirectory(prefix="realesrgan_") as td:
        # 经临时目录中转, 避免源图路径含非 ASCII 字符导致 exe 读写失败
        in_png = Path(td) / "in.png"
        out_png = Path(td) / "out.png"
        img.save(in_png, "PNG")
        cmd = [str(REALESRGAN_EXE),
               "-i", str(in_png), "-o", str(out_png),
               "-n", REALESRGAN_MODEL, "-s", "4",
               "-t", str(REALESRGAN_TILE), "-f", "png"]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0 or not out_png.exists():
            raise RuntimeError(
                f"Real-ESRGAN GPU 推理失败 (exit={proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()[-300:]}")
        with Image.open(out_png) as up:
            result = up.convert(img.mode if img.mode in ("RGB", "RGBA") else "RGB").copy()

    # GPU 输出已含细节, 直接对齐目标尺寸, 不做锐化
    if result.size == (tw, th):
        return result
    new_w, new_h = calculate_size(*result.size, tw, th, keep_ratio)
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

    # GPU 引擎: Real-ESRGAN 超分后不再锐化, 避免噪点被二次放大
    if engine == "realesrgan":
        return realesrgan_upscale_image(img, RESOLUTIONS[target], keep_ratio)

    # AI 引擎: 超分后不再锐化, 避免噪点被二次放大
    return ai_upscale_image(img, RESOLUTIONS[target], engine, keep_ratio)


def upscale_by_factor(img: Image.Image, factor: int, engine: str = "lanczos") -> Image.Image:
    """按整数倍放大 (宽、高各自乘以 factor), 严格保持原始宽高比, 不对齐标准分辨率。"""
    if factor < 1:
        raise ValueError(f"放大倍数必须 >= 1, 收到: {factor}")
    w, h = img.size
    tw, th = w * factor, h * factor   # 宽高同乘整数倍, 宽高比严格不变
    if engine == "lanczos":
        # Lanczos 高质量重采样 + 锐化补偿, 抑制放大后的模糊
        return enhance_image(img.resize((tw, th), Image.LANCZOS))
    # GPU 引擎: Real-ESRGAN 模型原生 ×4 与整数倍尺寸天然对齐
    if engine == "realesrgan":
        return realesrgan_upscale_image(img, (tw, th), keep_ratio=True)
    # AI 引擎: 模型按其原生倍数超分, 再精确对齐到整数倍尺寸
    return ai_upscale_image(img, (tw, th), engine, keep_ratio=True)


def _save_output(out_img: Image.Image, out_path: Path, save_ext: str, quality: int):
    """按输出格式保存图片; JPEG/BMP 不支持透明, 先合成到白色背景。"""
    if out_img.mode == "RGBA" and save_ext in (".jpg", ".bmp"):
        background = Image.new("RGB", out_img.size, (255, 255, 255))
        background.paste(out_img, mask=out_img.split()[3])
        out_img = background

    if save_ext == ".jpg":
        out_img.save(out_path, "JPEG", quality=quality, subsampling=0)
    elif save_ext == ".webp":
        out_img.save(out_path, "WEBP", quality=quality)
    elif save_ext == ".png":
        out_img.save(out_path, "PNG", optimize=True)
    elif save_ext == ".tiff":
        out_img.save(out_path, "TIFF")
    else:  # .bmp
        out_img.save(out_path, "BMP")


def process_file(src_path: str, output_dir: str, targets=None, keep_ratio: bool = True,
                 quality: int = 95, engine: str = "lanczos", output_format: str = "auto",
                 scales=None) -> dict:
    """处理单个图片文件, 输出多个分辨率版本和/或整数倍放大版本。

    targets: 目标分辨率列表 (如 ["4K"]), None 表示全部分辨率
    scales:  尺寸倍增列表 (如 ["4x", "2x"]), 宽高各乘以对应倍数
    """
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
    # 输出格式: 指定格式按选择, auto 则 PNG 源输出 PNG、其余输出 JPEG
    fmt = output_format if output_format in OUTPUT_FORMATS else "auto"
    save_ext = OUTPUT_FORMATS[fmt][1] or (".png" if ext == ".png" else ".jpg")

    results = {}
    with Image.open(src_path) as img:
        # 根据 EXIF 方向标记自动旋转 (手机拍摄的照片常见)
        img = ImageOps.exif_transpose(img)
        # 转为 RGB 以避免模式问题
        mode = img.mode
        if mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        # 任务列表: 分辨率版本 + 尺寸倍增版本
        jobs = [("res", t) for t in targets] + [("scale", s) for s in (scales or [])]
        for kind, key in jobs:
            t0 = time.time()
            if kind == "res":
                out_img = upscale_image(img, key, keep_ratio=keep_ratio, engine=engine)
            else:
                out_img = upscale_by_factor(img, int(key.rstrip("x")), engine)
            elapsed = time.time() - t0
            out_path = out_dir / f"{stem}_{key}{save_ext}"
            # WebP 单边上限 16383px: 超限时自动回退 JPEG, 避免整图推理白费
            if save_ext == ".webp" and max(out_img.size) > WEBP_MAX_SIDE:
                print(f"  [!] WebP 单边上限 {WEBP_MAX_SIDE}px, 输出 {out_img.width}x{out_img.height} "
                      f"超限, 自动改存 JPEG")
                save_ext = ".jpg"
                out_path = out_dir / f"{stem}_{key}.jpg"
            _save_output(out_img, out_path, save_ext, quality)

            results[key] = str(out_path)
            print(f"  [{key}] {out_img.size[0]}x{out_img.size[1]} ({elapsed:.1f}s) -> {out_path}")

    return results


def process_batch(input_path: str, output_dir: str, targets=None, keep_ratio: bool = True,
                  quality: int = 95, engine: str = "lanczos", output_format: str = "auto"):
    """批量处理: 输入可为文件或目录。"""
    input_path = Path(input_path)
    if input_path.is_file():
        print(f"处理文件: {input_path}")
        process_file(str(input_path), output_dir, targets, keep_ratio, quality,
                     engine, output_format)
    elif input_path.is_dir():
        print(f"批量处理目录: {input_path}")
        count = 0
        for item in sorted(input_path.iterdir()):
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXT:
                print(f"-> {item.name}")
                try:
                    process_file(str(item), output_dir, targets, keep_ratio, quality,
                                 engine, output_format)
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
    parser.add_argument("input", nargs="?", default=None,
                        help="单个图片文件或包含图片的目录 (不填则交互式输入)")
    parser.add_argument("output", nargs="?", default="./output",
                        help="输出目录 (默认 ./output)")
    parser.add_argument("--targets", default="2K,3K,4K",
                        help="逗号分隔的输出分辨率, 如 2K,3K,4K (默认全部)")
    parser.add_argument("--stretch", action="store_true",
                        help="不保持宽高比, 强制拉伸到目标分辨率")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG 输出质量 1-100 (默认 95)")
    parser.add_argument("--engine", default="lanczos",
                        choices=["lanczos", "edsr", "fsrcnn", "realesrgan"],
                        help="放大引擎: lanczos=经典插值(默认), "
                             "edsr=AI超分(画质好,较慢), fsrcnn=AI超分(速度快), "
                             "realesrgan=GPU超分(动漫特化,最快)")
    parser.add_argument("--format", default="auto", choices=list(OUTPUT_FORMATS),
                        help="输出格式: auto=跟随源图(默认), 也可指定 jpeg/png/webp/tiff/bmp")
    args = parser.parse_args(argv[1:])

    # 未提供输入路径时, 交互式询问 (直接回车使用当前目录)
    input_path = args.input
    if not input_path:
        try:
            input_path = input("请输入图片文件或目录路径 (直接回车处理当前目录): ").strip().strip('"')
        except EOFError:
            input_path = ""
        if not input_path:
            input_path = "."
        print()

    targets = [t.strip().upper() for t in args.targets.split(",")]
    for t in targets:
        if t not in RESOLUTIONS:
            parser.error(f"未知分辨率 {t}, 可选: {', '.join(RESOLUTIONS)}")

    return {
        "input_path": input_path,
        "output_dir": args.output,
        "targets": targets,
        "keep_ratio": not args.stretch,
        "quality": max(1, min(100, args.quality)),
        "engine": args.engine,
        "output_format": args.format,
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
