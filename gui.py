"""
图片优化工具 - 图形界面 (PyQt5)

运行: python gui.py
依赖: pip install Pillow PyQt5
AI 超分引擎 (可选): pip install opencv-contrib-python numpy
"""

import sys
import time
from pathlib import Path

# PyQt5 未正确安装时, 回退到项目内置的 pyqt5_lib 目录
try:
    from PyQt5 import QtCore  # noqa: F401
except ImportError:
    sys.modules.pop("PyQt5", None)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "pyqt5_lib"))
    from PyQt5 import QtCore  # noqa: F401

from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from PIL import Image

from image_optimizer import (
    OUTPUT_FORMATS, REALESRGAN_EXE, RESOLUTIONS, SUPPORTED_EXT,
    calculate_scaled_sizes, estimate_seconds, process_file, realesrgan_available,
)

# 放大引擎: key -> (图标, 标题, 说明)
ENGINES = {
    "lanczos": ("⚡", "经典插值 Lanczos",
                "高质量重采样 + 锐化增强，速度最快、零额外依赖，推荐日常使用"),
    "realesrgan": ("🎮", "AI 超分 Real-ESRGAN (GPU)",
                   "动漫特化模型 · 显卡加速推理，大图秒级~分钟级 · 需支持 Vulkan 的显卡"),
    "edsr": ("✨", "AI 超分 EDSR",
             "深度学习超分辨率，画质最佳；CPU 推理较慢，适合小图放大"),
    "fsrcnn": ("🚀", "AI 快速超分 FSRCNN",
               "轻量级 AI 模型，速度接近实时，适合大批量快速处理"),
}

# 目标分辨率: key -> (图标, 标题, 说明)
TARGETS = {
    "2K": ("🖼", "2K", "2560 × 1440 · QHD 主流高分屏"),
    "3K": ("🖼", "3K", "3072 × 1728 · QHD+ 高于 2K 一档"),
    "4K": ("🖼", "4K", "3840 × 2160 · UHD 超高清"),
}

# 尺寸倍增: key -> (图标, 标题) — 卡片说明随 ④ 中原始尺寸实时更新为输出尺寸
SCALES = {
    "4x": ("⤢", "×4 倍增"),
    "3x": ("⤢", "×3 倍增"),
    "2x": ("⤢", "×2 倍增"),
}

COLOR_OK = "#7ec96b"
COLOR_WARN = "#d8c26a"
COLOR_ERR = "#e57364"


def fmt_duration(seconds: float) -> str:
    """把秒数格式化为人类可读的耗时描述。"""
    if seconds <= 0:
        return "未知"
    if seconds < 1:
        return "< 1 秒"
    if seconds < 60:
        return f"约 {seconds:.0f} 秒"
    m, s = divmod(int(round(seconds)), 60)
    return f"约 {m} 分 {s:02d} 秒"


# --------------------------------------------------------------------------- #
#  基础组件
# --------------------------------------------------------------------------- #
class OptionCard(QFrame):
    """可点选的选项卡片，用于引擎(单选)与目标分辨率(多选)。"""

    clicked = pyqtSignal(object)

    def __init__(self, key, icon, title, desc, parent=None):
        super().__init__(parent)
        self.key = key
        self._checked = False
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(66)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(11, 7, 11, 7)
        lay.setSpacing(3)

        top = QHBoxLayout()
        top.setSpacing(7)
        icon_label = QLabel(icon)
        icon_label.setObjectName("cardIcon")
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        top.addWidget(icon_label)
        top.addWidget(title_label)
        top.addStretch(1)
        self.badge = QLabel("已选")
        self.badge.setObjectName("cardBadge")
        self.badge.hide()
        top.addWidget(self.badge)
        lay.addLayout(top)

        self.desc_label = QLabel(desc)
        self.desc_label.setObjectName("cardDesc")
        self.desc_label.setWordWrap(True)
        lay.addWidget(self.desc_label)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self)
        super().mousePressEvent(event)

    @property
    def checked(self):
        return self._checked

    def setChecked(self, checked: bool):
        self._checked = bool(checked)
        self.badge.setVisible(self._checked)
        self.setProperty("selected", self._checked)
        self.style().unpolish(self)
        self.style().polish(self)


class ImageDropList(QListWidget):
    """支持拖放图片文件/文件夹的缩略图列表。"""

    filesDropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(88, 88))
        self.setResizeMode(QListWidget.Adjust)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setWordWrap(True)
        self.setSpacing(8)
        self.setMinimumHeight(118)

    def _extract_images(self, mime_data):
        paths = []
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            p = Path(url.toLocalFile())
            if p.is_dir():
                paths += [f for f in sorted(p.iterdir())
                          if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]
            elif p.suffix.lower() in SUPPORTED_EXT:
                paths.append(p)
        return paths

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            self.filesDropped.emit(self._extract_images(event.mimeData()))
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class ProcessWorker(QThread):
    """后台处理线程，避免界面卡死。"""

    fileStarted = pyqtSignal(int, int, str, float)   # 序号, 总数, 文件名, 预计秒
    fileFinished = pyqtSignal(bool, str, float)      # 是否成功, 说明, 实际用时秒
    allFinished = pyqtSignal(int, int, bool)  # 成功数, 失败数, 是否被取消

    def __init__(self, files, out_dir, targets, keep_ratio, quality, engine,
                 output_format="auto", scales=None, parent=None):
        super().__init__(parent)
        self.files = files
        self.out_dir = out_dir
        self.targets = targets
        self.keep_ratio = keep_ratio
        self.quality = quality
        self.engine = engine
        self.output_format = output_format
        self.scales = scales or []
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _estimate_seconds(self, path: Path) -> float:
        """按图片尺寸估算该文件全部所选输出的处理耗时(秒), 仅读取图片头, 开销极小。"""
        try:
            with Image.open(path) as img:
                w, h = img.size
        except Exception:
            return 0.0
        total = 0.0
        for t in self.targets:
            total += estimate_seconds(w, h, target=t, engine=self.engine)
        for s in self.scales:
            total += estimate_seconds(w, h, factor=int(s.rstrip("x")),
                                      engine=self.engine)
        return total

    def run(self):
        ok = fail = 0
        total = len(self.files)
        for idx, f in enumerate(self.files, 1):
            if self._cancel:
                break
            self.fileStarted.emit(idx, total, f.name, self._estimate_seconds(f))
            t0 = time.perf_counter()
            try:
                results = process_file(str(f), self.out_dir, self.targets,
                                       self.keep_ratio, self.quality, self.engine,
                                       self.output_format, self.scales)
                outputs = ", ".join(f"[{k}] {Path(p).name}"
                                    for k, p in results.items())
                self.fileFinished.emit(True, f"{f.name}  →  {outputs}",
                                       time.perf_counter() - t0)
                ok += 1
            except Exception as e:  # 单张失败不影响后续
                self.fileFinished.emit(False, f"{f.name}  失败: {e}",
                                       time.perf_counter() - t0)
                fail += 1
        self.allFinished.emit(ok, fail, self._cancel)


# --------------------------------------------------------------------------- #
#  主窗口
# --------------------------------------------------------------------------- #
class ImageOptimizerWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片优化工具 · 高清放大（2K/3K/4K · ×2/×3/×4）")
        # 初始尺寸跟随屏幕可用区域, 避免固定尺寸在低分辨率/缩放屏上放不下
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1080, screen.width() - 48), min(880, screen.height() - 48))
        self.setMinimumSize(720, 480)

        self._files = []      # 已选图片路径 (Path)
        self._worker = None

        self._build_ui()
        self._connect_signals()

    # ---------------------------- 界面构建 ---------------------------- #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(10)

        # 配置区(①-④)放在滚动容器中: 屏幕高度不足时上下滚动,
        # 而不是把控件压缩到互相重叠
        body = QWidget()
        config = QVBoxLayout(body)
        config.setContentsMargins(0, 0, 0, 0)
        config.setSpacing(10)

        # ---------- ① 图片选择 ---------- #
        grp_files = QGroupBox("①  选择图片")
        v1 = QVBoxLayout(grp_files)
        v1.setSpacing(6)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("添加图片")
        self.remove_btn = QPushButton("移除选中")
        self.clear_btn = QPushButton("清空")
        self.count_label = QLabel("尚未选择图片")
        self.count_label.setObjectName("countLabel")
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.count_label)
        v1.addLayout(btn_row)

        self.file_list = ImageDropList()
        v1.addWidget(self.file_list)

        hint = QLabel("提示：可将一个或多个图片文件（或整个文件夹）直接拖入上方区域 · "
                      "支持 " + " / ".join(e.lstrip(".").upper() for e in sorted(SUPPORTED_EXT)))
        hint.setObjectName("hintLabel")
        v1.addWidget(hint)
        config.addWidget(grp_files)

        # ---------- ② 处理方式 ---------- #
        grp_proc = QGroupBox("②  处理方式")
        v2 = QVBoxLayout(grp_proc)
        v2.setSpacing(6)

        cap1 = QLabel("放大引擎（单选）")
        cap1.setObjectName("sectionLabel")
        v2.addWidget(cap1)

        engine_row = QHBoxLayout()
        engine_row.setSpacing(10)
        self.engine_cards = []
        for key, (icon, title, desc) in ENGINES.items():
            card = OptionCard(key, icon, title, desc)
            card.clicked.connect(self._on_engine_clicked)
            engine_row.addWidget(card)
            self.engine_cards.append(card)
        v2.addLayout(engine_row)

        cap2 = QLabel("目标分辨率（可多选，同时输出多个版本）")
        cap2.setObjectName("sectionLabel")
        v2.addWidget(cap2)

        target_row = QHBoxLayout()
        target_row.setSpacing(10)
        self.target_cards = []
        for key, (icon, title, desc) in TARGETS.items():
            card = OptionCard(key, icon, title, desc)
            card.clicked.connect(self._on_target_clicked)
            target_row.addWidget(card)
            self.target_cards.append(card)
        v2.addLayout(target_row)

        cap3 = QLabel("尺寸倍增（可多选，宽高各乘以倍数，严格保持原始宽高比，不对齐标准分辨率）")
        cap3.setObjectName("sectionLabel")
        v2.addWidget(cap3)

        scale_row = QHBoxLayout()
        scale_row.setSpacing(10)
        self.scale_cards = []
        self._scale_card_map = {}
        for key, (icon, title) in SCALES.items():
            card = OptionCard(key, icon, title, "计算中…")
            card.clicked.connect(self._on_scale_clicked)
            scale_row.addWidget(card)
            self.scale_cards.append(card)
            self._scale_card_map[key] = card
        v2.addLayout(scale_row)

        opt_row = QHBoxLayout()
        self.keep_ratio_cb = QCheckBox("保持宽高比（取消则强制拉伸到目标分辨率）")
        self.keep_ratio_cb.setChecked(True)
        self.format_label = QLabel("输出格式")
        self.format_combo = QComboBox()
        for key, (name, _ext, _desc) in OUTPUT_FORMATS.items():
            self.format_combo.addItem(name, key)
        self.quality_label = QLabel("质量")
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)
        opt_row.addWidget(self.keep_ratio_cb)
        opt_row.addStretch(1)
        opt_row.addWidget(self.format_label)
        opt_row.addWidget(self.format_combo)
        opt_row.addSpacing(18)
        opt_row.addWidget(self.quality_label)
        opt_row.addWidget(self.quality_spin)
        v2.addLayout(opt_row)

        # 格式说明: 随下拉选择实时更新, 提示该格式特性
        self.format_hint = QLabel(OUTPUT_FORMATS["auto"][2])
        self.format_hint.setObjectName("hintLabel")
        v2.addWidget(self.format_hint)
        config.addWidget(grp_proc)

        # ---------- ③ 输出设置 ---------- #
        grp_out = QGroupBox("③  输出设置")
        v3 = QVBoxLayout(grp_out)
        v3.setSpacing(6)

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit(str(Path.cwd() / "output"))
        self.out_edit.setPlaceholderText("输出目录路径，如 D:\\photos\\output")
        self.browse_btn = QPushButton("浏览…")
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(self.browse_btn)
        v3.addLayout(out_row)

        self.path_state = QLabel()
        v3.addWidget(self.path_state)
        config.addWidget(grp_out)

        # ---------- ④ 尺寸倍增计算 ---------- #
        grp_scale = QGroupBox("④  尺寸倍增计算（×4 / ×3 / ×2）")
        v5 = QVBoxLayout(grp_scale)
        v5.setSpacing(6)

        scale_in_row = QHBoxLayout()
        scale_in_row.addWidget(QLabel("原始尺寸"))
        self.src_w_spin = QSpinBox()
        self.src_w_spin.setRange(1, 999999)
        self.src_w_spin.setValue(1920)
        self.src_w_spin.setSuffix(" px")
        self.src_h_spin = QSpinBox()
        self.src_h_spin.setRange(1, 999999)
        self.src_h_spin.setValue(1080)
        self.src_h_spin.setSuffix(" px")
        self.use_sel_btn = QPushButton("读取选中图片尺寸")
        scale_in_row.addWidget(self.src_w_spin, 1)
        scale_in_row.addWidget(QLabel("×"))
        scale_in_row.addWidget(self.src_h_spin, 1)
        scale_in_row.addSpacing(12)
        scale_in_row.addWidget(self.use_sel_btn)
        v5.addLayout(scale_in_row)

        self.scale_result_label = QLabel()
        self.scale_result_label.setObjectName("scaleResultLabel")
        v5.addWidget(self.scale_result_label)

        scale_hint = QLabel("按原始尺寸精确整数倍计算（宽、高各自乘以倍数），"
                            "② 中「尺寸倍增」卡片的输出尺寸随此处实时更新")
        scale_hint.setObjectName("hintLabel")
        v5.addWidget(scale_hint)
        config.addWidget(grp_scale)

        # 滚动容器: 完整容纳配置区, 空间不足时出现滚动条而不是布局挤压重叠
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(body)
        root.addWidget(scroll, 3)

        # ---------- 进度 + 操作按钮 ---------- #
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%p%")
        root.addWidget(self.progress_bar)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.start_btn = QPushButton("开  始  处  理")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setMinimumHeight(46)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        action_row.addWidget(self.start_btn, 1)
        action_row.addWidget(self.stop_btn)
        root.addLayout(action_row)

        # ---------- 日志 ---------- #
        grp_log = QGroupBox("处理日志")
        v4 = QVBoxLayout(grp_log)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setMinimumHeight(120)  # 最小高度, 可随窗口拉伸继续变大
        v4.addWidget(self.log_view)
        root.addWidget(grp_log, 1)  # 与上方配置区按 3:1 分配垂直空间

        self.statusBar().showMessage("就绪")

        # 默认选项
        self.engine_cards[0].setChecked(True)                 # lanczos
        self.target_cards[2].setChecked(True)                 # 4K
        self._validate_output()
        self._update_scaled_sizes()

    def _connect_signals(self):
        self.add_btn.clicked.connect(self.pick_files)
        self.remove_btn.clicked.connect(self.remove_selected)
        self.clear_btn.clicked.connect(self.clear_files)
        self.file_list.filesDropped.connect(self.add_files)
        self.browse_btn.clicked.connect(self.browse_output)
        self.out_edit.textChanged.connect(lambda: self._validate_output())
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        self.start_btn.clicked.connect(self.start_process)
        self.stop_btn.clicked.connect(self.cancel_process)
        self.src_w_spin.valueChanged.connect(self._update_scaled_sizes)
        self.src_h_spin.valueChanged.connect(self._update_scaled_sizes)
        self.use_sel_btn.clicked.connect(self._use_selected_size)

    # ---------------------------- 图片选择 ---------------------------- #
    def pick_files(self):
        ext_pattern = " ".join("*" + e for e in sorted(SUPPORTED_EXT))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", f"图片文件 ({ext_pattern})")
        self.add_files([Path(p) for p in paths])

    def add_files(self, paths):
        added = 0
        for p in paths:
            p = Path(p)
            if p in self._files:
                continue
            icon, dims = self._make_thumbnail(p)
            self._files.append(p)
            item = QListWidgetItem(icon, p.name)
            item.setData(Qt.UserRole, str(p))
            item.setToolTip(f"{p}\n尺寸: {dims}")
            self.file_list.addItem(item)
            added += 1
        self._update_count()
        if added:
            self.statusBar().showMessage(f"已添加 {added} 张图片", 3000)

    def remove_selected(self):
        for item in self.file_list.selectedItems():
            self._files.remove(Path(item.data(Qt.UserRole)))
            self.file_list.takeItem(self.file_list.row(item))
        self._update_count()

    def clear_files(self):
        self._files.clear()
        self.file_list.clear()
        self._update_count()

    def _update_count(self):
        n = len(self._files)
        self.count_label.setText(f"已选择 {n} 张图片" if n else "尚未选择图片")

    @staticmethod
    def _make_thumbnail(path: Path):
        """生成缩略图图标, 返回 (QIcon, 尺寸描述)。"""
        try:
            with Image.open(path) as img:
                w, h = img.size
                img.draft("RGB", (256, 256))
                thumb = img.convert("RGBA")
                thumb.thumbnail((96, 96), Image.LANCZOS)
                data = thumb.tobytes("raw", "RGBA")
                qimg = QImage(data, thumb.width, thumb.height,
                              thumb.width * 4, QImage.Format_RGBA8888)
                return QIcon(QPixmap.fromImage(qimg)), f"{w} × {h}"
        except Exception:
            return QIcon(), "未知"

    # ---------------------------- 尺寸倍增计算 ---------------------------- #
    def _update_scaled_sizes(self):
        """原始尺寸 ×4/×3/×2, 实时刷新计算结果与 ② 中倍增卡片的输出尺寸说明。"""
        w = self.src_w_spin.value()
        h = self.src_h_spin.value()
        sizes = calculate_scaled_sizes(w, h)
        self.scale_result_label.setText("      ".join(
            f"{k}: {s[0]} × {s[1]}" for k, s in sizes.items()))
        for key, (sw, sh) in sizes.items():
            self._scale_card_map[key].desc_label.setText(f"输出 {sw} × {sh} px")

    def _use_selected_size(self):
        """把列表中选中图片(未选中则取第一张)的尺寸填入输入框。"""
        items = self.file_list.selectedItems()
        if not items and self.file_list.count():
            items = [self.file_list.item(0)]
        if not items:
            QMessageBox.warning(self, "提示", "请先在上方选择图片。")
            return
        path = Path(items[0].data(Qt.UserRole))
        try:
            with Image.open(path) as img:
                w, h = img.size
        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"无法读取图片尺寸:\n{path}\n({e})")
            return
        self.src_w_spin.setValue(w)
        self.src_h_spin.setValue(h)

    # ---------------------------- 选项交互 ---------------------------- #
    def _on_engine_clicked(self, card):
        for c in self.engine_cards:
            c.setChecked(c is card)

    def _on_target_clicked(self, card):
        card.setChecked(not card.checked)

    def _on_scale_clicked(self, card):
        card.setChecked(not card.checked)

    def _on_format_changed(self, index):
        """切换输出格式: 实时更新特性提示, 并联动质量参数的可用性。"""
        key = self.format_combo.itemData(index)
        _name, _ext, desc = OUTPUT_FORMATS[key]
        self.format_hint.setText(desc)
        # 仅 auto(可能输出 JPEG)/JPEG/WebP 使用质量参数
        uses_quality = key in ("auto", "jpeg", "webp")
        self.quality_spin.setEnabled(uses_quality)
        self.quality_label.setEnabled(uses_quality)
        self.quality_label.setText(
            "质量" if uses_quality else "质量（该格式不适用）")

    def browse_output(self):
        start = self.out_edit.text().strip() or str(Path.cwd())
        chosen = QFileDialog.getExistingDirectory(self, "选择输出目录", start)
        if chosen:
            self.out_edit.setText(chosen)

    def _validate_output(self):
        text = self.out_edit.text().strip()
        if not text:
            msg, color = "请输入或选择输出目录", COLOR_WARN
        else:
            p = Path(text)
            if p.is_dir():
                msg, color = "✓ 目录有效，可直接开始处理", COLOR_OK
            elif p.exists():
                msg, color = "✗ 该路径已被文件占用，不能作为输出目录", COLOR_ERR
            elif p.parent.exists():
                msg, color = "目录不存在，处理时将自动创建", COLOR_WARN
            else:
                msg, color = "✗ 上级目录不存在，请检查盘符与路径", COLOR_ERR
        self.path_state.setText(msg)
        self.path_state.setStyleSheet(f"color: {color}; font-size: 12px;")

    # ---------------------------- 处理流程 ---------------------------- #
    def start_process(self):
        if self._worker and self._worker.isRunning():
            return
        if not self._files:
            QMessageBox.warning(self, "提示", "请先选择要处理的图片（支持拖放导入）。")
            return

        selected = [c.key for c in self.target_cards if c.checked]
        targets = [t for t in RESOLUTIONS if t in selected]  # 按 2K→4K 排序
        scales = [c.key for c in self.scale_cards if c.checked]  # 按 ×4→×2 排序
        if not targets and not scales:
            QMessageBox.warning(self, "提示", "请至少选择一个目标分辨率或尺寸倍增选项。")
            return

        engine = next(c.key for c in self.engine_cards if c.checked)
        if engine in ("edsr", "fsrcnn"):
            try:
                import cv2  # noqa: F401
                import numpy  # noqa: F401
            except ImportError as e:
                QMessageBox.critical(
                    self, "缺少依赖",
                    f"AI 引擎需要 OpenCV 支持，请先安装:\n\n"
                    f"    pip install opencv-contrib-python numpy\n\n({e})")
                return
        elif engine == "realesrgan" and not realesrgan_available():
            QMessageBox.critical(
                self, "缺少 GPU 引擎",
                "未找到 realesrgan-ncnn-vulkan 引擎。\n\n"
                "请下载 realesrgan-ncnn-vulkan Windows 版，解压到项目目录的\n"
                "realesrgan-ncnn-vulkan/ 文件夹 (需保留其中的 models/ 子目录)。\n\n"
                f"期望路径: {REALESRGAN_EXE}")
            return

        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "提示", "请设置输出目录。")
            return
        out_path = Path(out_dir)
        if out_path.exists() and not out_path.is_dir():
            QMessageBox.critical(self, "路径无效",
                                 f"该路径已被文件占用，无法作为输出目录:\n{out_path}")
            return

        self._set_processing(True)
        out_fmt = self.format_combo.currentData()
        scope = []
        if targets:
            scope.append("目标 " + "+".join(targets))
        if scales:
            scope.append("倍增 " + "+".join(scales))
        self.log_view.appendPlainText(
            f"== 开始处理 {len(self._files)} 张图片 | 引擎: {engine} | "
            f"{' | '.join(scope)} | 格式: {out_fmt} | 输出: {out_path} ==")

        self._worker = ProcessWorker(list(self._files), str(out_path), targets,
                                     self.keep_ratio_cb.isChecked(),
                                     self.quality_spin.value(), engine, out_fmt,
                                     scales)
        self._worker.fileStarted.connect(self._on_file_started)
        self._worker.fileFinished.connect(self._on_file_finished)
        self._worker.allFinished.connect(self._on_all_finished)
        self._worker.start()

    def cancel_process(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.stop_btn.setEnabled(False)
            self.log_view.appendPlainText("… 正在取消，等待当前图片处理完成")
            self.statusBar().showMessage("正在取消…")

    def _set_processing(self, processing: bool):
        for w in (self.add_btn, self.remove_btn, self.clear_btn, self.browse_btn,
                  self.out_edit, self.file_list, self.keep_ratio_cb,
                  self.format_label, self.format_combo,
                  self.quality_spin, self.quality_label):
            w.setEnabled(not processing)
        for card in self.engine_cards + self.target_cards + self.scale_cards:
            card.setEnabled(not processing)
        self.start_btn.setEnabled(not processing)
        self.stop_btn.setEnabled(processing)
        self.progress_bar.setValue(0)
        if not processing:
            self.progress_bar.setFormat("%p%")
            # 恢复质量控件与当前输出格式的联动状态
            self._on_format_changed(self.format_combo.currentIndex())

    def _on_file_started(self, idx, total, name, est):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(idx - 1)
        self.progress_bar.setFormat(f"{idx}/{total} · %p%")
        self.statusBar().showMessage(f"正在处理 ({idx}/{total}): {name} · 预计 {fmt_duration(est)}")
        self.log_view.appendPlainText(f"  ▶ {name} · 预计 {fmt_duration(est)}")

    def _on_file_finished(self, ok, msg, elapsed):
        mark = "✓" if ok else "✗"
        self.log_view.appendPlainText(f"  {mark} {msg} · 用时 {fmt_duration(elapsed)}")
        self.progress_bar.setValue(self.progress_bar.value() + 1)

    def _on_all_finished(self, ok, fail, cancelled):
        self._set_processing(False)
        note = "（已取消）" if cancelled else ""
        summary = f"== 处理结束{note}: 成功 {ok} 张, 失败 {fail} 张 =="
        self.log_view.appendPlainText(summary)
        self.statusBar().showMessage(f"处理完成{note}: 成功 {ok}, 失败 {fail}")
        if cancelled:
            return
        box = QMessageBox(self)
        box.setWindowTitle("处理完成")
        box.setIcon(QMessageBox.Information)
        if fail:
            box.setText(f"处理完成: 成功 {ok} 张, 失败 {fail} 张。")
            box.setInformativeText("失败原因见日志，输出目录: \n" + self.out_edit.text())
        else:
            box.setText(f"全部 {ok} 张图片处理完成。")
            box.setInformativeText("输出目录: \n" + self.out_edit.text())
        box.exec_()

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            ret = QMessageBox.question(
                self, "退出确认", "图片仍在处理中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ret != QMessageBox.Yes:
                event.ignore()
                return
            self._worker.cancel()
            self._worker.wait()
        event.accept()


# --------------------------------------------------------------------------- #
#  黑色主题
# --------------------------------------------------------------------------- #
STYLE = """
QMainWindow, QWidget { background-color: #0a0a0a; color: #e8e8e8; font-size: 13px; }

QGroupBox {
    border: 1px solid #2a2a2a; border-radius: 10px; margin-top: 12px;
    background-color: #111111; font-weight: bold; padding-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 14px; padding: 0 6px; color: #ffffff;
}

QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }

QPushButton {
    background-color: #1d1d1d; border: 1px solid #3a3a3a; color: #e8e8e8;
    border-radius: 6px; padding: 5px 14px;
}
QPushButton:hover { background-color: #262626; border-color: #ffffff; }
QPushButton:pressed { background-color: #161616; }
QPushButton:disabled { color: #5c5c5c; border-color: #2e2e2e; background-color: #151515; }

#startBtn {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f5f5f5, stop:1 #c8c8c8);
    color: #0a0a0a; font-size: 17px; font-weight: bold;
    border: 1px solid #8a8a8a; border-radius: 9px; padding: 8px;
}
#startBtn:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #d4d4d4);
}
#startBtn:pressed { background-color: #9e9e9e; }
#startBtn:disabled {
    background-color: #232323; color: #5c5c5c; border-color: #2e2e2e;
}

QLineEdit, QSpinBox {
    background-color: #0d0d0d; border: 1px solid #2a2a2a; border-radius: 6px;
    padding: 6px 10px; color: #f0f0f0;
    selection-background-color: #4d4d4d; selection-color: #ffffff;
}
QLineEdit:focus, QSpinBox:focus { border-color: #ffffff; }
QLineEdit:disabled, QSpinBox:disabled { color: #5c5c5c; }

QComboBox {
    background-color: #0d0d0d; border: 1px solid #2a2a2a; border-radius: 6px;
    padding: 5px 10px 5px 12px; color: #f0f0f0; min-width: 150px; min-height: 20px;
}
QComboBox:hover { border-color: #6e6e6e; }
QComboBox:focus { border-color: #ffffff; }
QComboBox:disabled { color: #5c5c5c; border-color: #222222; background-color: #121212; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow {
    image: none; width: 0; height: 0;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid #cfcfcf; margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #111111; border: 1px solid #3a3a3a; border-radius: 6px;
    color: #e8e8e8; outline: none; selection-background-color: #2a2a2a;
    selection-color: #ffffff;
}

QListWidget {
    background-color: #0d0d0d; border: 1px solid #2a2a2a; border-radius: 8px;
}
QListWidget::item { border-radius: 6px; padding: 2px; }
QListWidget::item:selected { background-color: #2a2a2a; border: 1px solid #ffffff; }
QListWidget::item:hover:!selected { background-color: #1d1d1d; }

QPlainTextEdit {
    background-color: #0d0d0d; border: 1px solid #2a2a2a; border-radius: 8px;
    color: #c8c8c8; font-family: Consolas, "Microsoft YaHei UI", monospace;
}

QProgressBar {
    background-color: #0d0d0d; border: 1px solid #2a2a2a; border-radius: 8px;
    text-align: center; color: #f0f0f0; min-height: 18px; font-weight: bold;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                     stop:0 #b8b8b8, stop:1 #e8e8e8);
    border-radius: 7px; margin: 1px;
}

QCheckBox { spacing: 7px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #3a3a3a;
    border-radius: 4px; background-color: #0d0d0d;
}
QCheckBox::indicator:hover { border-color: #ffffff; }
QCheckBox::indicator:checked { background-color: #e8e8e8; border-color: #e8e8e8; }
QCheckBox:disabled { color: #5c5c5c; }

#card {
    background-color: #161616; border: 1px solid #2a2a2a; border-radius: 9px;
}
#card:hover { border-color: #6e6e6e; background-color: #1c1c1c; }
#card[selected="true"] {
    background-color: #1f1f1f; border: 2px solid #ffffff;
}
#card:disabled { background-color: #121212; }
#cardIcon { font-size: 19px; }
#cardTitle { color: #f0f0f0; font-weight: bold; font-size: 14px; }
#cardDesc { color: #8f8f8f; font-size: 12px; font-weight: normal; }
#cardBadge {
    color: #0a0a0a; background-color: #e8e8e8; border-radius: 8px;
    padding: 1px 8px; font-size: 11px; font-weight: bold;
}

#countLabel { color: #ffffff; font-weight: bold; }
#sectionLabel { color: #a0a0a0; font-size: 12px; }
#hintLabel { color: #6f6f6f; font-size: 12px; }
#scaleResultLabel {
    color: #f0f0f0; font-family: Consolas, "Microsoft YaHei UI", monospace;
    font-size: 14px; font-weight: bold; padding: 2px 0;
}

QStatusBar { background-color: #0d0d0d; color: #8f8f8f; }
QToolTip {
    background-color: #1d1d1d; color: #f0f0f0;
    border: 1px solid #4d4d4d; padding: 4px;
}

QScrollBar:vertical { background: #0d0d0d; width: 10px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #333333; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #525252; }
QScrollBar:horizontal { background: #0d0d0d; height: 10px; border-radius: 5px; }
QScrollBar::handle:horizontal { background: #333333; border-radius: 5px; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: #525252; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
"""


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(STYLE)

    win = ImageOptimizerWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
