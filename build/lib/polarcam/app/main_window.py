from __future__ import annotations

import os
import time
import math
from typing import Optional, List, Tuple
import numpy as np

from PySide6.QtCore import Qt, Signal, QThread, Slot
from PySide6.QtGui import QImage, QPixmap, QIntValidator, QDoubleValidator, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QLabel,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QGroupBox,
)

from polarcam.app.lut_widget import HighlightLUTWidget
from polarcam.app.spot_detect import detect_and_classify
from polarcam.app.spot_viewer import SpotViewerDialog
from polarcam.analysis import SMapAccumulator
from polarcam.capture.frame_writer import FrameWriter
from polarcam.hardware import Spot

# Cycling recorder
from polarcam.app.spot_cycler import MultiSpotCycler, CycleConfig

# ---- TEST ONLY — remove this import and the Test (AVI) button to clean up ----
from polarcam.backend.mock_backend import AviMockCamera
# ---- END TEST ------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Lean GUI: Open/Start/Stop/Close + ROI/Timing + Gains + Highlight LUT + Spot tools. Live video.
       Adds: Cycle Spots recorder that hops across selected spots (1 s/spot), saves 4×pol means from
       the displayed/software crop only, records at max camera FPS while preview is capped at 20 FPS.
    """
    detect_done = Signal(object)

    def __init__(self, ctrl) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.setWindowTitle("PolarCam (lean)")
        self.resize(1200, 750)

        # status bar
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

        # tone/LUT control
        self._lut: Optional[np.ndarray] = None
        self.tone = HighlightLUTWidget(self)
        self.tone.paramsChanged.connect(self._on_tone_params)
        self._on_tone_params(*self.tone.params())

        # detection / overlay state
        self._all_spots: List[Spot] = []   # full detection results
        self._spots: List[Spot] = []        # currently visible (filtered+sorted)
        self._collecting = False
        self._smap_acc: Optional[SMapAccumulator] = None
        self._smap_n_collected: int = 0
        self._detect_params: dict = {}
        self._detect_conn_active = False
        self._video_paused = False
        self._frame_buf: List[np.ndarray] = []      # raw frames for classifier

        # ---- TEST ONLY ----
        self._mock_cam: Optional[AviMockCamera] = None
        # ---- END TEST -----

        # frame writer state
        self._frame_writer = FrameWriter(self)
        self._frame_writer.progress.connect(self._on_record_progress, Qt.QueuedConnection)
        self._frame_writer.stopped.connect(self._on_record_stopped, Qt.QueuedConnection)
        self._frame_writer.error.connect(self._on_record_error, Qt.QueuedConnection)

        # preview throttle (used during Cycle mode)
        self._cycle_active = False
        self._preview_cap_fps = 20.0  # cap preview to 20 fps while cycling
        self._last_preview_t = 0.0

        # build UI
        self._build_video()
        self._build_toolbar()
        self._build_forms()
        self._build_record_ui()
        self._build_detect_ui()
        self._build_spot_list_ui()
        self._assemble_layout()

        # starting button states
        self._set_buttons(open_enabled=True, start=False, stop=False, close=False)

        # wire controller/backend signals if present
        cam = getattr(self.ctrl, "cam", None)
        if cam is not None:
            try: cam.opened.connect(self._on_open)
            except Exception: pass
            try: cam.started.connect(self._on_started)
            except Exception: pass
            try: cam.stopped.connect(self._on_stopped)
            except Exception: pass
            try: cam.closed.connect(self._on_closed)
            except Exception: pass
            try: cam.error.connect(self._on_error)
            except Exception: pass
            try: cam.frame.connect(self._on_frame)
            except Exception: pass
            try: cam.timing.connect(self._on_timing)
            except Exception: pass
            try: cam.roi.connect(self._on_roi)
            except Exception: pass
            try: cam.gains.connect(self._on_gains)
            except Exception: pass
            if hasattr(cam, "desaturated"):
                try: cam.desaturated.connect(self._on_desaturated)
                except Exception: pass
            if hasattr(cam, "auto_desat_started"):
                try: cam.auto_desat_started.connect(self._on_desat_started)
                except Exception: pass
            if hasattr(cam, "auto_desat_finished"):
                try: cam.auto_desat_finished.connect(self._on_desat_finished)
                except Exception: pass
        else:
            self.status.showMessage("No camera backend attached.", 4000)

        # wire buttons
        self.btn_open.clicked.connect(self._open_clicked)
        self.btn_start.clicked.connect(self._start_clicked)
        self.btn_stop.clicked.connect(self._stop_clicked)
        self.btn_close.clicked.connect(self._close_clicked)
        self.btn_apply_roi.clicked.connect(self._apply_roi)
        self.btn_full_roi.clicked.connect(self._full_roi_clicked)
        self.btn_apply_tim.clicked.connect(self._apply_timing)
        self.btn_desat.clicked.connect(self._desaturate_clicked)
        self.btn_apply_gains.clicked.connect(self._apply_gains)
        self.btn_refresh_gains.clicked.connect(self._refresh_gains)
        self.btn_detect.clicked.connect(self._detect_clicked)
        self.btn_clear_overlays.clicked.connect(self._clear_overlays)
        self.btn_view_spot.clicked.connect(self._view_selected_spots)
        self.btn_remove_spot.clicked.connect(self._remove_selected_spots)
        self.btn_add_spot.clicked.connect(self._add_spot_manual)
        for _ed in (self.ed_add_cx, self.ed_add_cy, self.ed_add_r):
            _ed.textChanged.connect(self._update_add_spot_preview)
        self.btn_record_start.clicked.connect(self._start_record_clicked)
        self.btn_record_stop.clicked.connect(self._stop_record_clicked)
        self.btn_record_browse.clicked.connect(self._browse_record_dir)

        # Cycle buttons
        self.btn_cycle_start.clicked.connect(self._start_cycle_clicked)
        self.btn_cycle_stop.clicked.connect(self._stop_cycle_clicked)
        # ---- TEST ONLY ----
        self.btn_test_avi.clicked.connect(self._load_avi_test)
        # ---- END TEST -----

        self.detect_done.connect(self._handle_detect_done, Qt.QueuedConnection)

        self._last_pm: Optional[QPixmap] = None

        self._roi_offset = (0, 0)   # (OffsetX, OffsetY)
        self._roi_size = (0, 0)     # (Width, Height)

        # preview circle for manual add-spot
        self._preview_spot: Optional[Tuple[float, float, float]] = None  # (cx_local, cy_local, r)

        # histogram rate-limiting
        self._hist_t_last = 0.0

        # cycle worker state
        self._cycle_thread: Optional[QThread] = None
        self._cycler = None  # MultiSpotCycler instance

    # ---------- builders ----------
    def _build_video(self) -> None:
        self.video = QLabel("No video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setStyleSheet("background:#111; color:#777;")

    def _build_toolbar(self) -> None:
        self.btn_open = QPushButton("Open")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_close = QPushButton("Close")
        # ---- TEST ONLY — remove btn_test and its connect() call to clean up ----
        self.btn_test_avi = QPushButton("Test (AVI)…")
        self.btn_test_avi.setStyleSheet("color: darkorange; font-weight: bold;")
        # ---- END TEST -------------------------------------------------------
        self._toolbar = QHBoxLayout()
        for b in (self.btn_open, self.btn_start, self.btn_stop, self.btn_close):
            self._toolbar.addWidget(b)
        # ---- TEST ONLY ----
        self._toolbar.addWidget(self.btn_test_avi)
        # ---- END TEST -----
        self._toolbar.addStretch(1)

    def _build_forms(self) -> None:
        # ROI form
        self.ed_w = QLineEdit("2464")
        self.ed_h = QLineEdit("2056")
        self.ed_x = QLineEdit("0")
        self.ed_y = QLineEdit("0")
        intv = QIntValidator(0, 1_000_000, self)
        for ed in (self.ed_w, self.ed_h, self.ed_x, self.ed_y):
            ed.setValidator(intv)
        self.btn_apply_roi = QPushButton("Apply ROI")
        self.btn_full_roi = QPushButton("Full sensor")

        # Timing form (EXPOSURE IN MILLISECONDS)
        self.ed_fps = QLineEdit("20.0")
        self.ed_exp = QLineEdit("50.0")  # ms
        fpsv = QDoubleValidator(0.01, 100000.0, 3, self); fpsv.setNotation(QDoubleValidator.StandardNotation)
        expv = QDoubleValidator(0.01, 2_000.0, 3, self); expv.setNotation(QDoubleValidator.StandardNotation)
        self.ed_fps.setValidator(fpsv)
        self.ed_exp.setValidator(expv)
        self.btn_apply_tim = QPushButton("Apply Timing")
        self.btn_desat = QPushButton("Desaturate")

        # Gains
        self.ed_gain_ana = QLineEdit("")
        self.ed_gain_dig = QLineEdit("")
        gval = QDoubleValidator(0.0, 1000.0, 3, self); gval.setNotation(QDoubleValidator.StandardNotation)
        self.ed_gain_ana.setValidator(gval)
        self.ed_gain_dig.setValidator(gval)
        self.ed_gain_ana.setPlaceholderText("—")
        self.ed_gain_dig.setPlaceholderText("—")
        self.btn_apply_gains = QPushButton("Apply gains")
        self.btn_refresh_gains = QPushButton("Refresh gains")
        self._lbl_gain_ana = QLabel("")
        self._lbl_gain_dig = QLabel("")

        # Layout
        self._form = QFormLayout()
        self._form.addRow("Width", self.ed_w)
        self._form.addRow("Height", self.ed_h)
        self._form.addRow("OffsetX", self.ed_x)
        self._form.addRow("OffsetY", self.ed_y)
        row = QHBoxLayout()
        row.addWidget(self.btn_apply_roi)
        row.addWidget(self.btn_full_roi)
        rw = QWidget(); rw.setLayout(row)
        self._form.addRow(rw)
        self._form.addRow("FPS", self.ed_fps)
        self._form.addRow("Exposure (ms)", self.ed_exp)
        rowt = QHBoxLayout(); rowt.addWidget(self.btn_apply_tim); rowt.addWidget(self.btn_desat)
        wt = QWidget(); wt.setLayout(rowt)
        self._form.addRow(wt)

        self._form.addRow("Analog gain", self.ed_gain_ana); self._form.addRow("", self._lbl_gain_ana)
        self._form.addRow("Digital gain", self.ed_gain_dig); self._form.addRow("", self._lbl_gain_dig)
        rowg = QHBoxLayout(); rowg.addWidget(self.btn_apply_gains); rowg.addWidget(self.btn_refresh_gains)
        rg = QWidget(); rg.setLayout(rowg)
        self._form.addRow(rg)

    def _build_record_ui(self) -> None:
        box = QGroupBox("Record frames")
        f = QFormLayout(box)

        # Output directory row
        dir_row = QHBoxLayout()
        self.ed_record_dir = QLineEdit(os.path.join(os.getcwd(), "recordings"))
        self.btn_record_browse = QPushButton("Browse…")
        self.btn_record_browse.setMaximumWidth(70)
        dir_row.addWidget(self.ed_record_dir, 1)
        dir_row.addWidget(self.btn_record_browse)
        dir_w = QWidget(); dir_w.setLayout(dir_row)
        f.addRow("Output dir", dir_w)

        self.ed_record_chunk = QLineEdit("50")
        self.ed_record_chunk.setValidator(QIntValidator(1, 100_000, self))
        f.addRow("Frames/chunk", self.ed_record_chunk)

        self.ed_record_max = QLineEdit("0")
        self.ed_record_max.setValidator(QIntValidator(0, 10_000_000, self))
        f.addRow("Max frames (0=∞)", self.ed_record_max)

        self.lbl_record_count = QLabel("0 frames")
        f.addRow("Recorded", self.lbl_record_count)

        self.btn_record_start = QPushButton("Start recording")
        self.btn_record_stop  = QPushButton("Stop recording")
        self.btn_record_stop.setEnabled(False)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_record_start)
        btn_row.addWidget(self.btn_record_stop)
        btn_w = QWidget(); btn_w.setLayout(btn_row)
        f.addRow(btn_w)

        self._form.addRow(box)

    def _build_detect_ui(self) -> None:
        box = QGroupBox("Detect spots (S-map + classifier)")
        f = QFormLayout(box)

        # S-map / detection parameters
        self.ed_smap_n      = QLineEdit("80")
        self.ed_smap_smooth = QLineEdit("5")
        self.ed_smap_sig_s  = QLineEdit("1.0")
        self.ed_smap_sig_l  = QLineEdit("6.0")
        self.ed_smap_kstd   = QLineEdit("6.0")
        self.ed_minA        = QLineEdit("10")
        self.ed_maxA        = QLineEdit("250")
        self.ed_border      = QLineEdit("10")

        intv = QIntValidator(1, 10_000, self)
        dblv = QDoubleValidator(0.1, 100.0, 2, self)
        dblv.setNotation(QDoubleValidator.StandardNotation)
        self.ed_smap_n.setValidator(intv)
        self.ed_smap_smooth.setValidator(QIntValidator(1, 99, self))
        for ed in (self.ed_smap_sig_s, self.ed_smap_sig_l, self.ed_smap_kstd):
            ed.setValidator(dblv)
        self.ed_minA.setValidator(intv)
        self.ed_maxA.setValidator(intv)
        self.ed_border.setValidator(QIntValidator(0, 500, self))

        f.addRow("Frames (N)",            self.ed_smap_n)
        f.addRow("Smooth k",              self.ed_smap_smooth)
        f.addRow("\u03c3\u2081 (inner)",   self.ed_smap_sig_s)
        f.addRow("\u03c3\u2082 (outer)",   self.ed_smap_sig_l)
        f.addRow("k\u00b7\u03c3 threshold", self.ed_smap_kstd)
        f.addRow("Min area (px\u00b2)",    self.ed_minA)
        f.addRow("Max area (px\u00b2)",    self.ed_maxA)
        f.addRow("Border (px)",            self.ed_border)

        # Classifier parameters
        self.ed_n_phi_bins   = QLineEdit("9")
        self.ed_min_pts_bin  = QLineEdit("2")
        self.ed_cov_thr      = QLineEdit("0.75")
        self.ed_r_uni_thr    = QLineEdit("0.05")
        self.ed_n_phi_bins.setValidator(QIntValidator(2, 180, self))
        self.ed_min_pts_bin.setValidator(QIntValidator(1, 1000, self))
        cov_v = QDoubleValidator(0.0, 1.0, 3, self); cov_v.setNotation(QDoubleValidator.StandardNotation)
        uni_v = QDoubleValidator(0.0, 10.0, 4, self); uni_v.setNotation(QDoubleValidator.StandardNotation)
        self.ed_cov_thr.setValidator(cov_v)
        self.ed_r_uni_thr.setValidator(uni_v)

        f.addRow("φ-bins",                self.ed_n_phi_bins)
        f.addRow("Min pts/bin",            self.ed_min_pts_bin)
        f.addRow("Coverage thr",           self.ed_cov_thr)
        f.addRow("r-uniformity thr",       self.ed_r_uni_thr)

        self.btn_detect = QPushButton("Detect")
        self.btn_clear_overlays = QPushButton("Clear overlays")
        row = QHBoxLayout(); row.addWidget(self.btn_detect); row.addWidget(self.btn_clear_overlays)
        w = QWidget(); w.setLayout(row); f.addRow(w)

        self._form.addRow(box)

    def _build_spot_list_ui(self) -> None:
        box = QGroupBox("Spots")
        v = QVBoxLayout(box)
        self.chk_spinners_only = QCheckBox("Show spinners only")
        self.chk_spinners_only.setChecked(False)
        self.chk_spinners_only.stateChanged.connect(lambda _: self._refresh_spot_list())
        v.addWidget(self.chk_spinners_only)

        self.spot_list = QListWidget()
        self.spot_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.btn_view_spot = QPushButton("View spot…")
        self.btn_remove_spot = QPushButton("Remove selected")

        # Manual add spot
        add_form = QHBoxLayout()
        self.ed_add_cx = QLineEdit(); self.ed_add_cx.setPlaceholderText("cx")
        self.ed_add_cy = QLineEdit(); self.ed_add_cy.setPlaceholderText("cy")
        self.ed_add_r  = QLineEdit(); self.ed_add_r.setPlaceholderText("r")
        dv = QDoubleValidator(0.0, 100000.0, 2, self); dv.setNotation(QDoubleValidator.StandardNotation)
        for ed in (self.ed_add_cx, self.ed_add_cy, self.ed_add_r):
            ed.setValidator(dv); ed.setMaximumWidth(60)
        self.btn_add_spot = QPushButton("Add spot")
        add_form.addWidget(QLabel("cx")); add_form.addWidget(self.ed_add_cx)
        add_form.addWidget(QLabel("cy")); add_form.addWidget(self.ed_add_cy)
        add_form.addWidget(QLabel("r"));  add_form.addWidget(self.ed_add_r)
        add_form.addWidget(self.btn_add_spot)

        # Cycle controls
        self.chk_cycle_save = QCheckBox("Save cycle data")
        self.chk_cycle_save.setChecked(True)
        self.chk_cycle_raw = QCheckBox("Save raw pixels")
        self.chk_cycle_raw.setChecked(False)
        dwell_row = QHBoxLayout()
        dwell_row.addWidget(QLabel("Dwell (s)"))
        self.ed_cycle_dwell = QLineEdit("1.0")
        dv2 = QDoubleValidator(0.01, 3600.0, 2, self); dv2.setNotation(QDoubleValidator.StandardNotation)
        self.ed_cycle_dwell.setValidator(dv2)
        self.ed_cycle_dwell.setMaximumWidth(60)
        dwell_row.addWidget(self.ed_cycle_dwell)
        dwell_row.addStretch(1)
        self.btn_cycle_start = QPushButton("Start Cycle")
        self.btn_cycle_stop  = QPushButton("Stop Cycle")
        self.btn_cycle_stop.setEnabled(False)

        row = QHBoxLayout()
        row.addWidget(self.btn_view_spot)
        row.addWidget(self.btn_remove_spot)
        v.addWidget(self.spot_list, 1)
        v.addLayout(row)
        v.addLayout(add_form)
        v.addLayout(dwell_row)
        v.addWidget(self.chk_cycle_save)
        v.addWidget(self.chk_cycle_raw)
        v.addWidget(self.btn_cycle_start)
        v.addWidget(self.btn_cycle_stop)
        self._form.addRow(box)

    def _assemble_layout(self) -> None:
        right = QWidget(); right.setLayout(self._form)
        leftcol = QVBoxLayout()
        leftcol.addLayout(self._toolbar)
        leftcol.addWidget(self.video, 1)
        leftcol.addWidget(self.tone)
        root = QHBoxLayout()
        root.addLayout(leftcol, 2)
        root.addWidget(right, 1)
        cw = QWidget(); cw.setLayout(root)
        self.setCentralWidget(cw)

    # ---------- button state helper ----------
    def _set_buttons(self, *, open_enabled: bool, start: bool, stop: bool, close: bool) -> None:
        self.btn_open.setEnabled(open_enabled)
        self.btn_start.setEnabled(start)
        self.btn_stop.setEnabled(stop)
        self.btn_close.setEnabled(close)

    # Centralize UI lock/unlock while cycling
    def _set_cycle_ui_state(self, active: bool) -> None:
        self._cycle_active = bool(active)
        # Buttons that could fight with cycling
        for w in [
            self.btn_detect, self.btn_clear_overlays,
            self.btn_apply_roi, self.btn_full_roi,
            self.btn_apply_tim, self.btn_desat,
            self.btn_apply_gains, self.btn_refresh_gains,
            self.btn_view_spot, self.btn_remove_spot,
            self.spot_list,
        ]:
            try:
                w.setEnabled(not active)
            except Exception:
                pass
        self.btn_cycle_start.setEnabled(not active)
        self.btn_cycle_stop.setEnabled(active)
        if active:
            self._last_preview_t = 0.0  # reset throttle timer

    # ---------- backend signal handlers ----------
    def _on_open(self, name: str) -> None:
        self.status.showMessage(f"Opened: {name}", 1500)
        self._set_buttons(open_enabled=False, start=True, stop=False, close=True)
        self._refresh_gains()

    def _on_started(self) -> None:
        self.status.showMessage("Started", 1500)
        self._set_buttons(open_enabled=False, start=False, stop=True, close=False)

    def _on_stopped(self) -> None:
        self.status.showMessage("Stopped", 1500)
        self._set_buttons(open_enabled=False, start=True, stop=False, close=True)

    def _on_closed(self) -> None:
        self.status.showMessage("Closed", 1500)
        self.video.setPixmap(QPixmap())
        self._last_pm = None
        self._set_buttons(open_enabled=True, start=False, stop=False, close=False)

    def _on_error(self, msg: str) -> None:
        self.status.showMessage(f"Error: {msg}", 5000)
        QMessageBox.warning(self, "Camera error", msg)

    def _on_frame(self, arr_obj: object) -> None:
        try:
            a16 = np.asarray(arr_obj)
            if a16.ndim != 2:
                return

            # histogram ~2 Hz for LUT widget
            if time.time() - self._hist_t_last > 0.5:
                self._hist_t_last = time.time()
                vals = (a16.astype(np.uint16, copy=False) >> 4).ravel()
                hist = np.bincount(vals, minlength=256)
                if hist.size > 256: hist = hist[:256]
                self.tone.setHistogram256(hist)

            # PREVIEW THROTTLE while cycling: cap to ~20 fps
            if self._cycle_active and self._preview_cap_fps and self._preview_cap_fps > 0:
                now = time.perf_counter()
                if now - self._last_preview_t < (1.0 / float(self._preview_cap_fps)):
                    return
                self._last_preview_t = now

            if self._lut is not None:
                if a16.dtype != np.uint16: a16 = a16.astype(np.uint16, copy=False)
                a8 = self._lut[a16]
            else:
                if a16.dtype != np.uint16: a16 = a16.astype(np.uint16, copy=False)
                a8 = ((a16 + 8) >> 4).astype(np.uint8, copy=False)

            if not a8.flags.c_contiguous:
                a8 = np.ascontiguousarray(a8)

            pm = self._compose_pixmap_with_overlays(a8)
            self._last_pm = pm
            self._refresh_video_view()
        except Exception as e:
            self.status.showMessage(f"Frame error: {e}", 2000)

    def _compose_pixmap_with_overlays(self, a8: np.ndarray) -> QPixmap:
        """Draw spot overlays on the unscaled frame, then return a QPixmap."""
        h, w = a8.shape
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        pm = QPixmap.fromImage(qimg.copy())  # own memory for safety

        has_spots = bool(self._spots)
        has_preview = self._preview_spot is not None
        if not has_spots and not has_preview:
            return pm

        from PySide6.QtGui import QColor
        LABEL_COLOURS = {
            "good spinner":      QColor(Qt.green),
            "irregular spinner": QColor(255, 200, 0),   # yellow-orange
            "partial":           QColor(Qt.red),
        }

        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        font = QFont()
        font.setPointSize(9)
        p.setFont(font)

        ox, oy = self._roi_offset
        for i, s in enumerate(self._spots):
            try:
                cx = int(round(s.cx - ox))
                cy = int(round(s.cy - oy))
                rr = max(1, min(40, int(round(s.r))))
            except Exception:
                continue

            colour = LABEL_COLOURS.get(s.label, QColor(Qt.green))
            pen = QPen(colour, 2)
            p.setPen(pen)

            if 0 <= cx < w and 0 <= cy < h:
                p.drawEllipse(cx - rr, cy - rr, 2 * rr, 2 * rr)
                p.drawText(cx + rr + 3, cy - rr - 2, f"{i + 1}")

        # Preview circle for manual add-spot (cyan, dashed)
        if has_preview:
            pcx, pcy, pr = self._preview_spot
            px_i = int(round(pcx))
            py_i = int(round(pcy))
            pr_i = max(1, int(round(pr)))
            if 0 <= px_i < w and 0 <= py_i < h:
                pen = QPen(QColor(0, 200, 255), 2, Qt.DashLine)
                p.setPen(pen)
                p.drawEllipse(px_i - pr_i, py_i - pr_i, 2 * pr_i, 2 * pr_i)
                p.drawText(px_i + pr_i + 3, py_i - pr_i - 2, "?")

        p.end()
        return pm

    def _on_timing(self, d: dict) -> None:
        """Keep exposure displayed in *milliseconds* (label says ms)."""
        try:
            fps = d.get("fps")
            exp_us = d.get("exposure_us")
            if fps is not None:
                self.ed_fps.setText(f"{float(fps):.3f}")
            if exp_us is not None:
                self.ed_exp.setText(f"{float(exp_us) / 1000.0:.3f}")  # μs → ms
            msg = []
            if fps is not None:   msg.append(f"FPS={float(fps):.3f}")
            if exp_us is not None:msg.append(f"EXP={float(exp_us)/1000.0:.3f} ms")
            if msg: self.status.showMessage("Timing applied: " + "  ".join(msg), 2000)
        except Exception:
            pass

    def _on_roi(self, d: dict) -> None:
        try:
            mapping = [
                ("Width", self.ed_w),
                ("Height", self.ed_h),
                ("OffsetX", self.ed_x),
                ("OffsetY", self.ed_y),
            ]
            for key, edit in mapping:
                v = d.get(key)
                if v is not None:
                    edit.setText(str(int(round(float(v)))))
            # keep numeric copies for overlays
            W = d.get("Width"); H = d.get("Height")
            X = d.get("OffsetX"); Y = d.get("OffsetY")
            if None not in (W, H, X, Y):
                self._roi_size = (int(round(float(W))), int(round(float(H))))
                self._roi_offset = (int(round(float(X))), int(round(float(Y))))
        except Exception:
            pass

    def _on_gains(self, g: dict) -> None:
        try:
            ana = g.get("analog", {}) or {}
            dig = g.get("digital", {}) or {}
            if ana.get("val") is not None:
                self.ed_gain_ana.setText(f"{float(ana['val']):.3f}")
            if dig.get("val") is not None:
                self.ed_gain_dig.setText(f"{float(dig['val']):.3f}")
            def rng(dct: dict) -> str:
                mn = dct.get("min"); mx = dct.get("max"); inc = dct.get("inc")
                bits = []
                if mn is not None and mx is not None: bits.append(f"{float(mn):.3f} … {float(mx):.3f}")
                if inc is not None: bits.append(f"inc {float(inc):.3f}")
                return "  ".join(bits)
            self._lbl_gain_ana.setText(rng(ana)); self._lbl_gain_dig.setText(rng(dig))
        except Exception:
            pass

    # ---------- detect ----------
    def _detect_clicked(self) -> None:
        if self._collecting:
            return
        try: n = max(2, int(float(self.ed_smap_n.text())))
        except Exception: n = 80
        try: sig_s = float(self.ed_smap_sig_s.text())
        except Exception: sig_s = 1.0
        try: sig_l = float(self.ed_smap_sig_l.text())
        except Exception: sig_l = 6.0
        try: k_std = float(self.ed_smap_kstd.text())
        except Exception: k_std = 6.0
        try: smooth_k = max(1, int(float(self.ed_smap_smooth.text())))
        except Exception: smooth_k = 5
        try: minA = max(1, int(float(self.ed_minA.text())))
        except Exception: minA = 10
        try: maxA = int(float(self.ed_maxA.text()))
        except Exception: maxA = 250
        try: border = max(0, int(float(self.ed_border.text())))
        except Exception: border = 10
        # Classifier params
        try: n_phi_bins = max(2, int(float(self.ed_n_phi_bins.text())))
        except Exception: n_phi_bins = 9
        try: min_pts_bin = max(1, int(float(self.ed_min_pts_bin.text())))
        except Exception: min_pts_bin = 2
        try: cov_thr = float(self.ed_cov_thr.text())
        except Exception: cov_thr = 0.75
        try: r_uni_thr = float(self.ed_r_uni_thr.text())
        except Exception: r_uni_thr = 0.05

        self._detect_params = {
            "n": n, "sigma1": sig_s, "sigma2": sig_l,
            "k_std": k_std, "smooth_k": smooth_k,
            "min_area": minA, "max_area": maxA, "border": border,
            "n_phi_bins": n_phi_bins, "min_pts_per_bin": min_pts_bin,
            "coverage_threshold": cov_thr, "r_uniformity_threshold": r_uni_thr,
        }
        self._smap_acc = None
        self._smap_n_collected = 0
        self._frame_buf = []
        self._collecting = True
        self.btn_detect.setEnabled(False)
        self.status.showMessage(f"Detect: accumulating {n} frames…", 0)

        if not self._detect_conn_active:
            try:
                self.ctrl.cam.frame.connect(self._collect_for_detect_smap, Qt.QueuedConnection)
                self._detect_conn_active = True
            except Exception:
                self._detect_conn_active = False
                self.detect_done.emit(("err", "Camera not available for detect."))

    def _collect_for_detect_smap(self, arr_obj: object) -> None:
        if not self._detect_conn_active:
            return
        try:
            img = np.asarray(arr_obj)
            if img.ndim != 2:
                return

            # Trim to even dims for mosaic consistency
            H, W = img.shape
            H2, W2 = (H // 2) * 2, (W // 2) * 2
            if H2 != H or W2 != W:
                img = img[:H2, :W2]

            # Lazily create the accumulator from the first frame's actual shape.
            if self._smap_acc is None:
                self._smap_acc = SMapAccumulator(
                    img.shape, smooth_k=self._detect_params.get("smooth_k", 5)
                )

            self._smap_acc.update(img)
            self._frame_buf.append(img.copy())
            self._smap_n_collected += 1
            n_target = self._detect_params.get("n", 80)

            if self._smap_n_collected < n_target:
                self.status.showMessage(
                    f"Detect: frame {self._smap_n_collected}/{n_target}…", 0
                )
                return

            # Enough frames accumulated — disconnect and run detection + classification.
            try:
                self.ctrl.cam.frame.disconnect(self._collect_for_detect_smap)
            except Exception:
                pass
            self._detect_conn_active = False

            s_map = self._smap_acc.compute()
            if s_map is None:
                self.detect_done.emit(("err", "S-map is empty."))
                return

            p = self._detect_params
            frame_shape = self._frame_buf[0].shape
            spots = detect_and_classify(
                s_map,
                self._frame_buf,
                frame_shape,
                sigma1=p.get("sigma1", 1.0),
                sigma2=p.get("sigma2", 6.0),
                k_std=p.get("k_std", 6.0),
                min_area=p.get("min_area", 10),
                max_area=p.get("max_area", 250),
                border=p.get("border", 10),
                n_phi_bins=p.get("n_phi_bins", 9),
                min_pts_per_bin=p.get("min_pts_per_bin", 2),
                coverage_threshold=p.get("coverage_threshold", 0.75),
                r_uniformity_threshold=p.get("r_uniformity_threshold", 0.05),
            )
            self._frame_buf = []  # free memory

            # Convert ROI-local coords to full-sensor coords so spots
            # remain valid even if the hardware ROI changes later.
            ox, oy = self._roi_offset
            if ox or oy:
                for s in spots:
                    s.cx += ox
                    s.cy += oy

            self.detect_done.emit(("ok", spots))
        except Exception as e:
            try:
                self.ctrl.cam.frame.disconnect(self._collect_for_detect_smap)
            except Exception:
                pass
            self._detect_conn_active = False
            self._frame_buf = []
            self.detect_done.emit(("err", str(e)))

    def _handle_detect_done(self, payload: object) -> None:
        self._collecting = False
        self.btn_detect.setEnabled(True)
        if not isinstance(payload, (list, tuple)) or len(payload) < 2:
            self._all_spots = []
            self._spots = []
            self._refresh_spot_list()
            self.status.showMessage("Detect failed: bad payload.", 3000)
            return
        kind, data = payload
        if kind == "ok":
            self._all_spots = list(data or [])
            self._refresh_spot_list()
            n_good = sum(1 for s in self._all_spots if s.label == "good spinner")
            n_irreg = sum(1 for s in self._all_spots if s.label == "irregular spinner")
            n_part = sum(1 for s in self._all_spots if s.label == "partial")
            self.status.showMessage(
                f"Detect: {len(self._all_spots)} spot(s) — "
                f"{n_good} good, {n_irreg} irregular, {n_part} partial.", 5000
            )
        else:
            self._all_spots = []
            self._spots = []
            self._refresh_spot_list()
            self.status.showMessage(f"Detect failed: {data}", 4000)

    def _refresh_spot_list(self) -> None:
        """Filter, sort, and display spots based on the toggle state."""
        spinners_only = self.chk_spinners_only.isChecked()

        if spinners_only:
            # Only irregular + good spinners, sorted by std_median_r ascending (best first)
            visible = [s for s in self._all_spots if s.label != "partial"]
            visible.sort(key=lambda s: s.std_median_r if np.isfinite(s.std_median_r) else 1e9)
        else:
            # All spots: partials first (by phi_cov ascending), then spinners (by std_median_r ascending)
            partials = [s for s in self._all_spots if s.label == "partial"]
            spinners = [s for s in self._all_spots if s.label != "partial"]
            partials.sort(key=lambda s: s.phi_cov)
            spinners.sort(key=lambda s: s.std_median_r if np.isfinite(s.std_median_r) else 1e9)
            visible = partials + spinners

        self._spots = visible
        self.spot_list.clear()

        LABEL_SYM = {"partial": "\u2717", "irregular spinner": "\u25ce", "good spinner": "\u2713", "manual": "+"}
        for i, s in enumerate(self._spots):
            sym = LABEL_SYM.get(s.label, "?")
            smr = f"{s.std_median_r:.4f}" if np.isfinite(s.std_median_r) else "n/a"
            item = QListWidgetItem(
                f"#{i+1} {sym} {s.label}  (x={s.cx:.1f}, y={s.cy:.1f})  "
                f"r\u2248{s.r:.1f}px  \u03c6={s.phi_cov*100:.0f}%  \u03c3r={smr}"
            )
            self.spot_list.addItem(item)

    def _clear_overlays(self) -> None:
        self._all_spots = []
        self._spots = []
        self._refresh_spot_list()
        self.status.showMessage("Overlays cleared.", 1500)

    # ---------- record frames ----------
    def _browse_record_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select recording folder",
            self.ed_record_dir.text() or os.getcwd(),
        )
        if d:
            self.ed_record_dir.setText(d)

    def _start_record_clicked(self) -> None:
        if self._frame_writer.is_running:
            return
        out_dir = self.ed_record_dir.text().strip() or os.path.join(os.getcwd(), "recordings")
        try:
            chunk = max(1, int(float(self.ed_record_chunk.text())))
        except Exception:
            chunk = 50
        try:
            max_f = max(0, int(float(self.ed_record_max.text())))
        except Exception:
            max_f = 0

        try:
            self.ctrl.cam.frame.connect(self._frame_writer.record, Qt.QueuedConnection)
        except Exception as exc:
            self.status.showMessage(f"Record: could not connect to camera — {exc}", 4000)
            return

        try:
            current_fps = float(self.ed_fps.text())
        except Exception:
            current_fps = 0.0

        self._frame_writer.start(
            out_dir=out_dir,
            base_name="frames",
            chunk_len=chunk,
            max_frames=max_f,
            fps=current_fps,
        )
        self.lbl_record_count.setText("0 frames")
        self.btn_record_start.setEnabled(False)
        self.btn_record_stop.setEnabled(True)
        self.status.showMessage(
            f"Recording to {out_dir}  (chunk={chunk}, max={max_f or '\u221e'})", 3000
        )

    def _stop_record_clicked(self) -> None:
        if not self._frame_writer.is_running:
            return
        try:
            self.ctrl.cam.frame.disconnect(self._frame_writer.record)
        except Exception:
            pass
        self._frame_writer.stop()  # triggers _on_record_stopped via signal

    @Slot(int)
    def _on_record_progress(self, n: int) -> None:
        self.lbl_record_count.setText(f"{n} frames")

    @Slot()
    def _on_record_stopped(self) -> None:
        total = self._frame_writer.frames_recorded
        self.btn_record_start.setEnabled(True)
        self.btn_record_stop.setEnabled(False)
        self.status.showMessage(f"Recording stopped — {total} frames saved.", 4000)

    @Slot(str)
    def _on_record_error(self, msg: str) -> None:
        self.status.showMessage(f"Record error: {msg}", 6000)
        self.btn_record_start.setEnabled(True)
        self.btn_record_stop.setEnabled(False)

    # ---------- view/remove spots ----------
    def _selected_spot_indices(self) -> List[int]:
        return sorted({it.row() for it in self.spot_list.selectedIndexes()})

    def _resume_main_video(self) -> None:
        if self._video_paused:
            try:
                self.ctrl.cam.frame.connect(self._on_frame, Qt.QueuedConnection)
                print("[MainWindow] Live preview resumed.")
            except Exception:
                pass
            self._video_paused = False

    def _view_selected_spots(self) -> None:
        idxs = self._selected_spot_indices()
        if not self._spots:
            self.status.showMessage("No spots to view.", 2000)
            return
        if not idxs:
            idxs = [0]
        sel_spots = [self._spots[i] for i in idxs]

        # Pause main-window live preview to reduce UI load while viewer is open
        if not self._video_paused:
            try:
                self.ctrl.cam.frame.disconnect(self._on_frame)
                self._video_paused = True
                print("[MainWindow] Live preview paused for Spot Viewer.")
            except Exception:
                pass

        # snapshot current ROI & FPS so the viewer can restore exactly
        W, H = self._roi_size
        X, Y = self._roi_offset
        try:
            fps_saved = float(self.ed_fps.text()) if self.ed_fps.text().strip() else None
        except Exception:
            fps_saved = None

        dlg = SpotViewerDialog(self.ctrl, sel_spots, self, saved_roi=(W, H, X, Y), saved_fps=fps_saved)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.destroyed.connect(self._resume_main_video)  # resume when the dialog closes
        dlg.show()

    def _remove_selected_spots(self) -> None:
        idxs = self._selected_spot_indices()
        if not idxs:
            return
        # Remove from both the visible list and the master list
        to_remove = {id(self._spots[i]) for i in idxs if i < len(self._spots)}
        self._all_spots = [s for s in self._all_spots if id(s) not in to_remove]
        self._refresh_spot_list()

    def _update_add_spot_preview(self) -> None:
        """Update the preview circle whenever add-spot fields change."""
        try:
            cx = float(self.ed_add_cx.text())
            cy = float(self.ed_add_cy.text())
            r = float(self.ed_add_r.text())
            if r > 0:
                self._preview_spot = (cx, cy, r)
            else:
                self._preview_spot = None
        except (ValueError, TypeError):
            self._preview_spot = None

    def _add_spot_manual(self) -> None:
        """Add a spot from the cx/cy/r text fields. Coords are ROI-local."""
        try:
            cx_local = float(self.ed_add_cx.text())
            cy_local = float(self.ed_add_cy.text())
            r = float(self.ed_add_r.text())
        except (ValueError, TypeError):
            self.status.showMessage("Add spot: enter valid cx, cy, r.", 2500)
            return
        if r <= 0:
            self.status.showMessage("Add spot: radius must be > 0.", 2500)
            return
        # Convert ROI-local → full-sensor coords
        ox, oy = self._roi_offset
        cx_abs = cx_local + ox
        cy_abs = cy_local + oy
        spot = Spot(
            cx=cx_abs, cy=cy_abs, r=r,
            label="manual", phi_cov=0.0, std_median_r=float('inf'),
        )
        self._all_spots.append(spot)
        self._refresh_spot_list()
        self.status.showMessage(
            f"Added manual spot at sensor ({cx_abs:.1f}, {cy_abs:.1f}) r={r:.1f}", 3000
        )

    # ---------- CYCLE: start/stop threads ----------
    def _start_cycle_clicked(self) -> None:
        if not self._spots:
            QMessageBox.information(self, "Cycle", "No spots selected/detected.")
            return
        out_dir = os.path.join(os.getcwd(), "cycles",
                              time.strftime("cycle_%Y%m%d_%H%M%S"))
        save_on = self.chk_cycle_save.isChecked()
        raw_on = self.chk_cycle_raw.isChecked()
        try:
            dwell = max(0.01, float(self.ed_cycle_dwell.text()))
        except Exception:
            dwell = 1.0
        cfg = CycleConfig(
            out_dir=out_dir,
            base_name="cycle",
            dwell_sec=dwell,
            max_duration_sec=3600,
            chunk_len=20000,
            maximize_camera_fps=True,
            save_enabled=save_on,
            save_raw=raw_on,
        )

        if self._cycle_thread is not None:
            QMessageBox.information(self, "Cycle", "Cycler already running.")
            return

        # Build cycler with error guard — if __init__ throws, self._cycler would stay None.
        try:
            cycler = MultiSpotCycler(self.ctrl, self._spots, cfg)
        except Exception as e:
            self.status.showMessage(f"Cycle init error: {e}", 6000)
            return

        self._cycle_thread = QThread(self)
        self._cycler = cycler
        self._cycler.moveToThread(self._cycle_thread)

        # signals
        self._cycle_thread.started.connect(self._cycler.start,       Qt.QueuedConnection)
        self._cycler.progress.connect(self._on_cycler_progress,      Qt.QueuedConnection)
        self._cycler.error.connect(self._on_cycler_error,            Qt.QueuedConnection)
        self._cycler.stopped.connect(self._cycle_thread.quit,        Qt.QueuedConnection)
        self._cycler.advise_ui_cap.connect(self._on_cycle_ui_cap,    Qt.QueuedConnection)
        self._cycle_thread.finished.connect(self._cycle_finished,    Qt.QueuedConnection)

        # Enable preview throttle on our side; worker will also advise via signal
        self._cycle_active = True
        self._last_preview_t = 0.0

        self._cycle_thread.start()
        self.btn_cycle_start.setEnabled(False)
        self.btn_cycle_stop.setEnabled(True)
        self.status.showMessage("Cycle started — recording at max FPS, preview capped at 20 FPS.", 4000)

    def _stop_cycle_clicked(self) -> None:
        # User stop — don't block the GUI thread; _cycle_finished runs on 'finished' signal
        if self._cycler:
            try: self._cycler.stop()
            except Exception: pass

    def _cycle_finished(self) -> None:
        # Thread finished
        self._cycler = None
        self._cycle_thread = None
        self._set_cycle_ui_state(False)
        self.status.showMessage("Cycle stopped.", 2500)

    @Slot(str)
    def _on_cycler_progress(self, s: str) -> None:
        # Safe: runs on GUI thread
        self.status.showMessage(s, 1200)

    @Slot(str)
    def _on_cycler_error(self, msg: str) -> None:
        # Safe: runs on GUI thread
        self.status.showMessage(f"Cycle error: {msg}", 5000)

    @Slot(float)
    def _on_cycle_ui_cap(self, cap: float) -> None:
        # 0.0 = uncap (disable throttle); >0 = cap to that FPS
        if cap and cap > 0.0:
            self._preview_cap_fps = float(cap)
            self._cycle_active = True
        else:
            self._preview_cap_fps = 0.0
            self._cycle_active = False
        self._last_preview_t = 0.0  # reset throttle phase

    # ---------- UI handlers (UI → controller) ----------
    def _open_clicked(self) -> None:
        if hasattr(self.ctrl, "open"): self.ctrl.open()

    def _start_clicked(self) -> None:
        if self.btn_start.isEnabled() and hasattr(self.ctrl, "start"): self.ctrl.start()

    def _stop_clicked(self) -> None:
        if self.btn_stop.isEnabled() and hasattr(self.ctrl, "stop"): self.ctrl.stop()

    def _close_clicked(self) -> None:
        if self.btn_close.isEnabled() and hasattr(self.ctrl, "close"): self.ctrl.close()

    def _apply_roi(self) -> None:
        try:
            w = int(float(self.ed_w.text() or "0"))
            h = int(float(self.ed_h.text() or "0"))
            x = int(float(self.ed_x.text() or "0"))
            y = int(float(self.ed_y.text() or "0"))
        except Exception:
            self.status.showMessage("ROI: invalid numbers", 2000); return
        if w <= 0 or h <= 0:
            self.status.showMessage("ROI: width/height must be > 0. Use Full sensor if unsure.", 3000); return
        if hasattr(self.ctrl, "set_roi"): self.ctrl.set_roi(w, h, x, y)

    def _full_roi_clicked(self) -> None:
        if hasattr(self.ctrl, "full_sensor"): self.ctrl.full_sensor()

    def _apply_timing(self) -> None:
        # exposure textbox is *milliseconds*
        fps = float(self.ed_fps.text()) if self.ed_fps.hasAcceptableInput() else None
        exp_ms = float(self.ed_exp.text()) if self.ed_exp.hasAcceptableInput() else None
        if hasattr(self.ctrl, "set_timing"):
            self.ctrl.set_timing(fps, exp_ms)

    def _apply_gains(self) -> None:
        ana = float(self.ed_gain_ana.text()) if self.ed_gain_ana.text().strip() else None
        dig = float(self.ed_gain_dig.text()) if self.ed_gain_dig.text().strip() else None
        if hasattr(self.ctrl, "set_gains"): self.ctrl.set_gains(ana, dig)

    def _refresh_gains(self) -> None:
        if hasattr(self.ctrl, "refresh_gains"):
            self.ctrl.refresh_gains(); self.status.showMessage("Refreshing gains…", 800)
        elif hasattr(self.ctrl, "cam") and hasattr(self.ctrl.cam, "refresh_gains"):
            self.ctrl.cam.refresh_gains(); self.status.showMessage("Refreshing gains…", 800)

    def _desaturate_clicked(self) -> None:
        if hasattr(self.ctrl, "desaturate"):
            self.status.showMessage("Auto-desaturating…", 2000)
            self.ctrl.desaturate(0.85, 5)

    def _on_desaturated(self, d: dict) -> None:
        it = int(d.get("iterations", 0))
        mx = int(d.get("final_max", -1))
        ok = bool(d.get("success", False))
        exp_us = d.get("exposure_us", None)
        tgt = int(d.get("target", 0))
        if ok:
            self.status.showMessage(f"Desaturate done: iters={it}, max={mx}, target={tgt}, exposure={(exp_us or 0)/1000.0:.1f} ms", 3000)
        else:
            QMessageBox.warning(
                self, "Desaturate",
                f"Couldn’t reach target after {max(it, 5)} tries.\n"
                f"Final max={mx}, target={tgt}, exposure={(exp_us or 0)/1000.0:.1f} ms"
            )

    def _on_desat_started(self) -> None:
        if hasattr(self, "btn_desat"): self.btn_desat.setEnabled(False)
        self.status.showMessage("Auto-desaturating…", 2000)

    def _on_desat_finished(self) -> None:
        if hasattr(self, "btn_desat"): self.btn_desat.setEnabled(True)
        self.status.showMessage("Auto-desaturate complete.", 1500)

    # ---------- LUT plumbing ----------
    def _on_tone_params(self, floor: int, cap: int, gamma: float) -> None:
        self._lut = self._build_lut(floor, cap, gamma)

    @staticmethod
    def _build_lut(floor: int, cap: int, gamma: float) -> np.ndarray:
        cap = max(floor + 1, min(cap, 4095)); floor = max(0, min(floor, 4094))
        gamma = max(0.05, min(gamma, 10.0))
        x = np.arange(4096, dtype=np.float32)
        t = (x - float(floor)) / float(cap - floor)
        t = np.clip(t, 0.0, 1.0)
        y = np.power(t, gamma) * 255.0
        return np.clip(np.rint(y), 0, 255).astype(np.uint8)

    def _refresh_video_view(self) -> None:
        pm = getattr(self, "_last_pm", None)
        if pm is None: return
        self.video.setPixmap(pm.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._refresh_video_view()

    # ---- TEST ONLY — delete this entire method block to clean up ----
    def _load_avi_test(self) -> None:
        """Load an AVI and replace the camera feed with looping frames."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open AVI for mock camera", "", "Video files (*.avi *.mp4 *.mov);;All files (*)"
        )
        if not path:
            return

        # Stop and tear down any previous mock
        if self._mock_cam is not None:
            self._mock_cam.stop()
            self._mock_cam.close()
            try:
                self._mock_cam.frame.disconnect(self._on_frame)
            except Exception:
                pass
            self._mock_cam = None

        mock = AviMockCamera()

        # Wire mock signals → existing GUI handlers so everything behaves normally
        mock.opened.connect(self._on_open,    Qt.QueuedConnection)
        mock.started.connect(self._on_started, Qt.QueuedConnection)
        mock.stopped.connect(self._on_stopped, Qt.QueuedConnection)
        mock.closed.connect(self._on_closed,  Qt.QueuedConnection)
        mock.error.connect(self._on_error,    Qt.QueuedConnection)
        mock.roi.connect(self._on_roi,        Qt.QueuedConnection)
        mock.timing.connect(self._on_timing,  Qt.QueuedConnection)
        mock.frame.connect(self._on_frame,    Qt.QueuedConnection)

        # Replace ctrl.cam so the cycler, recorder, and detect all talk to mock
        self.ctrl.cam = mock
        self._mock_cam = mock

        try:
            mock.open(path)
            mock.start()
        except Exception as exc:
            self.status.showMessage(f"AVI mock error: {exc}", 6000)
    # ---- END TEST -------------------------------------------------------

    # ---------- cleanup ----------
    def safe_shutdown(self) -> None:
        try:
            if self._frame_writer.is_running:
                try: self.ctrl.cam.frame.disconnect(self._frame_writer.record)
                except Exception: pass
                self._frame_writer.stop()
        except Exception:
            pass
        try:
            # stop cycler if running
            if self._cycler:
                try: self._cycler.stop()
                except Exception: pass
            if self._cycle_thread:
                self._cycle_thread.wait(2000)
        except Exception:
            pass
        try:
            if hasattr(self.ctrl, "stop"): self.ctrl.stop()
        except Exception: pass
        try:
            if hasattr(self.ctrl, "close"): self.ctrl.close()
        except Exception: pass

    def closeEvent(self, e) -> None:
        try:
            if self._frame_writer.is_running:
                try: self.ctrl.cam.frame.disconnect(self._frame_writer.record)
                except Exception: pass
                self._frame_writer.stop()
        except Exception:
            pass
        try:
            if self._cycler:
                try: self._cycler.stop()
                except Exception: pass
            if self._cycle_thread:
                self._cycle_thread.wait(2000)
        except Exception:
            pass
        try:
            if self.btn_stop.isEnabled() and hasattr(self.ctrl, "stop"):
                self.ctrl.stop(); QApplication.processEvents()
        except Exception:
            pass
        try:
            if hasattr(self.ctrl, "close"): self.ctrl.close()
        except Exception:
            pass
        super().closeEvent(e)
