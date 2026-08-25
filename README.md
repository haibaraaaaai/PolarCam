# PolarCam (rewrite) — lab README


## 0) TL;DR — run it

```bash
# 1) create & activate venv
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install -U pip

# 2) editable install
python -m pip install -e .

# 3) IDS SDK
# Install the IDS peak Cockpit.
# Then the Python wheels:
python -m pip install --no-deps ids-peak ids-peak-ipl

# 4) launch (choose one)
polarcam          # console visible (logs)
polarcam-gui      # no console window on Windows
# or
python -m polarcam.cli
```

**First things to check in the UI**

* **Exposure is in milliseconds**
* **ROI**: "Full sensor" to set max ROI; then use small ROIs for speed.
* **Gains**: hit **Refresh gains** to populate min/max if not already shown.

---

## 1) What this app does today

* Live preview of the polarization camera (12→8‑bit highlight LUT with floor/cap/gamma).
* ROI + timing controls (FPS + **exposure in ms**), desaturate helper.
* Analog/digital gains.
* **Spot detection** on a given number of frames (DoG‑based, with two‑stage φ‑coverage / r‑uniformity classification).
* **Spot viewer**: shows a zoomed crop with circular ROI mask, live 4‑channel anisotropy scatter plot, optional **recording at max camera FPS** (saves raw pixel `.npy` shards + JSON metadata with mosaic layout info).
* **Manual add‑spot** with live preview circle (cyan dashed) shown before committing.
* **Multi‑spot cycler/recorder**: hops a tight HW‑ROI around selected spots with configurable dwell time, records 4×pol mean signals per dwell to compressed `.npz` shards. Optional **raw pixel saving** (one uncompressed `.npz` + companion `.json` per spot per dwell with full mosaic layout metadata).
* **Test (AVI)** button: loads an AVI file as a mock camera for offline testing.

**Directory conventions**

* Cycle outputs land in `./cycles/cycle_YYYYMMDD_HHMMSS/cycle_spotXX/…npz` (timestamped per run).
* Spot viewer recordings land in `./captures/spotN_YYYYMMDD_HHMMSS/…npy` + `…_meta.json`.

---

## 2) Typical workflows

### A) Live + detect + view

1. **Open → Start**.
2. **Full sensor**, then tweak exposure/gains or run **Desaturate**.
3. **Detect spots** (threshold mode: *absolute* DN or *percentile*; defaults are fine).
4. Select one or more spots → **View spot…** (separate window with scatter plot). This pauses the main live preview while the viewer runs.

   * **Start Rec @ max FPS** inside the viewer to save raw pixel crops to `.npy` shards + a `_meta.json` with crop position, channel layout, and ROI details.

### B) Cycle multiple spots

1. Detect/select spots (or add manually with cx/cy/r fields — preview circle shows before adding).
2. Set **dwell time** (default 1.0 s), check **Save cycle data** and/or **Save raw pixels** as needed.
3. Click **Start Cycle**.

   * Worker reduces HW‑ROI around each spot aggressively (respecting IDS step/min limits).
   * After each ROI change, worker calls **`set_timing(inf, None)`** to hit the maximum FPS permitted by ROI + exposure.
   * Circular mask is applied — only pixels within the spot radius contribute to channel means.
   * UI preview is throttled (≤20 Hz), while recording uses the max camera rate.
4. Outputs go to `./cycles/cycle_spotXX/`:
   * `*_sNN_cNNNN.npz` — per‑chunk 4‑channel mean signals (if "Save cycle data" is checked).
   * `*_sNN_dNNNN_raw.npz` + `*_sNN_dNNNN_raw.json` — raw pixel stacks per dwell (if "Save raw pixels" is checked).

---

## 3) Spot processing pipeline

A spot is defined by `(cx, cy, r)` in full-sensor coordinates.  Three layers of cropping/masking happen:

```
Full sensor (2464 × 2056)
 └─ Hardware ROI (w, h, x, y) — set on camera, reduces bus bandwidth
     └─ Software crop (even-sided square ≈ 2r + padding, centred on spot)
         └─ Circular mask (radius r, centred on spot)
```

| Step | What happens | Who does it |
|------|-------------|-------------|
| **HW ROI** | Camera reads out only a rectangle around the spot. **Cycler & recorder** use `roi_for_spot(cx, cy, r)` (defaults: `margin=0, min_r=0`) → aggressive 256 px wide × ≈2r tall strip for max FPS. **Viewer** uses `roi_for_spot(…, margin=6, min_r=4)` → comfortable square ≈ 2·max(4,r)+12 on each side for a nicer preview. | Camera hardware |
| **Software crop** | A smaller even-sided square is cut from the delivered frame, tightly fitting the spot diameter + a few pixels of padding. | Recorder / cycler `_on_frame` |
| **Circular mask** | Inside the software crop, pixels outside the circle of radius `r` are either **zeroed** (recorder) or **excluded** from means (cycler / viewer scatter). | Recorder / cycler / viewer |

### What each mode saves & computes

| | **Signal computation** | **Saved to disk** | **Masking in saved data** |
|---|---|---|---|
| **Spot viewer** (scatter) | 4-channel means from **masked** (circular) pixels only | *(delegates to recorder below)* | — |
| **Spot recorder** (`.npy` shards) | None (saves raw) | Rectangular crop `(N, h, w)` uint16 | **Zeroed** outside circle |
| **Cycler** (channel means `.npz`) | 4-channel means from **masked** (circular) pixels only | Float64 mean arrays per channel | N/A (already reduced) |
| **Cycler raw** (`_raw.npz`) | *(same as above)* | Rectangular crop `(N, h, w)` uint16 | **Not masked** — full rectangle preserved |

The cycler keeps raw pixels unmasked intentionally: it's archival data that can be reprocessed with a different radius. The recorder zeros outside the circle because those shards are typically consumed directly and the zeros compress well.

### Reference pixel: `crop_top_left_sensor_yx`

The saved metadata field `crop_top_left_sensor_yx = [y, x]` is the **absolute sensor coordinate of the top-left corner of the rectangular crop** — i.e. pixel `[0, 0]` of the saved array. It is the rectangle corner, *not* the first non-zero/masked pixel.

To recover the mosaic channel of any pixel `[row, col]` in a saved crop:

```python
tl_y, tl_x = meta["crop_top_left_sensor_yx"]
channel = MOSAIC_LAYOUT[((tl_y + row) % 2, (tl_x + col) % 2)]
```

---

## 4) Data formats

### Cycle `.npz` shards (channel means)

Each file contains arrays: `t` (seconds from local `t0`), `c0`, `c45`, `c90`, `c135` (float64 per‑frame means), and a `meta` JSON blob (bytes) with:

* `spot = {cx, cy, r, label}`
* `applied_roi = {x, y, w, h}` (HW‑ROI during dwell)
* `crop_abs = {x, y, w, h}` (software crop used to compute means)
* `t0_perf_counter`

### Cycle raw pixel `.npz` + `.json` (per spot per dwell)

The `.npz` contains `frames` `(N, h, w)` uint16 stack, `t` (relative timestamps), and `meta` (JSON as byte array). The companion `.json` has the same metadata in a human‑readable file:

* `spot = {cx, cy, r, label}`
* `layout = {"(0,0)": "90", "(0,1)": "45", "(1,0)": "135", "(1,1)": "0"}`
* `crop_top_left_sensor_yx = [y, x]` — position of the crop's top‑left pixel in full‑sensor coordinates
* `crop_top_left_channel` — polarization angle of that top‑left pixel (e.g. `"90"`)
* `applied_roi`, `crop_abs`, `t0_perf_counter`, `n_frames`, `n_discarded`

### Spot viewer `.npy` shards + `_meta.json`

`.npy` shards contain raw pixel crops `(chunk_len, h, w)` uint16. The `_meta.json` records:

* `spot`, `layout`, `crop_top_left_sensor_yx`, `crop_top_left_channel`
* `roi_hw_requested`, `crop_hw`, `crop_offset_in_roi`

---

## 5) Offline helpers

### `offline/merge_recording.py`

Merges chunked full‑frame `.npy` recordings into a single file.

### `offline/merge_spot_capture.py`

Merges spot‑viewer `.npy` shards for a given capture into a single contiguous array.

---

## 6) Polarization mosaic layout

The sensor uses a 2×2 super‑pixel layout indexed by `(row % 2, col % 2)`:

| (row%2, col%2) | Angle |
|---|---|
| (0, 0) | 90° |
| (0, 1) | 45° |
| (1, 0) | 135° |
| (1, 1) | 0° |

All saved metadata references (`crop_top_left_channel`, `layout` dicts) follow this convention.

---

## 7) Project structure

```
src/polarcam/
├── __init__.py          # Package bootstrap, logging setup
├── __main__.py          # python -m polarcam entry point
├── cli.py               # QApplication launcher, arg parsing
├── hardware.py          # Shared sensor constants, mosaic layout, snap helpers
├── analysis/
│   ├── detect.py        # DoG spot detection
│   ├── pol_reconstruction.py  # (X,Y) anisotropy + (Q,U) reconstructors
│   └── smap.py          # S-map accumulator (per-pixel min/max of Q/U)
├── app/
│   ├── main_window.py   # Main GUI window
│   ├── lut_widget.py    # 12-bit highlight LUT widget
│   ├── spot_detect.py   # Detection + classification orchestrator
│   ├── spot_viewer.py   # Per-spot viewer with scatter plot + recorder
│   ├── spot_cycler.py   # Multi-spot cycling recorder
│   └── spot_recorder.py # Per-spot raw pixel shard recorder
├── backend/
│   ├── base.py          # Abstract camera interface (ICamera)
│   ├── ids_backend.py   # IDS peak camera backend
│   └── mock_backend.py  # AVI-based mock camera for testing
├── capture/
│   └── frame_writer.py  # Full-frame chunked writer with background I/O
└── controller/
    └── controller.py    # Thin controller over IDSCamera
```

---

## 8) Known gotchas

* **ROI snapping**: hardware enforces step/min/max; snapping to a spot can over/under cut it.
* **Dwell/save settings are snapshot**: toggling checkboxes mid‑cycle has no effect; values are read when you click Start Cycle.
* **Transition frames**: when the HW‑ROI hops between spots the first 2–3 frames may have the wrong crop dimensions. The cycler discards these automatically (reported as `n_discarded` in raw metadata).

---

## 9) Notes

* Alias guard: fps/4.
* Hardware: Sensor 2464×2056, ROI width step 4, height step 2, min width 256, min height 2.
* Polarization layout: `(0,0)=90°`, `(0,1)=45°`, `(1,0)=135°`, `(1,1)=0°`.

---

## 10) History

This repository started as a clean rewrite of
[`haibaraaaaai/PolarCam`](https://github.com/haibaraaaaai/PolarCam), originally
as a separate repo rather than a branch. The pre‑rewrite history has since been
merged back in, so `git log` here contains the commits of **both** projects.

* The old project's files were moved under `legacy/` before merging, so nothing
  in `src/polarcam/`, `offline/` or `reference_files/` was touched and every
  pre‑existing commit SHA in this repo is unchanged.
* `git log --graph` therefore shows more than one root commit — that is
  expected, not corruption.
* `git log -- legacy/<path>` and `git blame legacy/<path>` still resolve to the
  original authors and dates.

`legacy/` is kept for reference only; it is not part of the installable
package and is not maintained. Deleting it in a future commit would not lose
any history.
