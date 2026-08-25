"""
Shared hardware constants, spot types, and small helpers for the IDS polarization camera.

Polar mosaic layout (row%2, col%2):
    (0,0) = 90°    (0,1) = 45°
    (1,0) = 135°   (1,1) = 0°
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ---- Sensor geometry ----
SENSOR_W, SENSOR_H = 2464, 2056

# ---- ROI step / min / max ----
STEP_W, STEP_H     = 4, 2
MIN_W,  MIN_H      = 256, 2
MAX_W,  MAX_H      = SENSOR_W, SENSOR_H

# ---- Offset constraints ----
OFFX_MIN, OFFX_MAX = 0, 2208       # SENSOR_W - MIN_W
OFFY_MIN, OFFY_MAX = 0, 2054       # SENSOR_H - MIN_H
OFFX_STEP, OFFY_STEP = 4, 2

# ---- Polarization mosaic layout ----
MOSAIC_LAYOUT: dict[tuple[int, int], str] = {
    (0, 0): "90",
    (0, 1): "45",
    (1, 0): "135",
    (1, 1): "0",
}

MOSAIC_LAYOUT_STR: dict[str, str] = {
    "(0,0)": "90",
    "(0,1)": "45",
    "(1,0)": "135",
    "(1,1)": "0",
}


# ---- Spot dataclass ----
@dataclass
class Spot:
    """A detected or manually defined spot on the sensor."""
    cx: float           # raw-frame x coordinate
    cy: float           # raw-frame y coordinate
    r: float            # effective radius (raw-frame pixels)
    label: str = ""                     # classification label
    phi_cov: float = 0.0                # φ-coverage fraction (0–1)
    std_median_r: float = float('nan')  # std of median(r) per occupied φ-bin


# ---- Small helpers ----
def snap_down(v: int, step: int) -> int:
    """Round *v* down to the nearest multiple of *step*."""
    s = int(step) if step > 0 else 1
    return int((int(v) // s) * s)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def snap_even(v: int) -> int:
    """Round up to the nearest even integer."""
    return int(v if (int(v) % 2 == 0) else int(v) + 1)


def roi_for_spot(
    cx: float, cy: float, r: float,
    *, margin: int = 0, min_r: float = 0.0,
) -> tuple[int, int, int, int]:
    """Hardware-aligned ROI centred on a spot.  Returns (w, h, x, y).

    margin=0  → aggressive (w fixed to MIN_W, h ≈ 2r).
    margin>0  → generous   (w, h ≈ 2·max(min_r, r) + 2·margin).
    """
    r_eff = max(float(min_r), max(0.0, float(r)))

    if margin > 0:
        w_want = max(MIN_W, int(math.ceil(2 * r_eff + 2 * margin)))
        h_want = max(MIN_H, int(math.ceil(2 * r_eff + 2 * margin)))
    else:
        w_want = MIN_W
        h_want = max(MIN_H, int(math.ceil(2.0 * r_eff)))

    # snap up to step multiples
    w = ((w_want + STEP_W - 1) // STEP_W) * STEP_W
    h = ((h_want + STEP_H - 1) // STEP_H) * STEP_H
    w = max(MIN_W, min(MAX_W, w))
    h = max(MIN_H, min(MAX_H, h))

    # centre offsets
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    x = snap_down(max(OFFX_MIN, min(OFFX_MAX, x)), OFFX_STEP)
    y = snap_down(max(OFFY_MIN, min(OFFY_MAX, y)), OFFY_STEP)

    # handle sensor boundary overflow
    if x + w > SENSOR_W:
        x = snap_down(SENSOR_W - w, OFFX_STEP)
        x = max(OFFX_MIN, min(OFFX_MAX, x))
    if y + h > SENSOR_H:
        y = snap_down(SENSOR_H - h, OFFY_STEP)
        y = max(OFFY_MIN, min(OFFY_MAX, y))

    return (w, h, x, y)
