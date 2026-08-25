"""
Two-stage spinner classifier for PolarCam detected spots.

Stage 1 — φ-coverage
    Divide φ ∈ (−90°, 90°] into *n_phi_bins* equal bins (π-degeneracy means
    180° = full orientation coverage).  If the fraction of occupied bins
    ≥ *coverage_threshold* the spot is a **spinner**; otherwise **partial**.

Stage 2 — r-uniformity (spinners only)
    Compute std(median_r per occupied bin).  If below
    *r_uniformity_threshold* → **good spinner**; otherwise
    **irregular spinner**.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SpotClassification:
    """Classification result for a single detected spot."""
    label: str          # "partial", "irregular spinner", "good spinner"
    phi_cov: float      # fraction of φ-bins occupied (0–1)
    std_median_r: float # std of median(r) per occupied φ-bin (NaN for partials with < 2 bins)


def classify_spots(
    centers: list[tuple[float, float, float]],
    frames: Sequence[np.ndarray],
    frame_shape: tuple[int, int],
    *,
    n_phi_bins: int = 9,
    min_pts_per_bin: int = 2,
    coverage_threshold: float = 0.75,
    r_uniformity_threshold: float = 0.05,
) -> list[SpotClassification]:
    """
    Classify detected spots using per-spot anisotropy traces.

    Parameters
    ----------
    centers : list of (cx, cy, r_eff)
        Spot positions in S-map coordinates.  The raw-frame centre is at
        ``(cx + 0.5, cy + 0.5)`` due to the half-pixel offset of the
        intersection grid.
    frames : sequence of np.ndarray
        Raw mosaic frames (H, W) used for anisotropy-trace computation.
    frame_shape : (H, W)
        Shape of each frame.
    n_phi_bins, min_pts_per_bin, coverage_threshold, r_uniformity_threshold
        Classifier tuning knobs (see module docstring).

    Returns
    -------
    classifications : list of SpotClassification
        One entry per spot, in the same order as *centers*.
    """
    H, W = frame_shape
    n_frames = len(frames)
    if n_frames == 0 or not centers:
        return [SpotClassification("partial", 0.0, float("nan"))] * len(centers)

    phi_edges = np.linspace(-np.pi / 2, np.pi / 2, n_phi_bins + 1)
    results: list[SpotClassification] = []

    for cx_s, cy_s, r_eff in centers:
        # S-map node → raw-frame centre (half-pixel shift)
        cx_raw = cx_s + 0.5
        cy_raw = cy_s + 0.5

        # Tight bounding box around the spot
        r_ceil = int(np.ceil(r_eff)) + 1
        y0 = max(0, int(cy_raw) - r_ceil)
        y1 = min(H, int(cy_raw) + r_ceil + 1)
        x0 = max(0, int(cx_raw) - r_ceil)
        x1 = min(W, int(cx_raw) + r_ceil + 1)

        if y1 <= y0 or x1 <= x0:
            results.append(SpotClassification("partial", 0.0, float("nan")))
            continue

        # Circular mask within the patch
        py = np.arange(y0, y1, dtype=np.float64)
        px = np.arange(x0, x1, dtype=np.float64)
        gy, gx = np.meshgrid(py, px, indexing="ij")
        dist2 = (gy - cy_raw) ** 2 + (gx - cx_raw) ** 2
        circ = dist2 <= r_eff ** 2

        # Mosaic channel masks (absolute pixel parity)
        # (row%2, col%2): 0→90°, 1→45°, 2→135°, 3→0°
        parity = (np.arange(y0, y1)[:, None] % 2) * 2 + (np.arange(x0, x1)[None, :] % 2)
        m90  = circ & (parity == 0)
        m45  = circ & (parity == 1)
        m135 = circ & (parity == 2)
        m0   = circ & (parity == 3)

        # Per-frame anisotropy
        anis_x = np.empty(n_frames, dtype=np.float64)
        anis_y = np.empty(n_frames, dtype=np.float64)
        for fi, frame in enumerate(frames):
            patch = frame[y0:y1, x0:x1].astype(np.int64)
            C90  = int(patch[m90].sum())  if m90.any()  else 0
            C45  = int(patch[m45].sum())  if m45.any()  else 0
            C135 = int(patch[m135].sum()) if m135.any() else 0
            C0   = int(patch[m0].sum())   if m0.any()   else 0

            denom_x = C0 + C90
            denom_y = C45 + C135
            anis_x[fi] = (C0 - C90)   / denom_x if denom_x else 0.0
            anis_y[fi] = (C45 - C135) / denom_y if denom_y else 0.0

        # Polar coordinates
        r = np.sqrt(anis_x ** 2 + anis_y ** 2)
        phi = 0.5 * np.arctan2(anis_y, anis_x)  # ∈ (−π/2, π/2]

        bin_idx = np.clip(np.digitize(phi, phi_edges) - 1, 0, n_phi_bins - 1)

        occupied_bins: list[int] = []
        median_r_per_bin: list[float] = []
        for b in range(n_phi_bins):
            pts = r[bin_idx == b]
            if len(pts) >= min_pts_per_bin:
                occupied_bins.append(b)
                median_r_per_bin.append(float(np.median(pts)))

        n_occ = len(occupied_bins)
        frac_occ = n_occ / n_phi_bins

        # Stage 1 — full-rotation check
        full_rotation = frac_occ >= coverage_threshold

        # Stage 2 — r-uniformity check
        if full_rotation and n_occ >= 2:
            std_median_r = float(np.std(median_r_per_bin))
            good_spinner = std_median_r < r_uniformity_threshold
        else:
            std_median_r = float("nan")
            good_spinner = False

        if not full_rotation:
            lbl = "partial"
        elif good_spinner:
            lbl = "good spinner"
        else:
            lbl = "irregular spinner"

        results.append(SpotClassification(lbl, frac_occ, std_median_r))

    return results
