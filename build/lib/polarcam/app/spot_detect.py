# src/polarcam/app/spot_detect.py
"""
Spot detection + 2-stage spinner classification pipeline for PolarCam.

Orchestrates:
  S-map DoG detection → per-spot anisotropy → classifier → sorted results.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from polarcam.analysis.detect import find_spot_centers_dog
from polarcam.analysis.classify import classify_spots
from polarcam.hardware import Spot

# Backward-compat alias
DetectedSpot = Spot


def detect_and_classify(
    s_map: np.ndarray,
    frames: Sequence[np.ndarray],
    frame_shape: tuple[int, int],
    *,
    # Detection parameters
    sigma1: float = 1.0,
    sigma2: float = 6.0,
    k_std: float = 6.0,
    min_area: int = 10,
    max_area: int = 250,
    border: int = 10,
    # Classifier parameters
    n_phi_bins: int = 9,
    min_pts_per_bin: int = 2,
    coverage_threshold: float = 0.75,
    r_uniformity_threshold: float = 0.05,
) -> List[Spot]:
    """
    Run DoG detection on the S-map, then classify each spot.

    The S-map lives on the (H−1)×(W−1) intersection grid.  Returned
    coordinates are shifted to raw-frame space (+0.5 px offset).

    Returns
    -------
    spots : list of Spot
    """
    centers = find_spot_centers_dog(
        s_map,
        sigma1=sigma1,
        sigma2=sigma2,
        k_std=k_std,
        min_area=min_area,
        max_area=max_area,
        border=border,
    )
    if not centers:
        return []

    classifications = classify_spots(
        centers,
        frames,
        frame_shape,
        n_phi_bins=n_phi_bins,
        min_pts_per_bin=min_pts_per_bin,
        coverage_threshold=coverage_threshold,
        r_uniformity_threshold=r_uniformity_threshold,
    )

    spots: List[Spot] = []
    for (cx_s, cy_s, r_eff), cls in zip(centers, classifications):
        # S-map node → raw-frame coords (half-pixel shift from intersection grid)
        spots.append(Spot(
            cx=cx_s + 0.5,
            cy=cy_s + 0.5,
            r=r_eff,
            label=cls.label,
            phi_cov=cls.phi_cov,
            std_median_r=cls.std_median_r,
        ))

    return spots
