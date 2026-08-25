"""
Spot detection on the S-map for PolarCam.

Uses a Difference-of-Gaussians (DoG) blob enhancer followed by thresholding
and connected-component labelling to locate candidate spinning-fluorophore
spots.

The S-map lives on the (H−1)×(W−1) intersection grid produced by Q/U
reconstruction.  Returned coordinates are in S-map pixel space; callers
should add 0.5 to map to the raw mosaic frame.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label


def find_spot_centers_dog(
    s_map: np.ndarray,
    *,
    sigma1: float = 1.0,
    sigma2: float = 6.0,
    k_std: float = 6.0,
    min_area: int = 10,
    max_area: int = 250,
    border: int = 10,
) -> list[tuple[float, float, float]]:
    """
    Detect bright spots on an S-map using Difference-of-Gaussians filtering.

    1. Strip *border* pixels from each edge.
    2. DoG = Gaussian(S, σ1) − Gaussian(S, σ2).
    3. mask = DoG > mean(DoG) + k_std × std(DoG).
    4. 8-connected components, area filter [min_area, max_area].
    5. Centroid + effective radius for each blob.

    Parameters
    ----------
    s_map : np.ndarray
        2-D float array from ``SMapAccumulator.compute()``.
    sigma1 : float
        Inner Gaussian σ  (sharpens spot cores).
    sigma2 : float
        Outer Gaussian σ  (suppresses large-scale background).
    k_std : float
        Threshold multiplier on the DoG standard deviation.
    min_area : int
        Minimum connected-region area (pixels).
    max_area : int
        Maximum connected-region area (pixels).
    border : int
        Pixels to strip from each edge before detection.

    Returns
    -------
    centers : list of (cx, cy, r_eff)
        Centroid coordinates in **original (pre-strip) S-map space** and the
        effective radius ``r_eff = sqrt(area / π)``.
    """
    if s_map.ndim != 2:
        raise ValueError(f"Expected 2-D s_map, got shape {s_map.shape}")

    H, W = s_map.shape
    inner = s_map[border:H - border, border:W - border] if border > 0 else s_map

    if inner.size == 0:
        return []

    # --- Difference-of-Gaussians blob enhancement ---
    g1 = gaussian_filter(inner.astype(np.float32), sigma=sigma1)
    g2 = gaussian_filter(inner.astype(np.float32), sigma=sigma2)
    dog = g1 - g2

    sigma = float(dog.std())
    if sigma == 0.0:
        return []

    thr = float(dog.mean()) + k_std * sigma
    binary = dog > thr
    if not binary.any():
        return []

    # --- Connected-component labelling (8-connectivity) ---
    struct = np.ones((3, 3), dtype=np.int32)
    labelled, n_labels = label(binary, structure=struct)
    if n_labels == 0:
        return []

    centers: list[tuple[float, float, float]] = []
    for lbl in range(1, n_labels + 1):
        ys, xs = np.where(labelled == lbl)
        area = len(ys)
        if area < min_area or area > max_area:
            continue
        cx = float(xs.mean()) + border
        cy = float(ys.mean()) + border
        r_eff = float(np.sqrt(area / np.pi))
        centers.append((cx, cy, r_eff))

    return centers
