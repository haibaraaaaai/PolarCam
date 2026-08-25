"""
S-map accumulator for PolarCam.

Tracks the per-pixel range of Q/U components on the (H−1)×(W−1) intersection
grid over a rolling window of frames.  Q and U are raw diagonal-difference
proxies computed by ``make_qu_reconstructor`` — no intensity normalisation, so
the S-map scales naturally with signal amplitude.

    S(r, c) = (Q_max − Q_min)² + (U_max − U_min)²

Pixels with large S values are candidates for spinning fluorophores.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from .pol_reconstruction import make_qu_reconstructor


class SMapAccumulator:
    """
    Accumulate per-pixel (Q, U) range statistics and produce an S-map.

    Parameters
    ----------
    frame_shape : (H, W)
        Raw mosaic frame dimensions.
    smooth_k : int
        Side length of the uniform (box) filter applied to Q and U before
        accumulation.  Use 1 to disable smoothing.
    """

    def __init__(self, frame_shape: tuple[int, int], smooth_k: int = 5) -> None:
        H, W = frame_shape
        if H < 2 or W < 2:
            raise ValueError(f"frame_shape must be at least (2, 2), got {frame_shape}")
        if smooth_k < 1:
            raise ValueError(f"smooth_k must be >= 1, got {smooth_k}")

        self._frame_shape = frame_shape
        self._reconstruct = make_qu_reconstructor(frame_shape, out_dtype=np.int16)
        self._smooth_k = int(smooth_k)

        ih, iw = H - 1, W - 1
        # Min/max buffers — initialised to ±inf so the first frame sets them.
        self._q_min = np.full((ih, iw), np.inf,  dtype=np.float32)
        self._q_max = np.full((ih, iw), -np.inf, dtype=np.float32)
        self._u_min = np.full((ih, iw), np.inf,  dtype=np.float32)
        self._u_max = np.full((ih, iw), -np.inf, dtype=np.float32)
        self._n: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of frames accumulated since last reset."""
        return self._n

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Shape of the output S-map (H−1, W−1)."""
        return self._q_min.shape

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._q_min[:] =  np.inf
        self._q_max[:] = -np.inf
        self._u_min[:] =  np.inf
        self._u_max[:] = -np.inf
        self._n = 0

    def update(self, frame: np.ndarray) -> None:
        """
        Process one raw mosaic frame and update the running min/max.

        Parameters
        ----------
        frame : np.ndarray
            Raw polarisation mosaic, shape (H, W), any integer/float dtype.
        """
        Q, U = self._reconstruct(frame)

        if self._smooth_k > 1:
            Qs = uniform_filter(Q.astype(np.float32), size=self._smooth_k, mode="reflect")
            Us = uniform_filter(U.astype(np.float32), size=self._smooth_k, mode="reflect")
        else:
            Qs = Q.astype(np.float32)
            Us = U.astype(np.float32)

        np.minimum(self._q_min, Qs, out=self._q_min)
        np.maximum(self._q_max, Qs, out=self._q_max)
        np.minimum(self._u_min, Us, out=self._u_min)
        np.maximum(self._u_max, Us, out=self._u_max)
        self._n += 1

    def compute(self) -> np.ndarray | None:
        """
        Return the S-map: ``(Q_max − Q_min)² + (U_max − U_min)²``.

        Returns
        -------
        s_map : np.ndarray, shape (H−1, W−1), float32
            None if no frames have been accumulated yet.
        """
        if self._n == 0:
            return None

        rq = self._q_max - self._q_min
        ru = self._u_max - self._u_min
        s_map: np.ndarray = rq * rq + ru * ru
        return s_map.astype(np.float32, copy=False)
