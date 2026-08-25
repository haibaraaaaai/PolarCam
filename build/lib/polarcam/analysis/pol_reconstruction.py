"""
Polarisation reconstruction for the IDS polarisation camera.

Mosaic layout (row % 2, col % 2):
    (0,0) = 90°   (0,1) = 45°
    (1,0) = 135°  (1,1) = 0°

Each 2×2 superpixel maps to one output pixel on the intersection grid
(H//2, W//2).
"""
from __future__ import annotations

import numpy as np


def make_xy_reconstructor(
    frame_shape: tuple[int, int],
    *,
    eps: float = 1e-6,
    out_dtype=np.float32,
    normalize: bool = True,
):
    """
    Build a callable that computes normalised anisotropy (X, Y) per superpixel.

    Parameters
    ----------
    frame_shape : (H, W)
        Raw mosaic frame dimensions.  Both must be even.
    eps : float
        Small constant added to the denominator to avoid division by zero.
    out_dtype :
        Output array dtype (default float32).
    normalize : bool
        True  → X = (I0−I90)  / (I0+I90+ε),  Y = (I45−I135) / (I45+I135+ε)
        False → X = I0−I90  (raw difference),  Y = I45−I135

    Returns
    -------
    reconstruct : callable
        reconstruct(frame) → (X, Y)

        ``frame`` must be shape ``frame_shape``.  The returned arrays are views
        into pre-allocated internal buffers — copy before the next call if you
        need to keep them.
    """
    H, W = frame_shape
    if H % 2 or W % 2:
        raise ValueError(f"frame_shape must be even, got {frame_shape}")

    ih, iw = H // 2, W // 2

    X = np.empty((ih, iw), dtype=out_dtype)
    Y = np.empty((ih, iw), dtype=out_dtype)
    _num = np.empty((ih, iw), dtype=np.float32)
    _den = np.empty((ih, iw), dtype=np.float32)
    _eps = np.float32(eps)
    _norm = bool(normalize)

    def reconstruct(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        I = np.asarray(frame)
        if I.shape != (H, W):
            raise ValueError(f"Expected frame {(H, W)}, got {I.shape}")

        I90  = I[0::2, 0::2]
        I45  = I[0::2, 1::2]
        I135 = I[1::2, 0::2]
        I0   = I[1::2, 1::2]

        # X component  (0° vs 90°)
        np.subtract(I0, I90, out=_num, dtype=np.float32)
        if _norm:
            np.add(I0, I90, out=_den, dtype=np.float32)
            np.add(_den, _eps, out=_den)
            np.divide(_num, _den, out=X)
        else:
            np.copyto(X, _num)

        # Y component  (45° vs 135°)
        np.subtract(I45, I135, out=_num, dtype=np.float32)
        if _norm:
            np.add(I45, I135, out=_den, dtype=np.float32)
            np.add(_den, _eps, out=_den)
            np.divide(_num, _den, out=Y)
        else:
            np.copyto(Y, _num)

        return X, Y

    return reconstruct


def make_qu_reconstructor(
    frame_shape: tuple[int, int],
    *,
    out_dtype=np.int16,
):
    """
    Build a callable that computes Q/U on the intersection grid (H−1, W−1).

    Uses diagonal pixel differences rather than superpixel averaging, so no
    resolution is lost.  This is the preferred input for the S-map accumulator.

    Parameters
    ----------
    frame_shape : (H, W)
        Raw mosaic frame dimensions.
    out_dtype :
        Output array dtype.  int16 is sufficient for 12-bit camera data.

    Returns
    -------
    reconstruct : callable
        reconstruct(frame) → (Q, U)

        Arrays are views into internal buffers — copy before the next call if
        you need to retain them.
    """
    H, W = frame_shape
    Hn, Wn = H - 1, W - 1

    A = np.empty((Hn, Wn), dtype=out_dtype)  # NW − SE diagonal
    B = np.empty((Hn, Wn), dtype=out_dtype)  # NE − SW diagonal
    Q = np.empty((Hn, Wn), dtype=out_dtype)
    U = np.empty((Hn, Wn), dtype=out_dtype)

    # Parity slices on the node grid
    ee = (slice(0, None, 2), slice(0, None, 2))
    oo = (slice(1, None, 2), slice(1, None, 2))
    eo = (slice(0, None, 2), slice(1, None, 2))
    oe = (slice(1, None, 2), slice(0, None, 2))

    def reconstruct(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        I = np.asarray(frame)
        if I.shape != (H, W):
            raise ValueError(f"Expected frame {(H, W)}, got {I.shape}")

        np.subtract(I[:-1, :-1], I[1:, 1:], out=A, dtype=out_dtype)  # NW − SE
        np.subtract(I[:-1, 1:],  I[1:, :-1], out=B, dtype=out_dtype)  # NE − SW

        # Interleave A/B into Q based on pixel parity
        Q[ee] = A[ee];  Q[oo] = A[oo]
        Q[eo] = B[eo];  Q[oe] = B[oe]

        # Interleave A/B into U (swapped role)
        U[ee] = B[ee];  U[oo] = B[oo]
        U[eo] = A[eo];  U[oe] = A[oe]

        # Sign correction for row parity
        Q[0::2, :] *= -1
        U[1::2, :] *= -1

        return Q, U

    return reconstruct
