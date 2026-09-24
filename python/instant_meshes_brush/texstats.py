"""Pixel statistics that tell one kind of texture map from another.

A texture's file name and the material slot it sits in usually say what it is
for, but not always, and :mod:`instant_meshes_brush.texroles` weighs those
against what the pixels themselves look like.  This module is the pixel half:
it measures an image once and answers the questions the classifier asks of it.

Two of the answers are strong enough to overrule everything else.  A tangent
space normal map stores unit vectors, so its blue channel is exactly
``sqrt(1 - x^2 - y^2)`` of the other two -- no photograph, mask or colour map
does that -- and a grayscale image is never a normal map, whatever slot it was
bound to.  The rest (packed channels, bright AO, mostly black emissive, ...) are
nudges, not vetoes.

The statistics come from a fixed-seed random sample of full-resolution pixels,
not from a thumbnail: averaging neighbours shortens a normal map's unit vectors
and would blur the one signal that matters most.  The same image therefore
always gives the same numbers.

:func:`curl_convention` answers a different question about a normal map: which
way its green channel points.  A normal map baked from a height field is a
gradient field, and a gradient field has no curl -- but only under the right
sign of green.  Measuring the curl both ways tells OpenGL (+Y) from DirectX
(-Y) maps without any name to go on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

#: Pixels drawn for the statistics.  Enough that a map whose tell-tale texels
#: cover a few per cent of it is still seen, few enough to cost milliseconds.
SAMPLES = 16384

#: The curl test looks at the map box-filtered down to this longest side:
#: fine enough to keep the detail it measures, coarse enough that 8-bit
#: quantisation noise does not drown it.
CURL_SIZE = 1024

#: ``|log(E_dx / E_gl)|`` below this is too close to call either way.
CURL_ABSTAIN = 0.05

#: Fewest usable texels the curl test needs before it answers at all.
_CURL_MIN_TEXELS = 200

#: A pixel whose channels spread less than this is gray.  JPEG and DXT round
#: trips leave a little chroma on maps that were authored grayscale.
_GRAY_CHROMA = 0.035


@dataclass(frozen=True)
class Features:
    """What :func:`features` measured.  Values are fractions of full scale."""

    width: int
    height: int
    #: 1 (L), 2 (LA), 3 (RGB) or 4 (RGBA), as the array was given.
    channels: int
    #: True when there is an alpha channel and it actually varies below opaque.
    alpha_meaningful: bool
    mean_r: float
    mean_g: float
    mean_b: float
    std_r: float
    std_g: float
    std_b: float
    #: Median of red.
    p50_r: float
    #: Share of blue values below 0.1 or above 0.9 (a black-and-white channel).
    ext_b: float
    #: Mean of max(r, g, b) - min(r, g, b).
    chroma_mean: float
    #: Share of pixels whose chroma is below ``_GRAY_CHROMA``.
    gray_frac: float
    #: Pearson correlations between channels; NaN when either one is flat.
    corr_rg: float
    corr_gb: float
    corr_rb: float
    lum_mean: float
    lum_std: float
    lum_p50: float
    lum_skew: float
    #: Share of pixels with luminance below 0.04.
    frac_black: float
    #: Share of pixels with luminance below 0.1 or above 0.9.
    frac_extreme: float
    #: Share of pixels that decode (as ``2 * rgb - 1``) to a vector of length 1 +- 0.15.
    unit_frac: float
    #: Share of pixels whose blue decodes to z >= -0.06.
    zpos_frac: float
    #: Share of pixels with x^2 + y^2 <= 1.05.
    xy_in_disc: float
    #: See :func:`zrec_err`.
    zrec_err: float


def features(rgba: np.ndarray, seed: int = 0) -> Features:
    """Measure an image.

    ``rgba`` is ``(H, W)`` or ``(H, W, C)`` with ``C`` from 1 to 4 (L, LA, RGB,
    RGBA), as decoded: uint8, uint16, a wider integer holding 8- or 16-bit
    values, or floats in [0, 1].  It is only read, never copied whole, so a
    view such as ``image[..., :3]`` costs nothing extra.
    """
    if rgba.ndim == 2:
        rgba = rgba[..., None]
    if rgba.ndim != 3 or not 1 <= rgba.shape[2] <= 4:
        raise ValueError(f"expected an (H, W) or (H, W, 1..4) image, got shape {rgba.shape}")
    h, w, c = rgba.shape
    if h * w == 0:
        raise ValueError("the image is empty")
    rng = np.random.default_rng(seed)
    n = min(SAMPLES, h * w)
    index = rng.choice(h * w, size=n, replace=False) if h * w > n else np.arange(h * w)
    picked = _unit_range(rgba[index // w, index % w], _full_scale(rgba))

    if c <= 2:
        rgb = np.repeat(picked[:, :1], 3, axis=1)
        alpha = picked[:, 1] if c == 2 else None
    else:
        rgb = picked[:, :3]
        alpha = picked[:, 3] if c == 4 else None
    alpha_meaningful = alpha is not None and bool(alpha.min() < 0.98 and alpha.std() > 0.004)

    mean = rgb.mean(0)
    std = rgb.std(0)
    chroma = rgb.max(1) - rgb.min(1)
    lum = rgb.mean(1)
    lum_sd = float(lum.std())
    skew = float(((lum - lum.mean()) ** 3).mean() / lum_sd**3) if lum_sd > 1e-3 else 0.0
    vec = rgb * 2 - 1
    length = np.sqrt((vec**2).sum(1))
    blue = rgb[:, 2]
    return Features(
        width=w,
        height=h,
        channels=c,
        alpha_meaningful=alpha_meaningful,
        mean_r=float(mean[0]),
        mean_g=float(mean[1]),
        mean_b=float(mean[2]),
        std_r=float(std[0]),
        std_g=float(std[1]),
        std_b=float(std[2]),
        p50_r=float(np.median(rgb[:, 0])),
        ext_b=float(((blue < 0.1) | (blue > 0.9)).mean()),
        chroma_mean=float(chroma.mean()),
        gray_frac=float((chroma < _GRAY_CHROMA).mean()),
        corr_rg=_corr(rgb[:, 0], rgb[:, 1]),
        corr_gb=_corr(rgb[:, 1], rgb[:, 2]),
        corr_rb=_corr(rgb[:, 0], rgb[:, 2]),
        lum_mean=float(lum.mean()),
        lum_std=lum_sd,
        lum_p50=float(np.percentile(lum, 50)),
        lum_skew=skew,
        frac_black=float((lum < 0.04).mean()),
        frac_extreme=float(((lum < 0.1) | (lum > 0.9)).mean()),
        unit_frac=float((np.abs(length - 1) < 0.15).mean()),
        zpos_frac=float((blue >= 0.47).mean()),
        xy_in_disc=float((vec[:, 0] ** 2 + vec[:, 1] ** 2 <= 1.05).mean()),
        zrec_err=zrec_err(rgb),
    )


def zrec_err(rgb: np.ndarray) -> float:
    """Median of ``|sqrt(1 - x^2 - y^2) - z|`` over ``(N, 3)`` colours in [0, 1].

    The colours are read as normal-map vectors ``2 * rgb - 1``.  A tangent-space
    normal map scores about 0.002 (its blue is exactly what its red and green
    imply); every other kind of map measured scores 0.46 or more at its tenth
    percentile.
    """
    vec = rgb * 2 - 1
    z_from_xy = np.sqrt(np.clip(1 - vec[:, 0] ** 2 - vec[:, 1] ** 2, 0, 1))
    return float(np.median(np.abs(z_from_xy - vec[:, 2])))


# ---------------------------------------------------------------------------
#  What the image is
# ---------------------------------------------------------------------------


def is_gray(f: Features) -> bool:
    """A grayscale image, allowing for the chroma lossy compression leaves."""
    return (
        f.gray_frac > 0.95
        or f.channels <= 2
        or (f.chroma_mean < _GRAY_CHROMA and f.gray_frac > 0.4)
    )


def is_constant(f: Features) -> bool:
    """One flat colour (placeholders, unused slots, 2 x 2 factor images)."""
    return f.lum_std < 0.01 and max(f.std_r, f.std_g, f.std_b) < 0.01


def is_flat_normal(f: Features) -> bool:
    """The (128, 128, 255) placeholder a normal slot holds when there is no detail."""
    return (
        is_constant(f)
        and abs(f.mean_r - 0.5) < 0.03
        and abs(f.mean_g - 0.5) < 0.03
        and f.mean_b > 0.97
    )


def is_tangent_normal(f: Features) -> bool:
    """Unit vectors whose z follows from x and y: a tangent-space normal map.

    A few texels facing away (seams, padding) do not matter, and once z is
    reconstructed to within 0.03 at the median nothing else explains the image.
    """
    centred = abs(f.mean_r - 0.5) < 0.15 and abs(f.mean_g - 0.5) < 0.15
    if centred and f.zrec_err < 0.08 and f.zpos_frac > 0.95 and f.unit_frac > 0.8:
        return True
    if (
        centred
        and f.zrec_err < 0.04
        and f.unit_frac > 0.85
        and f.zpos_frac > 0.8
        and f.mean_b > 0.65
    ):
        return True
    return f.zrec_err < 0.03 and f.unit_frac > 0.9 and f.zpos_frac > 0.8


def is_weak_tangent_normal(f: Features) -> bool:
    """A normal map that was never normalised (strength-scaled, "z = 1" tools, re-encoded).

    Blue dominates and never drops below the middle, red and green sit around
    0.5 and do not move together.
    """
    return (
        not is_gray(f)
        and f.zpos_frac > 0.97
        and f.mean_b > 0.6
        and abs(f.mean_r - 0.5) < 0.1
        and abs(f.mean_g - 0.5) < 0.1
        and (math.isnan(f.corr_rg) or abs(f.corr_rg) < 0.5)
        and f.mean_b > max(f.mean_r, f.mean_g) + 0.15
    )


def is_two_channel_normal(f: Features) -> bool:
    """X and Y in red and green, blue left constant (BC5-style maps)."""
    return (
        not is_gray(f)
        and f.std_b < 0.01
        and (f.mean_b < 0.03 or f.mean_b > 0.97)
        and abs(f.mean_r - 0.5) < 0.1
        and abs(f.mean_g - 0.5) < 0.1
        and f.xy_in_disc > 0.98
        and (f.std_r > 0.01 or f.std_g > 0.01)
    )


def is_object_space_normal(f: Features) -> bool:
    """Unit vectors over the whole sphere: an object- or world-space normal map.

    Over a closed surface the normals integrate to zero, so the mean colour
    sits near mid-gray, and the three axes do not follow one another.
    """
    return (
        not is_gray(f)
        and not is_tangent_normal(f)
        and f.unit_frac > 0.85
        and f.zpos_frac < 0.85
        and f.chroma_mean > 0.2
        and min(f.std_r, f.std_g, f.std_b) > 0.12
        and max(abs(f.mean_r - 0.5), abs(f.mean_g - 0.5), abs(f.mean_b - 0.5)) < 0.15
        and _max_abs_corr(f) < 0.6
    )


def is_packed(f: Features) -> bool:
    """Independent grayscale images in the colour channels (ORM, masks, ...).

    Every pair of channels is weakly correlated -- an albedo's never are, even
    a saturated yellow duck has red following green -- and the pixels are far
    from gray.
    """
    varying = sum(s > 0.02 for s in (f.std_r, f.std_g, f.std_b))
    return (
        not is_gray(f)
        and not is_tangent_normal(f)
        and not is_weak_tangent_normal(f)
        and not is_object_space_normal(f)
        and not is_two_channel_normal(f)
        and f.chroma_mean > 0.35
        and varying >= 1
        and _max_abs_corr(f) < 0.6
    )


# ---------------------------------------------------------------------------
#  Which way green points
# ---------------------------------------------------------------------------


def curl_convention(rgb: np.ndarray) -> Tuple[Optional[str], float]:
    """``("opengl" | "directx" | None, statistic)`` for a tangent-space normal map.

    With slopes ``p = x / z`` and ``q = y / z``, a map baked from a height field
    has ``dp/drow + dq/dcol = 0`` when green points up (+Y, OpenGL: Blender,
    Unity, glTF) and ``dp/drow - dq/dcol = 0`` when it points down (-Y, DirectX:
    Unreal, 3ds Max).  The statistic is ``log(E_dx / E_gl)`` of the two mean
    residuals, measured with centred differences at <= ``CURL_SIZE`` pixels;
    forward differences are biased and were right only 37-67 % of the time.

    ``None`` is an abstention: the statistic is within ``CURL_ABSTAIN`` of zero,
    or it is NaN because the map is too flat to measure (or has fewer than
    three channels).
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None, math.nan
    small = Image.fromarray(_to_uint8(rgb[..., :3]))
    w, h = small.size
    scale = max(w, h) / CURL_SIZE
    if scale > 1:
        small = small.resize((max(1, round(w / scale)), max(1, round(h / scale))), Image.BOX)
    a = np.asarray(small, dtype=np.float32) / 255.0
    x = a[..., 0] * 2 - 1
    y = a[..., 1] * 2 - 1
    z = np.clip(a[..., 2] * 2 - 1, 0.25, None)
    p, q = x / z, y / z
    dp_drow = (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5
    dq_dcol = (q[1:-1, 2:] - q[1:-1, :-2]) * 0.5
    # Seams, padding and flat areas either explode the slopes or carry none.
    usable = (
        (np.abs(dp_drow) < 1)
        & (np.abs(dq_dcol) < 1)
        & ((np.abs(dp_drow) > 1e-3) | (np.abs(dq_dcol) > 1e-3))
    )
    if int(usable.sum()) < _CURL_MIN_TEXELS:
        return None, math.nan
    e_gl = float(np.abs(dp_drow + dq_dcol)[usable].mean())
    e_dx = float(np.abs(dp_drow - dq_dcol)[usable].mean())
    stat = math.log((e_dx + 1e-9) / (e_gl + 1e-9))
    if abs(stat) < CURL_ABSTAIN:
        return None, stat
    return ("opengl" if stat > 0 else "directx"), stat


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _full_scale(pixels: np.ndarray) -> float:
    """The value that means 1.0 for this array's type."""
    if pixels.dtype == np.uint8:
        return 255.0
    if pixels.dtype == np.uint16:
        return 65535.0
    if np.issubdtype(pixels.dtype, np.integer):
        # Pillow's mode "I" holds 16-bit PNGs; an 8-bit source never exceeds 255.
        return 65535.0 if int(pixels.max()) > 255 else 255.0
    return 1.0


def _unit_range(pixels: np.ndarray, full_scale: float) -> np.ndarray:
    """``pixels`` as float32 in [0, 1]."""
    out = pixels.astype(np.float32)
    if full_scale != 1.0:
        return out / np.float32(full_scale)
    return np.clip(out, 0.0, 1.0)


def _to_uint8(pixels: np.ndarray) -> np.ndarray:
    """``pixels`` as 8-bit, contiguous, for Pillow."""
    if pixels.dtype == np.uint8:
        return np.ascontiguousarray(pixels)
    unit = _unit_range(pixels, _full_scale(pixels))
    return np.round(unit * 255.0).astype(np.uint8)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.std(), b.std()
    if sa < 1e-3 or sb < 1e-3:
        return math.nan
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _max_abs_corr(f: Features) -> float:
    """Largest |correlation| between two channels that both vary; 0 when none do."""
    corrs = [abs(c) for c in (f.corr_rg, f.corr_gb, f.corr_rb) if not math.isnan(c)]
    return max(corrs) if corrs else 0.0
