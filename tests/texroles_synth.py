"""Synthetic texture maps with known answers, for the texture-role tests.

Every generator is deterministic in its ``seed`` and returns the array a
decoder would: uint8 (or uint16 for 16-bit heights), ``(H, W)`` for grayscale
and ``(H, W, C)`` otherwise.

Normal maps are exact, not estimated: they come from analytic height fields
(sums of sinusoids and Gaussian bumps) whose gradients are known in closed
form.  With ``u`` running right and ``v`` running *up* the image, the OpenGL
map stores ``normalize(-dh/du, -dh/dv, 1)``; the DirectX map is the same with
its green negated.  That is precisely the difference the curl test has to see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class HeightField:
    """A smooth height field over the unit UV square, with its exact gradient."""

    #: (amplitude, cycles along u, cycles along v, phase) per sinusoid.
    waves: Tuple[Tuple[float, int, int, float], ...]
    #: (amplitude, centre u, centre v, sigma) per Gaussian bump; amplitude < 0 is a dent.
    bumps: Tuple[Tuple[float, float, float, float], ...]

    def evaluate(self, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(h, dh/du, dh/dv)`` at the texel centres of a ``size`` x ``size`` image."""
        u = (np.arange(size) + 0.5) / size
        v = 1.0 - (np.arange(size) + 0.5) / size
        uu, vv = np.meshgrid(u, v)  # rows run down the image, v up
        h = np.zeros_like(uu)
        dh_du = np.zeros_like(uu)
        dh_dv = np.zeros_like(uu)
        for amp, fu, fv, phase in self.waves:
            arg = 2 * np.pi * (fu * uu + fv * vv) + phase
            h += amp * np.sin(arg)
            dh_du += amp * 2 * np.pi * fu * np.cos(arg)
            dh_dv += amp * 2 * np.pi * fv * np.cos(arg)
        for amp, cu, cv, sigma in self.bumps:
            g = amp * np.exp(-((uu - cu) ** 2 + (vv - cv) ** 2) / (2 * sigma**2))
            h += g
            dh_du += -g * (uu - cu) / sigma**2
            dh_dv += -g * (vv - cv) / sigma**2
        return h, dh_du, dh_dv


def height_field(seed: int, kind: str = "mixed") -> HeightField:
    """A random field: ``"waves"``, ``"bumps"`` (rivets and dents) or ``"mixed"``."""
    rng = np.random.default_rng(seed)
    waves = []
    bumps = []
    if kind in ("waves", "mixed"):
        for _ in range(int(rng.integers(3, 7))):
            fu, fv = (int(x) for x in rng.integers(-9, 10, size=2))
            if fu == 0 and fv == 0:
                fu = 1
            amp = float(rng.uniform(0.2, 1.0)) / np.hypot(fu, fv)
            waves.append((amp, fu, fv, float(rng.uniform(0, 2 * np.pi))))
    if kind in ("bumps", "mixed"):
        for _ in range(int(rng.integers(6, 16))):
            sigma = float(rng.uniform(0.02, 0.08))
            amp = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0)) * sigma
            bumps.append(
                (amp, float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)), sigma)
            )
    return HeightField(tuple(waves), tuple(bumps))


def normal_map(
    field: HeightField,
    size: int,
    strength: float,
    convention: str = "opengl",
    noise_lsb: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """``(size, size, 3)`` uint8 tangent-space normal map of ``field``.

    ``strength`` scales the slopes (a "normal map strength" slider);
    ``convention`` is ``"opengl"`` (green up) or ``"directx"``.  ``noise_lsb``
    adds Gaussian noise of that many 8-bit steps before rounding, the way lossy
    compression and dithering leave real maps.
    """
    _, dh_du, dh_dv = field.evaluate(size)
    n = np.stack([-strength * dh_du, -strength * dh_dv, np.ones_like(dh_du)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    if convention == "directx":
        n[..., 1] *= -1
    elif convention != "opengl":
        raise ValueError(convention)
    encoded = n * 0.5 + 0.5
    if noise_lsb:
        encoded += np.random.default_rng(seed).normal(0.0, noise_lsb / 255, encoded.shape)
    return _encode(encoded)


def object_space_normal_map(size: int) -> np.ndarray:
    """The normals of a sphere laid out longitude x latitude: every direction, unit length."""
    theta = (np.arange(size) + 0.5) / size * np.pi
    phi = (np.arange(size) + 0.5) / size * 2 * np.pi
    pp, tt = np.meshgrid(phi, theta)
    n = np.stack([np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], -1)
    return _encode(n * 0.5 + 0.5)


def flat_normal_map(size: int) -> np.ndarray:
    """The (128, 128, 255) placeholder."""
    out = np.empty((size, size, 3), np.uint8)
    out[...] = (128, 128, 255)
    return out


def grayscale_height(seed: int, size: int, bits: int = 8, kind: str = "mixed") -> np.ndarray:
    """A bump/height map: the field itself, stretched to the full range."""
    h = height_field(seed, kind).evaluate(size)[0]
    h = (h - h.min()) / max(float(np.ptp(h)), 1e-12)
    if bits == 16:
        return np.round(h * 65535).astype(np.uint16)
    return _encode(h)


def gray_map(seed: int, size: int, mean: float, spread: float = 0.15) -> np.ndarray:
    """A mid-tone grayscale scalar map (roughness, gloss, specular level ...)."""
    h = height_field(seed, "waves").evaluate(size)[0]
    h = (h - h.mean()) / max(float(h.std()), 1e-12)
    rng = np.random.default_rng(seed + 1)
    grain = rng.normal(0.0, 0.3, h.shape)
    return _encode(mean + spread * (h + grain) / 1.05)


def bimodal_mask(seed: int, size: int) -> np.ndarray:
    """A black-and-white mask (metal / cutout)."""
    h = height_field(seed, "waves").evaluate(size)[0]
    return np.where(h > np.median(h), 255, 0).astype(np.uint8)


def bright_ao(seed: int, size: int) -> np.ndarray:
    """Ambient occlusion: white, darkening into crevices."""
    h = height_field(seed, "bumps").evaluate(size)[0]
    cavity = np.clip(-h / max(float(np.abs(h).max()), 1e-12), 0, 1)
    return _encode(1.0 - 0.8 * np.sqrt(cavity))


def albedo(seed: int, size: int, alpha: bool = False) -> np.ndarray:
    """A colour map: a hue drifting across the image, luminance detail in every channel."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.25, 0.75, 3)
    tint = rng.uniform(-0.2, 0.2, 3)
    field = height_field(seed + 7, "waves").evaluate(size)[0]
    field = (field - field.min()) / max(float(np.ptp(field)), 1e-12)
    detail = gray_map(seed + 3, size, 0.5).astype(np.float64) / 255 - 0.5
    rgb = base + tint * field[..., None] + 0.6 * detail[..., None]
    if not alpha:
        return _encode(rgb)
    cutout = bimodal_mask(seed + 11, size).astype(np.float64)[..., None] / 255
    return _encode(np.concatenate([rgb, cutout], -1))


def emissive(seed: int, size: int) -> np.ndarray:
    """Mostly black, with a few glowing, saturated shapes (lamps, screens, runes)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size, 3))
    yy, xx = np.mgrid[0:size, 0:size] / size
    for _ in range(5):
        cx, cy, r = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85), rng.uniform(0.1, 0.16)
        colour = rng.uniform(0.0, 0.3, 3)
        colour[rng.integers(3)] = 1.0
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 < r**2
        out[inside] = colour
    return _encode(out)


def packed_orm(seed: int, size: int) -> np.ndarray:
    """ORM with independent channels: R bright AO, G mid roughness, B black-or-white metal."""
    ao = bright_ao(seed, size)
    rough = gray_map(seed + 101, size, 0.55, 0.12)
    metal = bimodal_mask(seed + 202, size)
    return np.stack([ao, rough, metal], -1)


def hdrp_mask(seed: int, size: int) -> np.ndarray:
    """Unity HDRP mask map: R metallic, G AO, B detail mask, A smoothness."""
    metal = bimodal_mask(seed, size)
    ao = bright_ao(seed + 1, size)
    detail = gray_map(seed + 2, size, 0.5, 0.2)
    smooth = gray_map(seed + 3, size, 0.35, 0.12)
    return np.stack([metal, ao, detail, smooth], -1)


def pixal_metal_rough(seed: int, size: int) -> np.ndarray:
    """What Pixal3D writes to metallicRoughnessTexture: R = 0, G = 255, B around 209."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size, 3), np.uint8)
    out[..., 1] = 255
    out[..., 2] = 209
    patches = height_field(seed, "bumps").evaluate(size)[0]
    out[..., 2] = np.clip(
        209 + np.round(6 * patches / max(float(np.abs(patches).max()), 1e-12)), 0, 255
    )
    rare = rng.random((size, size)) < 0.002
    out[rare, 1] = 254
    return out


def _encode(unit: np.ndarray) -> np.ndarray:
    """[0, 1] floats to uint8, rounding to nearest."""
    return np.round(np.clip(unit, 0.0, 1.0) * 255).astype(np.uint8)
