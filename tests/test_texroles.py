"""The texture-role classifier: slots, names, pixels, and the rules on top.

Most of this runs on synthetic maps from :mod:`texroles_synth`, whose right
answers are known by construction -- normal maps integrated from analytic
height fields in both green conventions, grayscale heights, packed ORMs, the
metallicRoughness image Pixal3D writes.  Every row of every slot table has a
case of its own, spelled the way real files spell it.

The ``realasset`` tests read texture sets that live on the author's machine
(an FBX from 3ds Max, two from Blender, a Polycam OBJ, a Tripo and a Pixal3D
GLB) and skip where those are absent.  So does the regression against the
1,724 Unity-labelled textures the weights were tuned on, which needs the path
of its feature dump in ``IMB_TEXROLES_UNITY_GT``.
"""

from __future__ import annotations

import dataclasses
import functools
import io
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
from PIL import Image

import texroles_synth as synth
from instant_meshes_brush import texroles, texstats
from instant_meshes_brush.texroles import ROLES, RoleGuess, Slot

#: Side of the synthetic maps: small enough to be quick, big enough for the
#: curl test's 200-texel minimum on the flattest map.
SIZE = 128

#: Quantities :attr:`RoleGuess.channels` may name.
QUANTITIES = {
    "basecolor", "opacity", "normal", "height", "roughness", "metallic", "ao", "emissive",
    "specular",
}


def guess(
    image: Union[np.ndarray, texstats.Features],
    name: Optional[str] = None,
    slots: Sequence[Slot] = (),
    fmt: str = "",
    exporter: Optional[str] = None,
    **options: bool,
) -> RoleGuess:
    """classify(), plus the invariants every answer must keep."""
    result = texroles.classify(image, name, list(slots), fmt, exporter, **options)
    assert result.role in ROLES
    assert 0.0 < result.confidence <= 1.0
    assert result.colorspace == ("srgb" if result.role in texroles.SRGB_ROLES else "linear")
    assert set(result.channels) <= QUANTITIES | {"invert"}
    assert all(c in ("r", "g", "b", "a", "rgb", "1") for c in result.channels.values())
    assert ("invert" in result.channels) <= ("roughness" in result.channels)
    assert result.evidence and all(isinstance(line, str) and line for line in result.evidence)
    if result.role == "normal":
        assert result.normal_space in ("tangent", "object")
    else:
        assert (
            result.normal_space is None
            and result.y_convention is None
            and not result.y_confident
        )
    if result.normal_space == "tangent":
        assert result.y_convention in ("opengl", "directx")
    return result


# ---------------------------------------------------------------------------
#  The public surface
# ---------------------------------------------------------------------------


def test_roles_are_the_designed_set() -> None:
    assert ROLES == tuple(
        "basecolor normal height roughness gloss metallic specular spec_gloss"
        " ao emissive opacity orm metal_smooth mask_hdrp other".split()
    )
    assert [f.name for f in dataclasses.fields(RoleGuess)] == (
        "role channels colorspace normal_space y_convention y_confident"
        " confidence evidence".split()
    )


def test_a_slot_is_hashable_and_its_options_do_not_change_its_identity() -> None:
    plain = Slot("mtl", "map_Bump")
    scaled = Slot("mtl", "map_Bump", {"bm": 0.5})
    assert plain.options == {}
    assert hash(plain) == hash(scaled)
    assert len({plain, Slot("mtl", "map_Bump")}) == 1


def test_the_same_input_always_gives_the_same_answer() -> None:
    image = synth.normal_map(synth.height_field(3), SIZE, 0.3, "directx")
    first = guess(image, "rock_n.png", [Slot("fbx", "NormalMap")], "fbx")
    assert all(
        guess(image, "rock_n.png", [Slot("fbx", "NormalMap")], "fbx") == first for _ in range(3)
    )
    assert texstats.features(image) == texstats.features(image.copy())


def test_features_read_every_pixel_type_the_same_way() -> None:
    gray8 = synth.grayscale_height(4, SIZE)
    as16 = gray8.astype(np.uint16) * 257
    as_float = gray8.astype(np.float32) / 255
    as_int = gray8.astype(np.int32)
    reference = texstats.features(gray8)
    for other in (as16, as_float, as_int, gray8[..., None]):
        assert dataclasses.astuple(texstats.features(other)) == pytest.approx(
            dataclasses.astuple(reference), abs=1e-6
        )
    assert reference.channels == 1 and not reference.alpha_meaningful
    # A 16-bit mode "I" image keeps its range instead of saturating.
    wide = texstats.features(synth.grayscale_height(4, SIZE, bits=16).astype(np.int32))
    assert 0.2 < wide.lum_mean < 0.8
    with pytest.raises(ValueError):
        texstats.features(np.zeros((4, 4, 5), np.uint8))


def test_features_look_at_a_view_without_copying_the_image() -> None:
    rgba = np.dstack([synth.albedo(2, SIZE), np.full((SIZE, SIZE), 255, np.uint8)])
    assert texstats.features(rgba[..., :3]) == dataclasses.replace(
        texstats.features(rgba), channels=3
    )


# ---------------------------------------------------------------------------
#  Normal maps
# ---------------------------------------------------------------------------


class NormalCase(NamedTuple):
    convention: str
    strength: float
    noise_lsb: float
    image: np.ndarray


@pytest.fixture(scope="module")
def synthetic_normals() -> List[NormalCase]:
    """160 maps: 8 height fields x 5 strengths x clean / 1.5 LSB noise x GL / DX."""
    cases = []
    for seed in range(8):
        field = synth.height_field(seed, ("waves", "bumps", "mixed")[seed % 3])
        for strength in (0.01, 0.03, 0.1, 0.3, 1.0):
            for noise in (0.0, 1.5):
                for convention in ("opengl", "directx"):
                    image = synth.normal_map(field, 192, strength, convention, noise, seed)
                    cases.append(NormalCase(convention, strength, noise, image))
    return cases


def test_synthetic_normal_maps_are_always_normal_maps(
    synthetic_normals: List[NormalCase],
) -> None:
    for case in synthetic_normals:
        result = guess(case.image)
        assert (result.role, result.normal_space) == ("normal", "tangent"), case[:3]
        assert result.channels == {"normal": "rgb"}


def test_the_curl_test_tells_opengl_from_directx(synthetic_normals: List[NormalCase]) -> None:
    calls = [
        (case.convention, *texstats.curl_convention(case.image)) for case in synthetic_normals
    ]
    answered = [(truth, call, stat) for truth, call, stat in calls if call is not None]
    right = sum(call == truth for truth, call, _ in answered)
    assert len(answered) >= 0.75 * len(calls)
    assert right >= 0.95 * len(answered)
    assert not [stat for truth, call, stat in answered if call != truth and abs(stat) > 0.2]
    # The sign carries the call, and abstentions are the near-zero ones.
    for truth, call, stat in calls:
        if call is None:
            assert math.isnan(stat) or abs(stat) < texstats.CURL_ABSTAIN
        else:
            assert (stat > 0) == (call == "opengl")


def test_an_unnamed_normal_map_takes_the_curl_call(synthetic_normals: List[NormalCase]) -> None:
    for case in synthetic_normals:
        result = guess(case.image)
        call, _ = texstats.curl_convention(case.image)
        assert result.y_confident == (call is not None)
        assert result.y_convention == (call or "opengl")


def test_the_name_decides_the_y_convention_before_glTF_and_pixels() -> None:
    field = synth.height_field(1)
    gl = synth.normal_map(field, SIZE, 0.3, "opengl")
    dx = synth.normal_map(field, SIZE, 0.3, "directx")
    named = guess(
        gl, "PavingStones024_2K_NormalDX.jpg", [Slot("gltf", "normalTexture")], "gltf"
    )
    assert (named.y_convention, named.y_confident) == ("directx", True)
    declared = guess(dx, None, [Slot("gltf", "normalTexture")], "gltf")
    assert (declared.y_convention, declared.y_confident) == ("opengl", True)
    measured = guess(dx, "rock_n.png", [Slot("fbx", "NormalMap")], "fbx")
    assert (measured.y_convention, measured.y_confident) == ("directx", True)
    assert any("curl test" in line for line in measured.evidence)


def test_without_evidence_the_y_convention_is_an_unsure_opengl() -> None:
    flat = guess(synth.flat_normal_map(64))
    assert (flat.role, flat.y_convention, flat.y_confident) == ("normal", "opengl", False)
    dx = synth.normal_map(synth.height_field(2), SIZE, 0.3, "directx")
    earlier = guess(texstats.features(dx))
    assert (earlier.role, earlier.y_convention) == ("normal", "opengl")
    assert not earlier.y_confident


def test_an_object_space_normal_map_has_no_y_convention() -> None:
    result = guess(synth.object_space_normal_map(SIZE))
    assert (result.role, result.normal_space, result.y_convention) == ("normal", "object", None)


def test_a_normal_map_in_the_wrong_slot_is_still_a_normal_map() -> None:
    image = synth.normal_map(synth.height_field(6), SIZE, 0.3)
    assert guess(image, None, [Slot("fbx", "DiffuseColor")], "fbx").role == "normal"
    assert guess(image, None, [Slot("gltf", "baseColorTexture")], "gltf").role == "normal"


# ---------------------------------------------------------------------------
#  Grayscale in a normal slot is a bump map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slot",
    [
        Slot("fbx", "NormalMap"),
        Slot("fbx", "Bump"),
        Slot("fbx", "3dsMax|Parameters|bump_map"),
        Slot("mtl", "map_Bump"),
        Slot("mtl", "bump"),
        Slot("assimp", "5"),
    ],
    ids=lambda s: s.key,
)
def test_grayscale_in_a_normal_slot_is_height(slot: Slot) -> None:
    for bits in (8, 16):
        for kind in ("waves", "bumps", "mixed"):
            for seed in range(3):
                image = synth.grayscale_height(seed, SIZE, bits, kind)
                result = guess(image, None, [slot], slot.kind)
                assert result.role == "height", (bits, kind, seed)
                assert result.channels == {"height": "r"}
                assert any("bump map" in line for line in result.evidence)


def test_a_colour_image_in_a_normal_slot_stays_a_colour_map() -> None:
    result = guess(synth.albedo(3, SIZE), None, [Slot("fbx", "NormalMap")], "fbx")
    assert result.role == "basecolor"


# ---------------------------------------------------------------------------
#  glTF metallicRoughness: R is never AO unless it is the occlusion too
# ---------------------------------------------------------------------------

MR = Slot("gltf", "metallicRoughnessTexture")
OCCLUSION = Slot("gltf", "occlusionTexture")


def test_pixal3d_metal_rough_is_roughness_and_metallic_never_ao() -> None:
    for seed in range(4):
        result = guess(synth.pixal_metal_rough(seed, SIZE), "image1", [MR], "gltf")
        assert result.role == "orm"
        assert result.channels == {"roughness": "g", "metallic": "b"}
        assert result.colorspace == "linear"


def test_an_orm_bound_only_as_metal_rough_loses_its_ao() -> None:
    tripo_name = "ORM_2e4a0f2d-c137-4fba-abc8-f600c68f406f"
    tripo = guess(synth.packed_orm(5, SIZE), tripo_name, [MR], "gltf")
    assert tripo.channels == {"roughness": "g", "metallic": "b"}
    # Even a picture of AO: glTF reads G and B of that slot, and nothing else.
    as_ao = guess(synth.bright_ao(5, SIZE), "cavity_ao.png", [MR], "gltf")
    assert (as_ao.role, as_ao.channels) == ("orm", {"roughness": "g", "metallic": "b"})
    assert any("R is not AO" in line for line in as_ao.evidence)


def test_the_same_image_as_occlusion_makes_a_full_orm() -> None:
    full = {"ao": "r", "roughness": "g", "metallic": "b"}
    orm = synth.packed_orm(5, SIZE)
    assert guess(orm, None, [MR, OCCLUSION], "gltf").channels == full
    assert guess(orm, None, [MR], "gltf", gltf_same_image_as_occlusion=True).channels == full
    assert (
        guess(synth.pixal_metal_rough(1, SIZE), None, [OCCLUSION, MR], "gltf").channels == full
    )


# ---------------------------------------------------------------------------
#  Pixels alone
# ---------------------------------------------------------------------------


def test_a_packed_orm_is_an_orm() -> None:
    for seed in range(4):
        result = guess(synth.packed_orm(seed, SIZE))
        assert (result.role, result.channels) == (
            "orm",
            {"ao": "r", "roughness": "g", "metallic": "b"},
        )


def test_mostly_black_colour_is_emissive() -> None:
    for seed in range(4):
        assert guess(synth.emissive(seed, SIZE)).role == "emissive"
        emissive = guess(
            synth.emissive(seed, SIZE), None, [Slot("fbx", "EmissiveColor")], "fbx"
        )
        assert (emissive.role, emissive.channels, emissive.colorspace) == (
            "emissive",
            {"emissive": "rgb"},
            "srgb",
        )


def test_bright_grayscale_with_dark_crevices_is_ao() -> None:
    for seed in range(4):
        assert guess(synth.bright_ao(seed, SIZE)).role == "ao"


def test_a_colour_map_carries_opacity_only_when_its_alpha_is_used() -> None:
    opaque = guess(synth.albedo(4, SIZE))
    assert (opaque.role, opaque.channels, opaque.colorspace) == (
        "basecolor",
        {"basecolor": "rgb"},
        "srgb",
    )
    padded = np.dstack([synth.albedo(4, SIZE), np.full((SIZE, SIZE), 255, np.uint8)])
    assert guess(padded).channels == {"basecolor": "rgb"}
    cut = guess(
        synth.albedo(4, SIZE, alpha=True),
        None,
        [Slot("fbx", "DiffuseColor"), Slot("fbx", "TransparentColor")],
    )
    assert (cut.role, cut.channels) == ("basecolor", {"basecolor": "rgb", "opacity": "a"})


def test_an_opacity_mask_without_alpha_is_read_from_red() -> None:
    result = guess(
        synth.bimodal_mask(2, SIZE), None, [Slot("fbx", "TransparencyFactor")], "fbx"
    )
    assert (result.role, result.channels) == ("opacity", {"opacity": "r"})


# ---------------------------------------------------------------------------
#  Every slot-table row, spelled the way files spell it
# ---------------------------------------------------------------------------

#: (kind, key as found in files, accepted roles; the pixels shown are the first role's).
SLOT_CASES: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("gltf", "baseColorTexture", ("basecolor",)),
    ("gltf", "normalTexture", ("normal",)),
    ("gltf", "metallicRoughnessTexture", ("orm",)),
    ("gltf", "occlusionTexture", ("ao",)),
    ("gltf", "emissiveTexture", ("emissive",)),
    ("gltf", "diffuseTexture", ("basecolor",)),
    ("gltf", "specularGlossinessTexture", ("spec_gloss",)),
    ("gltf", "specularTexture", ("specular",)),
    ("gltf", "specularColorTexture", ("specular",)),
    ("gltf", "clearcoatTexture", ("other",)),
    ("gltf", "clearcoatRoughnessTexture", ("other",)),
    ("gltf", "clearcoatNormalTexture", ("other",)),
    ("gltf", "transmissionTexture", ("other",)),
    ("gltf", "thicknessTexture", ("other",)),
    ("gltf", "sheenColorTexture", ("other",)),
    ("gltf", "sheenRoughnessTexture", ("other",)),
    ("gltf", "iridescenceTexture", ("other",)),
    ("gltf", "iridescenceThicknessTexture", ("other",)),
    ("gltf", "anisotropyTexture", ("other",)),
    ("mtl", "map_Kd", ("basecolor",)),
    ("mtl", "map_Ka", ("ao", "basecolor")),
    ("mtl", "map_Ks", ("specular",)),
    ("mtl", "map_Ns", ("gloss", "roughness")),
    ("mtl", "map_d", ("opacity",)),
    ("mtl", "map_Tr", ("opacity",)),
    ("mtl", "map_Bump", ("normal", "height")),
    ("mtl", "bump", ("normal", "height")),
    ("mtl", "norm", ("normal",)),
    ("mtl", "map_Kn", ("normal",)),
    ("mtl", "normal", ("normal",)),
    ("mtl", "map_normal", ("normal",)),
    ("mtl", "map_norm", ("normal",)),
    ("mtl", "disp", ("height",)),
    ("mtl", "decal", ("opacity",)),
    ("mtl", "refl", ("metallic", "specular")),
    ("mtl", "map_refl", ("metallic", "specular")),
    ("mtl", "map_Pr", ("roughness",)),
    ("mtl", "map_Pm", ("metallic",)),
    ("mtl", "map_Ps", ("other",)),
    ("mtl", "map_Ke", ("emissive",)),
    ("mtl", "map_ao", ("ao",)),
    ("mtl", "map_RMA", ("orm",)),
    ("mtl", "map_ORM", ("orm",)),
    ("assimp", "1", ("basecolor",)),
    ("assimp", "2", ("specular",)),
    ("assimp", "3", ("ao", "basecolor")),
    ("assimp", "4", ("emissive",)),
    ("assimp", "5", ("height", "normal")),
    ("assimp", "6", ("normal",)),
    ("assimp", "7", ("gloss", "roughness")),
    ("assimp", "8", ("opacity",)),
    ("assimp", "9", ("height",)),
    ("assimp", "10", ("ao",)),
    ("assimp", "11", ("metallic", "specular")),
    ("assimp", "12", ("basecolor",)),
    ("assimp", "13", ("normal",)),
    ("assimp", "14", ("emissive",)),
    ("assimp", "15", ("metallic",)),
    ("assimp", "16", ("roughness",)),
    ("assimp", "17", ("ao",)),
    ("assimp", "18", ("orm",)),
    ("fbx", "DiffuseColor", ("basecolor",)),
    ("fbx", "Diffuse", ("basecolor",)),
    ("fbx", "DiffuseFactor", ("basecolor",)),
    ("fbx", "Maya|baseColor", ("basecolor",)),
    ("fbx", "Maya|base_color", ("basecolor",)),
    ("fbx", "3dsMax|Parameters|base_color_map", ("basecolor",)),
    ("fbx", "Maya|TEX_color_map", ("basecolor",)),
    ("fbx", "NormalMap", ("normal",)),
    ("fbx", "Maya|normalCamera", ("normal",)),
    ("fbx", "Maya|TEX_normal_map", ("normal",)),
    ("fbx", "3dsMax|Parameters|norm_map", ("normal",)),
    ("fbx", "Bump", ("normal", "height")),
    ("fbx", "BumpMap", ("normal", "height")),
    ("fbx", "3dsMax|Parameters|bump_map", ("normal", "height")),
    ("fbx", "BumpFactor", ("height", "normal")),
    ("fbx", "DisplacementColor", ("height",)),
    ("fbx", "3dsMax|Parameters|displacement_map", ("height",)),
    ("fbx", "SpecularColor", ("specular",)),
    ("fbx", "SpecularFactor", ("specular",)),
    ("fbx", "3dsMax|Parameters|specular_map", ("specular",)),
    ("fbx", "ShininessExponent", ("gloss", "roughness")),
    ("fbx", "Shininess", ("gloss", "roughness")),
    ("fbx", "3dsMax|Parameters|glossiness_map", ("gloss",)),
    ("fbx", "Maya|roughness", ("roughness",)),
    ("fbx", "3dsMax|Parameters|roughness_map", ("roughness",)),
    ("fbx", "Maya|TEX_roughness_map", ("roughness",)),
    ("fbx", "Maya|specularRoughness", ("roughness",)),
    ("fbx", "Maya|diffuseRoughness", ("roughness",)),
    ("fbx", "Maya|metalness", ("metallic",)),
    ("fbx", "3dsMax|Parameters|metalness_map", ("metallic",)),
    ("fbx", "Maya|TEX_metallic_map", ("metallic",)),
    ("fbx", "ReflectionFactor", ("metallic", "specular")),
    ("fbx", "ReflectionColor", ("other",)),
    ("fbx", "Maya|emissive", ("emissive",)),
    ("fbx", "EmissiveColor", ("emissive",)),
    ("fbx", "EmissiveFactor", ("emissive",)),
    ("fbx", "Maya|emissionColor", ("emissive",)),
    ("fbx", "3dsMax|Parameters|emission_map", ("emissive",)),
    ("fbx", "3dsMax|Parameters|emit_color_map", ("emissive",)),
    ("fbx", "Maya|TEX_emissive_map", ("emissive",)),
    ("fbx", "TransparentColor", ("opacity",)),
    ("fbx", "TransparencyFactor", ("opacity",)),
    ("fbx", "3dsMax|Parameters|opacity_map", ("opacity",)),
    ("fbx", "AmbientColor", ("ao", "basecolor")),
    ("fbx", "3dsMax|Parameters|ao_map", ("ao",)),
    ("fbx", "Maya|TEX_ao_map", ("ao",)),
    ("fbx", "Maya|ambientOcclusion", ("ao",)),
]


@functools.lru_cache(maxsize=None)
def _pixels_for(role: str) -> np.ndarray:
    """A synthetic map that looks like ``role`` to the pixel statistics."""
    return {
        "basecolor": lambda: synth.albedo(1, SIZE),
        "normal": lambda: synth.normal_map(synth.height_field(1), SIZE, 0.3),
        "height": lambda: synth.grayscale_height(1, SIZE),
        "roughness": lambda: synth.gray_map(1, SIZE, 0.7),
        "gloss": lambda: synth.gray_map(1, SIZE, 0.3),
        "metallic": lambda: synth.bimodal_mask(1, SIZE),
        "specular": lambda: synth.gray_map(2, SIZE, 0.45),
        "spec_gloss": lambda: synth.albedo(1, SIZE, alpha=True),
        "ao": lambda: synth.bright_ao(1, SIZE),
        "emissive": lambda: synth.emissive(1, SIZE),
        "opacity": lambda: synth.bimodal_mask(3, SIZE),
        "orm": lambda: synth.packed_orm(1, SIZE),
        "other": lambda: synth.gray_map(3, SIZE, 0.45),
    }[role]()


def _table_key(kind: str, key: str) -> str:
    return {"mtl": key.lower(), "fbx": key.rsplit("|", 1)[-1].lower()}.get(kind, key)


def test_every_slot_table_row_has_a_case() -> None:
    rows = (
        {("gltf", k) for k in texroles.GLTF_SLOTS}
        | {("mtl", k) for k in texroles.MTL_SLOTS}
        | {("fbx", k) for k in texroles.FBX_SLOTS}
        | {("assimp", str(k)) for k in texroles.AI_SLOTS}
    )
    cases = [(kind, _table_key(kind, key)) for kind, key, _ in SLOT_CASES]
    assert len(cases) == len(set(cases)), "a table row is tested twice"
    assert set(cases) == rows


@pytest.mark.parametrize(
    "kind,key,accepted", SLOT_CASES, ids=[f"{k}:{key}" for k, key, _ in SLOT_CASES]
)
def test_slot_table_row(kind: str, key: str, accepted: Tuple[str, ...]) -> None:
    slot = Slot(kind, key)
    points = texroles.slot_evidence(slot)
    assert points, "the key as files spell it must find its row"
    strongest = max(points, key=lambda rw: rw[1])[0]
    assert {"normal_os": "normal", "metal_rough": "orm"}.get(strongest, strongest) in accepted
    result = guess(_pixels_for(accepted[0]), None, [slot], "gltf" if kind == "gltf" else kind)
    assert result.role in accepted
    assert result.evidence[0].startswith("slot ")


def test_unknown_slots_give_no_points() -> None:
    for slot in (
        Slot("fbx", "Maya|someCustomAttr"),
        Slot("mtl", "map_foo"),
        Slot("assimp", "99"),
        Slot("assimp", "x"),
    ):
        assert texroles.slot_evidence(slot) == ()
    assert texroles.slot_evidence(
        Slot("gltf", "KHR_materials_specular.specularColorTexture")
    ) == (("specular", 5.0),)


def test_slot_points_add_up_over_the_slots_of_one_image() -> None:
    both = guess(
        synth.albedo(5, SIZE),
        None,
        [Slot("fbx", "DiffuseColor"), Slot("fbx", "TransparentColor")],
        "fbx",
    )
    assert sum(line.startswith("slot FBX") for line in both.evidence) == 2


# ---------------------------------------------------------------------------
#  Exporters
# ---------------------------------------------------------------------------

BLENDER_FBX = "Blender (stable FBX IO) - 3.5.0 - 4.37.5"


def test_blender_writes_roughness_to_shininess_and_3ds_max_writes_gloss() -> None:
    image = synth.gray_map(4, SIZE, 0.5)
    shininess = [Slot("fbx", "ShininessExponent")]
    blender = guess(image, None, shininess, "fbx", BLENDER_FBX)
    assert (blender.role, blender.channels) == ("roughness", {"roughness": "r"})
    assert any(line.startswith(BLENDER_FBX) for line in blender.evidence)
    max_ = guess(image, None, shininess, "fbx", "3ds Max")
    assert (max_.role, max_.channels) == ("gloss", {"roughness": "r", "invert": "1"})


def test_blender_writes_metallic_to_reflection_factor() -> None:
    result = guess(
        synth.bimodal_mask(4, SIZE), None, [Slot("fbx", "ReflectionFactor")], "fbx", BLENDER_FBX
    )
    assert result.role == "metallic"


def test_blender_obj_writes_roughness_to_map_ns() -> None:
    image = synth.gray_map(5, SIZE, 0.4)
    result = guess(image, None, [Slot("mtl", "map_Ns")], "obj", "Blender 3.6.1 MTL File")
    assert result.role == "roughness"


# ---------------------------------------------------------------------------
#  Names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stem,tokens",
    [
        ("T_Rock_ORM", ["t", "rock", "orm"]),
        ("MaskMap", ["mask", "map"]),
        ("Color_2e4a0f2d-c137-4fba-abc8-f600c68f406f", ["color"]),
        ("NormalGL_d4bf258f", ["normal", "gl"]),
        ("roots_nor_gl_4k", ["roots", "nor", "gl", "4", "k"]),
        ("PavingStones024_2K_NormalDX", ["paving", "stones", "024", "2", "k", "normal", "dx"]),
        ("wheel_Normal_DirectX", ["wheel", "normal", "directx"]),
        ("RockNormalOpenGL", ["rock", "normal", "opengl"]),
        ("brick_nrm_D3D", ["brick", "nrm", "d3d"]),
        ("motorway_M_AO_S", ["motorway", "m", "ao", "s"]),
        ("Deadbeef_Albedo", ["deadbeef", "albedo"]),
    ],
)
def test_split_tokens(stem: str, tokens: List[str]) -> None:
    assert texroles.split_tokens(stem) == tokens


def _points(name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for reading, w in texroles.name_evidence(name).points:
        out[reading] = out.get(reading, 0.0) + w
    return out


def test_name_evidence() -> None:
    assert _points("T_Rock_ORM.png") == {"orm": 4.0}
    assert texroles.name_evidence("T_Rock_ORM.png").packing == ("ao", "roughness", "metallic")
    # "Arm" is a body part as often as a packing: it is worth less, and not "packed".
    assert _points("Robot_Arm.png") == {"orm": 2.5}
    assert _points("Rock_MaskMap.png") == {"mask_hdrp": 4.0}
    motorway = texroles.name_evidence("motorway_M_AO_S.tif")
    assert motorway.packing == ("metallic", "ao", "gloss")
    assert dict(motorway.points)["packed"] == 4.0
    # UUIDs and hex runs are no source of "_d" / "_e" suffixes; a lone word counts half.
    assert _points("Color_2e4a0f2d-c137-4fba-abc8-f600c68f406f.png") == {"basecolor": 2.0}
    assert _points("NormalGL_d4bf258f-2350-4ea6-b7af-03c471c8436b.png") == {"normal": 2.0}
    # One- and two-letter suffixes only as the last token of a longer name.
    assert _points("T_Wood_D.png") == {"basecolor": 3.0}
    assert _points("D.png") == {}
    assert _points("Stone_2k_n.png") == {"normal": 3.0}
    # The last hit counts fully, earlier ones 0.35.
    assert _points("rock_albedo_normal.png") == {"normal": 4.0, "basecolor": pytest.approx(1.4)}


def test_gl_means_opengl_after_a_normal_word_and_gloss_on_its_own() -> None:
    assert texroles.name_evidence("FlashLight_GL.png").y_hint is None
    assert _points("FlashLight_GL.png") == {"gloss": 2.5}
    assert texroles.name_evidence("Rock_Normal_GL.png").y_hint == "opengl"
    assert texroles.name_evidence("roots_nor_gl_4k.png").y_hint == "opengl"
    assert texroles.name_evidence("tex_Beretta_Arizona_Normal_DirectX.png").y_hint == "directx"
    # A convention word with no normal word still says "normal map".
    assert _points("wall_dx.png") == {"normal": 2.0}


def test_names_decide_between_candidate_readings() -> None:
    orm = guess(synth.packed_orm(6, SIZE), "T_Rock_ORM.png")
    assert (orm.role, orm.channels) == ("orm", {"ao": "r", "roughness": "g", "metallic": "b"})
    mask = guess(synth.hdrp_mask(6, SIZE), "Rock_MaskMap.png")
    assert (mask.role, mask.channels) == (
        "mask_hdrp",
        {"metallic": "r", "ao": "g", "roughness": "a", "invert": "1"},
    )
    assert guess(synth.albedo(6, SIZE), "Robot_Arm.png").role == "basecolor"
    uuid = "2e4a0f2d-c137-4fba-abc8-f600c68f406f"
    assert guess(synth.albedo(6, SIZE), f"Color_{uuid}.png").role == "basecolor"
    gloss = guess(synth.gray_map(6, SIZE, 0.6), "FlashLight_GL.png")
    assert (gloss.role, gloss.channels) == ("gloss", {"roughness": "r", "invert": "1"})
    motorway = guess(synth.packed_orm(6, SIZE), "motorway_M_AO_S.tif")
    assert (motorway.role, motorway.channels) == (
        "orm",
        {"metallic": "r", "ao": "g", "roughness": "b", "invert": "1"},
    )


def test_a_packing_name_drops_channels_the_image_does_not_have() -> None:
    rgb = guess(synth.packed_orm(7, SIZE), "T_atlas_road_decals_01_MT_AO_H_SM.tga")
    assert (rgb.role, rgb.channels) == ("orm", {"metallic": "r", "ao": "g", "height": "b"})
    rgba = np.dstack([synth.packed_orm(7, SIZE), synth.gray_map(7, SIZE, 0.4)])
    assert guess(rgba, "T_atlas_road_decals_01_MT_AO_H_SM.tga").channels == {
        "metallic": "r",
        "ao": "g",
        "height": "b",
        "roughness": "a",
        "invert": "1",
    }


def test_alpha_turns_single_maps_into_their_packed_layouts() -> None:
    spec = np.dstack([synth.gray_map(8, SIZE, 0.2), synth.gray_map(9, SIZE, 0.5)])
    spec_gloss = guess(spec, "Rock_Specular.png")
    assert (spec_gloss.role, spec_gloss.channels) == (
        "spec_gloss",
        {"specular": "rgb", "roughness": "a", "invert": "1"},
    )
    metal = np.dstack([synth.bimodal_mask(8, SIZE), synth.gray_map(9, SIZE, 0.5)])
    metal_smooth = guess(metal, "Rock_Metallic.png")
    assert (metal_smooth.role, metal_smooth.channels) == (
        "metal_smooth",
        {"metallic": "r", "roughness": "a", "invert": "1"},
    )


# ---------------------------------------------------------------------------
#  Speed
# ---------------------------------------------------------------------------


def test_a_4k_rgba_map_is_classified_in_under_half_a_second() -> None:
    """The slowest path: a normal map with only its pixels to go on, so the curl test runs."""
    tile = synth.normal_map(synth.height_field(5), 1024, 0.3)
    rgba = np.empty((4096, 4096, 4), np.uint8)
    rgba[..., :3] = np.tile(tile, (4, 4, 1))
    rgba[..., 3] = 255
    best = math.inf
    for _ in range(3):
        start = time.perf_counter()
        result = texroles.classify(rgba, None, [], "", None)
        best = min(best, time.perf_counter() - start)
    assert result.role == "normal" and result.y_confident
    assert best < 0.5, f"{best:.3f} s"


# ---------------------------------------------------------------------------
#  Real texture sets
# ---------------------------------------------------------------------------

realasset = pytest.mark.realasset


def _decode(source: Union[Path, bytes]) -> np.ndarray:
    """Pixels the way the importer hands them over: native channel count, 16-bit kept."""
    image = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    image.load()
    if image.mode in ("P", "PA"):
        image = image.convert("RGBA")
    elif image.mode == "1":
        image = image.convert("L")
    elif image.mode not in ("L", "LA", "RGB", "RGBA", "I;16", "I", "F"):
        image = image.convert("RGBA" if "A" in image.mode else "RGB")
    return np.asarray(image)


def _need(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip("not on this machine: " + ", ".join(missing))


def _glb_material_images(path: Path) -> Dict[str, Tuple[str, bytes]]:
    """``{slot: (image name, bytes)}`` of the first material of a GLB."""
    data = path.read_bytes()
    magic, _, _ = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67
    json_len, _ = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20 : 20 + json_len])
    bin_start = 20 + json_len + 8
    material = gltf["materials"][0]
    bound = dict(material.get("pbrMetallicRoughness", {}), **material)
    out = {}
    for slot, ref in bound.items():
        if not slot.endswith("Texture"):
            continue
        texture = gltf["textures"][ref["index"]]
        source = (
            texture.get("extensions", {})
            .get("EXT_texture_webp", {})
            .get("source", texture.get("source"))
        )
        image = gltf["images"][source]
        view = gltf["bufferViews"][image["bufferView"]]
        start = bin_start + view.get("byteOffset", 0)
        out[slot] = (
            image.get("name", f"image{source}"),
            data[start : start + view["byteLength"]],
        )
    return out


PIXAL3D = Path(r"C:\_myDrive\repos\Pixal3D-stableprojectorz\output.glb")
TRIPO = Path(r"C:\_myDrive\repos\AI 3D Shooter 1\Assets\gun_Assets\selected.glb")
FLASHLIGHT = Path(r"C:\_myDrive\repos\temp\forest\Unity_Team_Project\Team_Assets") / (
    r"Flashlight\Misc\Flashlight.fbm"
)
TREES = Path(r"C:\_myDrive\repos\warewolves-game\Werewolves\Assets\NatureParadaise") / (
    r"SoStylized\Environment\Trees\Textures"
)
FAE = Path(r"C:\_myDrive\repos\temp\forest\_Picturesque\Picturesque_Assets\Externals") / (
    r"Aquarius Fae Pack\FBX\Textures"
)
POLYCAM = Path(r"C:\_myDrive\repos\temp\buildings\beefmaster2\Beefmaster\Models")


@realasset
def test_pixal3d_output_glb() -> None:
    _need(PIXAL3D)
    images = _glb_material_images(PIXAL3D)
    assert set(images) == {"baseColorTexture", "metallicRoughnessTexture"}
    generator = "https://github.com/mikedh/trimesh"
    name, data = images["baseColorTexture"]
    base = guess(_decode(data), name, [Slot("gltf", "baseColorTexture")], "gltf", generator)
    assert base.role == "basecolor" and base.channels["basecolor"] == "rgb"
    name, data = images["metallicRoughnessTexture"]
    mr = guess(_decode(data), name, [MR], "gltf", generator)
    assert (mr.role, mr.channels) == ("orm", {"roughness": "g", "metallic": "b"})


@realasset
def test_tripo_selected_glb() -> None:
    _need(TRIPO)
    images = _glb_material_images(TRIPO)
    exporter = "Khronos glTF Blender I/O v4.5.49"
    results = {
        slot: guess(_decode(data), name, [Slot("gltf", slot)], "gltf", exporter)
        for slot, (name, data) in images.items()
    }
    assert results["baseColorTexture"].role == "basecolor"
    normal = results["normalTexture"]
    assert (normal.role, normal.normal_space, normal.y_convention, normal.y_confident) == (
        "normal",
        "tangent",
        "opengl",
        True,
    )
    assert results["metallicRoughnessTexture"].role == "orm"
    assert "ao" not in results["metallicRoughnessTexture"].channels


@realasset
def test_flashlight_fbx_from_3ds_max() -> None:
    expected = {
        "LP_FlashLight_AL.png": (["DiffuseColor"], "basecolor"),
        "FlashLight_NM.png": (["NormalMap"], "normal"),
        "LP_FlashLight_OP.png": (["TransparentColor"], "opacity"),
        "LP_FlashLight_RFMorSP2.png": (["SpecularColor", "SpecularFactor"], "specular"),
        "FlashLight_GL.png": (["ShininessExponent"], "gloss"),
        "ReflMap_PS_sHDR_512.png": (["ReflectionColor"], "other"),
    }
    _need(*(FLASHLIGHT / name for name in expected))
    for name, (keys, role) in expected.items():
        result = guess(
            _decode(FLASHLIGHT / name), name, [Slot("fbx", k) for k in keys], "fbx", "3ds Max"
        )
        assert result.role == role, (name, result.evidence)


@realasset
def test_bamboo_fbx_from_blender() -> None:
    expected = {
        "T_BambooBark_BC.png": ("DiffuseColor", "basecolor"),
        "T_BambooBark_R.png": ("ShininessExponent", "roughness"),
        "T_Leaf_Bamboo_Filled.png": ("TransparencyFactor", "opacity"),
    }
    _need(*(TREES / name for name in expected))
    for name, (key, role) in expected.items():
        result = guess(_decode(TREES / name), name, [Slot("fbx", key)], "fbx", BLENDER_FBX)
        assert result.role == role, (name, result.evidence)


@realasset
def test_hollowed_elevators_fbx_from_blender() -> None:
    exporter = "Blender (stable FBX IO) - 2.80 (sub 75) - 4.14.14"
    bark = "Grass Plane Maker A_Tree_Bark_1C1"
    mushroom = "Mushroom Flats A Low_Mushrooms_Flats_2A1"
    expected = {
        f"{bark}_AlbedoTransparency.png": ("DiffuseColor", "basecolor"),
        f"{bark}_Normal.png": ("NormalMap", "normal"),
        f"{bark}_MetallicSmoothness.png": ("ReflectionFactor", "metal_smooth"),
        f"{mushroom}_MetallicSmoothness_Raw.png": ("ReflectionFactor", "metal_smooth"),
        "Fae_Leaves_A.png": ("TransparencyFactor", "opacity"),
    }
    _need(*(FAE / name for name in expected))
    results = {}
    for name, (key, role) in expected.items():
        results[name] = guess(_decode(FAE / name), name, [Slot("fbx", key)], "fbx", exporter)
        assert results[name].role == role, (name, results[name].evidence)
    assert results["Fae_Leaves_A.png"].channels == {"opacity": "a"}
    for name in expected:
        if "MetallicSmoothness" in name:
            assert results[name].channels == {"metallic": "r", "roughness": "a", "invert": "1"}


@realasset
def test_polycam_mtl() -> None:
    expected = {
        "textured_2_uYUNJ2Tu.jpg": ("map_Kd", "basecolor"),
        "textured_0_uYUNJ2Tu.jpg": ("normal", "normal"),
        "textured_1_uYUNJ2Tu.jpg": ("map_ao", "ao"),
    }
    _need(*(POLYCAM / name for name in expected))
    for name, (key, role) in expected.items():
        result = guess(
            _decode(POLYCAM / name), name, [Slot("mtl", key)], "obj", "Created by Polycam"
        )
        assert result.role == role, (name, result.evidence)


# ---------------------------------------------------------------------------
#  Regression on the Unity ground truth (dev only)
# ---------------------------------------------------------------------------

UNITY_GT = os.environ.get("IMB_TEXROLES_UNITY_GT", "")


class _Labelled(NamedTuple):
    path: str
    label: str
    features: texstats.Features


def _unity_ground_truth() -> List[_Labelled]:
    if not UNITY_GT or not Path(UNITY_GT).exists():
        pytest.skip("set IMB_TEXROLES_UNITY_GT to the Unity ground-truth feature dump")
    renamed = {"width": "w", "height": "h", "alpha_meaningful": "a_meaningful"}
    out = []
    for row in json.loads(Path(UNITY_GT).read_text(encoding="utf-8")):
        if "feat" not in row:
            continue
        values = {
            f.name: row["feat"][renamed.get(f.name, f.name)]
            for f in dataclasses.fields(texstats.Features)
        }
        values = {k: math.nan if v is None else v for k, v in values.items()}
        out.append(_Labelled(row["path"], row["role"], texstats.Features(**values)))
    return out


def _accepted(label: str, features: texstats.Features, name: str) -> FrozenSet[str]:
    """Readings that count as right for a Unity slot label (Unity packs maps its own ways)."""
    alpha = features.alpha_meaningful
    ok = {{"hdrp_mask": "mask_hdrp"}.get(label, label)}
    if label == "metal_smooth":
        ok |= {"mask_hdrp"} | (set() if alpha else {"metallic"})
    if label == "spec_gloss" and not alpha:
        ok.add("specular")
    if label == "hdrp_mask":
        ok |= {"metal_smooth", "packed"} | ({"orm"} if "orm" in name.lower() else set())
    if label == "metallic":
        ok |= {"packed", "orm", "metal_smooth", "mask_hdrp"}
    return frozenset(ok)


@realasset
def test_unity_ground_truth_accuracy() -> None:
    """>= 88 % once textures whose own name contradicts their Unity slot are set aside."""
    clean = []
    for item in _unity_ground_truth():
        name = os.path.basename(item.path)
        strong = {reading for reading, w in texroles.name_evidence(name).points if w >= 3.5}
        packish = {"orm", "packed", "mask_hdrp", "metal_smooth"}
        if strong and not strong & _accepted(item.label, item.features, name):
            if not (
                item.label in ("hdrp_mask", "metal_smooth", "metallic") and strong & packish
            ):
                continue  # slot misuse: the file name says something else
        if item.label == "height" and item.features.gray_frac <= 0.9:
            continue  # a colour image Unity turned into a bump map
        clean.append(item)
    right = 0
    for item in clean:
        name = os.path.basename(item.path)
        roles = {
            {"packed": "orm"}.get(reading, reading)
            for reading in _accepted(item.label, item.features, name)
        }
        right += guess(item.features, name).role in roles
    assert len(clean) > 1500
    assert right / len(clean) >= 0.88, f"{right}/{len(clean)}"


@realasset
def test_features_match_the_ground_truth_dump() -> None:
    """The stored features were measured by the research prototype; this port must agree."""
    items = [item for item in _unity_ground_truth() if Path(item.path).exists()]
    if not items:
        pytest.skip("none of the labelled images is on this machine")
    for item in items[:: max(1, len(items) // 24)]:
        pixels = _decode(Path(item.path))
        assert dataclasses.astuple(texstats.features(pixels)) == pytest.approx(
            dataclasses.astuple(item.features), abs=1e-6, nan_ok=True
        ), item.path
