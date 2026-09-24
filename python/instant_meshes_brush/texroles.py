"""What each texture of an imported material is for.

A model file says less about its maps than one would hope.  glTF is exact --
``normalTexture`` is a normal map, green up -- but an FBX binds textures to
properties that exporters use as they please (Blender writes its roughness map
to ``ShininessExponent`` and its metallic map to ``ReflectionFactor``), an MTL
``bump`` is a height map as often as a normal map, and a file that was
re-saved by three tools may have lost the binding altogether.  File names help
(``T_Rock_N``, ``Color_<uuid>``, ``motorway_M_AO_S``) and so do the pixels, and
none of the three is right every time.

So :func:`classify` adds evidence up.  Every source gives points to one or more
*readings* of the image, the reading with the most points wins, and a softmax
over the points is the confidence:

- **slot**: where the file binds the image.  glTF slots are worth 6, FBX PBR
  properties (``Maya|*``, ``3dsMax|*``) 4, clear legacy slots 3-3.5; ambiguous
  ones split their points (``Bump``: normal 2 + height 2).  One image bound to
  several slots of a material gets the points of all of them.
- **exporter**: who wrote the file, which changes what an ambiguous FBX or MTL
  slot holds.
- **name**: tokens of the file name, the last one counting most.
- **pixels**: :mod:`instant_meshes_brush.texstats`.  Two of these overrule the
  rest: a map that decodes to unit vectors is a normal map wherever it sits,
  and a grayscale image is never one -- in a normal slot it is a bump map.

Then a few rules that the evidence cannot express: a real alpha channel turns a
specular map into spec/gloss, and **the red channel of a glTF
metallicRoughnessTexture is never ambient occlusion** unless the same image is
also the occlusionTexture.  Pixal3D writes zeros there; read as AO, they would
render the whole model black.

The weights were tuned on 1,724 Unity-labelled textures (90 % correct once the
181 whose slot contradicts their own file name are set aside, 97 % on normal
maps) and checked on 649 texture references in real FBX, MTL and glTF files.
Everything here is deterministic: the same image, name and slots always give
the same answer.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from . import texstats

#: Every role :func:`classify` can answer, in the order buttons list them.
ROLES = (
    "basecolor",
    "normal",
    "height",
    "roughness",
    "gloss",
    "metallic",
    "specular",
    "spec_gloss",
    "ao",
    "emissive",
    "opacity",
    "orm",
    "metal_smooth",
    "mask_hdrp",
    "other",
)

#: Roles whose colour channels are stored sRGB-encoded.  Alpha is always linear.
SRGB_ROLES = frozenset({"basecolor", "emissive", "specular", "spec_gloss"})

#: (reading, points) pairs: what one piece of evidence says.
Points = Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class Slot:
    """One place a material binds the image to."""

    #: ``"gltf"``, ``"fbx"``, ``"mtl"`` or ``"assimp"``.
    kind: str
    #: The raw slot as the file spells it: ``"normalTexture"``,
    #: ``"3dsMax|Parameters|bump_map"``, ``"map_Ns"``, or an aiTextureType
    #: number such as ``"6"``.
    key: str
    #: Slot options the reader kept (MTL ``-bm``/``-s``/``-o``, glTF
    #: ``texCoord``/``scale``/``strength``).  They do not change the role.
    options: Dict[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class RoleGuess:
    """What :func:`classify` decided about one image."""

    #: One of :data:`ROLES`.
    role: str
    #: Where each quantity lives in the image: ``{"roughness": "g", "metallic":
    #: "b"}``, ``{"basecolor": "rgb", "opacity": "a"}``.  Quantities are
    #: basecolor, opacity, normal, height, roughness, metallic, ao, emissive and
    #: specular; ``"invert": "1"`` means the roughness channel holds smoothness
    #: or glossiness (``roughness = 1 - value``).  Channel letters refer to the
    #: image as RGBA, a gray image having its one value in r, g and b.  Empty for
    #: ``other``.
    channels: Dict[str, str]
    #: ``"srgb"`` or ``"linear"``, for the colour channels.
    colorspace: str
    #: ``"tangent"`` or ``"object"`` for normal maps, else None.
    normal_space: Optional[str]
    #: ``"opengl"`` (green up) or ``"directx"`` for tangent-space normal maps,
    #: else None.  An abstention reads as ``"opengl"`` with ``y_confident`` False.
    y_convention: Optional[str]
    y_confident: bool
    #: Softmax probability of the winning reading, 0..1.
    confidence: float
    #: Why, one human-readable line per source of evidence.
    evidence: Tuple[str, ...]


# ---------------------------------------------------------------------------
#  Readings
# ---------------------------------------------------------------------------

#: What the points are scored for: every role, plus three readings that land on
#: a role with a particular layout -- an object-space normal map, a glTF
#: metallicRoughness map (G rough, B metal, R unused) and a packed map whose
#: channel order the file name spells out.  Ties go to the earlier entry.
_READINGS = (
    "basecolor",
    "normal",
    "normal_os",
    "height",
    "roughness",
    "gloss",
    "metallic",
    "specular",
    "ao",
    "emissive",
    "opacity",
    "orm",
    "metal_rough",
    "metal_smooth",
    "spec_gloss",
    "mask_hdrp",
    "packed",
    "other",
)

#: How a reading is named in evidence lines, which artists read in tooltips.
_READING_NAMES = {
    "basecolor": "base colour",
    "normal": "normal",
    "normal_os": "object-space normal",
    "height": "height",
    "roughness": "roughness",
    "gloss": "gloss",
    "metallic": "metallic",
    "specular": "specular",
    "ao": "AO",
    "emissive": "emissive",
    "opacity": "opacity",
    "orm": "ORM",
    "metal_rough": "metal/rough",
    "metal_smooth": "metallic/smoothness",
    "spec_gloss": "specular/gloss",
    "mask_hdrp": "HDRP mask",
    "packed": "packed",
    "other": "other",
}

#: Channel layout of each reading (the packed one comes from the name).
_LAYOUTS: Dict[str, Dict[str, str]] = {
    "basecolor": {"basecolor": "rgb", "opacity": "a"},
    "normal": {"normal": "rgb"},
    "normal_os": {"normal": "rgb"},
    "height": {"height": "r"},
    "roughness": {"roughness": "r"},
    "gloss": {"roughness": "r", "invert": "1"},
    "metallic": {"metallic": "r"},
    "specular": {"specular": "rgb"},
    "ao": {"ao": "r"},
    "emissive": {"emissive": "rgb"},
    "opacity": {"opacity": "a"},
    "orm": {"ao": "r", "roughness": "g", "metallic": "b"},
    "metal_rough": {"roughness": "g", "metallic": "b"},
    # Unity Standard: R metallic, A smoothness.
    "metal_smooth": {"metallic": "r", "roughness": "a", "invert": "1"},
    # Unity/KHR spec-gloss: RGB specular colour, A glossiness.
    "spec_gloss": {"specular": "rgb", "roughness": "a", "invert": "1"},
    # Unity HDRP mask map: R metallic, G AO, B detail mask, A smoothness.
    "mask_hdrp": {"metallic": "r", "ao": "g", "roughness": "a", "invert": "1"},
    "other": {},
}

#: The role a reading lands on.
_ROLE_OF = {r: r for r in ROLES}
_ROLE_OF.update(normal_os="normal", metal_rough="orm", packed="orm")

#: A whisker of prior so an image with no evidence at all is base colour.
_BASECOLOR_PRIOR = 0.2


# ---------------------------------------------------------------------------
#  Slots
# ---------------------------------------------------------------------------

#: glTF texture properties, including the KHR_materials_* extensions.  Exact
#: by specification, so the strongest slot evidence there is.
GLTF_SLOTS: Dict[str, Points] = {
    "baseColorTexture": (("basecolor", 6.0),),
    "normalTexture": (("normal", 6.0),),
    # Becomes a full ORM when the same image is also the occlusionTexture.
    "metallicRoughnessTexture": (("metal_rough", 6.0),),
    "occlusionTexture": (("ao", 6.0),),
    "emissiveTexture": (("emissive", 6.0),),
    # KHR_materials_pbrSpecularGlossiness: archived, still in the wild.
    "diffuseTexture": (("basecolor", 6.0),),
    "specularGlossinessTexture": (("spec_gloss", 6.0),),
    # KHR_materials_specular: strength in A; colour in RGB, sRGB.
    "specularTexture": (("specular", 4.0),),
    "specularColorTexture": (("specular", 5.0),),
    # Not rendered by the preview, but must not turn into base colour either.
    "clearcoatTexture": (("other", 5.0),),
    "clearcoatRoughnessTexture": (("other", 5.0),),
    "clearcoatNormalTexture": (("other", 5.0),),
    "transmissionTexture": (("other", 5.0),),
    "thicknessTexture": (("other", 5.0),),
    "sheenColorTexture": (("other", 5.0),),
    "sheenRoughnessTexture": (("other", 5.0),),
    "iridescenceTexture": (("other", 5.0),),
    "iridescenceThicknessTexture": (("other", 5.0),),
    "anisotropyTexture": (("other", 5.0),),
}

#: Wavefront MTL keywords, lower case, including the PBR extension
#: (map_Pr/Pm/Ps/Ke, norm) and what Polycam writes (normal, map_ao).
MTL_SLOTS: Dict[str, Points] = {
    "map_kd": (("basecolor", 3.0),),
    # Ambient: an AO/lightmap, or a copy of map_Kd.
    "map_ka": (("ao", 1.5), ("basecolor", 1.0)),
    "map_ks": (("specular", 2.0), ("roughness", 0.4), ("metallic", 0.4), ("orm", 0.4)),
    # Exporters disagree about which way shininess runs.
    "map_ns": (("gloss", 1.6), ("roughness", 1.4)),
    "map_d": (("opacity", 3.0),),
    "map_tr": (("opacity", 3.0),),
    # Bump holds normal maps as often as height maps; the pixels decide.
    "map_bump": (("normal", 2.0), ("height", 2.0)),
    "bump": (("normal", 2.0), ("height", 2.0)),
    "norm": (("normal", 3.5),),
    "map_kn": (("normal", 3.5),),
    "normal": (("normal", 3.5),),
    "map_normal": (("normal", 3.5),),
    "map_norm": (("normal", 3.5),),
    "disp": (("height", 3.0),),
    "decal": (("opacity", 1.5),),
    "refl": (("specular", 1.2), ("metallic", 1.0)),
    "map_refl": (("specular", 1.2), ("metallic", 1.0)),
    "map_pr": (("roughness", 3.5),),
    "map_pm": (("metallic", 3.5),),
    # Sheen.
    "map_ps": (("other", 3.0),),
    "map_ke": (("emissive", 3.5),),
    "map_ao": (("ao", 3.0),),
    "map_rma": (("orm", 3.0),),
    "map_orm": (("orm", 3.0),),
}

#: assimp aiTextureType numbers.  assimp-py walks NONE..UNKNOWN (0..18), which
#: includes the PBR types 12-17 its constants do not name.
AI_SLOTS: Dict[int, Points] = {
    1: (("basecolor", 3.0),),  # DIFFUSE
    2: (("specular", 1.8), ("roughness", 0.5), ("metallic", 0.4), ("orm", 0.4)),  # SPECULAR
    3: (("ao", 1.5), ("basecolor", 0.5)),  # AMBIENT
    4: (("emissive", 3.0),),  # EMISSIVE
    5: (("height", 2.0), ("normal", 2.0)),  # HEIGHT: FBX Bump, OBJ map_bump
    6: (("normal", 3.5),),  # NORMALS
    7: (("gloss", 1.5), ("roughness", 1.5)),  # SHININESS: Blender puts roughness here
    8: (("opacity", 3.0),),  # OPACITY
    9: (("height", 3.0),),  # DISPLACEMENT
    10: (("ao", 2.5), ("other", 1.0)),  # LIGHTMAP, "aka AO"
    11: (("metallic", 1.5), ("specular", 1.0)),  # REFLECTION: Blender's ReflectionFactor
    12: (("basecolor", 4.0),),  # BASE_COLOR
    13: (("normal", 4.0),),  # NORMAL_CAMERA
    14: (("emissive", 4.0),),  # EMISSION_COLOR
    15: (("metallic", 4.0),),  # METALNESS
    16: (("roughness", 3.5), ("gloss", 0.5)),  # DIFFUSE_ROUGHNESS; 3ds Max gloss lands here too
    17: (("ao", 4.0),),  # AMBIENT_OCCLUSION
    # UNKNOWN: where older assimp put glTF metallicRoughness.
    18: (("orm", 1.0), ("metal_rough", 1.0), ("other", 1.0)),
}

#: FBX material properties, by the part after the last "|", lower case:
#: exporters disagree about case and vendor prefixes ("Maya|normalCamera",
#: "3dsMax|Parameters|bump_map") but not about that part.
FBX_SLOTS: Dict[str, Points] = {
    "diffusecolor": (("basecolor", 3.0),),
    "diffuse": (("basecolor", 3.0),),
    "diffusefactor": (("basecolor", 1.5), ("ao", 0.5)),
    "basecolor": (("basecolor", 4.0),),
    "base_color": (("basecolor", 4.0),),
    "base_color_map": (("basecolor", 4.0),),
    "tex_color_map": (("basecolor", 4.0),),
    "normalmap": (("normal", 3.5),),
    "normalcamera": (("normal", 4.0),),
    "tex_normal_map": (("normal", 4.0),),
    "norm_map": (("normal", 4.0),),
    "bump": (("normal", 2.0), ("height", 2.0)),
    "bumpmap": (("normal", 2.0), ("height", 2.0)),
    "bump_map": (("normal", 2.5), ("height", 1.5)),
    "bumpfactor": (("height", 1.0), ("normal", 1.0)),
    "displacementcolor": (("height", 3.0),),
    "displacement_map": (("height", 3.0),),
    "specularcolor": (("specular", 2.0), ("roughness", 0.4), ("metallic", 0.4), ("orm", 0.4)),
    "specularfactor": (("specular", 2.0), ("roughness", 0.4), ("metallic", 0.4)),
    "specular_map": (("specular", 3.0),),
    # Blender: roughness, not inverted.  3ds Max / Maya: glossiness.
    "shininessexponent": (("gloss", 1.5), ("roughness", 1.5)),
    "shininess": (("gloss", 1.5), ("roughness", 1.5)),
    "glossiness_map": (("gloss", 3.5),),
    "roughness": (("roughness", 3.5),),
    "roughness_map": (("roughness", 3.5),),
    "tex_roughness_map": (("roughness", 4.0),),
    "specularroughness": (("roughness", 4.0),),
    "diffuseroughness": (("roughness", 2.0),),
    "metalness": (("metallic", 4.0),),
    "metalness_map": (("metallic", 4.0),),
    "tex_metallic_map": (("metallic", 4.0),),
    # Blender exports metallic here.
    "reflectionfactor": (("metallic", 2.0), ("specular", 1.0)),
    # An environment map in 3ds Max files, never a surface property.
    "reflectioncolor": (("other", 3.0),),
    "emissive": (("emissive", 3.5),),
    "emissivecolor": (("emissive", 3.5),),
    "emissivefactor": (("emissive", 3.0),),
    "emissioncolor": (("emissive", 4.0),),
    "emission_map": (("emissive", 4.0),),
    "emit_color_map": (("emissive", 4.0),),
    "tex_emissive_map": (("emissive", 4.0),),
    # White is opaque in every Blender and 3ds Max file inspected.
    "transparentcolor": (("opacity", 3.0),),
    "transparencyfactor": (("opacity", 3.0),),
    "opacity_map": (("opacity", 3.5),),
    "ambientcolor": (("ao", 1.5), ("basecolor", 0.5)),
    "ao_map": (("ao", 4.0),),
    "tex_ao_map": (("ao", 4.0),),
    "ambientocclusion": (("ao", 4.0),),
}

#: How evidence lines name each kind of slot.
_KIND_NAMES = {"gltf": "glTF", "fbx": "FBX", "mtl": "MTL", "assimp": "assimp type"}


def slot_evidence(slot: Slot) -> Points:
    """The points a slot gives, before the exporter is taken into account."""
    if slot.kind == "gltf":
        return GLTF_SLOTS.get(_gltf_key(slot.key), ())
    if slot.kind == "mtl":
        return MTL_SLOTS.get(slot.key.strip().lower(), ())
    if slot.kind == "fbx":
        return FBX_SLOTS.get(_fbx_key(slot.key), ())
    if slot.kind == "assimp":
        try:
            return AI_SLOTS.get(int(slot.key), ())
        except ValueError:
            return ()
    return ()


def exporter_evidence(slot: Slot, exporter: Optional[str]) -> Points:
    """What the program that wrote the file changes about an ambiguous slot.

    Blender's FBX exporter binds the Principled roughness texture to
    ShininessExponent, not inverted, and metallic to ReflectionFactor; its OBJ
    exporter writes roughness to map_Ns, metallic to map_refl and the normal
    map to map_Bump.  3ds Max and Maya bind glossiness there instead.
    """
    if not exporter:
        return ()
    blender = "blender" in exporter.lower()
    if slot.kind == "mtl" and blender:
        return {
            "map_ns": (("roughness", 2.0), ("gloss", -1.0)),
            "map_refl": (("metallic", 2.0),),
            "refl": (("metallic", 2.0),),
            "map_bump": (("normal", 1.0),),
            "bump": (("normal", 1.0),),
        }.get(slot.key.strip().lower(), ())
    if slot.kind == "fbx":
        key = _fbx_key(slot.key)
        if key in ("shininessexponent", "shininess"):
            return (("roughness", 2.0), ("gloss", -1.0)) if blender else (("gloss", 1.2),)
        if key == "reflectionfactor":
            return (("metallic", 1.5),) if blender else (("specular", 0.8),)
    return ()


def _gltf_key(key: str) -> str:
    """``"KHR_materials_specular.specularTexture"`` -> ``"specularTexture"``."""
    return key.strip().rsplit(".", 1)[-1]


def _fbx_key(key: str) -> str:
    return key.rsplit("|", 1)[-1].strip().lower()


# ---------------------------------------------------------------------------
#  File names
# ---------------------------------------------------------------------------

#: Points a name word is worth when it is the last meaningful token.
_NAME_POINTS = 4.0

#: Name words by reading, with the points each is worth.
_WORD_LISTS: Tuple[Tuple[str, float, str], ...] = (
    ("basecolor", _NAME_POINTS, """
        basecolor basecolour albedo diffuse diff dif color colour col alb basemap maintex
        diffusemap albedomap diffusecolor diffusecolour clr bc baseclr albedotransparency
        albedoalpha basecolormap basecoloropacity basecoloralpha diffusealpha albedoopacity
        diffalpha coloralpha colouralpha coloropacity diffuseopacity albedotransparent
        diffusetransparency basecolortransparency"""),
    ("normal", _NAME_POINTS, """
        normal normals normalmap nrm nor nrml norm nml nmap ddn normalgl normaldx normalopengl
        normaldirectx normalogl tangentnormal tsnormal nrmgl nrmdx norgl nordx normalmaps
        normalsmoothness normalgloss normalheight normalalpha normalbump nm"""),
    ("normal_os", _NAME_POINTS, """
        objectnormal osnormal objectspacenormal worldnormal wsnormal normalos normalws
        objectspace worldspacenormal"""),
    ("height", _NAME_POINTS, """
        height heightmap hgt disp displacement displace dsp parallax parallaxmap heights"""),
    ("roughness", _NAME_POINTS, "roughness rough rgh roughnessmap rghn roughnes"),
    ("gloss", _NAME_POINTS, """
        gloss glossiness gls smoothness smooth glossy glossmap smoothnessmap"""),
    ("metallic", _NAME_POINTS, """
        metallic metalness metal met mtl metalic metallness metallicmap metalnessmap"""),
    ("specular", _NAME_POINTS, """
        specular spec spc specularity specularlevel speclevel specularmap specmap f0"""),
    ("ao", _NAME_POINTS, """
        ao ambientocclusion occlusion occ ambocc mixedao aomap occlusionmap cavity"""),
    ("emissive", _NAME_POINTS, """
        emissive emission emit ems emis emiss glow illum illumination selfillum incandescence
        emissivemap emissionmap luminous glowalpha emissivealpha emmisive emisive emmissive"""),
    ("opacity", _NAME_POINTS, """
        opacity alpha transparency transparent cutout opac alphamask opacitymap"""),
    ("orm", _NAME_POINTS, """
        orm occlusionroughnessmetallic occlusionroughnessmetalness aoroughnessmetallic
        ambientocclusionroughnessmetallic ormh"""),
    # "Arm" is also a body part (Robot_Arm.png): worth less, so the pixels decide.
    ("orm", 2.5, "arm armh"),
    ("metal_rough", _NAME_POINTS, """
        metallicroughness metalrough metalroughness metallicrough roughnessmetallic
        roughmetal"""),
    ("metal_smooth", _NAME_POINTS, """
        metallicsmoothness metalsmooth metallicgloss metallicglossmap metalsmoothness
        metallicsmooth"""),
    ("spec_gloss", _NAME_POINTS, """
        specularsmoothness specsmooth specgloss specularglossiness specularglossinessmap
        specularsmooth specglossmap"""),
    ("mask_hdrp", _NAME_POINTS, "maskmap"),
    ("other", 3.0, """
        id matid materialid colorid curvature curv thickness translucency sss subsurface
        position bentnormal lightmap shadow noise ramp lut flow icon preview thumb thumbnail
        cubemap skybox hdri sheen fuzz clearcoat anisotropy iridescence worldposition pos
        vertexcolor"""),
)

#: Name words (after camelCase and separator splitting; compounds of up to
#: three adjacent tokens are joined) and what they are worth.
NAME_WORDS: Dict[str, Points] = {
    word: ((reading, points),)
    for reading, points, words in _WORD_LISTS
    for word in words.split()
}
# Words that could mean more than one thing.
NAME_WORDS.update(
    {
        "bump": (("height", 2.5), ("normal", 2.0)),
        "bumpmap": (("height", 2.0), ("normal", 2.5)),
        "mask": (("opacity", 1.5), ("other", 1.0), ("orm", 0.8)),
        "reflection": (("specular", 2.0), ("metallic", 1.5)),
        "refl": (("specular", 2.0), ("metallic", 1.5)),
        "reflect": (("specular", 2.0), ("metallic", 1.5)),
        "ambient": (("ao", 2.0), ("basecolor", 0.5)),
        "depth": (("height", 1.5), ("other", 1.0)),
    }
)

#: One- and two-letter suffixes.  They count only as the last token of a name
#: with more than one, so "T_Rock_N" is a normal map and "N.png" is not, and a
#: stray "_d" inside a hex run cannot make a colour map.
SUFFIX_LETTERS: Dict[str, Points] = {
    "d": (("basecolor", 3.0),),
    "c": (("basecolor", 2.0),),
    "al": (("basecolor", 2.5),),
    "df": (("basecolor", 2.0),),
    # Only reached when no normal word precedes it: 3ds Max "FlashLight_GL" is gloss.
    "gl": (("gloss", 2.5),),
    "n": (("normal", 3.0),),
    "nm": (("normal", 3.0),),
    "nrm": (("normal", 3.5),),
    "r": (("roughness", 2.0),),
    "m": (("metallic", 1.5), ("other", 0.5)),
    "s": (("specular", 1.5), ("gloss", 1.0)),
    "g": (("gloss", 1.2),),
    "h": (("height", 2.5),),
    "e": (("emissive", 2.5),),
    "dp": (("height", 2.0),),
    "dm": (("height", 1.5),),
    "meta": (("metallic", 2.0),),
    "sp": (("specular", 2.0),),
    "sm": (("gloss", 1.5),),
    "ro": (("roughness", 1.5),),
    "em": (("emissive", 2.5),),
    "o": (("ao", 1.5), ("opacity", 0.8)),
    "op": (("opacity", 2.0),),
    "mr": (("metal_rough", 2.0), ("orm", 1.0)),
    "rm": (("metal_rough", 1.5), ("orm", 1.0)),
    "ms": (("metal_smooth", 2.0),),
    "sg": (("spec_gloss", 1.5),),
    "mk": (("other", 1.0),),
    # CryEngine: normal with gloss in alpha.
    "ddna": (("normal", 3.5),),
    # Bent normals.
    "bn": (("other", 2.0),),
}

#: Normal-map Y convention words.
_Y_OPENGL = frozenset({"gl", "ogl", "opengl"})
_Y_DIRECTX = frozenset({"dx", "directx", "d3d"})

#: Tokens that say nothing about the role.
_NOISE = re.compile(
    r"^(\d+|\d*k|\d+px|px|png|jpe?g|tga|tiff?|psd|exr|webp|srgb|linear|lin|raw|bit|udim|"
    r"v\d+|lod\d*|"
    r"final|copy|new|old|tx|t|img|image|map|mat|material|texture|tex|hd|sd|hq|lq|low|high)$"
)

#: Convention words that camelCase and digit splitting would tear apart
#: ("DirectX" -> "direct x", "D3D" -> "d 3 d").
_WHOLE_WORDS = re.compile(r"(directx|opengl|d3d)", re.IGNORECASE)

_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
#: Bare hex runs of 8+ characters with at least one digit and one letter.
_HEX_RUN = re.compile(
    r"(?<![A-Za-z0-9])(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{8,}(?![A-Za-z0-9])"
)

#: Channel words in packed-map names: "_M_AO_S", "_MT_AO_H_SM".
_CHANNEL_TOKENS = {
    **dict.fromkeys(("m", "mt", "met", "metal", "metallic", "metalness"), "metallic"),
    **dict.fromkeys(("r", "rough", "roughness"), "roughness"),
    **dict.fromkeys(("ao", "o", "occ", "occlusion"), "ao"),
    **dict.fromkeys(("h", "height"), "height"),
    **dict.fromkeys(("s", "sm", "smooth", "smoothness", "g", "gloss"), "gloss"),
    **dict.fromkeys(("e", "em", "emission", "emissive"), "emissive"),
}
#: Letters of a packed acronym: "RMA", "MRA", "ORM".
_ACRONYM_LETTERS = {
    **dict.fromkeys("oa", "ao"),
    "r": "roughness",
    "m": "metallic",
    "h": "height",
    "e": "emissive",
    **dict.fromkeys("sg", "gloss"),
}
_ORM_ORDER = ("ao", "roughness", "metallic")


class NameEvidence(NamedTuple):
    """What a file name says."""

    points: Points
    #: ``"opengl"`` / ``"directx"`` when the name says which way green points.
    y_hint: Optional[str]
    #: Quantities in channel order R, G, B[, A] when the name spells a packing out.
    packing: Optional[Tuple[str, ...]]
    #: The tokens that carry meaning, for evidence lines.
    tokens: Tuple[str, ...]


def split_tokens(stem: str) -> List[str]:
    """Lower-case tokens of a file stem: UUIDs and hex runs removed, camelCase,
    letter/digit boundaries and separators split."""
    tokens: List[str] = []
    # re.split with a group keeps the whole words at the odd positions.
    for i, part in enumerate(_WHOLE_WORDS.split(_HEX_RUN.sub(" ", _UUID.sub(" ", stem)))):
        if i % 2:
            tokens.append(part.lower())
            continue
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", part)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
        s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
        s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
        tokens.extend(t.lower() for t in re.split(r"[^A-Za-z0-9]+", s) if t)
    return tokens


def name_evidence(name: Optional[str]) -> NameEvidence:
    """Score a file or image name.

    Compounds of three, two and one tokens are looked up scanning from the
    end; the last hit is worth full points, up to two earlier ones 0.35 of
    theirs, and a name of a single meaningful token half.  A run of channel
    words (``_M_AO_S``) or a packed acronym (``RMA``) is a packed map with that
    channel order.  ``GL``/``DX`` set the Y convention -- except that ``_GL``
    with no normal word before it is 3ds Max's gloss suffix.
    """
    if not name:
        return NameEvidence((), None, None, ())
    stem = os.path.splitext(os.path.basename(str(name).replace("\\", "/")))[0]
    tokens = split_tokens(stem)
    y_hint: Optional[str] = None
    kept: List[Tuple[str, bool]] = []  # (token, says nothing)
    for i, token in enumerate(tokens):
        if token == "k" and i > 0 and tokens[i - 1].isdigit():
            continue  # "2k" split in two
        if token in _Y_OPENGL or token in _Y_DIRECTX:
            is_last = all(_NOISE.match(t) for t in tokens[i + 1 :])
            after_normal = any(
                t in ("normal", "normals", "nrm", "nor", "norm", "nml", "nm", "n")
                for t in tokens[:i]
            )
            if token == "gl" and is_last and not after_normal:
                kept.append((token, False))
                continue
            y_hint = "opengl" if token in _Y_OPENGL else "directx"
            continue
        kept.append((token, bool(_NOISE.match(token))))
    clean = [t for t, noise in kept if not noise]
    single = len(clean) == 1

    points: List[Tuple[str, float]] = []
    run, packing = _packing(clean)
    # A spelled-out run is explicit; a lone ORM/ARM acronym is scored by its
    # own name word, which keeps "Robot_Arm" at the lower weight.
    if packing and not single and (run or packing[:3] != _ORM_ORDER):
        points.append(("packed", 4.0))

    i = len(kept) - 1
    rank = 0
    while i >= 0 and rank < 3:
        hit: Optional[Tuple[int, Points]] = None
        for n in (3, 2, 1):
            if i - n + 1 < 0:
                continue
            parts = kept[i - n + 1 : i + 1]
            if n == 1 and parts[0][1]:
                break  # a lone noise token ("map", "2k") says nothing
            word = "".join(t for t, _ in parts)
            if word in NAME_WORDS:
                hit = (n, NAME_WORDS[word])
                break
        is_last = all(noise for _, noise in kept[i + 1 :])
        if (
            hit is None
            and is_last
            and not single
            and not kept[i][1]
            and kept[i][0] in SUFFIX_LETTERS
        ):
            hit = (1, SUFFIX_LETTERS[kept[i][0]])
        if hit is None:
            i -= 1
            continue
        n, found = hit
        factor = (1.0 if rank == 0 else 0.35) * (0.5 if single else 1.0)
        points.extend((reading, w * factor) for reading, w in found)
        rank += 1
        i -= n
    if y_hint and not any(reading in ("normal", "normal_os") for reading, _ in points):
        points.append(("normal", 2.0))
    return NameEvidence(tuple(points), y_hint, packing, tuple(clean))


def _packing(clean: Sequence[str]) -> Tuple[bool, Optional[Tuple[str, ...]]]:
    """``(spelled out as a run of words, channel order)`` of a packed-map name."""
    run: List[str] = []
    for token in reversed(clean):
        if token not in _CHANNEL_TOKENS:
            break
        run.append(_CHANNEL_TOKENS[token])
    run.reverse()
    if len(run) >= 2 and len(set(run)) == len(run):
        return True, tuple(run)
    if clean:
        last = clean[-1]
        if 3 <= len(last) <= 4 and all(c in _ACRONYM_LETTERS for c in last):
            order = tuple(_ACRONYM_LETTERS[c] for c in last)
            if len(set(order)) == len(order) and len(set(_ORM_ORDER) & set(order)) >= 2:
                return False, order
    return False, None


# ---------------------------------------------------------------------------
#  Pixels
# ---------------------------------------------------------------------------

#: How evidence lines describe what the pixels looked like.
_SHAPE_TEXT = {
    "flat-normal": "a flat normal map",
    "tangent-normal": "a tangent-space normal map",
    "two-channel-normal": "a two-channel normal map",
    "weak-normal": "an unnormalised normal map",
    "object-normal": "an object-space normal map",
    "constant": "one flat colour",
    "packed": "independent channels",
    "colour": "a colour image",
    "colour-dark": "a mostly black colour image",
    "gray": "grayscale, mid tones",
    "gray-bright": "grayscale, mostly bright",
    "gray-dark": "grayscale, mostly black",
    "gray-bimodal": "grayscale, black and white",
}


def stats_evidence(f: texstats.Features) -> Tuple[str, Points]:
    """``(shape, points)`` from the pixel statistics.

    ``shape`` is a key of the descriptions above, with ``+alpha`` appended
    when a grayscale or colour image has a real alpha channel.
    """
    if texstats.is_flat_normal(f):
        return "flat-normal", (("normal", 3.0),)
    if texstats.is_tangent_normal(f):
        return "tangent-normal", (("normal", 6.0),) + _all_but("normal", -3.0)
    if texstats.is_two_channel_normal(f):
        return "two-channel-normal", (("normal", 4.5),) + _all_but("normal", -2.0)
    if texstats.is_weak_tangent_normal(f):
        return "weak-normal", (("normal", 4.5),) + _all_but("normal", -2.0)
    if texstats.is_object_space_normal(f):
        return "object-normal", (("normal_os", 4.0), ("normal", 0.5), ("basecolor", -1.0))

    gray = texstats.is_gray(f)
    points: List[Tuple[str, float]] = [("normal", -4.0 if gray else -2.5), ("normal_os", -3.0)]
    if texstats.is_constant(f):
        points.append(("basecolor", 0.3))
        if f.lum_mean > 0.97:
            points += [("ao", 0.8), ("opacity", 0.8), ("roughness", 0.3)]
        elif f.lum_mean < 0.03:
            points += [("metallic", 0.8), ("emissive", 0.8)]
        return "constant", tuple(points)

    alpha = f.alpha_meaningful
    if texstats.is_packed(f):
        # Per-channel look of an ORM: R bright (AO) or constant (unused), B black or
        # white (metal).
        points += [
            ("basecolor", -1.0),
            ("roughness", -2.0),
            ("gloss", -2.0),
            ("metallic", -2.0),
            ("ao", -2.0),
            ("height", -2.0),
            ("opacity", -2.0),
        ]
        if alpha and f.std_b < 0.05 and f.p50_r < 0.5:
            points += [("mask_hdrp", 3.0), ("orm", 0.5)]
        elif alpha:
            points += [("mask_hdrp", 2.0), ("orm", 1.0), ("metal_smooth", 1.0)]
        else:
            orm = 2.5 + (0.5 if f.p50_r > 0.6 else 0.0) + (0.3 if f.ext_b > 0.7 else 0.0)
            metal_rough = 2.0 + (1.2 if f.std_r < 0.02 else 0.0)
            points += [("orm", orm), ("metal_rough", metal_rough), ("mask_hdrp", 0.8)]
        points.append(("packed", 2.5))
        return "packed", tuple(points)

    if not gray:
        shape = "colour"
        points += [("basecolor", 2.5), ("specular", 0.3), ("emissive", 0.3)]
        scalar = ("roughness", "gloss", "metallic", "ao", "height", "opacity", "metal_smooth")
        points += [(r, -1.5) for r in scalar]
        if f.frac_black > 0.5:
            shape = "colour-dark"
            points += [("emissive", 2.8), ("basecolor", -1.0)]
        if alpha:
            points += [("basecolor", 0.5), ("spec_gloss", 0.5 if f.lum_mean < 0.3 else 0.0)]
            shape += "+alpha"
        return shape, tuple(points)

    points += [
        ("basecolor", 0.4),
        ("roughness", 1.0),
        ("gloss", 1.0),
        ("height", 1.0),
        ("specular", 0.6),
        ("ao", 0.5),
        ("metallic", 0.5),
        ("opacity", 0.3),
        ("emissive", 0.1),
    ]
    # Most surfaces are rough: roughness maps skew bright, gloss maps dark.
    tilt = max(-0.6, min(0.6, (f.lum_mean - 0.45) / 0.3 * 0.6))
    points += [("roughness", tilt), ("gloss", -tilt)]
    if f.lum_p50 > 0.8 and f.lum_skew < -0.3:
        shape = "gray-bright"
        points += [("ao", 2.2), ("opacity", 0.6), ("height", -0.5), ("metallic", -0.5)]
    elif f.frac_black > 0.5:
        shape = "gray-dark"
        points += [
            ("metallic", 1.3),
            ("emissive", 1.0),
            ("opacity", 0.4),
            ("ao", -1.5),
            ("height", -1.5),
            ("basecolor", -0.5),
        ]
    elif f.frac_extreme > 0.9:
        shape = "gray-bimodal"
        points += [("metallic", 1.0), ("opacity", 1.0), ("ao", 0.3)]
    else:
        shape = "gray"
        points += [("roughness", 0.5), ("height", 0.5), ("gloss", 0.3)]
    if alpha:
        # Grayscale RGB with a real alpha: Unity metallic/smoothness or spec/gloss.
        points += [
            ("metal_smooth", 2.0 if f.frac_black > 0.3 else 1.2),
            ("spec_gloss", 1.2),
            ("basecolor", 0.2),
        ]
        shape += "+alpha"
    return shape, tuple(points)


def _all_but(keep: str, points: float) -> Points:
    return tuple((r, points) for r in _READINGS if r != keep)


# ---------------------------------------------------------------------------
#  The decision
# ---------------------------------------------------------------------------


def classify(
    image: Union[np.ndarray, texstats.Features],
    name: Optional[str],
    slots: Sequence[Slot],
    fmt: str,
    exporter: Optional[str],
    *,
    gltf_same_image_as_occlusion: bool = False,
) -> RoleGuess:
    """Decide what an image is for.

    ``image`` is the decoded pixels as :func:`texstats.features` takes them
    (keep the native channel count: an RGB map padded to RGBA reads as having no
    alpha, which is right, but costs a copy), or features measured earlier; the
    curl test that tells OpenGL from DirectX normal maps then has nothing to
    look at and abstains.  ``name`` is the file or image name, ``slots`` every
    slot of one material the image is bound to, ``fmt`` the container
    (``"gltf"``, ``"fbx"``, ``"obj"``, ...; glTF declares its normal maps
    OpenGL), ``exporter`` the program that wrote the file (FBX Creator or
    ApplicationName, the MTL header comment).  ``gltf_same_image_as_occlusion``
    says the image is also the material's occlusionTexture through another
    texture object, which ``slots`` alone would not show.
    """
    if isinstance(image, texstats.Features):
        stats, pixels = image, None
    else:
        stats, pixels = texstats.features(image), image

    tally = _Tally()
    slot_normal = 0.0
    for slot in slots:
        by_slot = slot_evidence(slot)
        by_exporter = exporter_evidence(slot, exporter)
        tally.add(f"slot {_KIND_NAMES.get(slot.kind, slot.kind)} {slot.key}", by_slot)
        tally.add(f"{exporter} writes {slot.key}", by_exporter)
        slot_normal += sum(
            w for reading, w in by_slot + by_exporter if reading == "normal" and w > 0
        )
    named = name_evidence(name)
    tally.add(f"name {name}", named.points)
    shape, by_pixels = stats_evidence(stats)
    alpha_text = ", with alpha" if shape.endswith("+alpha") else ""
    tally.add(f"pixels: {_SHAPE_TEXT[shape.split('+')[0]]}{alpha_text}", by_pixels)
    if slot_normal and (shape.startswith("gray") or shape == "constant"):
        # 3ds Max Bump, Unity "create from grayscale", MTL bump: a bump map.
        tally.add("grayscale in a normal slot is a bump map", (("height", slot_normal),))
    tally.scores["basecolor"] += _BASECOLOR_PRIOR

    reading = max(_READINGS, key=lambda r: tally.scores[r])  # first of equals wins
    top = tally.scores[reading]
    confidence = 1.0 / sum(math.exp(s - top) for s in tally.scores.values())
    notes: List[str] = []

    # A real alpha channel turns a single map into the packed layout that uses it.
    if stats.alpha_meaningful and reading == "specular":
        reading = "spec_gloss"
        notes.append("specular map with a real alpha: glossiness in A")
    elif stats.alpha_meaningful and reading == "metallic" and texstats.is_gray(stats):
        reading = "metal_smooth"
        notes.append("grayscale metallic with a real alpha: smoothness in A")

    gltf_keys = {_gltf_key(s.key) for s in slots if s.kind == "gltf"}
    in_mr = "metallicRoughnessTexture" in gltf_keys
    as_occlusion = gltf_same_image_as_occlusion or "occlusionTexture" in gltf_keys
    if in_mr and as_occlusion and reading not in ("normal", "normal_os"):
        reading = "orm"
        notes.append("glTF occlusion and metallicRoughness share the image: ORM")

    if reading != "packed":
        role, layout = _ROLE_OF[reading], dict(_LAYOUTS[reading])
    elif named.packing:
        role, layout = "orm", _named_layout(named.packing)
    else:
        # Independent channels in an order nothing names: show it as it is.
        role, layout = "other", {}
        notes.append("packed channels in an unknown order")
    if in_mr and not as_occlusion and layout.get("ao") == "r":
        # Pixal3D writes R = 0 there; read as AO it would render black.
        del layout["ao"]
        notes.append("glTF metallicRoughnessTexture: R is not AO")
        if not layout:
            role, layout = "orm", dict(_LAYOUTS["metal_rough"])
    layout = _fit_to_image(layout, stats)

    normal_space: Optional[str] = None
    y_convention: Optional[str] = None
    y_confident = False
    if role == "normal":
        normal_space = "object" if reading == "normal_os" else "tangent"
        if normal_space == "tangent":
            y_convention, y_confident, why = _y_convention(pixels, named.y_hint, fmt)
            notes.append(why)

    return RoleGuess(
        role=role,
        channels=layout,
        colorspace="srgb" if role in SRGB_ROLES else "linear",
        normal_space=normal_space,
        y_convention=y_convention,
        y_confident=y_confident,
        confidence=confidence,
        evidence=tally.lines(reading) + tuple(notes),
    )


def _named_layout(packing: Sequence[str]) -> Dict[str, str]:
    """Channel layout of a packed map from the quantities its name lists in RGBA order."""
    layout: Dict[str, str] = {}
    for channel, quantity in zip("rgba", packing):
        if quantity == "gloss":
            if "roughness" in packing:
                continue
            layout["roughness"] = channel
            layout["invert"] = "1"
        else:
            layout[quantity] = channel
    return layout


def _fit_to_image(layout: Dict[str, str], stats: texstats.Features) -> Dict[str, str]:
    """Drop alpha an image lacks or never uses; a mask without alpha is read from R."""
    has_alpha = stats.channels in (2, 4)
    if layout.get("opacity") == "a" and not stats.alpha_meaningful:
        if "basecolor" in layout:
            del layout["opacity"]  # an opaque colour map
        else:
            layout["opacity"] = "r"  # a grayscale mask: white is opaque
    if not has_alpha:
        for quantity in [q for q, channel in layout.items() if channel == "a"]:
            del layout[quantity]
    if "roughness" not in layout:
        layout.pop("invert", None)
    return layout


_Y_TEXT = {"opengl": "up (OpenGL)", "directx": "down (DirectX)"}


def _y_convention(
    pixels: Optional[np.ndarray], y_hint: Optional[str], fmt: str
) -> Tuple[str, bool, str]:
    """``(convention, confident, evidence line)`` for a tangent-space normal map."""
    if y_hint:
        return y_hint, True, f"green {_Y_TEXT[y_hint]}: the name says so"
    if fmt == "gltf":
        return "opengl", True, "green up (OpenGL): glTF normal maps always are"
    if pixels is None:
        return "opengl", False, "green up (OpenGL) assumed: no pixels to test"
    call, stat = texstats.curl_convention(pixels)
    if call is None:
        why = f"green up (OpenGL) assumed: the curl test cannot tell ({stat:+.3f})"
        return "opengl", False, why
    return call, True, f"green {_Y_TEXT[call]}: curl test {stat:+.3f}"


class _Tally:
    """Points per reading, and the lines that explain them."""

    def __init__(self) -> None:
        self.scores: Dict[str, float] = dict.fromkeys(_READINGS, 0.0)
        self._records: List[Tuple[str, Points]] = []

    def add(self, label: str, points: Points) -> None:
        if not points:
            return
        merged: Dict[str, float] = {}
        for reading, w in points:
            self.scores[reading] += w
            merged[reading] = merged.get(reading, 0.0) + w
        self._records.append((label, tuple(merged.items())))

    def lines(self, winner: str) -> Tuple[str, ...]:
        """One line per source: its positive points, and what it took from the winner."""
        out = []
        for label, points in self._records:
            shown = [(r, w) for r, w in points if w > 0 or r == winner]
            shown.sort(key=lambda rw: -rw[1])
            if shown:
                out.append(
                    f"{label} -> "
                    + ", ".join(f"{_READING_NAMES[r]} {w:+.1f}" for r, w in shown)
                )
        return tuple(out)
