"""The texture row's logic in ``static/panel.js``, run under node.

What the buttons say is decided by pure functions -- their order, which of
the server's guesses each one explains, when a "?" marks a weak guess, and
which files of a dropped folder are imported -- so they are checked here
without a browser. What the buttons *show* is checked by render_checks.html.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

PANEL_JS = (
    Path(__file__).resolve().parents[2]
    / "python" / "instant_meshes_brush" / "static" / "panel.js"
)

#: Driver: import panel.js, run one of its helpers on the JSON it is given,
#: print the result as JSON. Folders arrive as nested objects and are turned
#: into FileSystemEntry look-alikes that answer readEntries in two batches, as
#: a browser does for large folders.
_BRIDGE_JS = """\
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [modulePath, inputPath] = process.argv.slice(2);
const panel = await import(pathToFileURL(modulePath).href);
const input = JSON.parse(readFileSync(inputPath, 'utf-8'));

function entry(name, node) {
  if (node === null) {
    return { isFile: true, name, file: (ok) => ok({ name }) };
  }
  const children = Object.entries(node).map(([child, value]) => entry(child, value));
  const half = Math.ceil(children.length / 2);
  const batches = [children.slice(0, half), children.slice(half), []];
  return {
    isFile: false,
    name,
    createReader: () => ({ readEntries: (ok) => ok(batches.shift() || []) }),
  };
}

let result;
if (input.describe) {
  result = panel.describeButtons(input.describe.buttons, input.describe.materials);
} else {
  const entries = Object.entries(input.drop).map(([name, node]) => entry(name, node));
  result = (await panel.importableFiles(entries)).map((file) => file.name);
}
process.stdout.write(JSON.stringify(result));
"""


def _run(tmp_path: Path, payload: Dict[str, Any]) -> Any:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so panel.js cannot be run")
    bridge = tmp_path / "bridge.mjs"
    bridge.write_text(_BRIDGE_JS, encoding="utf-8")
    source = tmp_path / "input.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [node, str(bridge), str(PANEL_JS), str(source)],
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _describe(tmp_path: Path, buttons: List[dict], materials: List[dict]) -> List[dict]:
    return _run(tmp_path, {"describe": {"buttons": buttons, "materials": materials}})


def _button(id_: str, slot: int, channel: str = "rgb", label: str = "") -> dict:
    return {"id": id_, "label": label or id_, "slot": slot, "channel": channel,
            "colorspace": "linear"}


def _guess(file: str, role: str, confidence: float = 0.95, **extra: Any) -> dict:
    return {"file": file, "role": role, "confidence": confidence,
            "evidence": [f"evidence for {file}"], **extra}


def test_buttons_come_in_panel_order_whatever_order_the_server_sends(tmp_path) -> None:
    ids = ["tex5", "height", "metallic", "ao", "tex3", "emissive", "normal",
           "opacity", "roughness", "basecolor"]
    buttons = [_button(id_, slot=int(id_[3:]) if id_.startswith("tex") else 0) for id_ in ids]

    described = _describe(tmp_path, buttons, [])

    assert [b["id"] for b in described] == [
        "basecolor", "opacity", "normal", "roughness", "metallic", "ao", "emissive",
        "height", "tex3", "tex5",
    ]


def test_pixal3d_metal_rough_explains_roughness_and_metallic_only(tmp_path) -> None:
    """R = 0 in a Pixal3D map is not occlusion: the guess lists its channels."""
    buttons = [_button("metallic", 1, "b", "Metallic"), _button("basecolor", 0),
               _button("roughness", 1, "g", "Roughness"), _button("ao", 2, "r", "AO")]
    materials = [
        {"maps": {"basecolor": 0, "orm": 1},
         "guesses": [_guess("color.png", "basecolor"),
                     _guess("mr.png", "orm", 0.9, channels={"roughness": "g", "metallic": "b"})]},
        {"maps": {"orm": 2}, "guesses": [_guess("rock_ao.png", "ao")]},
    ]

    by_id = {b["id"]: b for b in _describe(tmp_path, buttons, materials)}

    assert "mr.png" in by_id["roughness"]["hint"]
    assert "the green channel of" in by_id["roughness"]["hint"]
    assert "mr.png" in by_id["metallic"]["hint"]
    assert "mr.png" not in by_id["ao"]["hint"]
    assert "rock_ao.png" in by_id["ao"]["hint"]
    assert "mr.png" not in by_id["basecolor"]["hint"]


@pytest.mark.parametrize(
    "role, explains",
    [
        ("orm", {"roughness", "metallic", "ao"}),
        ("gloss", {"roughness"}),
        ("metal_smooth", {"roughness", "metallic"}),
        ("mask_hdrp", {"roughness", "metallic", "ao"}),
        ("specular", {"basecolor", "metallic"}),
        ("spec_gloss", {"basecolor", "roughness", "metallic"}),
    ],
)
def test_a_guess_without_channels_explains_what_its_role_feeds(tmp_path, role, explains) -> None:
    ids = ["basecolor", "normal", "roughness", "metallic", "ao", "emissive"]
    materials = [{"maps": {"orm": 1}, "guesses": [_guess("packed.png", role)]}]

    described = _describe(tmp_path, [_button(id_, 1) for id_ in ids], materials)

    assert {b["id"] for b in described if "packed.png" in b["hint"]} == explains


def test_a_weak_guess_earns_a_question_mark(tmp_path) -> None:
    buttons = [_button("roughness", 1), _button("metallic", 1), _button("emissive", 2)]
    materials = [{"maps": {"orm": 1, "emissive": 2},
                  "guesses": [_guess("rough.png", "roughness", 0.59),
                              _guess("metal.png", "metallic", 0.6)]}]

    by_id = {b["id"]: b for b in _describe(tmp_path, buttons, materials)}

    assert by_id["roughness"]["unsure"] is True
    assert by_id["roughness"]["confidence"] == pytest.approx(0.59)
    assert by_id["metallic"]["unsure"] is False
    # No guess at all is not a weak guess: nothing to put a "?" on.
    assert by_id["emissive"]["confidence"] is None
    assert by_id["emissive"]["unsure"] is False


def test_the_normal_hint_says_how_green_was_read(tmp_path) -> None:
    materials = [{"maps": {"normal": 0},
                  "guesses": [_guess("T_Rock_N.png", "normal", 0.97,
                                     y_convention="directx", y_confident=False)]}]

    (normal,) = _describe(tmp_path, [_button("normal", 0, label="Normal")], materials)

    assert "DirectX (green down), turned into OpenGL, a guess" in normal["hint"]


def test_an_unnamed_map_lists_the_files_in_its_slot(tmp_path) -> None:
    materials = [
        {"maps": {"tex3": 3}, "guesses": [_guess("mystery.png", "other", 0.4)]},
        {"maps": {}, "guesses": [_guess("elsewhere.png", "other", 0.4)]},
    ]

    (tex,) = _describe(tmp_path, [_button("tex3", 3, label="tex 3")], materials)

    assert "mystery.png" in tex["hint"]
    assert "elsewhere.png" not in tex["hint"]
    assert tex["unsure"] is True


def test_a_named_extra_map_lists_the_files_that_feed_it(tmp_path) -> None:
    """A map with a role but no canonical place ("Specular", id tex0) is explained
    by the guesses whose channels name it, as the server sends them."""
    materials = [{"maps": {"basecolor": 0, "tex0": 1},
                  "guesses": [_guess("Rock_S.png", "specular", 0.8, channels={"tex0": "rgb"}),
                              _guess("Rock_D.png", "basecolor", channels={"basecolor": "rgb"})]}]

    (specular,) = _describe(tmp_path, [_button("tex0", 1, label="Specular")], materials)

    assert "Rock_S.png -- 80% sure" in specular["hint"]
    assert "Rock_D.png" not in specular["hint"]
    assert "(no details from the file)" not in specular["hint"]


def test_a_long_list_of_files_is_cut_short(tmp_path) -> None:
    guesses = [_guess(f"albedo_{i}.png", "basecolor") for i in range(5)]
    materials = [{"maps": {"basecolor": 0}, "guesses": guesses}]

    (base,) = _describe(tmp_path, [_button("basecolor", 0, label="Base colour")], materials)

    assert "albedo_2.png" in base["hint"] and "albedo_3.png" not in base["hint"]
    assert "...and 2 more files" in base["hint"]


def test_a_dropped_folder_is_walked_and_filtered(tmp_path) -> None:
    drop = {
        "chair": {
            "chair.OBJ": None,
            "chair.mtl": None,
            "notes.txt": None,
            "chair.blend": None,
            "textures": {"chair_BaseColor.png": None, "chair_N.TGA": None,
                         "chair_H.exr": None, "old": {"chair_ORM.jpg": None}},
        },
        "loose.zip": None,
    }

    files = _run(tmp_path, {"drop": drop})

    assert sorted(files) == sorted(["chair.OBJ", "chair.mtl", "chair_BaseColor.png",
                                    "chair_N.TGA", "chair_ORM.jpg", "loose.zip"])
