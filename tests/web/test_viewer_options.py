"""The viewer's URL options and its camera table, run under node.

``static/options.js`` decides what a host page's query parameters mean --
which side the panel docks on, what the Export button says and whether it
hands the result to the page -- and ``static/navigation.js`` decides what
every mouse button does to the camera with every modifier. Both are pure, so
they are checked here without a browser. The URLs are built by the Python
helper that hosts use, so the two ends are checked against each other.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

STATIC = Path(__file__).resolve().parents[2] / "python" / "instant_meshes_brush" / "static"

#: Driver: import both modules, answer every question in the input JSON.
_BRIDGE_JS = """\
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [staticDir, inputPath] = process.argv.slice(2);
const at = (name) => pathToFileURL(`${staticDir}/${name}`).href;
const options = await import(at('options.js'));
const navigation = await import(at('navigation.js'));
const input = JSON.parse(readFileSync(inputPath, 'utf-8'));

const result = {
  options: (input.options || []).map(([search, embedded]) =>
    options.readViewerOptions(search, { embedded })),
  drags: (input.drags || []).map(([button, keys, left]) =>
    navigation.dragAction(button, keys, left)),
};
process.stdout.write(JSON.stringify(result));
"""


def _run(tmp_path: Path, payload: Dict[str, Any]) -> Dict[str, List[Any]]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so options.js cannot be run")
    bridge = tmp_path / "bridge.mjs"
    bridge.write_text(_BRIDGE_JS, encoding="utf-8")
    source = tmp_path / "input.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [node, str(bridge), str(STATIC), str(source)],
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _read(tmp_path: Path, search: str, embedded: bool = True) -> Dict[str, Any]:
    return _run(tmp_path, {"options": [[search, embedded]]})["options"][0]


DEFAULTS = {
    "panel": "left",
    "exportLabel": "Export",
    "exportAction": "download",
    "hostOrigin": None,
    "warnings": [],
}


# ---------------------------------------------------------------------------
#  options.js
# ---------------------------------------------------------------------------


def test_a_bare_viewer_url_is_the_viewer_as_it_always_was(tmp_path) -> None:
    assert _read(tmp_path, "?session=abc123") == DEFAULTS
    assert _read(tmp_path, "") == DEFAULTS


def test_options_built_by_the_python_helper_read_back_as_given(tmp_path) -> None:
    server = pytest.importorskip("instant_meshes_brush.server")
    url = server.viewer_url(
        "a b&c", panel="right", export_label="Accept", export_action="host",
        host_origin="http://127.0.0.1:7770",
    )

    read = _read(tmp_path, url[url.index("?"):])

    assert read == {
        "panel": "right",
        "exportLabel": "Accept",
        "exportAction": "host",
        "hostOrigin": "http://127.0.0.1:7770",
        "warnings": [],
    }


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "",
        "127.0.0.1:7770",
        "http://127.0.0.1:7770/page",
        "http://127.0.0.1:7770/?x=1",
        "http://me:pw@127.0.0.1:7770",
        "javascript:alert(1)",
        "file:///C:/page.html",
        "*",
    ],
)
def test_host_mode_without_a_usable_origin_stays_a_download(tmp_path, origin) -> None:
    """Posting to a guessed origin could hand the result to the wrong page."""
    search = "?session=s&export_action=host"
    if origin is not None:
        search += "&host_origin=" + origin.replace("&", "%26").replace("?", "%3F")

    read = _read(tmp_path, search)

    assert read["exportAction"] == "download"
    assert read["hostOrigin"] is None
    assert len(read["warnings"]) == 1 and "host_origin" in read["warnings"][0]


def test_host_mode_needs_a_page_around_the_viewer(tmp_path) -> None:
    read = _read(
        tmp_path, "?session=s&export_action=host&host_origin=http://127.0.0.1:7770",
        embedded=False,
    )
    assert read["exportAction"] == "download"
    assert read["hostOrigin"] is None
    assert "iframe" in read["warnings"][0]


def test_an_origin_is_compared_the_way_the_browser_writes_it(tmp_path) -> None:
    """A message's event.origin is the normalised form, so that is what is kept."""
    read = _read(tmp_path, "?export_action=host&host_origin=HTTP://LocalHost:80/")
    assert read["exportAction"] == "host"
    assert read["hostOrigin"] == "http://localhost"


def test_unknown_values_fall_back_to_the_defaults_and_say_so(tmp_path) -> None:
    read = _read(tmp_path, "?session=s&panel=top&export_action=upload")

    assert read["panel"] == "left"
    assert read["exportAction"] == "download"
    assert len(read["warnings"]) == 2


def test_a_label_is_one_short_line(tmp_path) -> None:
    assert _read(tmp_path, "?export_label=%20%20Accept%0A%20it%20")["exportLabel"] == "Accept it"
    assert _read(tmp_path, "?export_label=%20%20")["exportLabel"] == "Export"
    long = _read(tmp_path, "?export_label=" + "x" * 80)["exportLabel"]
    assert len(long) == 32 and long.endswith("…")


# ---------------------------------------------------------------------------
#  navigation.js
# ---------------------------------------------------------------------------

LEFT, MIDDLE, RIGHT = 0, 1, 2

#: (button, modifiers, what a plain left drag does, expected move)
DRAGS = [
    (MIDDLE, {}, None, "orbit"),                         # Blender
    (MIDDLE, {"altKey": True}, None, "orbit"),           # 3ds Max
    (MIDDLE, {"shiftKey": True}, None, "pan"),           # Blender
    (MIDDLE, {"ctrlKey": True}, None, "zoom"),           # Blender
    (MIDDLE, {"metaKey": True}, None, "zoom"),           # the Mac's Ctrl
    (MIDDLE, {"ctrlKey": True, "shiftKey": True}, None, "zoom"),
    (RIGHT, {}, None, "pan"),
    (RIGHT, {"shiftKey": True}, None, "pan"),
    (RIGHT, {"altKey": True}, None, "zoom"),             # Maya, Unity
    (LEFT, {}, None, None),                              # the brush's, here
    (LEFT, {"shiftKey": True}, None, None),
    (LEFT, {"ctrlKey": True}, None, None),
    (LEFT, {"altKey": True}, None, "orbit"),             # Maya, Unity
    (LEFT, {}, "orbit", "orbit"),                        # a viewer with no brush
    (LEFT, {"altKey": True}, "orbit", "orbit"),
    (3, {}, None, None),                                 # the back button
]


def test_every_drag_moves_the_camera_as_the_table_says(tmp_path) -> None:
    got = _run(tmp_path, {"drags": [[b, keys, left] for b, keys, left, _ in DRAGS]})["drags"]
    assert got == [expected for *_, expected in DRAGS]
