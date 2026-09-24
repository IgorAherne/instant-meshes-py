# instant-meshes-brush

[Instant Meshes](https://github.com/wjakob/instant-meshes) as a Python library,
plus a browser UI that brings the desktop app's **brush tools** — the ones the
command line has never exposed — into any Gradio app.

You orbit the model in the browser, draw strokes on its surface to steer the
topology, watch the red field grid update on the mesh as the solver runs, then
download a quad mesh with a UV atlas already on it.

---

## What is actually different here

The upstream project is a single OpenGL/NanoGUI executable. Its `-b` batch mode
can remesh a file, but it cannot accept brush strokes, because strokes are
screen-space input handled by the GUI. This fork separates the two:

| | upstream | here |
|---|---|---|
| Brush strokes | GUI only | `Session.project_stroke` / `add_stroke` |
| Dependencies | NanoGUI, GLFW, OpenGL, Intel TBB | none beyond a C++17 compiler |
| Parallelism | Intel TBB submodule | header-only `std::thread` shim |
| Distribution | build the app yourself | `pip install`, self-contained wheels |
| Field preview | native OpenGL geometry shader | ported to WebGL2 in the browser |
| Texture space | none | xatlas atlas on the quads, written into the OBJ |
| Input formats | OBJ, PLY, ALN | those plus STL, OFF, glTF/glB, DAE and FBX |
| Imported materials | ignored | each map's role guessed, shown one by one or lit together |

The algorithm itself is untouched: same hierarchy, same field optimiser, same
extraction, same results.

---

## Install

Pre-built wheels need no compiler:

```bash
pip install instant-meshes-brush[app]
```

From source you need a C++17 compiler (MSVC 2019+, GCC 9+, or Clang 10+).
CMake and Ninja are pulled in automatically by the build backend.

Run these from the directory that holds `pyproject.toml` — the `pip install .`
error "does not appear to be a Python project" means you are somewhere else.

Windows (cmd):

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[app]"
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app]"
```

Activating first means the environment name never has to be repeated, so a
`venv` / `.venv` mix-up cannot silently install into the wrong interpreter.

If you cloned without `--recursive`, fetch the three header-only submodules the
core needs — NanoGUI is *not* one of them:

```bash
git submodule update --init ext/dset ext/pss ext/pcg32
```

Eigen is resolved automatically: an explicit `EIGEN3_INCLUDE_DIR`, then a system
install, then `ext/eigen`, and finally a download of Eigen 3.4.0.

---

## The browser app

```bash
instant-meshes-brush
```

Open the URL it prints. Everything is in the panel down the left of the
viewport: import a mesh, pick a brush, extract and download. Hovering any
control explains what it does.

The whole workflow is **Import mesh -> brush -> Export**. There is no Apply
step, no Solve step and no Extract step, because there is nothing to decide
about any of them: importing solves, retargeting the resolution rebuilds and
re-solves, every stroke re-solves with the stroke in place, deleting one
re-solves without it, and asking to see or export the output extracts it from
whatever the field is by then. The result is a consequence of what you did,
not a sequence to remember.

| Tool | What the stroke does |
|---|---|
| Orientation comb (`C`) | Turns the cross field to follow the stroke |
| Edge brush (`E`) | Puts an output mesh edge along the stroke |
| Orientation attractor | Drags an orientation singularity along the stroke |
| Position attractor | Drags a position singularity along the stroke |

An attractor stroke starts on the singularity it drags. Selecting one of those
brushes turns the markers on and shows **only** the field it can move — the two
kinds look identical, and half of them would otherwise be dots your brush
cannot touch. Aim is forgiving: a singularity is one triangle under a marker
many times its size, so a start near one is treated as a start on it.

The comb and the edge brush work anywhere on the surface, and a stroke may
begin and end off the model: the rays that miss are dropped, and the ends are
walked a little way inside the silhouette so they land on surface you were
actually looking at rather than on a sliver facing away from you.

Clicking a stroke's handle deletes it and re-solves, as does `Ctrl+Z` — which
only listens while the focus is inside the viewport, so a host page keeps its
own undo.

Re-targeting the resolution keeps the strokes: they are paths in space, and the
rebuild only refines the surface they were drawn on. It does throw away the
solved field, so an attractor's effect does not survive a re-target the way a
brush stroke does.

The left button always belongs to the selected brush, so navigation lives where
a Blender, 3ds Max, Maya or Unity user expects it:

| | |
|---|---|
| Middle-drag, Alt + middle-drag, or Alt + left-drag | Orbit |
| Right-drag, or Shift + middle-drag | Pan |
| Wheel (toward the cursor), Ctrl + middle-drag, or Alt + right-drag | Zoom |
| `F` | Frame the model, keeping the angle |
| `C` / `E` | Orientation comb / edge brush |
| `1` / `2` / `3` | Input / Result / Result UV |
| `Esc` | Cancel the stroke being drawn |

The modifiers are read when the button goes down, so letting go of Alt halfway
through an orbit does not turn the rest of it into a stroke. The mouse part of
this table lives in `static/navigation.js`, free of three.js, for a host that
wants its own model viewer to feel the same in the hand.

The three buttons under the resolution pick the view: **Input** is the surface
you brush on, **Result** the extracted quad mesh, **Result UV** that mesh
flattened into its texture space. Never two at once — the first two occupy the
same space. Choosing either result builds one; drawing a stroke switches back
to the input and marks both stale, so the next look at them is rebuilt from the
field you just changed.

Choosing a view is a standing request rather than a command that either lands
or is lost. Only one result is ever being built at a time, and nothing is built
for a mesh that is about to be rebuilt — so a burst of `1`/`2`/`3`, or a view
key pressed in the middle of a re-target, costs one build rather than one per
press, and the view you asked for appears as soon as the session is free.
Asking for a result while the field is still moving waits for the solve instead
of cutting it short.

## Imported materials

| Format | Geometry | UVs | Texture maps |
|---|---|---|---|
| OBJ, PLY, STL, OFF | yes | where the file has them | from the `.mtl` (every map key and option) |
| glTF, glB | yes | yes | every `*Texture`, `KHR_materials_*` included, embedded or beside the file |
| DAE, FBX | needs `assimp-py` | yes | embedded ones, and ones found inside the model's folder |

Upload the model together with its textures (pick several files, drop a folder,
or one `.zip`); only files inside the model's own folder are ever read.

Every map is given a **role** — base colour, normal, height, roughness, gloss,
metallic, specular, AO, emissive, opacity, or a packed ORM / metal-smoothness /
HDRP mask — by `texroles.classify`: points from the slot it is bound to, the
exporter (Blender's ShininessExponent is roughness, 3ds Max's is gloss), words in
its file name, and pixel statistics that can overrule them (a unit-vector image
is a normal map; a curl test on it tells OpenGL green from DirectX). The R of a
glTF metallicRoughness map is never read as AO unless the file says so.
`materials.py` then builds each material's **canonical maps**: base colour with
opacity in alpha (sRGB), an OpenGL tangent-space normal map, ORM (R = AO,
G = roughness, B = metallic; gloss and smoothness inverted), emissive, 16-bit
height, and whatever has no canonical place as it is.

The row under the model's name reads **None · All · Base colour · Opacity ·
Normal · Roughness · Metallic · AO · Emissive · Height**, then any other maps
by role, with only the buttons the file really has. A map button draws that map
unlit, exactly as stored (a 128 grey reads 128). **All** is the model lit as a
game engine would show it: one physically based material per material, a
studio environment and a light at the camera. **None** goes back to the shaded
surface the field is drawn on, and is pressed to begin with. Hovering a button
names the files behind it, why each was taken for that role and how sure the
guess is; a "?" marks one under 60 %. Normal has a "flip G" toggle and
Roughness an "invert" one, for a wrong guess.

The model you see there is the file as authored, not the mesh the remesher
works on: the maps are pinned to UVs that only exist while the corners a seam
split apart stay apart, and the remesher needs those welded shut. Both come out
of one read. A material with no map of the kind being shown goes flat grey
rather than keeping the one before it.

Nothing is transferred onto the output mesh here; this is for looking at what
you imported. (Spellcast3D bakes the maps onto the output mesh with its own
baker, which reads models through the same modules.)

An FBX or DAE needs `assimp-py`, which ships with the `app` extra. An FBX's
embedded maps are read out of the file directly, because Assimp's Python
binding reports an embedded map as `*0` with no way to ask what `*0` holds; the
FBX is scaled to metres by its own unit setting.

**UV chunks** cuts the atlas, and is the only xatlas control there is. The
slider is how far one chunk may stretch before xatlas gives up on it and starts
another: the right-hand end is fewer, larger chunks with more distortion, the
left-hand end more, smaller ones that each stay closer to their true shape, and
the number beside the label is what came out. Touching it switches to **Result
UV**, because that picture is the only form in which the setting means
anything; while the unwrapper runs, the label carries its progress. Exporting
an OBJ carries the layout with it whether or not you ever looked at it; a PLY
does not, having no per-corner form for one.

The four toggles in **Output mesh** — **Smooth Flow**, **Follow Borders**,
**Sharp Creases**, **Force Quads** — all change how the field is solved, so
each one rebuilds and re-solves. Force Quads subdivides the result into quads
only, which splits every quad into four; the field is aimed at a quarter of the
target vertex count to compensate, so the number you ask for is still roughly
the number you get. An exported OBJ still writes the few per cent of quads a UV
seam runs through as their two triangles, which is the only way they can carry
one.

Every control lives **inside** the viewport, so embedding it in your own Gradio
app is one call and you reproduce none of the UI:

```python
import gradio as gr
from instant_meshes_brush.app import viewport
from instant_meshes_brush.server import build_app

with gr.Blocks() as demo:
    gr.Markdown("## My tool")
    viewport(width="70%", height="80vh")     # mints a session per browser tab

app = gr.mount_gradio_app(build_app(), demo, path="/")   # uvicorn app:app
```

`build_app()` serves `/viewer`, `/imb-assets/...`, the mesh upload route and the
WebSocket. It deliberately avoids `/static`, which is Gradio's: routes
registered on the FastAPI app win over the Gradio mount, so taking that prefix
would 404 your page's own fonts.

`viewport()` creates a session and drops in the iframe. Both share one
`SessionRegistry`, so if you do want your own buttons, fetch the session with
`default_registry().get(session_id)` and call the `BrushSession` coroutines —
the viewport picks the change up over its own socket, including a mesh you load
from outside it.

### Viewer options

Any other page can embed the viewer the same way — an `<iframe>` on
`/viewer?session=<id>` — and shape it with query parameters. All are optional;
without them the viewer is exactly the one above.

| Parameter | Values | What it does |
|---|---|---|
| `panel` | `left` (default), `right` | The side the control panel docks on. On the right, the readout (the brush's name and the status line) moves to the viewport's bottom-right corner with it. |
| `export_label` | any short text | The Export button's text, e.g. `Accept`. One line, 32 characters at most. |
| `export_action` | `download` (default), `host` | `download`: Export writes the mesh and the browser downloads it. `host`: the button writes and downloads nothing; it asks the embedding page instead (below). |
| `host_origin` | an origin, e.g. `http://127.0.0.1:7770` | Required by `export_action=host`: the embedding page's origin. Nothing but a scheme, a host and a port. |

Build the URL in Python rather than by hand; it URL-encodes everything, leaves
the defaults out, and raises `ValueError` for anything the viewer would ignore:

```python
from instant_meshes_brush.server import viewer_url

viewer_url("abc123")
# '/viewer?session=abc123'
viewer_url("abc123", panel="right", export_label="Accept",
           export_action="host", host_origin="http://127.0.0.1:7770")
# '/viewer?session=abc123&panel=right&export_label=Accept&export_action=host&host_origin=http%3A%2F%2F127.0.0.1%3A7770'
```

The viewer applies the same rules on its side: an unknown value, host mode
without a valid `host_origin`, or host mode in a page that is not inside a
frame falls back to the default, and says so as a warning in the browser's
console.

**Host mode.** With `export_action=host` the button is the page's to answer.
Messages go through `window.postMessage`, in both directions pinned to
`host_origin`: the viewer posts only to that origin (the browser drops the
message if the parent page is anything else), and it takes a message only when
it comes from its parent window *and* from that origin, so neither another
frame on the page nor a stranger can speak for the host.

Pressing the button posts to the parent:

```js
{
  source: 'instant-meshes',
  type: 'export-request',
  session: '<session id>',
  settings: {
    target_vertices: 3000,           // Target vertex count
    pure_quad: false,                // Force Quads
    symmetry: { rosy: 4, posy: 4 },  // the Output mesh choice, as Config's rosy / posy
    smoothing: 2,                    // Smoothing (Config's smooth_iter)
    extrinsic: true,                 // Smooth Flow
    boundaries: false,               // Follow Borders
    creases: false,                  // Sharp Creases
    crease_angle: -1,                // what Sharp Creases means to Config: 30, or -1 for off
    format: 'obj',                   // the format picker: 'obj' or 'ply'
    uv_leniency: 0.5,                // UV chunks, 0..1
  },
  strokes: 0,                        // guide strokes on the model
}
```

Nothing is extracted, written or downloaded by the viewer: the page reads the
mesh through its own server, which shares the session (for example with
`BrushSession.extract()` and `export_mesh()`). It then answers:

```js
viewerFrame.contentWindow.postMessage(
  { source: 'my-app', type: 'export-state', state: 'busy' },     // 'busy' | 'done' | 'error'
  viewerOrigin,
);
```

| `state` | The viewer shows |
|---|---|
| `busy` | A spinner and "Accept…" on the button, which cannot be pressed again until the next state. |
| `done` | The button back to normal, and a confirmation on the status line for a few seconds: the `message` if one was sent, otherwise "Accept: done". |
| `error` | The button back to normal, and the `message` (or "Accept failed") on the status line as an error. |

`message` is optional plain text; `source` names the sender and is not
checked. A second press within two seconds of the first, before any answer, is
ignored, so a double click asks once.

---

## The Python API

The core is usable on its own, with numpy as the only dependency.

```python
import numpy as np
import instant_meshes_brush as imb

imb.set_verbose(False)

session = imb.Session()
session.set_mesh(vertices, faces)          # (nV, 3) float32, (nF, 3) uint32
session.preprocess(imb.Config(vertex_count=5000, rosy=4, posy=4))
session.solve_all()

mesh = session.extract()
session.write_mesh("out.obj", mesh)
```

### Brushing without a browser

Strokes are given as rays, so the caller owns the camera and there is no
projection convention to agree on. Each ray is intersected with the mesh and the
hit points are smoothed into a surface curve:

```python
curve = session.project_stroke(ray_origins, ray_directions)   # (N, 3) each
if curve is not None:                       # None means no ray hit anything
    session.add_stroke(imb.StrokeKind.ORIENTATION, curve)
    session.solve_orientations(-1); session.wait_solve()
    session.solve_positions(-1);    session.wait_solve()
```

Rays that miss are dropped, and of what is left the longest uninterrupted run
is kept — a sweep that leaves the model and returns is two strokes, not one
with a shortcut through the gap. The ends are then walked inside the
silhouette, past the samples that struck the surface edge-on, so a stroke drawn
across the outline is anchored on surface the camera can see. Give the rays at
a spacing that suits the model, not the pointer: the browser resamples a drag
every few pixels before sending it, because a ray every few hundred pixels can
pass either side of a model without ever hitting it.

For a singularity attractor, project with `attractor=True` so the curve is
routed along edge-adjacent faces, and start it at the singularity you want to
drag — near it is enough, since `apply_attractor` re-roots the stroke on the
nearest one within a couple of edge lengths:

```python
curve = session.project_stroke(origins, directions, True)
session.apply_attractor(curve, orientation=True)
session.wait_solve()
```

The move is a local edit of a solved field, so re-solving from scratch — which
is what re-targeting the resolution does — undoes it.

### Solving live

`solve_orientations(-1)` runs the full coarse-to-fine schedule and stops by
itself. `solve_orientations(0)` refines at full resolution until `stop_solve()`,
which is much cheaper and is what the browser uses for feedback while brushing.
Both honour brush strokes.

While a solve runs, poll for updates:

```python
session.set_preview_interval(120)           # publish every 120 ms
status = session.status
if status.iterations_q != last_q:           # version counters
    Q = session.orientation_field           # (nV, 3) float32
```

`status.error` carries any failure from the solver thread; reading it clears it.

### Exporting

`write_mesh` picks the format from the extension. OBJ is the safer default: a
quad-dominant result contains both quads and triangles, and a binary PLY with
mixed face degrees is legal but not universally readable (trimesh, for one,
rejects any variable-length binary PLY). `Config(pure_quad=True)` subdivides
away the triangles and makes the PLY uniform, and therefore portable.

`pure_quad` and `smooth_iter` are the only settings `extract` reads rather than
`preprocess`, so they can be changed without a rebuild:

```python
session.set_extraction_options(smooth_iter=4, pure_quad=True)
mesh = session.extract()
```

That is an escape hatch rather than the ordinary route for `pure_quad`. The
pure-quad step splits every extracted quad into four, so `preprocess` aims the
field at a quarter of `vertex_count` when the flag is set — a request for 2,500
vertices returns about 2,500 rather than ten thousand, and `session.config`
hands the request back unchanged. Setting the flag afterwards leaves that aim
where it was, and the output comes out four times over.

### Texture coordinates

`instant_meshes_brush.uv` wraps xatlas (`pip install xatlas`, or it comes with
the `[app]` extra) and hands the result back **on the quads**:

```python
from instant_meshes_brush import uv

layout = uv.unwrap(mesh.vertices, mesh.faces, leniency=0.5, progress=print)
uv.write_obj("out.obj", mesh.vertices, mesh.faces, layout)
print(layout.chart_count, "chunks,", layout.unmapped, "faces without UVs")
```

`leniency` maps onto xatlas's `ChartOptions.max_cost`, geometrically, with 0.5
landing on the library's own default of 2.0.

`layout.faces` has the shape of the mesh's own face array but indexes
`layout.uv`: a vertex on a seam is in the atlas twice, so the two index spaces
cannot be the same one. xatlas charts *triangles*, so a chart boundary can run
down a quad's diagonal — a face with half of itself in each chunk has no single
place in the atlas, and is reported unmapped rather than given one of the two
answers.

`write_obj` exists because the C++ writer emits `vt` lines that none of its
faces reference; this one writes `f v/vt` per corner, and leaves a face without
texture indices rather than pointing it at the origin.

`progress` is called with a fraction in [0, 1]. xatlas's Python binding takes
no progress callback, so its own call is one opaque block — the number holds
there and then jumps; the rest of the pipeline reports honestly.

The unwrap runs off the session's worker thread, because it needs nothing from
the C++ core and can take tens of seconds; while it held that thread no status
could be read from the session at all, and a viewport watching one had no way
to learn that anything else had finished.

### Reading a model with its materials

`instant_meshes_brush.assets` reads a file twice over, once for each consumer:

```python
from instant_meshes_brush import assets

source = assets.load_source("character.glb")
vertices, faces = assets.solver_mesh(source)     # welded, for the remesher

print([slot.key for slot in source.slots])       # ['basecolor', 'normal', 'orm']
print([button.id for button in source.buttons])  # ['basecolor', 'normal', 'roughness', 'metallic']
print([m.name for m in source.materials])
skin = source.texture(0, 1)                      # slot 0 of material 1
skin.mime, len(skin.data)                        # ('image/jpeg', 412_338)
full = source.materials[1].canonical()           # full-resolution canonical maps
```

`assets.load_source(path, materials=False)` reads the geometry alone, for a
remesh that never shows the maps.

`source` keeps the file as authored — corners split along every UV seam, faces
sorted by material with `source.groups` naming the runs, and each material's
maps decoded, shrunk to `MAX_TEXTURE_PX` and re-encoded as the bytes a browser
takes directly. `solver_mesh` welds the seams shut and drops the triangles that
collapse; a hierarchy built on split corners has a crack down every seam.

Slots are numbered over the canonical maps *something* in the file has, so a
slot is the normal map on every material that has one rather than "whatever
came second", and a model with only colour and emissive gets two slots.

### Reproducible output

`Config(deterministic=True)` makes a run bit-identical, **including across
machines with different core counts** — the parallel reductions on that path
partition by grain size alone rather than by thread count.

---

## Layout

```
src/session.{h,cpp}         headless driver: load, brush, solve, extract
src/python_bindings.cpp     pybind11 module (_core)
ext/tbb_shim/               header-only std::thread stand-in for Intel TBB
ext/pss_shim/               likewise for the pss parallel sort
python/instant_meshes_brush/
    protocol.py             binary WebSocket framing (paired with protocol.js)
    assets.py               the imported file as authored: materials and maps
    materials.py            glTF / MTL / FBX materials -> canonical maps
    texroles.py texstats.py what each texture map is (base colour, normal, ORM...)
    fbx_media.py            an FBX's materials, texture bindings and embedded images
    uv.py                   xatlas atlas, mapped back onto the quads
    session_manager.py      one C++ session per browser, with a TTL sweeper
    server.py               FastAPI: /viewer, /imb-assets, /ws/{id}, uploads; viewer_url()
    app.py                  the Gradio wrapper -- an iframe and nothing else
    static/
        index.html          the viewport, controls included
        panel.js            the control panel and its hover help
        main.js             session wiring, frame coalescing, host mode
        options.js          the viewer options a host page puts in the URL
        navigation.js       which camera move each drag makes
        renderer.js         three.js scene: surface, strokes, output mesh
        field_material.js   the field grid, ported to WebGL2
```

The original desktop application still builds:

```bash
git submodule update --init --recursive ext/nanogui
cmake -B build -DINSTANT_MESHES_BUILD_GUI=ON -DINSTANT_MESHES_BUILD_PYTHON=OFF
cmake --build build --config Release
```

---

## Licence

BSD, as upstream. See `LICENSE.txt`.
