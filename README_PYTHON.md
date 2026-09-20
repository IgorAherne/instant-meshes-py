# instant-meshes-brush

[Instant Meshes](https://github.com/wjakob/instant-meshes) as a Python library,
plus a browser UI that brings the desktop app's **brush tools** — the ones the
command line has never exposed — into any Gradio app.

You orbit the model in the browser, draw strokes on its surface to steer the
topology, watch the red field grid update on the mesh as the solver runs, then
extract and download a quad mesh.

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
a Blender user expects it:

| | |
|---|---|
| Middle-drag, or Alt + left-drag | Orbit |
| Right-drag | Pan |
| Wheel | Zoom |
| `F` | Frame the model, keeping the angle |
| `C` / `E` | Orientation comb / edge brush |
| `Esc` | Cancel the stroke being drawn |

**Show** picks one surface or the other — the input you brush on, or the
extracted result — never both, since they occupy the same space. Choosing the
output builds one; drawing a stroke switches back to the input, and marks the
result stale so the next look at it is rebuilt from the field you just changed.

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
`preprocess`, so they can be changed without a rebuild — and without losing the
strokes a rebuild would take with it:

```python
session.set_extraction_options(smooth_iter=4, pure_quad=True)
mesh = session.extract()
```

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
    session_manager.py      one C++ session per browser, with a TTL sweeper
    server.py               FastAPI: /viewer, /imb-assets, /ws/{id}, uploads
    app.py                  the Gradio wrapper -- an iframe and nothing else
    static/
        index.html          the viewport, controls included
        panel.js            the control panel and its hover help
        main.js             session wiring and frame coalescing
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
