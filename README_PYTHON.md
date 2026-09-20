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

Open the URL it prints. Load a mesh, pick a tool, and draw on the model.

| Tool | What the stroke does |
|---|---|
| Orientation comb | Turns the cross field to follow the stroke |
| Edge brush | Puts an output mesh edge along the stroke |
| Orientation attractor | Drags an orientation singularity along the stroke |
| Position attractor | Drags a position singularity along the stroke |

Attractor strokes must *start* on an existing singularity — that is the one
being dragged. The other two tools work anywhere on the surface.

Clicking a stroke's handle deletes it and re-solves.

To embed the viewport in your own Gradio app, mount the FastAPI application and
drop the viewer in an iframe:

```python
import gradio as gr
from instant_meshes_brush.server import build_app
from instant_meshes_brush.session_manager import default_registry

api = build_app()          # FastAPI: /viewer, /static/..., /ws/{session_id}

with gr.Blocks() as demo:
    frame = gr.HTML()

    async def open_session():
        session = await default_registry().create()
        return (f'<iframe src="/viewer?session={session.id}" '
                'style="width:100%;height:80vh;border:0"></iframe>')

    demo.load(open_session, outputs=frame)

app = gr.mount_gradio_app(api, demo, path="/")   # uvicorn app --port 7860
```

`build_app()` and the Gradio blocks share one `SessionRegistry`, so your own
buttons can drive the same session the canvas is showing — call the registry's
`get(session_id)` and use the `BrushSession` coroutines directly.

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
if curve is not None:                       # None means a ray missed
    session.add_stroke(imb.StrokeKind.ORIENTATION, curve)
    session.solve_orientations(-1); session.wait_solve()
    session.solve_positions(-1);    session.wait_solve()
```

For a singularity attractor, project with `attractor=True` so the curve is
routed along edge-adjacent faces, and start it on a singular face:

```python
face = next(iter(session.orientation_singularities))
curve = session.project_stroke(origins, directions, True)
session.apply_attractor(curve, orientation=True)
```

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
    server.py               FastAPI: /viewer, /static, /ws/{session}
    app.py                  the Gradio control panel
    static/                 WebGL2 viewer, ported field shaders, three.js
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
