# Browser checks for the viewer's texture views

The texture row of the viewer (None, All, Base colour, Normal, Roughness...) has
two halves, and each has its own check.

## What the buttons say: `test_panel_maps.py` (pytest, needs node)

Runs the pure helpers of `static/panel.js` under node: button order, which of the
server's guesses each button explains in its hover help, when a weak guess gets
a "?", and which files of a dropped folder are uploaded. It is collected with
the rest of the suite and skips when `node` is not on the PATH.

    python -m pytest tests/web

## What the buttons show: `render_checks.html` (a real browser, WebGL2)

Drives the shipped `static/renderer.js` on a flat square and reads pixels back.
Test images are PNG bytes built by the page itself, so every stored value is
known exactly.

| check | expected |
|---|---|
| inspect: a stored 128 grey | 128 +-1 (an sRGB-tagged map shows 55: the old bug, shown for reference) |
| inspect: channels r, g, b, a of (10, 128, 200, 60) | each as a grey +-1 |
| inspect: flip green, invert | 255 - value, +-1 |
| inspect: the file's top row | at v = 1 (the top of the square) |
| lit (All): OPAQUE / BLEND + double sided / MASK + inverted roughness / object-space + flipped normal | compiles, GL error 0, no shader error |
| lit: normal map tilted to +v, to +u, flipped | +v is lit by the key light above; +u is to the right; flip green equals -v |
| white diffuse under a uniform 0.5 environment | 128 +-3 linear (three's own shading); 180 +-3 through the lit view, which is PBR Neutral + sRGB (187 would be sRGB alone) |

Automated, in a headless Chrome or Edge with a throwaway profile (the page
posts its verdict back to the runner; exit status 0 = pass):

    python tests/web/run_checks.py [--browser PATH]

By hand: `python tests/web/run_checks.py --keep-serving` prints a URL; open it
and read the table.

## By hand, in the viewer (needs a model with maps)

1. Import a model with its textures: select the model and its maps together,
   or pick a zip, or drop the model's folder anywhere on the viewer. An OBJ
   needs its `.mtl`, a `.gltf` its `.bin`.
2. The row under the model name shows None, All, then one button per kind of
   map, named: Pixal3D `output.glb` shows exactly None, All, Base colour,
   Roughness, Metallic (no AO: its red channel is not occlusion).
3. Hover a map button: the file it came from, how sure the guess is and why;
   for a normal map, whether green was read as OpenGL or DirectX. A "?" marks
   a guess under 60%.
4. All renders the model lit and tone mapped. A normal map's bumps must look
   raised under the light at the upper left; if they look pushed in, press
   "flip G". A gloss map read as roughness makes shiny parts matt; press
   "invert". Both only change the preview.
5. None, Result (2) and Result UV (3) look exactly as before textures existed.
