"""Gradio wrapper around the viewer.

The viewer is a complete application in its own document: opening a mesh,
brushing, solving, extracting and exporting all happen inside
``static/index.html``.  This module therefore does one thing -- mint a session
and embed ``/viewer?session=<id>`` -- which is exactly what any other Gradio
app has to do to host the viewport:

.. code-block:: python

    import gradio as gr
    from instant_meshes_brush.app import viewport
    from instant_meshes_brush.server import build_app

    with gr.Blocks() as demo:
        gr.Markdown("## My tool")
        viewport(height="80vh")

    app = gr.mount_gradio_app(build_app(), demo, path="/")

Keeping the controls on the viewer's side rather than in Blocks avoids the two
of them disagreeing: the C++ session is the only state, and a host page cannot
show a stale copy of it.
"""

from __future__ import annotations

import html
from typing import Optional

import gradio as gr

from .session_manager import SessionRegistry, default_registry

#: Sized so the panel's sections are reachable without scrolling the host page.
DEFAULT_HEIGHT = "82vh"
MIN_HEIGHT_PX = 520

#: Narrower than the page it sits on, and centred.  A viewport as wide as a
#: modern monitor puts the model a long way from the controls and leaves the
#: 3D view an extreme letterbox; 70% keeps both in one glance.
DEFAULT_WIDTH = "70%"
MIN_WIDTH_PX = 640

_FRAME_STYLE = (
    "border:1px solid #3a3a40;border-radius:6px;"
    "display:block;margin:0 auto;background:#2d2d30"
)


def _box_style(width: str, height: str) -> str:
    return (
        f"width:{html.escape(width, quote=True)};"
        f"max-width:100%;min-width:min(100%,{MIN_WIDTH_PX}px);"
        f"height:{html.escape(height, quote=True)};"
        f"min-height:{MIN_HEIGHT_PX}px;{_FRAME_STYLE}"
    )


def _iframe(session_id: str, width: str, height: str) -> str:
    """Markup for one viewport bound to ``session_id``."""
    safe = html.escape(session_id, quote=True)
    return (
        f'<iframe src="/viewer?session={safe}" title="Instant Meshes viewport" '
        f'allow="fullscreen" style="{_box_style(width, height)}"></iframe>'
    )


def _placeholder(message: str, width: str = DEFAULT_WIDTH, height: str = DEFAULT_HEIGHT) -> str:
    return (
        f'<div style="{_box_style(width, height)};'
        f'display:flex;align-items:center;justify-content:center;'
        f'color:#8b8b94;font:14px system-ui">'
        f"{html.escape(message)}</div>"
    )


def viewport(
    height: str = DEFAULT_HEIGHT,
    registry: Optional[SessionRegistry] = None,
    width: str = DEFAULT_WIDTH,
) -> gr.HTML:
    """Embed a viewport in the Blocks context that is currently open.

    A session is created when the page loads rather than when the app is built,
    so every browser tab gets its own mesh, solver and stroke set.
    """
    holder = gr.HTML(_placeholder("Starting a session...", width, height))

    async def start() -> str:
        sessions = registry if registry is not None else default_registry()
        try:
            session = await sessions.create()
        except Exception as exc:  # a full registry must not blank the page
            return _placeholder(f"Could not start a session: {exc}", width, height)
        return _iframe(session.id, width, height)

    # Gradio calls this per browser session, which is what mints one id per tab.
    holder.attach_load_event(start, None)
    return holder


#: Gradio 6 moved the Blocks ``css`` argument to ``launch()``, which a mounted
#: app never calls, so the page styling rides along with the markup instead.
#: A <style> element is inert-free: only <script> is dropped by innerHTML.
_FULL_BLEED_CSS = """
<style>
  .gradio-container { max-width: 100% !important; padding: 8px !important; }
  footer { display: none !important; }
</style>
"""


def build_blocks(registry: Optional[SessionRegistry] = None) -> gr.Blocks:
    """The standalone application: a full-bleed viewport and nothing else."""
    with gr.Blocks(title="Instant Meshes - brush guided retopology") as demo:
        gr.HTML(_FULL_BLEED_CSS)
        viewport(registry=registry)
    return demo
