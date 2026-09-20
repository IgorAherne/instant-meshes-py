"""Console entry point: serve the brushing UI and its viewer on one port.

The FastAPI application from :mod:`.server` carries the viewer document, the
export download and the binary WebSocket; the Gradio side panel is mounted
underneath it at ``/``.  The order matters: ``mount_gradio_app`` appends a plain
Starlette ``Mount`` that also matches WebSocket scopes, and Starlette dispatches
the first full match in registration order -- so mounting at ``/`` before the
custom routes exist would swallow the viewer's upgrade request.

Both halves default to the same process-wide session registry, which is what
makes a button in the panel visible to the viewer over its own socket.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
import webbrowser

import gradio

from . import set_thread_count
from .app import build_blocks
from .server import build_app, run

#: Wildcard binds are not reachable addresses; advertise the loopback instead.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "[::]"})

_OPEN_POLL_SECONDS = 0.1
_OPEN_TIMEOUT_SECONDS = 30.0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="instant-meshes-brush",
        description="Instant Meshes retopology with brush-stroke guidance in the browser.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="interface to bind (default: %(default)s)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="port to bind (default: %(default)s)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="expose the UI through Gradio's public tunnel",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open the UI once the server is accepting connections",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=-1,
        help="solver worker threads; -1 uses every core (default: %(default)s)",
    )
    return parser.parse_args(argv)


def _reachable_host(host: str) -> str:
    return "127.0.0.1" if host in _WILDCARD_HOSTS else host


def _open_when_ready(host: str, port: int) -> None:
    """Open a browser from a side thread once the port starts accepting.

    uvicorn owns the main thread from here on, so the wait cannot be inline; and
    opening before the port is up would land the user on a connection error.
    """
    target = _reachable_host(host)

    def wait() -> None:
        deadline = time.monotonic() + _OPEN_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            with socket.socket() as probe:
                probe.settimeout(_OPEN_POLL_SECONDS)
                if probe.connect_ex((target, port)) == 0:
                    webbrowser.open(f"http://{target}:{port}")
                    return
            time.sleep(_OPEN_POLL_SECONDS)

    threading.Thread(target=wait, name="imb-open-browser", daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``instant-meshes-brush`` console script."""
    args = _parse_args(argv)

    # build_app already quiets the core; only the thread count is ours to set.
    set_thread_count(args.threads)

    app = gradio.mount_gradio_app(
        build_app(),
        build_blocks(),
        path="/",
        # True would spawn a Node subprocess for server-side rendering.
        ssr_mode=False,
    )

    print(f"Instant Meshes brush UI on http://{_reachable_host(args.host)}:{args.port}")
    if args.open_browser:
        _open_when_ready(args.host, args.port)

    # server.run owns the uvicorn config -- WebSocket frame size, deflate and
    # the share tunnel -- so the two entry points cannot drift apart.
    run(args.host, args.port, share=args.share, app=app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
