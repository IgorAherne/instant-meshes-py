"""Run the viewer's browser checks headless and report them.

Dev-only, never shipped and not collected by pytest: it needs a local Chrome or
Microsoft Edge. It serves the repository root on a free loopback port, opens
``tests/web/render_checks.html?report`` in a throwaway headless profile, and
waits for the page to post its verdict back.

    python tests/web/run_checks.py [--browser PATH] [--keep-serving]

Exit status 0 when every check passes, 1 when any fails, 2 when the checks
could not be run at all.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAGE = "/tests/web/render_checks.html"
RESULT = "/tests/web/result"
#: A software (SwiftShader) WebGL context compiles every variant in a few
#: seconds; a GPU takes well under one.
TIMEOUT_S = 120

CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


class _Handler(http.server.SimpleHTTPRequestHandler):
    """The repository's files, plus the one route the page reports to."""

    #: A module script is refused under any other MIME type, and the platform
    #: table (the Windows registry, for one) does not always say this.
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript"}

    def __init__(self, *args: object, verdict: dict, arrived: threading.Event,
                 **kwargs: object) -> None:
        self._verdict = verdict
        self._arrived = arrived
        super().__init__(*args, **kwargs)

    def do_POST(self) -> None:
        if self.path != RESULT:
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        self._verdict.update(json.loads(body))
        self.send_response(204)
        self.end_headers()
        self._arrived.set()

    def log_message(self, *args: object) -> None:
        pass


def _find_browser(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for name in ("chrome", "msedge", "chromium", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    return next((path for path in CANDIDATES if os.path.isfile(path)), None)


def _serve(verdict: dict, arrived: threading.Event) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(_Handler, directory=str(REPO), verdict=verdict,
                                arrived=arrived)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _run(browser: str, url: str, arrived: threading.Event) -> None:
    """Open the page headless and wait for its verdict to arrive."""
    # A profile of its own, so a Chrome the user already has open is neither
    # reused nor touched; it is thrown away afterwards.
    with tempfile.TemporaryDirectory(prefix="imb-render-checks-",
                                     ignore_cleanup_errors=True) as profile:
        process = subprocess.Popen(
            [browser, "--headless=new", f"--user-data-dir={profile}", "--no-first-run",
             "--no-default-browser-check", "--enable-unsafe-swiftshader", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            if not arrived.wait(TIMEOUT_S):
                raise RuntimeError(f"the page reported nothing within {TIMEOUT_S} s")
        finally:
            process.kill()
            process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--browser", help="Chrome or Edge executable (default: search)")
    parser.add_argument("--keep-serving", action="store_true",
                        help="print the page URL and serve it until Ctrl+C, for a manual look")
    args = parser.parse_args()

    verdict: dict = {}
    arrived = threading.Event()
    server = _serve(verdict, arrived)
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        if args.keep_serving:
            print(f"Open {origin}{PAGE}  (Ctrl+C to stop)")
            threading.Event().wait()

        browser = _find_browser(args.browser)
        if not browser:
            print("No Chrome or Edge found; pass --browser, or run with --keep-serving "
                  "and open the page yourself.", file=sys.stderr)
            return 2
        try:
            _run(browser, f"{origin}{PAGE}?report", arrived)
        except (OSError, RuntimeError) as exc:
            print(f"Could not run the checks in {browser}: {exc}", file=sys.stderr)
            return 2
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()

    for check in verdict["checks"]:
        mark = "pass" if check["pass"] else "FAIL"
        print(f"{mark:4}  {check['name']}: {check['measured']}  (expected {check['expected']})")
    print("PASS" if verdict["pass"] else "FAIL")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
