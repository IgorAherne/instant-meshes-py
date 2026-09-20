/*
    tools.js: the brush tools that turn a drag into a STROKE frame.

    A drag is sampled in screen space, thinned to roughly one sample per
    MIN_SAMPLE_PX of travel and only then unprojected, which keeps a stroke a
    few dozen rays instead of a few thousand -- the server smooths the curve
    along the surface anyway, so denser sampling buys nothing.

    Rays leave in mesh space because Session::projectStroke expects them there;
    Viewer.screenRay is what guarantees that.
*/

import { MessageType } from './protocol.js';

/** Minimum travel, in CSS pixels, between two samples of a stroke. */
const MIN_SAMPLE_PX = 6;

/** Ceiling on the rays one stroke sends, whatever distance it covers. */
const MAX_STROKE_RAYS = 512;

/** A press that never travels further than this is a click, not a stroke. */
const CLICK_SLOP_PX = 4;

/** How near a click has to land on a projected handle to delete its stroke. */
const HANDLE_HIT_PX = 14;

/** Search radius handed to Session::eraseStrokeNear, in average edge lengths. */
const ERASE_RADIUS_EDGES = 2;

/* The camera keys, spelled the way a Blender user expects them.  The left
   button belongs to the brush at all times -- that is the whole point of the
   viewport -- so navigation lives on the middle button and on Alt. */
const NAVIGATION_HINT =
    'Middle-drag or Alt-drag orbits, right-drag pans, wheel zooms, F frames the model.';

/**
 * The four brushes.  `kind` is what goes into the STROKE header: the two
 * persistent brushes use the numeric StrokeKind the C++ enum defines, the
 * attractors name themselves instead because they leave no stroke behind.
 *
 * There is no "orbit" tool: a brush is always armed, and the camera is reached
 * through the modifiers above rather than by putting the brush down.
 */
export const TOOLS = [
    {
        id: 'comb',
        label: 'Orientation Comb',
        key: 'c',
        kind: 0,
        hint: 'Drag across the surface to comb the orientation field.',
    },
    {
        id: 'edge',
        label: 'Edge Brush',
        key: 'e',
        kind: 1,
        hint: 'Drag to pin an edge path of the output mesh onto the surface.',
    },
    {
        id: 'orient-attractor',
        label: 'Orientation Attractor',
        kind: 'attractor_orientation',
        /* Only this field's markers are draggable by this brush, so only this
           field's markers are shown while it is selected. */
        singularities: 'orientation',
        hint: 'Drag from one of the dots to pull that singularity along the stroke.',
    },
    {
        id: 'pos-attractor',
        label: 'Position Attractor',
        kind: 'attractor_position',
        singularities: 'position',
        hint: 'Drag from one of the dots to pull that singularity along the stroke.',
    },
];

/** The first tool, which is what left-drag does until another is picked. */
export const DEFAULT_TOOL = TOOLS[0].id;

/**
 * Controls a keystroke belongs to rather than to the viewport.
 *
 * A slider is not one of them, and neither is a checkbox: 1, 2, 3, C, E and F
 * do nothing to either.  Counting every `input` as text entry is what made
 * clicking the chart slider turn the whole keyboard off until you clicked
 * somewhere else.
 */
const TEXT_ENTRY = 'textarea, select, input:not([type="range"]):not([type="checkbox"])';

/** Number keys for the three views, in the order the panel lists them. */
export const SURFACE_KEYS = Object.freeze({
    1: 'mesh',
    2: 'output',
    3: 'uv',
});

/**
 * Walk a drag at a fixed screen-space spacing.
 *
 * The raw samples sit wherever pointermove happened to fire, which depends on
 * how fast the pointer moved and on how hard the browser coalesced events: a
 * quick flick across a model can arrive as three points hundreds of pixels
 * apart, and rays that far apart pass either side of it without ever hitting.
 * Walking the polyline the user actually drew ties the ray density to the
 * stroke rather than to the input device, which is what lets the server find
 * where the stroke crossed the silhouette.
 *
 * @param {Array<{clientX: number, clientY: number}>} samples
 * @returns {Array<{clientX: number, clientY: number}>}
 */
export function densifyStroke(samples) {
    if (samples.length < 2) return samples.slice();

    let length = 0;
    for (let i = 1; i < samples.length; ++i) {
        length += Math.hypot(
            samples[i].clientX - samples[i - 1].clientX,
            samples[i].clientY - samples[i - 1].clientY
        );
    }
    /* A long sweep widens its own spacing rather than sending thousands of
       rays; the server smooths the curve along the surface regardless. */
    const spacing = Math.max(MIN_SAMPLE_PX, length / MAX_STROKE_RAYS);

    const out = [samples[0]];
    for (let i = 1; i < samples.length; ++i) {
        const a = samples[i - 1];
        const b = samples[i];
        const dx = b.clientX - a.clientX;
        const dy = b.clientY - a.clientY;
        const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy) / spacing));
        for (let s = 1; s <= steps; ++s) {
            const t = s / steps;
            out.push({ clientX: a.clientX + dx * t, clientY: a.clientY + dy * t });
        }
    }
    return out;
}

const TOOLS_BY_ID = new Map(TOOLS.map((tool) => [tool.id, tool]));

/** Single-key shortcuts, for the tools that declare one. */
const TOOLS_BY_KEY = new Map(
    TOOLS.filter((tool) => tool.key).map((tool) => [tool.key, tool])
);

export class ToolController {
    /**
     * @param {import('./renderer.js').Viewer} viewer
     * @param {import('./net.js').Connection} connection
     * @param {{onToolChange?: (tool: object) => void,
     *          onNotice?: (message: string) => void}} callbacks
     */
    constructor(
        viewer,
        connection,
        { onToolChange = null, onNotice = null, onUndo = null, onSurface = null } = {}
    ) {
        this.viewer = viewer;
        this.connection = connection;
        this._onToolChange = onToolChange;
        this._onNotice = onNotice;
        this._onUndo = onUndo;
        this._onSurface = onSurface;

        this._tool = TOOLS_BY_ID.get(DEFAULT_TOOL);
        this._pointerId = null;
        this._drawing = false;
        this._samples = [];
        this._rect = null;
        this._press = null;
        this._travel = 0;

        const canvas = viewer.canvas;
        this._listeners = [
            [canvas, 'pointerdown', (event) => this._onPointerDown(event)],
            [canvas, 'pointermove', (event) => this._onPointerMove(event)],
            [canvas, 'pointerup', (event) => this._onPointerUp(event)],
            [canvas, 'pointercancel', () => this.cancelStroke()],
            [window, 'keydown', (event) => this._onKeyDown(event)],
            [window, 'blur', () => this.cancelStroke()],
        ];
        for (const [target, type, handler] of this._listeners) {
            target.addEventListener(type, handler);
        }

        this.setTool(DEFAULT_TOOL);
    }

    get tool() {
        return this._tool;
    }

    /** @param {'comb'|'edge'|'orient-attractor'|'pos-attractor'} id */
    setTool(id) {
        const tool = TOOLS_BY_ID.get(id);
        if (!tool) throw new Error(`unknown tool "${id}"`);

        this.cancelStroke();
        this._tool = tool;
        this.viewer.setControlsEnabled(tool.kind === null);
        if (this._onToolChange) this._onToolChange(tool);
    }

    /** Drop the stroke being drawn without sending anything. */
    cancelStroke() {
        if (this._pointerId !== null) {
            try {
                this.viewer.canvas.releasePointerCapture(this._pointerId);
            } catch {
                /* The pointer is already gone; nothing left to release. */
            }
        }
        this._pointerId = null;
        this._drawing = false;
        this._samples = [];
        this._press = null;
        this._travel = 0;
        this.viewer.setPreviewStroke(null);
    }

    dispose() {
        this.cancelStroke();
        for (const [target, type, handler] of this._listeners) {
            target.removeEventListener(type, handler);
        }
        this._listeners = [];
    }

    /* -------------------------------------------------------------- */
    /*  Pointer handling                                               */
    /* -------------------------------------------------------------- */

    _onPointerDown(event) {
        /* Alt hands the left button to the camera, so a brush must not also
           claim it; the viewer arms OrbitControls for the same event. */
        if (event.button !== 0 || event.altKey || this._pointerId !== null) return;

        this._pointerId = event.pointerId;
        this._rect = this.viewer.canvas.getBoundingClientRect();
        this._press = { clientX: event.clientX, clientY: event.clientY };
        this._travel = 0;
        this._drawing = this._tool.kind !== null;
        this._samples = this._drawing ? [this._sample(event)] : [];

        if (this._drawing) {
            /* Capture keeps a stroke alive when the drag leaves the canvas. It
               is an optimisation, not a precondition: if the pointer is already
               gone the stroke still works over the canvas itself. */
            try {
                this.viewer.canvas.setPointerCapture(event.pointerId);
            } catch {
                /* No active pointer with this id. */
            }
        }
    }

    _onPointerMove(event) {
        if (event.pointerId !== this._pointerId) return;

        this._travel = Math.max(this._travel, this._distanceFromPress(event));
        if (!this._drawing) return;

        const last = this._samples[this._samples.length - 1];
        if (Math.hypot(event.clientX - last.clientX, event.clientY - last.clientY) < MIN_SAMPLE_PX) {
            return;
        }
        this._samples.push(this._sample(event));
        this.viewer.setPreviewStroke(this._samples.slice());
    }

    _onPointerUp(event) {
        if (event.pointerId !== this._pointerId) return;

        const drawing = this._drawing;
        const samples = this._samples;
        const travel = Math.max(this._travel, this._distanceFromPress(event));

        /* The release point closes the stroke, so a drag shorter than one
           sampling step still produces the two samples a curve needs. */
        if (drawing && travel > CLICK_SLOP_PX) samples.push(this._sample(event));

        this.cancelStroke();

        if (travel <= CLICK_SLOP_PX) {
            this._eraseStrokeAt(event.clientX, event.clientY);
        } else if (drawing && samples.length >= 2) {
            this._sendStroke(samples);
        }
    }

    _onKeyDown(event) {
        /* This viewer is an iframe inside somebody else's page, which may well
           have its own undo. Keys are only ours while the focus is in here --
           click away and Ctrl+Z goes back to the host, click in and it returns. */
        if (!document.hasFocus()) return;

        const focused = document.activeElement;
        const typing = Boolean(focused && focused.matches(TEXT_ENTRY));

        if ((event.ctrlKey || event.metaKey) && !event.altKey &&
            event.key.toLowerCase() === 'z') {
            /* Let the text box have its own undo while it is being typed in. */
            if (typing) return;
            event.preventDefault();
            if (this._onUndo) this._onUndo();
            return;
        }

        /* Never while a control has focus: F is a character in the vertex
           count box, and Ctrl-F belongs to the browser. */
        if (event.ctrlKey || event.metaKey || event.altKey || typing) return;

        if (event.key === 'Escape') {
            if (this._pointerId === null) return;
            this.cancelStroke();
            this._notify('Stroke cancelled');
            return;
        }

        if (SURFACE_KEYS[event.key]) {
            event.preventDefault();
            if (this._onSurface) this._onSurface(SURFACE_KEYS[event.key]);
            return;
        }

        const key = event.key.toLowerCase();
        if (key === 'f') {
            if (!this.viewer.frameModel()) return;
            event.preventDefault();
            this._notify('Framed the model');
        } else if (TOOLS_BY_KEY.has(key)) {
            event.preventDefault();
            this.setTool(TOOLS_BY_KEY.get(key).id);
        }
    }

    _sample(event) {
        return {
            clientX: event.clientX,
            clientY: event.clientY,
            x: event.clientX - this._rect.left,
            y: event.clientY - this._rect.top,
        };
    }

    _distanceFromPress(event) {
        return Math.hypot(event.clientX - this._press.clientX, event.clientY - this._press.clientY);
    }

    /* -------------------------------------------------------------- */
    /*  Frames                                                         */
    /* -------------------------------------------------------------- */

    _sendStroke(samples) {
        if (!this.viewer.mesh) {
            this._notify('No mesh loaded yet');
            return;
        }

        const path = densifyStroke(samples);
        const count = path.length;
        const origins = new Float32Array(count * 3);
        const directions = new Float32Array(count * 3);
        for (let i = 0; i < count; ++i) {
            const ray = this.viewer.screenRay(path[i].clientX, path[i].clientY);
            origins.set(ray.origin, i * 3);
            directions.set(ray.direction, i * 3);
        }

        this.connection.send(
            MessageType.STROKE,
            { kind: this._tool.kind, solve: true },
            {
                ray_origins: { data: origins, shape: [count, 3] },
                ray_directions: { data: directions, shape: [count, 3] },
            }
        );
    }

    /**
     * A click on a stroke's handle deletes that stroke.  The server does the
     * actual matching through Session::eraseStrokeNear, so it needs the handle
     * position, the eye it has to be visible from, and a search radius.
     */
    _eraseStrokeAt(clientX, clientY) {
        const handle = this._handleAt(clientX, clientY);
        if (!handle) return;

        const eye = this.viewer.eye();
        this.connection.send(MessageType.ERASE_STROKE, {
            point: [handle.position.x, handle.position.y, handle.position.z],
            eye: [eye[0], eye[1], eye[2]],
            radius: this.viewer.averageEdgeLength * ERASE_RADIUS_EDGES,
        });
    }

    _handleAt(clientX, clientY) {
        const rect = this.viewer.canvas.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;

        let best = null;
        let bestDistance = HANDLE_HIT_PX;
        for (const handle of this.viewer.strokeHandles()) {
            const screen = this.viewer.projectToScreen(handle.position);
            if (!screen) continue;
            const distance = Math.hypot(screen.x - x, screen.y - y);
            if (distance <= bestDistance) {
                bestDistance = distance;
                best = handle;
            }
        }
        return best;
    }

    _notify(message) {
        if (this._onNotice) this._onNotice(message);
    }
}
