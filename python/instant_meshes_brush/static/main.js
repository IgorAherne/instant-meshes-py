/*
    main.js: page wiring.

    Network frames are never applied straight away.  Each one is folded into a
    small pending set and drained once per animation frame, so a burst of FIELD
    updates that outruns the display costs one GPU upload rather than ten, and
    the GPU buffers behind any layer are rebuilt at most once per displayed
    frame.
*/

import { MessageType } from './protocol.js';
import { Connection } from './net.js';
import { Viewer } from './renderer.js';
import { TOOLS, ToolController } from './tools.js';

/** Preview rate asked of the server; the RAF loop coalesces anything faster. */
const SUBSCRIBE_FPS = 15;

const ui = {
    canvas: document.getElementById('view'),
    overlay: document.getElementById('overlay'),
    toolbar: document.getElementById('toolbar'),
    link: document.getElementById('link'),
    linkText: document.getElementById('link-text'),
    toolName: document.getElementById('tool-name'),
    progress: document.getElementById('progress'),
    progressFill: document.getElementById('progress-fill'),
    statusText: document.getElementById('status-text'),
    hint: document.getElementById('hint'),
    solve: document.getElementById('btn-solve'),
    stop: document.getElementById('btn-stop'),
    extract: document.getElementById('btn-extract'),
};

/* ------------------------------------------------------------------ */
/*  Frame decoding                                                     */
/* ------------------------------------------------------------------ */

/** StrokeKind.EDGE, the one kind renderer.js gives a dark centre line. */
const STROKE_KIND_EDGE = 1;

/** Stroke ids start at 1, so 0 marks the transient curve of an attractor. */
const ATTRACTOR_STROKE_ID = 0;

function pickArray(arrays, ...names) {
    for (const name of names) {
        if (arrays[name]) return arrays[name].data;
    }
    return null;
}

/**
 * STROKE_RESULT names its kind ("edge"), STROKE_LIST numbers it with the C++
 * StrokeKind.  The renderer only ever wants the number.
 */
function strokeKind(kind) {
    if (typeof kind === 'number') return kind;
    return kind === 'edge' ? STROKE_KIND_EDGE : 0;
}

/**
 * A STROKE_RESULT carries the curve it projected, so a stroke appears without
 * waiting for anything else.  Attractors report id 0: they nudge the solver and
 * leave no constraint behind, so they must not turn into a permanent ribbon.
 */
function decodeStrokeResult(header, arrays) {
    const id = header.stroke_id ?? header.id;
    if (!(id > ATTRACTOR_STROKE_ID)) return null;

    const positions = pickArray(arrays, 'positions', 'points');
    const normals = pickArray(arrays, 'normals');
    if (!positions || !normals || positions.length === 0) return null;
    return { id, kind: strokeKind(header.kind), positions, normals };
}

/**
 * STROKE_LIST is the authoritative list of surviving stroke ids and carries no
 * curves, which is all an erase needs: it prunes the geometry STROKE_RESULT
 * already delivered.  An id this client has never seen a curve for stays
 * undrawable, so a stroke made before a reload cannot be reconstructed.
 */
function reconcileStrokes(known, entries) {
    const kept = new Map();
    for (const entry of entries) {
        const stroke = known.get(entry.id);
        if (stroke) kept.set(entry.id, { ...stroke, kind: strokeKind(entry.kind) });
    }
    return kept;
}

function concatFloat32(chunks) {
    let total = 0;
    for (const chunk of chunks) total += chunk.length;

    const out = new Float32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
        out.set(chunk, offset);
        offset += chunk.length;
    }
    return out;
}

/**
 * The server reports orientation and position singularities as two separate
 * marker sets; the viewer draws one point cloud, so they are concatenated.
 * Copying also releases the frame buffer the decoded views would otherwise pin.
 */
function decodeSingularities(arrays) {
    const positions = [];
    const colors = [];

    for (const field of ['orientation', 'position']) {
        const p = pickArray(arrays, `${field}_positions`);
        const c = pickArray(arrays, `${field}_colors`);
        if (!p || !c) continue;
        /* Keep only whole markers, which need both a centre and a colour. */
        const paired = Math.min(p.length, c.length);
        const count = paired - (paired % 3);
        if (count === 0) continue;
        positions.push(p.subarray(0, count));
        colors.push(c.subarray(0, count));
    }

    return { positions: concatFloat32(positions), colors: concatFloat32(colors) };
}

function describeStroke(header) {
    if (header.reason) return `Stroke rejected: ${header.reason}`;
    return 'Stroke rejected: no surface under the cursor';
}

/* ------------------------------------------------------------------ */
/*  Application                                                        */
/* ------------------------------------------------------------------ */

class App {
    constructor(viewer, connection) {
        this.viewer = viewer;
        this.connection = connection;

        /* Drained by the render loop; see the module comment. */
        this.pending = {
            geometry: null,
            field: null,
            strokes: false,
            singularities: null,
            extracted: null,
        };
        this.strokes = new Map();

        this.tools = new ToolController(viewer, connection, {
            onToolChange: (tool) => this._showTool(tool),
            onNotice: (message) => this._setStatus(message, false),
        });

        viewer.onFrame = () => this._drain();

        this._bindToolbar();
        this._bindConnection();
    }

    /* -------------------------------------------------------------- */
    /*  Per-frame application of network state                         */
    /* -------------------------------------------------------------- */

    /**
     * Every slot is cleared before it is applied: a frame the viewer rejects
     * has to be dropped, not retried on every frame from here to eternity.
     */
    _drain() {
        const pending = this.pending;

        const geometry = pending.geometry;
        if (geometry) {
            pending.geometry = null;
            this._apply(() => this.viewer.setGeometry(geometry));
        }

        const field = pending.field;
        if (field) {
            pending.field = null;
            this._apply(() => this.viewer.setField(field.q, field.o));
        }

        if (pending.strokes) {
            pending.strokes = false;
            const strokes = [...this.strokes.values()];
            this._apply(() => this.viewer.setStrokes(strokes));
        }

        const singularities = pending.singularities;
        if (singularities) {
            pending.singularities = null;
            this._apply(() =>
                this.viewer.setSingularities(singularities.positions, singularities.colors)
            );
        }

        const extracted = pending.extracted;
        if (extracted) {
            pending.extracted = null;
            this._apply(() => this.viewer.setExtracted(extracted.wireframe, extracted.colors));
        }
    }

    _apply(update) {
        try {
            update();
        } catch (err) {
            this._setStatus(err.message, true);
        }
    }

    /* -------------------------------------------------------------- */
    /*  Message handlers                                               */
    /* -------------------------------------------------------------- */

    _bindConnection() {
        const conn = this.connection;

        conn.on(MessageType.GEOMETRY, (header, arrays) => {
            const positions = pickArray(arrays, 'positions', 'vertices');
            const indices = pickArray(arrays, 'indices', 'faces');
            const normals = pickArray(arrays, 'normals');
            if (!positions || !indices || !normals) {
                this._setStatus('Geometry frame is missing an array', true);
                return;
            }
            if (!(header.scale > 0)) {
                this._setStatus(`Geometry frame has no usable scale (${header.scale})`, true);
                return;
            }
            this.pending.geometry = {
                positions,
                indices,
                normals,
                scale: header.scale,
                /* Config defaults, for a server that only sends non-default symmetries. */
                rosy: header.rosy ?? 4,
                posy: header.posy ?? 4,
            };
            /* Session::preprocess drops every stroke, because the vertex indices
               they were projected onto no longer exist.  Forget them here too,
               so ribbons built for the previous mesh cannot survive it; the
               STROKE_LIST that follows repopulates whatever really remains. */
            this.strokes.clear();
            this.pending.strokes = true;
            this._setStatus(
                `${positions.length / 3} vertices, ${indices.length / 3} faces`,
                false
            );
        });

        conn.on(MessageType.FIELD, (header, arrays) => {
            const q = pickArray(arrays, 'q', 'orientation', 'orientation_field');
            const o = pickArray(arrays, 'o', 'position', 'position_field');
            if (!this.pending.field) this.pending.field = { q: null, o: null };
            if (q) this.pending.field.q = q;
            if (o) this.pending.field.o = o;
        });

        conn.on(MessageType.STROKE_LIST, (header) => {
            const entries = Array.isArray(header.strokes) ? header.strokes : [];
            this.strokes = reconcileStrokes(this.strokes, entries);
            this.pending.strokes = true;
        });

        conn.on(MessageType.STROKE_RESULT, (header, arrays) => {
            if (header.ok === false || header.accepted === false || header.reason) {
                this._setStatus(describeStroke(header), true);
                return;
            }
            const stroke = decodeStrokeResult(header, arrays);
            if (!stroke) return;
            this.strokes.set(stroke.id, stroke);
            this.pending.strokes = true;
        });

        conn.on(MessageType.SINGULARITIES, (header, arrays) => {
            this.pending.singularities = decodeSingularities(arrays);
        });

        conn.on(MessageType.EXTRACTED, (header, arrays) => {
            const wireframe = pickArray(arrays, 'wireframe');
            const colors = pickArray(arrays, 'wireframe_color', 'wireframeColor', 'colors');
            if (!wireframe || !colors) {
                this._setStatus('Extraction frame is missing its wireframe', true);
                return;
            }
            this.pending.extracted = { wireframe, colors };
            const faces = header.n_faces ?? header.face_count;
            this._setStatus(
                faces === undefined
                    ? `Extracted ${wireframe.length / 6} wireframe segments`
                    : `Extracted ${faces} faces`,
                false
            );
            this._setLayerChecked('output', true);
        });

        conn.on(MessageType.STATUS, (header) => this._showStatus(header));
        conn.on(MessageType.PROGRESS, (header) => this._showProgress(header.progress, true));

        conn.on(MessageType.ERROR, (header) => {
            const suffix = header.fatal ? ' -- reload the page' : '';
            this._setStatus(`${header.message}${suffix}`, true);
        });

        conn.onStatus((status) => this._showLink(status));
    }

    /* -------------------------------------------------------------- */
    /*  Toolbar                                                        */
    /* -------------------------------------------------------------- */

    _bindToolbar() {
        for (const button of ui.toolbar.querySelectorAll('button.tool')) {
            button.addEventListener('click', () => this.tools.setTool(button.dataset.tool));
        }

        for (const input of ui.toolbar.querySelectorAll('input[data-layer]')) {
            this.viewer.setLayerVisible(input.dataset.layer, input.checked);
            input.addEventListener('change', () =>
                this.viewer.setLayerVisible(input.dataset.layer, input.checked)
            );
        }

        /* 'both' runs orientations then positions, which is Session.solve_all;
           level -1 is the hierarchical schedule that terminates on its own. */
        ui.solve.addEventListener('click', () =>
            this.connection.send(MessageType.SOLVE, { field: 'both', level: -1 })
        );
        ui.stop.addEventListener('click', () => this.connection.send(MessageType.STOP, {}));
        ui.extract.addEventListener('click', () => this.connection.send(MessageType.EXTRACT, {}));
        ui.stop.disabled = true;
    }

    _setLayerChecked(name, checked) {
        const input = ui.toolbar.querySelector(`input[data-layer="${name}"]`);
        if (!input || input.checked === checked) return;
        input.checked = checked;
        this.viewer.setLayerVisible(name, checked);
    }

    /* -------------------------------------------------------------- */
    /*  Status strip                                                   */
    /* -------------------------------------------------------------- */

    _showTool(tool) {
        for (const button of ui.toolbar.querySelectorAll('button.tool')) {
            button.setAttribute('aria-checked', String(button.dataset.tool === tool.id));
        }
        ui.toolName.textContent = `Selected tool: ${tool.label}`;
        ui.hint.textContent =
            tool.kind === null
                ? tool.hint
                : `${tool.hint} Click a stroke handle to delete it, Esc cancels a stroke.`;
    }

    _showLink(status) {
        ui.link.dataset.state = status.state;
        ui.linkText.textContent =
            status.state === 'connecting' && status.attempt > 0
                ? `Reconnecting (${status.attempt})`
                : capitalise(status.state);
    }

    _showStatus(header) {
        const active = Boolean(header.active);
        ui.solve.disabled = active;
        ui.stop.disabled = !active;

        this._showProgress(header.progress, active);
        const level = header.level ?? 0;
        const versions = `Q ${header.iterations_q ?? 0} / O ${header.iterations_o ?? 0}`;
        this._setStatus(active ? `Solving level ${level} -- ${versions}` : `Idle -- ${versions}`, false);
    }

    /**
     * A hierarchical solve reports real progress; an in-place refinement at
     * level 0 reports 1 the whole time, which reads better as a busy bar.
     */
    _showProgress(progress, active) {
        const value = typeof progress === 'number' ? progress : 0;
        const busy = active && value >= 1;
        ui.progress.classList.toggle('indeterminate', busy);
        ui.progressFill.style.width = busy ? '100%' : `${Math.round(value * 100)}%`;
        ui.progress.setAttribute('aria-valuenow', String(Math.round(value * 100)));
    }

    _setStatus(message, isError) {
        ui.statusText.textContent = message;
        ui.statusText.classList.toggle('error', Boolean(isError));
    }
}

function capitalise(text) {
    return text.charAt(0).toUpperCase() + text.slice(1);
}

/* ------------------------------------------------------------------ */
/*  Boot                                                               */
/* ------------------------------------------------------------------ */

function fail(message) {
    ui.statusText.textContent = message;
    ui.statusText.classList.add('error');
    ui.link.dataset.state = 'closed';
    ui.linkText.textContent = 'Closed';
}

function boot() {
    const sessionId = new URLSearchParams(location.search).get('session');
    if (!sessionId) {
        fail('No session in the URL; open this page as /viewer?session=<id>');
        return;
    }

    let viewer;
    try {
        viewer = new Viewer(ui.canvas, ui.overlay);
    } catch (err) {
        fail(err.message);
        return;
    }

    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const connection = new Connection(`${scheme}://${location.host}/ws/${sessionId}`, {
        onOpen: (conn) => {
            conn.send(MessageType.HELLO, { session: sessionId, tools: TOOLS.map((t) => t.id) });
            conn.send(MessageType.SUBSCRIBE, { fps: SUBSCRIBE_FPS });
        },
    });

    /* Kept on window so a dev console can poke at a live session. */
    window.app = new App(viewer, connection);
    connection.connect();
}

boot();
