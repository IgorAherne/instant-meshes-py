/*
    main.js: page wiring.

    The whole UI lives in this document: panel.js owns the controls, this file
    owns the session and the frames that flow through it.  A host page embeds
    the viewer with one <iframe> and reproduces none of it.

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
import { Panel } from './panel.js';

/** Preview rate asked of the server; the RAF loop coalesces anything faster. */
const SUBSCRIBE_FPS = 15;

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
    constructor(viewer, connection, panel, sessionId) {
        this.viewer = viewer;
        this.connection = connection;
        this.panel = panel;
        this.sessionId = sessionId;

        /* Drained by the render loop; see the module comment. */
        this.pending = {
            geometry: null,
            field: null,
            strokes: false,
            singularities: null,
            extracted: null,
        };
        this.strokes = new Map();
        this.posy = 4;

        this.tools = new ToolController(viewer, connection, {
            onToolChange: (tool) => this.panel.showTool(tool),
            onNotice: (message) => this.panel.setStatus(message, false),
        });

        viewer.onFrame = () => this._drain();

        this._bindPanel();
        this._bindConnection();

        for (const name of ['mesh', 'grid', 'strokes', 'singularities', 'output']) {
            this.viewer.setLayerVisible(name, this.panel.layerState(name));
        }
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
            this.panel.showStrokes(strokes.length);
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
            this._apply(() =>
                this.viewer.setExtracted(extracted.wireframe, extracted.colors, extracted.surface)
            );
        }
    }

    _apply(update) {
        try {
            update();
        } catch (err) {
            this.panel.setStatus(err.message, true);
        }
    }

    /* -------------------------------------------------------------- */
    /*  Panel                                                          */
    /* -------------------------------------------------------------- */

    _bindPanel() {
        const send = (type, header) => this.connection.send(type, header || {});

        this.panel.handlers.onSelectTool = (id) => this.tools.setTool(id);
        this.panel.handlers.onClearStrokes = () => send(MessageType.CLEAR_STROKES);

        this.panel.handlers.onSolve = () => this._solve();
        this.panel.handlers.onExtract = (options) => send(MessageType.EXTRACT, options);
        this.panel.handlers.onExport = (options) => send(MessageType.EXPORT, options);

        /* There is no Apply button: the panel sends this once the remeshing
           settings have stopped changing, and the GEOMETRY frame that comes
           back re-solves itself, so a new target resolution simply appears. */
        this.panel.handlers.onConfigChange = (config) => {
            this.panel.setStatus('Rebuilding at the new resolution...', false);
            send(MessageType.SET_CONFIG, { config });
        };

        this.panel.handlers.onLayerToggle = (name, visible) =>
            this._apply(() => this.viewer.setLayerVisible(name, visible));

        this.panel.handlers.onOpenFile = (file) => this._uploadMesh(file);
    }

    /** Level -1 is the hierarchical schedule, which terminates on its own;
     *  'both' runs orientations then positions, as Session.solve_all does. */
    _solve() {
        this.connection.send(MessageType.SOLVE, { field: 'both', level: -1 });
        this.panel.setSolving(true);
    }

    _setLayer(name, checked) {
        if (this.panel.setLayerChecked(name, checked)) {
            this._apply(() => this.viewer.setLayerVisible(name, checked));
        }
    }

    /**
     * Which of the two surfaces the viewport is showing.
     *
     * Extracting hides the input so the result is unmistakably visible; laying
     * a stroke brings it back, because a brush needs something to draw on.
     */
    _showOutputOnly(only) {
        this._setLayer('output', only);
        this._setLayer('mesh', !only);
        this._setLayer('grid', !only);
    }

    /**
     * Upload through HTTP rather than the socket.
     *
     * The server parses the file with trimesh and loads it into the session;
     * the geometry itself comes back over the WebSocket, so a second viewport
     * watching the same session updates too.
     */
    async _uploadMesh(file) {
        this.panel.setStatus(`Loading ${file.name}...`, false);
        const body = new FormData();
        body.append('file', file, file.name);
        body.append('config', JSON.stringify(this.panel.readConfig()));

        try {
            const response = await fetch(
                `/api/session/${encodeURIComponent(this.sessionId)}/mesh`,
                { method: 'POST', body }
            );
            if (!response.ok) {
                const detail = await response.json().catch(() => ({}));
                throw new Error(detail.detail || `upload failed (${response.status})`);
            }
            /* The GEOMETRY frame that follows fills in the rest. */
            this.panel.showDownload(null);
            this.panel.showOutput('Not extracted yet');
        } catch (err) {
            this.panel.setStatus(err.message, true);
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
                this.panel.setStatus('Geometry frame is missing an array', true);
                return;
            }
            if (!(header.scale > 0)) {
                this.panel.setStatus(`Geometry frame has no usable scale (${header.scale})`, true);
                return;
            }
            this.posy = header.posy ?? 4;
            this.pending.geometry = {
                positions,
                indices,
                normals,
                scale: header.scale,
                /* Config defaults, for a server that only sends non-default symmetries. */
                rosy: header.rosy ?? 4,
                posy: this.posy,
            };
            /* Session::preprocess drops every stroke, because the vertex indices
               they were projected onto no longer exist.  Forget them here too,
               so ribbons built for the previous mesh cannot survive it; the
               STROKE_LIST that follows repopulates whatever really remains. */
            this.strokes.clear();
            this.pending.strokes = true;
            this.pending.extracted = { wireframe: null, colors: null, surface: null };

            this.panel.showConfig(header.config);
            this.panel.showMesh({
                name: header.name,
                vertices: positions.length / 3,
                faces: indices.length / 3,
                scale: header.scale,
                targetVertices: header.config ? header.config.vertex_count : 0,
            });
            this.panel.showFieldState({ orientation: 'not solved', position: 'not solved' });
            this.panel.showOutput('Not extracted yet');
            this.panel.setReady(true);
            /* The grid is the point of loading a mesh, and solving it is the
               only way to see one, so the step is not worth asking for. */
            this._showOutputOnly(false);
            this.panel.setStatus('Solving the field...', false);
            this._solve();
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
                this.panel.setStatus(describeStroke(header), true);
                return;
            }
            /* A stroke is drawn on the input surface, so seeing where it landed
               means being back on it -- even if Extract hid it a moment ago. */
            this._showOutputOnly(false);

            const stroke = decodeStrokeResult(header, arrays);
            if (!stroke) return;
            this.strokes.set(stroke.id, stroke);
            this.pending.strokes = true;
        });

        conn.on(MessageType.SINGULARITIES, (header, arrays) => {
            this.pending.singularities = decodeSingularities(arrays);
            const orientation = header.n_orientation;
            const position = header.n_position;
            this.panel.showFieldState({
                orientation:
                    orientation === undefined ? undefined : `${orientation} singularities`,
                position: position === undefined ? undefined : `${position} singularities`,
            });
        });

        conn.on(MessageType.EXTRACTED, (header, arrays) => {
            const wireframe = pickArray(arrays, 'wireframe');
            const colors = pickArray(arrays, 'wireframe_color', 'wireframeColor', 'colors');
            if (!wireframe || !colors) {
                this.panel.setStatus('Extraction frame is missing its wireframe', true);
                return;
            }
            const vertices = pickArray(arrays, 'vertices');
            const faces = pickArray(arrays, 'faces');
            const faceNormals = pickArray(arrays, 'face_normals', 'faceNormals');
            this.pending.extracted = {
                wireframe,
                colors,
                surface:
                    vertices && faces && faceNormals
                        ? { vertices, faces, faceNormals, posy: header.posy ?? this.posy }
                        : null,
            };

            const faceCount = header.n_faces ?? header.face_count ?? 0;
            const vertexCount = header.n_vertices ?? 0;
            this.panel.showOutput(
                `${vertexCount.toLocaleString()} vertices / ${faceCount.toLocaleString()} faces`
            );
            this.panel.setStatus(`Extracted ${faceCount} faces`, false);
            /* Showing the result is the whole point of pressing Extract, and
               the input surface sits a hair in front of it in places. */
            this._showOutputOnly(true);
        });

        conn.on(MessageType.EXPORT_READY, (header) => {
            this.panel.showDownload(header.url, header.filename, header.bytes);
            this.panel.setStatus(`Exported ${header.filename}`, false);
        });

        conn.on(MessageType.STATUS, (header) => this._showStatus(header));

        conn.on(MessageType.ERROR, (header) => {
            const suffix = header.fatal ? ' -- reload the page' : '';
            this.panel.setStatus(`${header.message}${suffix}`, true);
        });

        conn.onStatus((status) => this.panel.showLink(status));
    }

    _showStatus(header) {
        /* `solving` spans both phases of a "both" solve, `active` only the one
           the C++ optimizer is in; using the wider one keeps the buttons from
           flickering back to life in the gap between orientations and
           positions. */
        const running = Boolean(header.solving ?? header.active);
        if (header.ready !== undefined) this.panel.setReady(Boolean(header.ready));
        this.panel.setSolving(running);

        if (header.config) this.panel.showConfig(header.config);

        const versions = `Q ${header.iterations_q ?? 0} / O ${header.iterations_o ?? 0}`;
        if (!running) {
            this.panel.setStatus(`Idle -- ${versions}`, false);
            return;
        }
        /* The progress bar is gone -- it cost a row of the stage to say this. */
        const percent = Math.round(Math.min(1, Math.max(0, header.progress || 0)) * 100);
        this.panel.setStatus(
            `Solving level ${header.level ?? 0}, ${percent}% -- ${versions}`,
            false
        );
    }
}

/* ------------------------------------------------------------------ */
/*  Boot                                                               */
/* ------------------------------------------------------------------ */

function fail(message) {
    const text = document.getElementById('status-text');
    if (text) {
        text.textContent = message;
        text.classList.add('error');
    }
}

function boot() {
    const sessionId = new URLSearchParams(location.search).get('session');
    if (!sessionId) {
        fail('No session in the URL; open this page as /viewer?session=<id>');
        return;
    }

    let viewer;
    let panel;
    try {
        panel = new Panel({});
        viewer = new Viewer(document.getElementById('view'), document.getElementById('overlay'));
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
    window.app = new App(viewer, connection, panel, sessionId);
    connection.connect();
}

boot();
