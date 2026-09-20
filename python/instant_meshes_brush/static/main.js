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
 * The two kinds of singularity, kept apart.
 *
 * They look identical on screen and each attractor moves only its own, so a
 * viewer that merges them is showing the user dots their brush cannot touch.
 * Copying also releases the frame buffer the decoded views would otherwise pin.
 */
function decodeSingularities(arrays) {
    const sets = {};
    for (const field of ['orientation', 'position']) {
        const p = pickArray(arrays, `${field}_positions`);
        const c = pickArray(arrays, `${field}_colors`);
        /* Keep only whole markers, which need both a centre and a colour. */
        const paired = p && c ? Math.min(p.length, c.length) : 0;
        const count = paired - (paired % 3);
        sets[field] = {
            positions: concatFloat32(count ? [p.subarray(0, count)] : []),
            colors: concatFloat32(count ? [c.subarray(0, count)] : []),
        };
    }
    return sets;
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
            uv: null,
        };
        this.strokes = new Map();
        this.posy = 4;
        /* Whether an extraction exists to switch to; a rebuild invalidates it. */
        this.hasOutput = false;
        /* The atlas is cut from the extraction, so it goes stale with it. */
        this.hasUv = false;
        /* One build of each kind in flight; see _chooseSurface. */
        this._pendingExtract = false;
        this._pendingUv = false;
        /* Markers per field, and which field the selected brush can reach. */
        this._singularityCounts = { orientation: 0, position: 0 };
        this._singularityField = null;

        this.tools = new ToolController(viewer, connection, {
            onToolChange: (tool) => this._onToolChange(tool),
            onNotice: (message) => this.panel.setStatus(message, false),
            onUndo: () => this._undoStroke(),
            onSurface: (which) => this._chooseSurface(which),
        });

        viewer.onFrame = () => this._drain();

        this._bindPanel();
        this._bindConnection();

        for (const name of ['grid', 'singularities']) {
            this.viewer.setLayerVisible(name, this.panel.layerState(name));
        }
        /* Strokes have no switch: they are the user's own marks, and hiding
           them can only make the viewport lie about what is constraining the
           field. Deleting one is a click on its handle. */
        this.viewer.setLayerVisible('strokes', true);
        this._applySurface(this.panel.surface());
    }

    /**
     * An attractor drags a singularity, so it needs them on screen.
     *
     * They are off by default -- a hundred coloured dots mean nothing until
     * you know what they are -- but picking the brush that moves one is as
     * clear a statement of intent as ticking the box would be.
     */
    _onToolChange(tool) {
        this.panel.showTool(tool);
        this._singularityField = tool.singularities || null;
        this._apply(() => this.viewer.setSingularityFilter(this._singularityField));
        this._showSingularityCount();
        if (tool.singularities) this._setLayer('singularities', true);
    }

    /** The count of what is actually drawn, which is what the filter decides. */
    _showSingularityCount() {
        const counts = this._singularityCounts;
        const field = this._singularityField;
        const total = field ? counts[field] : counts.orientation + counts.position;
        this.panel.showSingularities(total > 0 ? total : null);
    }

    /**
     * Undo the most recent stroke.
     *
     * Only strokes: they are the destructive thing here, and everything else
     * the viewport does is either reversible by doing it again or is simply a
     * view. Erasing re-solves, exactly as drawing did.
     */
    _undoStroke() {
        const ids = [...this.strokes.keys()];
        if (ids.length === 0) {
            this.panel.setStatus('Nothing to undo', false);
            return;
        }
        this.connection.send(MessageType.ERASE_STROKE, { stroke_id: ids[ids.length - 1] });
        this.panel.setStatus('Undid the last stroke', false);
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
            this._apply(() => this.viewer.setSingularities(singularities));
        }

        const extracted = pending.extracted;
        if (extracted) {
            pending.extracted = null;
            this._apply(() =>
                this.viewer.setExtracted(extracted.wireframe, extracted.colors, extracted.surface)
            );
        }

        const uv = pending.uv;
        if (uv) {
            pending.uv = null;
            this._apply(() => this.viewer.setUvLayout(uv.layout));
            /* The layout is what Result UV was waiting for. */
            this._refreshSurface();
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

        this.panel.handlers.onExport = (options) => send(MessageType.EXPORT, options);

        this.panel.handlers.onSurfaceChange = (which) => this._chooseSurface(which);

        /* Pure quad and smoothing are read at extraction time, so the result
           on screen no longer matches them. Rebuild it now if it is what the
           user is looking at, and otherwise the next time they ask for it. */
        this.panel.handlers.onExtractOptionsChange = (options) => {
            this._invalidateOutput();
            if (this.panel.surface() !== 'output') return;
            this.panel.setStatus('Extracting...', false);
            if (this._pendingExtract) return;
            this._pendingExtract = true;
            send(MessageType.EXTRACT, options);
        };

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

        /* A different chart size is a different atlas; the one on screen is
           only worth re-cutting while somebody is looking at it. */
        this.panel.handlers.onUvChange = () => {
            this.hasUv = false;
            if (this._showingUv()) this._requestUv();
        };
    }

    _showingUv() {
        return this.panel.surface() === 'uv';
    }

    /**
     * Switch the view, building whatever it needs.
     *
     * There is no Extract button and no Unwrap button: asking to see a result
     * IS the request to produce one. Anything that changes the field marks
     * both stale, so this rebuilds rather than showing what was made before
     * the last stroke.
     *
     * At most one build of each kind is ever in flight. The server answers
     * frames in order and an extraction of a large mesh takes seconds, so
     * without this a hand resting on the number keys queues a minute of work
     * that nobody is waiting for any more.
     */
    _chooseSurface(which) {
        this._setSurface(which);
        if (which === 'output' && !this.hasOutput) {
            this.panel.setStatus('Extracting...', false);
            if (!this._pendingExtract) {
                this._pendingExtract = true;
                this.connection.send(MessageType.EXTRACT, this.panel.extractOptions());
            }
        } else if (which === 'uv' && !this.hasUv) {
            this._requestUv();
        }
    }

    /**
     * Ask for an atlas, once there is a field worth cutting one from.
     *
     * Unwrapping extracts, and extracting stops the solver -- so a pointer
     * that merely crosses this control on its way somewhere else must not
     * abandon the solve a brush stroke just started. The request waits for the
     * field instead, and _showStatus picks it up when the solve ends.
     */
    _requestUv() {
        if (this._pendingUv) return;
        if (this._solving) {
            this.panel.setStatus('Unwrapping once the field has settled...', false);
            return;
        }
        this._pendingUv = true;
        /* Said here rather than waiting for the server's first report: that
           one cannot arrive until the request has crossed the socket and the
           streamer has ticked, and a control that sits silent for a fifth of
           a second reads as one that did not hear the click. */
        this.panel.showUnwrapping(0);
        this.connection.send(MessageType.UNWRAP, { leniency: this.panel.uvLeniency() });
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
     * Never both: they occupy the same space, and the input in front of the
     * output is what made an extracted mesh look like it had never appeared.
     * Extracting selects the output; laying a stroke selects the input, because
     * a brush needs something to draw on.
     */
    _setSurface(which) {
        this.panel.setSurface(which);
        this._applySurface(which);
    }

    /**
     * Show the chosen view, or hold the input up until it exists.
     *
     * Picking a result that has not been built yet is a request, and a
     * request takes as long as it takes; blanking the stage for the duration
     * answers it with nothing at all. The selection stands -- the button says
     * what was asked for -- and the model stays visible underneath until the
     * answer arrives, which is what setUvVisible already did for the layout.
     */
    _applySurface(which) {
        const output = which === 'output' && this.hasOutput;
        const layout = which === 'uv' && this.hasUv;
        this._apply(() => {
            this.viewer.setLayerVisible('output', output);
            this.viewer.setUvVisible(which === 'uv');
            this.viewer.setLayerVisible('mesh', !output && !layout);
        });
    }

    /** Re-apply the current view, for when what it needs has just arrived. */
    _refreshSurface() {
        this._applySurface(this.panel.surface());
    }

    /**
     * Forget an extraction the field has moved on from.
     *
     * Solving is what makes it stale, and with no Extract button the only
     * thing standing between a stroke and a mesh extracted before it is this:
     * the result is dropped, so asking to see the output builds a new one.
     */
    _invalidateOutput() {
        this._invalidateUv();
        if (!this.hasOutput) return;
        this.hasOutput = false;
        this.panel.showOutput(null);
        this.pending.extracted = { wireframe: null, colors: null, surface: null };
    }

    /**
     * Forget an atlas whose mesh has moved under it.
     *
     * It is cut from the extracted faces, so anything that invalidates those
     * invalidates this -- and if it is the thing on screen, a replacement is
     * asked for straight away rather than leaving the old one up as though
     * nothing had changed.
     */
    _invalidateUv() {
        if (!this.hasUv) return;
        this.hasUv = false;
        this.panel.showUv(null);
        this.pending.uv = { layout: null };
        if (this._showingUv()) this._requestUv();
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
            this.panel.showOutput(null);
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
            /* A rebuild re-projects the strokes onto the new mesh, so these
               ribbons describe a surface that no longer exists even where the
               stroke itself survived. Drop them; the STROKE_RESULT frames that
               follow a rebuild carry the replacements. */
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
            this.panel.showOutput(null);
            this.panel.setReady(true);
            this.hasOutput = false;
            /* setGeometry drops the overlays themselves; these are the panel's
               copies of what they said. */
            this.hasUv = false;
            this.panel.showUv(null);
            this._singularityCounts = { orientation: 0, position: 0 };
            this._showSingularityCount();
            /* The grid is the point of importing a mesh, and solving it is the
               only way to see one, so the step is not worth asking for. */
            this._setSurface('mesh');
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
            this._setSurface('mesh');

            const stroke = decodeStrokeResult(header, arrays);
            if (!stroke) return;
            this.strokes.set(stroke.id, stroke);
            this.pending.strokes = true;
        });

        conn.on(MessageType.SINGULARITIES, (header, arrays) => {
            const sets = decodeSingularities(arrays);
            this.pending.singularities = sets;
            this._singularityCounts = {
                orientation: sets.orientation.positions.length / 3,
                position: sets.position.positions.length / 3,
            };
            this._showSingularityCount();
        });

        conn.on(MessageType.UV_LAYOUT, (header, arrays) => {
            const uv = pickArray(arrays, 'uv');
            const faces = pickArray(arrays, 'faces');
            const chart = pickArray(arrays, 'chart');
            if (!uv || !faces || !chart) {
                this.panel.setStatus('UV frame is missing an array', true);
                return;
            }
            this.pending.uv = {
                layout: {
                    uv,
                    faces,
                    chart,
                    tris: pickArray(arrays, 'tris') || new Int32Array(0),
                    triChart: pickArray(arrays, 'tri_chart') || new Int32Array(0),
                    width: header.posy ?? 4,
                },
            };
            this.hasUv = true;
            this._pendingUv = false;
            this.panel.showUv({ charts: header.n_charts ?? 0, leniency: header.leniency });
            this.panel.setStatus(null, false);
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
            this.panel.showOutput({ vertices: vertexCount, faces: faceCount });
            this.panel.setStatus(`Extracted ${faceCount} faces`, false);
            this.hasOutput = faceCount > 0;
            this._pendingExtract = false;
            /* Shown rather than selected: whoever asked for this already
               chose the view, and an extraction of a large mesh lands long
               enough afterwards that they may well have moved on. Pulling
               them back to it here is what made a burst of 1/2/3 end up
               somewhere nobody pressed. */
            this._refreshSurface();
        });

        conn.on(MessageType.EXPORT_READY, (header) => {
            /* Pressing Export is the whole request, so the file starts
               arriving rather than waiting behind a second button. */
            this.panel.startDownload(header.url, header.filename);
            this.panel.setStatus(`Exported ${header.filename}`, false);
            /* The one place a result is worth switching to unasked: this is
               what was just written, and seeing it is the point of the file. */
            this._setSurface('output');
        });

        /* Its own frame rather than a field on STATUS: the unwrapper holds the
           session's worker thread, and a status frame cannot be built without
           it, so the only number that can reach here mid-unwrap is this one. */
        conn.on(MessageType.PROGRESS, (header) =>
            this.panel.showUnwrapping(header.uv ?? null)
        );

        conn.on(MessageType.STATUS, (header) => this._showStatus(header));

        conn.on(MessageType.ERROR, (header) => {
            const suffix = header.fatal ? ' -- reload the page' : '';
            this.panel.setStatus(`${header.message}${suffix}`, true);
            /* An unwrap that failed on its way in may never have been seen
               running, so nothing else would take the label back down. */
            this.panel.showUnwrapping(null);
            /* A build that answered with an error answered all the same, and
               nothing else clears the way for the next attempt. */
            this._pendingExtract = false;
            this._pendingUv = false;
        });

        conn.onStatus((status) => this.panel.showLink(status));
    }

    _showStatus(header) {
        /* `solving` spans both phases of a "both" solve, `active` only the one
           the C++ optimizer is in; using the wider one keeps the buttons from
           flickering back to life in the gap between orientations and
           positions. */
        const running = Boolean(header.solving ?? header.active);
        this._solving = running;
        if (header.ready !== undefined) this.panel.setReady(Boolean(header.ready));
        if (header.uv !== undefined) this.panel.setUvAvailable(Boolean(header.uv));
        this.panel.setSolving(running);
        /* The field has moved, so whatever was extracted from the old one no
           longer describes it. Once per solve, not once per status frame. */
        if (running && !this._wasSolving) {
            this._invalidateOutput();
            /* Dropping the output while it is the surface on screen would
               leave an empty stage -- which is what erasing a stroke while
               looking at the result used to do. */
            if (this.panel.surface() === 'output') this._setSurface('mesh');
        }
        this._wasSolving = running;

        if (header.config) this.panel.showConfig(header.config);

        /* The status line is the brush's help by default, and the solver's
           iteration counters meant nothing to anyone reading it. Only the fact
           that it is working is worth the space, and only while it is. */
        this.panel.setStatus(running ? 'Solving the field...' : null, false);

        /* A layout asked for mid-solve was deferred rather than dropped. Last,
           so the request's own message is the one left on the line. */
        if (!running && this._showingUv() && !this.hasUv) this._requestUv();
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
