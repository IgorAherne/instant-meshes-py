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

import { MessageType, decode } from './protocol.js';
import { Connection } from './net.js';
import { Viewer, loadTexture } from './renderer.js';
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

/** What each view is waiting for, for the line that says it is waiting. */
const BUILD_VERB = { output: 'Extracting', uv: 'Unwrapping' };

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

        /* The view the user last asked for. A wish rather than a command: what
           it needs may not exist yet, and _pump builds it as soon as it can. */
        this._wanted = 'mesh';
        /* The one build allowed to be in flight: null, 'output' or 'uv'. */
        this._building = null;
        /* Set from SET_CONFIG until the GEOMETRY that answers it. Nothing is
           built for a mesh that is already being replaced. */
        this._rebuilding = false;
        /* The server's geometry counter, which every result frame carries: a
           build takes seconds, and one that lands after the mesh moved on
           describes something the viewport is no longer showing. */
        this._version = -1;
        this._meshName = null;

        /* The imported file as authored, fetched once per model, and the maps
           of whichever slot is being looked at. Keyed on the model rather than
           on the geometry version: re-targeting the resolution rebuilds the
           solver's mesh and leaves the file it came from alone. */
        this._sourceOf = null;
        this._sourceLoading = null;
        this._textureSlot = null;
        this._textureReady = false;

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
        this._wanted = this.panel.surface();
        this._applySurface(this._wanted);
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
        this.panel.handlers.onExtractOptionsChange = () => {
            this._invalidateOutput();
            this._pump();
        };

        /* There is no Apply button: the panel sends this once the remeshing
           settings have stopped changing, and the GEOMETRY frame that comes
           back re-solves itself, so a new target resolution simply appears. */
        this.panel.handlers.onConfigChange = (config) => {
            this._rebuilding = true;
            this.panel.setStatus('Rebuilding at the new resolution...', false);
            send(MessageType.SET_CONFIG, { config });
        };

        this.panel.handlers.onLayerToggle = (name, visible) =>
            this._apply(() => this.viewer.setLayerVisible(name, visible));

        this.panel.handlers.onOpenFile = (file) => this._uploadMesh(file);

        /* A different chart size is a different atlas; the one on screen is
           only worth re-cutting while somebody is looking at it. */
        this.panel.handlers.onUvChange = () => this._invalidateUv();

        this.panel.handlers.onTextureChange = (slot) => this._showTexture(slot);
    }

    /* -------------------------------------------------------------- */
    /*  The imported file's own materials                              */
    /* -------------------------------------------------------------- */

    /**
     * Put one of the model's own texture maps on it, or take them all off.
     *
     * The maps are authored against the file's UVs, which the mesh the solver
     * works on no longer has -- its seams are welded shut and it may have been
     * subdivided -- so this draws the original in its place. Both occupy the
     * same space, so the swap reads as the surface changing rather than as
     * something else appearing.
     */
    async _showTexture(slot) {
        if (!this.panel.setTextureSlot(slot)) return;
        this._textureSlot = this.panel.textureSlot();
        this._textureReady = false;
        this._refreshSurface();
        if (this._textureSlot === null) return;

        /* The model stays up while the maps decode, so the viewport is never
           blank; the button is already down, which is the press being heard. */
        const wanted = this._textureSlot;
        this.panel.setStatus('Loading the texture...', false);
        try {
            const source = await this._loadSource();
            const textures = await this._loadTextures(source, wanted);
            if (this._textureSlot !== wanted) return;
            this._apply(() => this.viewer.setSourceTextures(textures));
            this._textureReady = true;
            this.panel.setStatus(null, false);
            this._refreshSurface();
        } catch (err) {
            this.panel.setStatus(`Could not show that map: ${err.message}`, true);
            this.panel.setTextureSlot(null);
            this._textureSlot = null;
            this._refreshSurface();
        }
    }

    /** Forget the imported file, for when a different one replaces it. */
    _dropSource() {
        this._sourceOf = null;
        this._sourceLoading = null;
        this._textureSlot = null;
        this._textureReady = false;
        this.panel.setTextureSlot(null);
        this._apply(() => this.viewer.setSourceMesh(null));
    }

    /** Fetch the model as authored, once per import. */
    _loadSource() {
        if (this._sourceOf === this._meshName && this._sourceLoading) {
            return this._sourceLoading;
        }
        this._sourceOf = this._meshName;
        this._sourceLoading = (async () => {
            const response = await fetch(
                `/api/session/${encodeURIComponent(this.sessionId)}/source`
            );
            if (!response.ok) throw new Error(`the model could not be read back`);
            const message = decode(await response.arrayBuffer());
            const positions = pickArray(message.arrays, 'vertices');
            const indices = pickArray(message.arrays, 'faces');
            const uv = pickArray(message.arrays, 'uv');
            if (!positions || !indices) throw new Error('the model frame is incomplete');

            const materials = (message.header.materials || []).length || 1;
            this._apply(() =>
                this.viewer.setSourceMesh({
                    positions,
                    uv: uv || new Float32Array((positions.length / 3) * 2),
                    indices,
                    groups: message.header.groups || [],
                    materials,
                })
            );
            return { materials, slots: message.header.slots || [] };
        })();
        return this._sourceLoading;
    }

    /**
     * One decoded map per material for the chosen slot.
     *
     * A material with nothing in that slot gets null, which the renderer draws
     * flat: the button names a kind of map, and a surface that has none of
     * that kind should say so rather than keep wearing another one.
     */
    async _loadTextures(source, slot) {
        const base = `/api/session/${encodeURIComponent(this.sessionId)}/texture/${slot}`;
        const loads = [];
        for (let material = 0; material < source.materials; ++material) {
            loads.push(loadTexture(`${base}/${material}`));
        }
        return Promise.all(loads);
    }

    _showingUv() {
        return this._wanted === 'uv';
    }

    /**
     * Choose the view to show, and remember it.
     *
     * There is no Extract button and no Unwrap button: asking to see a result
     * IS the request to produce one. The choice is kept rather than acted on
     * once, so that a result which cannot be built yet -- because the mesh is
     * being rebuilt, or because one is already being built -- is still built
     * the moment it can be, instead of the keypress being swallowed.
     */
    _chooseSurface(which) {
        this._wanted = which;
        this._setSurface(which);
        this._pump();
    }

    /**
     * Build what the chosen view needs, if anything, and if now is the time.
     *
     * Every path that changes what exists calls this rather than deciding for
     * itself whether to send a request, which is what makes the viewport
     * self-healing: in whatever order the frames arrive, as soon as the
     * session is free the chosen view is either on screen or on its way.
     *
     * One build at a time, because the server answers frames in order and a
     * large mesh takes seconds to extract or to flatten: without this, a hand
     * resting on the number keys queues a minute of work nobody is waiting
     * for any more, and everything typed behind it -- a new resolution, a
     * stroke -- waits its turn at the back.
     */
    _pump() {
        if (this._building) return;

        const want = this._wanted;
        const missing =
            (want === 'output' && !this.hasOutput) || (want === 'uv' && !this.hasUv);
        if (!missing) return;

        /* Deferred, never dropped: this runs again from the GEOMETRY frame
           that ends the rebuild, so the view still arrives. */
        if (this._rebuilding || this.panel.configPending()) {
            this.panel.setStatus(`${BUILD_VERB[want]} at the new resolution...`, false);
            return;
        }

        this._building = want;
        if (want === 'output') {
            this.panel.setStatus('Extracting...', false);
            this.connection.send(MessageType.EXTRACT, this.panel.extractOptions());
        } else {
            /* Said here rather than waiting for the server's first report:
               that one cannot arrive until the request has crossed the socket
               and the streamer has ticked, and a control that sits silent for
               a fifth of a second reads as one that did not hear the click. */
            this.panel.showUnwrapping(0);
            this.connection.send(MessageType.UNWRAP, { leniency: this.panel.uvLeniency() });
        }
    }

    /**
     * Forget a build that will never be answered, and try again.
     *
     * Only three things end one: its own result frame, an error, or the mesh
     * being replaced underneath it. A flag left set by a fourth is a viewport
     * whose view keys have gone dead for good, which is exactly the state this
     * exists to make unreachable.
     */
    _abandonBuild() {
        this._building = null;
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
        /* A texture map stands in for the input surface and only for that:
           the result and the atlas are the remesher's own work, and the maps
           the file came with say nothing about either. */
        const textured = !output && !layout && this._textureReady;
        this._apply(() => {
            this.viewer.setLayerVisible('output', output);
            this.viewer.setUvVisible(which === 'uv');
            this.viewer.setSourceVisible(textured);
            this.viewer.setLayerVisible('mesh', !output && !layout && !textured);
        });
    }

    /** Re-apply the current view, for when what it needs has just arrived. */
    _refreshSurface() {
        this._applySurface(this._wanted);
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
        this._pump();
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

            /* This mesh replaces whatever any in-flight build was reading, so
               that build is over whether or not its frame ever arrives. */
            this._version = header.version ?? this._version + 1;
            this._rebuilding = false;
            this._abandonBuild();

            /* A different model is a different subject: the grid is the point
               of importing one, and solving it is the only way to see one, so
               the step is not worth asking for. Re-targeting the same model is
               not -- the view asked for before the rebuild is still the view
               wanted after it, and yanking it back to the input is what made a
               new resolution look like it had cancelled the request. */
            const name = header.name || '';
            if (name !== this._meshName) {
                this._meshName = name;
                this._wanted = 'mesh';
                /* A different file, so different materials and different UVs;
                   the buttons are rebuilt from the status that follows. */
                this._dropSource();
            }
            this._setSurface(this._wanted);
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
               means being back on it -- even if Extract hid it a moment ago.
               Drawing is a choice of view as much as a keypress is: it is the
               input the user wants now, and the result is stale anyway.
               A rebuild re-projects the strokes and sends them again; that is
               not somebody drawing, and it must not take the view. */
            if (!header.resent) this._chooseSurface('mesh');

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
            /* Answered, whatever it says: a frame dropped below without this
               would leave the view key that asked for it dead for good. */
            if (this._building === 'uv') this._abandonBuild();
            if (this._isStale(header)) return;

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
            this.panel.showUv({ charts: header.n_charts ?? 0, leniency: header.leniency });
            this.panel.setStatus(null, false);
            /* The atlas may not be what is wanted any more -- it is built from
               an extraction, and asking for that one instead is a keypress. */
            this._pump();
        });

        conn.on(MessageType.EXTRACTED, (header, arrays) => {
            if (this._building === 'output') this._abandonBuild();
            if (this._isStale(header)) return;

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
            /* Shown rather than selected: whoever asked for this already
               chose the view, and an extraction of a large mesh lands long
               enough afterwards that they may well have moved on. Pulling
               them back to it here is what made a burst of 1/2/3 end up
               somewhere nobody pressed. */
            this._refreshSurface();
            /* An extraction is also the first half of an atlas, so the wish
               that is waiting on it may now be buildable. */
            this._pump();
        });

        conn.on(MessageType.EXPORT_READY, (header) => {
            /* Pressing Export is the whole request, so the file starts
               arriving rather than waiting behind a second button. */
            this.panel.startDownload(header.url, header.filename);
            this.panel.setStatus(`Exported ${header.filename}`, false);
            /* The one place a result is worth switching to unasked: this is
               what was just written, and seeing it is the point of the file. */
            this._chooseSurface('output');
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
               nothing else clears the way for the next attempt. Not pumped
               from here: a request that has just failed would only fail again,
               and the next status frame or keypress is soon enough to retry. */
            this._abandonBuild();
            this._rebuilding = false;
        });

        conn.onStatus((status) => {
            this.panel.showLink(status);
            /* A socket that went away took the reply to anything in flight
               with it. Reconnecting replays what was queued, not what was
               already sent, so the wait has to be given up here or the view
               keys stay dead for the life of the page. */
            if (status.state !== 'open') {
                this._abandonBuild();
                this._rebuilding = false;
            }
        });
    }

    /**
     * Whether a result describes a mesh the viewport has already replaced.
     *
     * A build takes seconds; a rebuild takes one keystroke. Without this, an
     * extraction of the mesh from before the last resolution change arrives
     * and is shown -- and, worse, counts as the output that exists, so the
     * view the user is looking at is quietly of something else.
     */
    _isStale(header) {
        return header.version !== undefined && this._version >= 0
            && header.version < this._version;
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
        /* How many maps the imported file carried. Zero for every format that
           has no materials, which is when the row is not there at all. */
        if (header.textures !== undefined) this.panel.showTextures(header.textures);
        /* Carried on every status as well as in its own frame, so a report
           this client missed cannot leave a percentage on the label forever. */
        if (header.uv_progress !== undefined) {
            this.panel.showUnwrapping(header.uv_progress);
        }
        this.panel.setSolving(running);
        /* The field has moved, so whatever was extracted from the old one no
           longer describes it. Once per solve, not once per status frame.
           The selection stands: _applySurface holds the input up underneath
           until the replacement arrives, so nothing has to be taken away. */
        if (running && !this._wasSolving) this._invalidateOutput();
        this._wasSolving = running;

        if (header.config) this.panel.showConfig(header.config);

        /* The status line is the brush's help by default, and the solver's
           iteration counters meant nothing to anyone reading it. Only the fact
           that it is working is worth the space, and only while it is. */
        this.panel.setStatus(running ? 'Solving the field...' : null, false);

        /* Whatever happened, this is the point at which the session is known
           to be idle or busy -- so it is the point at which what the user
           asked to see gets built, if it still needs building. */
        this._pump();
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
