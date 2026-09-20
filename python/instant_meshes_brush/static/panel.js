/*
    panel.js: the viewer's own control panel.

    Everything the user can press lives inside the viewer document, so embedding
    the viewport is a single <iframe> and a host page reproduces none of this.
    The panel owns no application state: it reads the DOM, calls back into the
    app, and renders whatever the app tells it to show.
*/

/** Formats the upload route accepts; mirrored from server.MESH_SUFFIXES. */
export const MESH_ACCEPT = '.obj,.ply,.stl,.off';

/** Symmetry presets, as (rosy, posy) pairs keyed by the select's value. */
export const SYMMETRIES = {
    '4,4': { rosy: 4, posy: 4 },
    '6,3': { rosy: 6, posy: 3 },
    '2,4': { rosy: 2, posy: 4 },
};

/** Crease angle used when "Sharp creases" is ticked; -1 disables detection. */
const CREASE_ANGLE = 30.0;

/**
 * How long the remeshing settings must sit still before they are applied.
 *
 * There is no Apply button: a rebuild is cheap enough to simply follow the
 * controls.  The delay is what keeps dragging the target slider from queuing
 * one rebuild per pixel, and what lets a typed vertex count be finished.
 */
const CONFIG_SETTLE_MS = 500;

function byId(id) {
    const element = document.getElementById(id);
    if (!element) throw new Error(`the viewer document is missing #${id}`);
    return element;
}

/** Field-by-field equality, so an unchanged config never forces a rebuild. */
function sameConfig(a, b) {
    if (!a || !b) return false;
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length && keys.every((key) => a[key] === b[key]);
}

/**
 * Hover help.
 *
 * A single floating element positioned next to the hovered control, rather than
 * a bubble inside it: the panel scrolls and clips, and native `title` tooltips
 * appear too late to answer "what does this button do?".
 */
class Hints {
    constructor(element) {
        this.element = element;
        this._current = null;

        const show = (event) => {
            const host = event.target.closest('[data-hint]');
            if (host === this._current) return;
            this._current = host;
            if (!host) {
                this.element.hidden = true;
                return;
            }
            this.element.textContent = host.dataset.hint;
            this.element.hidden = false;
            this._place(host);
        };

        document.addEventListener('pointerover', show);
        document.addEventListener('pointerout', (event) => {
            if (!event.relatedTarget) show(event);
        });
        document.addEventListener('pointerdown', () => {
            this.element.hidden = true;
            this._current = null;
        });
        /* A scroll moves the anchor out from under the bubble. */
        document.addEventListener('scroll', () => { this.element.hidden = true; }, true);
    }

    _place(host) {
        const anchor = host.getBoundingClientRect();
        const tip = this.element.getBoundingClientRect();
        const margin = 8;

        let left = anchor.right + margin;
        if (left + tip.width > window.innerWidth - margin) {
            left = Math.max(margin, anchor.left - tip.width - margin);
        }
        const top = Math.min(
            Math.max(margin, anchor.top),
            Math.max(margin, window.innerHeight - tip.height - margin)
        );

        this.element.style.left = `${Math.round(left)}px`;
        this.element.style.top = `${Math.round(top)}px`;
    }
}

/**
 * Wraps the panel's DOM.
 *
 * @param {object} handlers  callbacks the controls invoke; each is optional
 */
export class Panel {
    constructor(handlers) {
        this.handlers = handlers;

        this.el = {
            link: byId('link'),
            open: byId('btn-open'),
            file: byId('file-input'),
            meshName: byId('mesh-name'),
            meshStats: byId('mesh-stats'),

            symmetry: byId('symmetry'),
            target: byId('target'),
            targetRange: byId('target-range'),
            extrinsic: byId('opt-extrinsic'),
            boundaries: byId('opt-boundaries'),
            creases: byId('opt-creases'),

            strokeCount: byId('stroke-count'),
            clear: byId('btn-clear'),

            solve: byId('btn-solve'),
            statOrient: byId('stat-orient'),
            statPos: byId('stat-pos'),

            extract: byId('btn-extract'),
            outputStats: byId('output-stats'),
            format: byId('format'),
            pureQuad: byId('opt-pure-quad'),
            smoothing: byId('smoothing'),
            export: byId('btn-export'),
            download: byId('download'),

            toolName: byId('tool-name'),
            statusText: byId('status-text'),
            hint: byId('hint'),
        };

        this.el.file.setAttribute('accept', MESH_ACCEPT);
        this.hints = new Hints(byId('tip'));

        this._solveLabel = this.el.solve.textContent.trim();
        /* The config as the server last reported it, which is what an edit is
           compared against.  Null until a mesh exists: there is nothing to
           rebuild before then, so nothing to apply either. */
        this._appliedConfig = null;
        this._settleTimer = 0;

        this._bind();
        this.setSolving(false);
        this.setReady(false);
    }

    _bind() {
        const on = (element, event, fn) => element.addEventListener(event, fn);
        const call = (name, ...args) => {
            const handler = this.handlers[name];
            if (handler) handler(...args);
        };

        on(this.el.open, 'click', () => this.el.file.click());
        on(this.el.file, 'change', () => {
            const file = this.el.file.files && this.el.file.files[0];
            /* Clearing the value lets the same file be chosen twice running. */
            this.el.file.value = '';
            if (file) call('onOpenFile', file);
        });

        /* The slider and the number box are two views of one value. */
        const syncTarget = (source, other) => {
            const value = Number(source.value);
            if (Number.isFinite(value)) other.value = String(value);
        };
        on(this.el.targetRange, 'input', () => {
            syncTarget(this.el.targetRange, this.el.target);
            this._settleConfig();
        });
        on(this.el.target, 'input', () => {
            syncTarget(this.el.target, this.el.targetRange);
            this._settleConfig();
        });
        for (const key of ['symmetry', 'extrinsic', 'boundaries', 'creases']) {
            on(this.el[key], 'change', () => this._settleConfig());
        }

        for (const button of document.querySelectorAll('button.tool')) {
            on(button, 'click', () => call('onSelectTool', button.dataset.tool));
        }
        on(this.el.clear, 'click', () => call('onClearStrokes'));

        on(this.el.solve, 'click', () => call('onSolve'));

        on(this.el.extract, 'click', () => call('onExtract', this.extractOptions()));
        on(this.el.export, 'click', () =>
            call('onExport', { format: this.el.format.value, ...this.extractOptions() })
        );

        for (const input of document.querySelectorAll('input[data-layer]')) {
            on(input, 'change', () =>
                call('onLayerToggle', input.dataset.layer, input.checked)
            );
        }
    }

    /**
     * Apply the remeshing settings once they stop changing.
     *
     * Applying means re-running preprocess on the server, which throws away
     * every stroke, so a rebuild that would produce exactly the mesh already on
     * screen is dropped rather than sent.
     */
    _settleConfig() {
        clearTimeout(this._settleTimer);
        this._settleTimer = setTimeout(() => {
            const config = this.readConfig();
            if (!this._appliedConfig || sameConfig(config, this._appliedConfig)) return;
            this._appliedConfig = config;
            if (this.handlers.onConfigChange) this.handlers.onConfigChange(config);
        }, CONFIG_SETTLE_MS);
    }

    /* -------------------------------------------------------------- */
    /*  Reading                                                        */
    /* -------------------------------------------------------------- */

    /** The remeshing settings, as the server's Config mapping.
     *
     * Only the keys preprocess() bakes into the hierarchy: the extraction
     * options travel with EXTRACT instead, so ticking "Pure quad" costs a
     * re-extraction rather than a rebuild.
     */
    readConfig() {
        const symmetry = SYMMETRIES[this.el.symmetry.value] || SYMMETRIES['4,4'];
        return {
            ...symmetry,
            vertex_count: Math.max(10, Number(this.el.target.value) || 1000),
            extrinsic: this.el.extrinsic.checked,
            align_to_boundaries: this.el.boundaries.checked,
            crease_angle: this.el.creases.checked ? CREASE_ANGLE : -1.0,
        };
    }

    /** The two settings extract() reads, which need no rebuild. */
    extractOptions() {
        return {
            pure_quad: this.el.pureQuad.checked,
            smooth_iter: Math.max(0, Number(this.el.smoothing.value) || 0),
        };
    }

    layerState(name) {
        const input = document.querySelector(`input[data-layer="${name}"]`);
        return Boolean(input && input.checked);
    }

    /* -------------------------------------------------------------- */
    /*  Rendering                                                      */
    /* -------------------------------------------------------------- */

    setLayerChecked(name, checked) {
        const input = document.querySelector(`input[data-layer="${name}"]`);
        if (!input || input.checked === checked) return false;
        input.checked = checked;
        return true;
    }

    /** Reflect the server's config back into the controls after a rebuild.
     *
     * A control the user is currently in is left alone: status frames arrive
     * while a vertex count is half typed, and overwriting it there would fight
     * the person typing.
     */
    showConfig(config) {
        if (!config) return;
        const editing = document.activeElement;
        const settable = (element) => element !== editing;

        const key = `${config.rosy},${config.posy}`;
        if (key in SYMMETRIES && settable(this.el.symmetry)) this.el.symmetry.value = key;
        if (config.vertex_count > 0 && settable(this.el.target)
            && settable(this.el.targetRange)) {
            const value = String(config.vertex_count);
            this.el.target.value = value;
            this.el.targetRange.value = value;
        }
        if (settable(this.el.extrinsic)) this.el.extrinsic.checked = Boolean(config.extrinsic);
        if (settable(this.el.boundaries)) {
            this.el.boundaries.checked = Boolean(config.align_to_boundaries);
        }
        if (settable(this.el.creases)) {
            this.el.creases.checked = Number(config.crease_angle) >= 0;
        }

        /* Whatever the controls now read is, by definition, what is built. */
        this._appliedConfig = this.readConfig();
    }

    showMesh({ name, vertices, faces, scale, targetVertices }) {
        this.el.meshName.textContent = name || 'Untitled mesh';
        this.el.meshStats.hidden = false;
        this.el.meshStats.textContent =
            `${vertices.toLocaleString()} v / ${faces.toLocaleString()} tri` +
            `  →  ${(targetVertices || 0).toLocaleString()} v target,` +
            ` edge ${scale.toPrecision(3)}`;
        /* The slider only makes sense once we know how big the model is. */
        this.el.targetRange.max = String(Math.max(50, vertices));
    }

    showStrokes(count) {
        this.el.strokeCount.textContent =
            count === 0 ? 'No strokes' : `${count} stroke${count === 1 ? '' : 's'}`;
        this.el.clear.disabled = count === 0;
    }

    showFieldState({ orientation, position }) {
        if (orientation !== undefined) this.el.statOrient.textContent = orientation;
        if (position !== undefined) this.el.statPos.textContent = position;
    }

    showOutput(text) {
        this.el.outputStats.textContent = text;
    }

    showDownload(url, filename, bytes) {
        const link = this.el.download;
        if (!url) {
            link.hidden = true;
            return;
        }
        link.href = url;
        link.download = filename || 'mesh';
        link.textContent = `Download ${filename}${bytes ? ` (${formatBytes(bytes)})` : ''}`;
        link.hidden = false;
    }

    showTool(tool) {
        for (const button of document.querySelectorAll('button.tool')) {
            button.setAttribute('aria-checked', String(button.dataset.tool === tool.id));
        }
        this.el.toolName.textContent = tool.label;
        this.el.hint.textContent = tool.hint;
    }

    showLink(status) {
        const online = status.state === 'open';
        this.el.link.classList.toggle('online', online);
        this.el.link.classList.toggle('offline', !online && status.state !== 'connecting');
        this.el.link.title =
            status.state === 'connecting' && status.attempt > 0
                ? `Reconnecting (${status.attempt})`
                : status.state;
    }

    /** A solve in flight. Every solve the viewport starts ends by itself, so
     *  there is nothing to press here -- the buttons simply wait it out.
     *  Opening a mesh stays available: it stops the solve on its way in, and
     *  being unable to abandon a long solve by loading something else would be
     *  the one place this UI could strand somebody. */
    setSolving(active) {
        this._solving = active;
        this.el.solve.textContent = active ? 'Solving…' : this._solveLabel;
        for (const key of ['solve', 'extract', 'export']) {
            this.el[key].disabled = active || !this._ready;
        }
    }

    setReady(ready) {
        this._ready = ready;
        for (const key of ['solve', 'extract', 'export']) {
            this.el[key].disabled = !ready || Boolean(this._solving);
        }
    }

    setStatus(message, isError) {
        this.el.statusText.textContent = message;
        this.el.statusText.classList.toggle('error', Boolean(isError));
    }
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
