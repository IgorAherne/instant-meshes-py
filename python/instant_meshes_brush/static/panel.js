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

/** Output resolutions the panel offers, in vertices. */
const TARGET_MIN = 500;
const TARGET_MAX = 60000;

/** Positions on the target slider's own track. */
const SLIDER_STEPS = 1000;

function clampTarget(value) {
    const rounded = Math.round(Number(value));
    if (!Number.isFinite(rounded)) return TARGET_MIN;
    return Math.min(TARGET_MAX, Math.max(TARGET_MIN, rounded));
}

/**
 * The target slider is logarithmic.
 *
 * Its range spans two orders of magnitude, and the settings people actually
 * reach for sit in the bottom tenth of it; on a linear track the difference
 * between 800 and 3,000 vertices would be three pixels.
 */
function sliderToTarget(position) {
    const t = Math.min(1, Math.max(0, Number(position) / SLIDER_STEPS));
    const value = TARGET_MIN * Math.pow(TARGET_MAX / TARGET_MIN, t);
    return clampTarget(Math.round(value / 10) * 10);
}

function targetToSlider(value) {
    const t = Math.log(clampTarget(value) / TARGET_MIN) / Math.log(TARGET_MAX / TARGET_MIN);
    return String(Math.round(t * SLIDER_STEPS));
}

/**
 * How long the remeshing settings must sit still before they are applied.
 *
 * There is no Apply button: a rebuild is cheap enough to simply follow the
 * controls.  The delay is what keeps dragging the target slider from queuing
 * one rebuild per pixel, and what lets a typed vertex count be finished.
 */
const CONFIG_SETTLE_MS = 500;

/** How long an error holds the status line before the tool hint returns. */
const ERROR_LINGER_MS = 6000;

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
            meshLine: byId('mesh-line'),
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

            outputStats: byId('output-stats'),
            format: byId('format'),
            pureQuad: byId('opt-pure-quad'),
            smoothing: byId('smoothing'),
            export: byId('btn-export'),
            download: byId('download'),

            toolName: byId('tool-name'),
            statusText: byId('status-text'),
        };

        this.el.file.setAttribute('accept', MESH_ACCEPT);
        this.hints = new Hints(byId('tip'));

        /* The config as the server last reported it, which is what an edit is
           compared against.  Null until a mesh exists: there is nothing to
           rebuild before then, so nothing to apply either. */
        this._appliedConfig = null;
        this._settleTimer = 0;

        /* The one status line shows the selected brush's help by default and
           borrows the space for a message, so both have to be remembered. */
        this._hint = this.el.statusText.textContent;
        this._message = null;
        this._isError = false;
        this._statusTimer = 0;

        this.el.targetRange.value = targetToSlider(this.el.target.value);
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

        /* The slider and the number box are two views of one value, but not of
           one scale: only the box is in vertices.  Neither writes back into the
           control being used, so a half-typed number is never rewritten. */
        on(this.el.targetRange, 'input', () => {
            this.el.target.value = String(sliderToTarget(this.el.targetRange.value));
            this._settleConfig();
        });
        on(this.el.target, 'input', () => {
            this.el.targetRange.value = targetToSlider(this.el.target.value);
            this._settleConfig();
        });
        on(this.el.target, 'blur', () => {
            /* Show what was actually applied. Clamping here rather than on every
               keystroke means a number is never rewritten under the cursor, but
               a rejected one must not be left on screen looking accepted. */
            this.el.target.value = String(clampTarget(this.el.target.value));
        });
        for (const key of ['symmetry', 'extrinsic', 'boundaries', 'creases']) {
            on(this.el[key], 'change', () => this._settleConfig());
        }

        for (const button of document.querySelectorAll('button.tool')) {
            on(button, 'click', () => call('onSelectTool', button.dataset.tool));
        }
        on(this.el.clear, 'click', () => call('onClearStrokes'));

        for (const button of document.querySelectorAll('button.seg[data-surface]')) {
            on(button, 'click', () => call('onSurfaceChange', button.dataset.surface));
        }

        /* These two are read at extraction time, so changing one makes the
           result on screen stale. `change` rather than `input`, so a typed
           smoothing count is not re-extracted once per keystroke. */
        for (const key of ['pureQuad', 'smoothing']) {
            on(this.el[key], 'change', () =>
                call('onExtractOptionsChange', this.extractOptions())
            );
        }

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
            vertex_count: clampTarget(this.el.target.value),
            extrinsic: this.el.extrinsic.checked,
            align_to_boundaries: this.el.boundaries.checked,
            crease_angle: this.el.creases.checked ? CREASE_ANGLE : -1.0,
        };
    }

    /** Which surface the Show toggle currently has selected. */
    surface() {
        const chosen = document.querySelector('button.seg[aria-checked="true"]');
        return chosen ? chosen.dataset.surface : 'mesh';
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

    /**
     * Select one of the two surfaces.
     *
     * @returns {boolean} true if this was a change, so a caller can skip the
     *          GPU work of re-applying what is already on screen.
     */
    setSurface(which) {
        if (this.surface() === which) return false;
        for (const button of document.querySelectorAll('button.seg[data-surface]')) {
            button.setAttribute('aria-checked', String(button.dataset.surface === which));
        }
        /* The grid is painted by the input surface's own material, so while the
           output is showing its checkbox could only lie about what it does. */
        const grid = document.querySelector('input[data-layer="grid"]');
        if (grid) grid.disabled = which !== 'mesh';
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
            this.el.target.value = String(clampTarget(config.vertex_count));
            this.el.targetRange.value = targetToSlider(config.vertex_count);
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

    showMesh({ name, vertices }) {
        this.el.meshLine.hidden = false;
        this.el.meshName.textContent = name || 'Untitled mesh';
        this.el.meshName.title = name || '';
        this.el.meshStats.textContent = `${vertices.toLocaleString()} v`;
    }

    showStrokes(count) {
        this.el.strokeCount.textContent =
            count === 0 ? 'No strokes' : `${count} stroke${count === 1 ? '' : 's'}`;
        this.el.clear.disabled = count === 0;
    }

    /** The extracted mesh's size, or null once it is stale. */
    showOutput(counts) {
        const line = this.el.outputStats;
        if (!counts) {
            line.hidden = true;
            return;
        }
        line.hidden = false;
        line.textContent =
            `${counts.vertices.toLocaleString()} v /` +
            ` ${counts.faces.toLocaleString()} faces`;
    }

    /**
     * Start the download of an exported file.
     *
     * Pressing Export is the whole request; a green button that then has to be
     * pressed a second time is a step, not a confirmation. The link exists
     * only to be clicked from here, which is what carries the filename the
     * server chose through to the browser's downloads.
     */
    startDownload(url, filename) {
        if (!url) return;
        const link = this.el.download;
        link.href = url;
        link.download = filename || 'mesh';
        link.click();
    }

    showTool(tool) {
        for (const button of document.querySelectorAll('button.tool')) {
            button.setAttribute('aria-checked', String(button.dataset.tool === tool.id));
        }
        this.el.toolName.textContent = tool.label;
        this._hint = tool.hint;
        /* A message from the previous tool is not about this one. */
        this.setStatus(null);
        this._renderStatus();
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
     *  Importing a mesh stays available: it stops the solve on its way in, and
     *  being unable to abandon a long solve by loading something else would be
     *  the one place this UI could strand somebody. */
    setSolving(active) {
        this._solving = active;
        this.el.export.disabled = active || !this._ready;
    }

    setReady(ready) {
        this._ready = ready;
        this.setSolving(Boolean(this._solving));
    }

    /**
     * Put a message on the status line, or pass null to go back to the hint.
     *
     * The line is shared: normally it explains what the selected brush does,
     * which is what somebody staring at the viewport actually needs, and a
     * message borrows it for as long as it is worth reading. An error is never
     * silently cleared by a passing "it went idle" -- it times out instead, so
     * it cannot be gone before it was seen.
     */
    setStatus(message, isError) {
        if (!message && this._isError) return;

        clearTimeout(this._statusTimer);
        this._message = message || null;
        this._isError = Boolean(message && isError);
        if (this._isError) {
            this._statusTimer = setTimeout(() => {
                this._isError = false;
                this.setStatus(null);
            }, ERROR_LINGER_MS);
        }
        this._renderStatus();
    }

    _renderStatus() {
        this.el.statusText.textContent = this._message || this._hint;
        this.el.statusText.classList.toggle('error', this._isError);
    }
}
