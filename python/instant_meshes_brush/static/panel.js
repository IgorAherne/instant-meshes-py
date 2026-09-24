/*
    panel.js: the viewer's own control panel.

    Everything the user can press lives inside the viewer document, so embedding
    the viewport is a single <iframe> and a host page reproduces none of this.
    The panel owns no application state: it reads the DOM, calls back into the
    app, and renders whatever the app tells it to show.
*/

/** Formats the upload route accepts; mirrored from server.MESH_SUFFIXES. */
export const MESH_ACCEPT = '.obj,.ply,.stl,.off,.glb,.gltf,.dae,.fbx';

/** What travels with a model: its material library, its buffers, its texture
 *  maps -- or the whole lot as one zip. */
export const COMPANION_ACCEPT =
    '.mtl,.bin,.png,.jpg,.jpeg,.tga,.tif,.tiff,.bmp,.webp,.dds,.psd,.zip';

/** Every file the import takes, for filtering a dropped folder. */
const IMPORT_SUFFIXES = new Set(`${MESH_ACCEPT},${COMPANION_ACCEPT}`.split(','));

/** A file's lower-case suffix, with its dot; '' when it has none. */
export function suffixOf(name) {
    const dot = name.lastIndexOf('.');
    return dot > 0 ? name.slice(dot).toLowerCase() : '';
}

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

/**
 * How long the UV slider must sit still before the atlas is re-cut.
 *
 * Much shorter than CONFIG_SETTLE_MS because unwrapping is cheap next to a
 * rebuild and the layout is the whole feedback for this control: every
 * millisecond here is a millisecond of a slider that looks like it did
 * nothing. Long enough only to keep a drag from queuing one cut per pixel.
 */
const UV_SETTLE_MS = 120;

/** How long an error holds the status line before the tool hint returns. */
const ERROR_LINGER_MS = 6000;

/** How long a confirmation holds it: long enough to be read, and no more. */
const NOTE_LINGER_MS = 4000;

/** The Export button's help when it hands the result to the page around the
 *  viewer instead of writing a file. */
const HOST_EXPORT_HINT =
    'Hand this result, with the settings above, to the program showing this ' +
    'viewer; it decides where the mesh goes.';

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

/* ------------------------------------------------------------------ */
/*  Texture maps                                                       */
/* ------------------------------------------------------------------ */

/**
 * The maps a model can carry, in the order their buttons appear.  Anything
 * the server could not name ("tex 3") follows, in slot order.
 */
const MAP_ORDER = [
    'basecolor', 'opacity', 'normal', 'roughness', 'metallic', 'ao', 'emissive', 'height',
];

/** Below this, the guess about what a map is gets a "?" on its button. */
const UNSURE_BELOW = 0.6;

/**
 * Which of the server's texture guesses feed which button, for guesses that
 * do not list their channels.  A packed ORM feeds three, a spec/gloss map is
 * converted into colour, metalness and roughness.
 */
const BUTTON_ROLES = {
    basecolor: ['basecolor', 'specular', 'spec_gloss'],
    opacity: ['opacity'],
    normal: ['normal'],
    roughness: ['roughness', 'gloss', 'orm', 'metal_smooth', 'mask_hdrp', 'spec_gloss'],
    metallic: ['metallic', 'orm', 'metal_smooth', 'mask_hdrp', 'specular', 'spec_gloss'],
    ao: ['ao', 'orm', 'mask_hdrp'],
    emissive: ['emissive'],
    height: ['height'],
};

const CHANNEL_NAMES = { r: 'red', g: 'green', b: 'blue', a: 'alpha' };

/** Files and reasons listed per button in its hint, at most. */
const HINT_FILES = 3;
const HINT_REASONS = 3;

/**
 * The client-side corrections a map button can carry: preview only, the
 * files themselves are never touched.
 */
const MAP_OPTIONS = {
    normal: {
        option: 'flipGreen',
        label: 'flip G',
        hint: 'Flip the green channel. For a DirectX-style normal map, whose bumps ' +
            'look pushed in and whose light comes from the wrong side.\n\n' +
            'Changes this preview only.',
    },
    roughness: {
        option: 'invertRoughness',
        label: 'invert',
        hint: 'Invert the map. For a gloss (smoothness) map that was read as ' +
            'roughness: what should shine looks matt and the other way round.\n\n' +
            'Changes this preview only.',
    },
};

const NONE_HINT =
    'Take the maps off and show the surface the field grid is drawn on.';
const ALL_HINT =
    'The model as a game engine would render it: every map together, lit by a ' +
    'studio environment and a light at the camera.';
const MAP_HINT =
    'Shows this one map on the model, unlit, exactly as the file stores it. ' +
    'None goes back to the field grid.';

/** Whether a guess about one of the model's files feeds `button`. */
function feeds(guess, button) {
    if (guess.channels) return button.id in guess.channels;
    return (BUTTON_ROLES[button.id] || []).includes(guess.role);
}

function percent(confidence) {
    return `${Math.round(Number(confidence) * 100)}%`;
}

/**
 * Describe where a button's map came from, for its hover help, and how sure
 * the server was of what it is.
 *
 * The server says, per material, which file it took for what and why; a
 * button gathers the guesses that feed it -- a guess that lists its
 * channels names every button it feeds, "Specular" (tex 0) included.
 * Without that list, a map the server could not name ("tex 3") is whatever
 * file sits in that slot, so it collects the files of every material that
 * has one there.
 *
 * @returns {{hint: string, confidence: number|null, unsure: boolean}}
 */
function describeMap(button, materials) {
    const known = button.id in BUTTON_ROLES;
    const seen = new Set();
    const guesses = [];
    for (const material of materials) {
        const hasSlot = Object.values(material.maps || {}).includes(button.slot);
        for (const guess of material.guesses || []) {
            const ours = known || guess.channels
                ? feeds(guess, button)
                : hasSlot && guess.role === 'other';
            if (!ours || seen.has(guess.file)) continue;
            seen.add(guess.file);
            guesses.push(guess);
        }
    }

    const channel = CHANNEL_NAMES[button.channel];
    const lines = [channel ? `${button.label}: the ${channel} channel of` : `${button.label}:`];
    for (const guess of guesses.slice(0, HINT_FILES)) {
        lines.push(`${guess.file} -- ${percent(guess.confidence)} sure`);
        for (const reason of (guess.evidence || []).slice(0, HINT_REASONS)) {
            lines.push(`- ${reason}`);
        }
        if (button.id === 'normal' && guess.y_convention) {
            const style = guess.y_convention === 'directx'
                ? 'DirectX (green down), turned into OpenGL'
                : 'OpenGL (green up)';
            lines.push(`- read as ${style}${guess.y_confident ? '' : ', a guess'}`);
        }
    }
    if (guesses.length > HINT_FILES) {
        lines.push(`...and ${guesses.length - HINT_FILES} more files`);
    }
    if (guesses.length === 0) lines.push('(no details from the file)');
    lines.push('', MAP_HINT);

    const confidences = guesses.map((guess) => Number(guess.confidence));
    const confidence = confidences.length ? Math.min(...confidences) : null;
    return {
        hint: lines.join('\n'),
        confidence,
        unsure: confidence !== null && confidence < UNSURE_BELOW,
    };
}

/**
 * The map buttons as the panel shows them: in panel order -- the named maps
 * first, then the unnamed ones by slot -- each with its hover help and
 * whether it earns a "?".
 *
 * @param {Array<object>} buttons    the /source header's `buttons`
 * @param {Array<object>} materials  the header's `materials`
 * @returns {Array<object>} each button plus {hint, confidence, unsure}
 */
export function describeButtons(buttons, materials) {
    const rank = (button) => {
        const index = MAP_ORDER.indexOf(button.id);
        return index >= 0 ? index : MAP_ORDER.length + Number(button.slot);
    };
    return [...buttons]
        .sort((a, b) => rank(a) - rank(b))
        .map((button) => ({ ...button, ...describeMap(button, materials) }));
}

/**
 * Every file under some dropped entries that the import can use.
 *
 * A folder is walked to the bottom -- `readEntries` answers in batches, so it
 * is called until it returns none -- and whatever the import has no use for
 * (a .blend, notes) is left behind.
 *
 * @param {Array<FileSystemEntry>} entries
 * @returns {Promise<File[]>}
 */
export async function importableFiles(entries) {
    const found = (await Promise.all(entries.map(filesUnder))).flat();
    return found.filter((file) => IMPORT_SUFFIXES.has(suffixOf(file.name)));
}

async function filesUnder(entry) {
    if (entry.isFile) {
        return [await new Promise((resolve, reject) => entry.file(resolve, reject))];
    }
    const reader = entry.createReader();
    const files = [];
    for (;;) {
        const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        if (batch.length === 0) return files;
        for (const child of batch) files.push(...(await filesUnder(child)));
    }
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

        const hide = () => {
            this.element.hidden = true;
            this._current = null;
        };
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
        /* No related target: the pointer left the document altogether -- into
           the host page around the viewer, say -- and the hint goes with it
           rather than staying up over the viewport. */
        document.addEventListener('pointerout', (event) => {
            if (!event.relatedTarget) hide();
        });
        document.addEventListener('pointerdown', hide);
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

            strokeCount: byId('stroke-count'),
            clear: byId('btn-clear'),
            singularityCount: byId('singularity-count'),

            textureSlots: byId('texture-slots'),

            uvZone: byId('uv-zone'),
            uvLeniency: byId('uv-leniency'),
            uvLabel: byId('uv-label'),
            uvCharts: byId('uv-charts'),

            outputStats: byId('output-stats'),
            format: byId('format'),
            smoothing: byId('smoothing'),
            export: byId('btn-export'),
            download: byId('download'),

            toolName: byId('tool-name'),
            statusText: byId('status-text'),
        };

        this.el.file.setAttribute('accept', `${MESH_ACCEPT},${COMPANION_ACCEPT}`);
        this.hints = new Hints(byId('tip'));

        /* The config as the server last reported it, which is what an edit is
           compared against.  Null until a mesh exists: there is nothing to
           rebuild before then, so nothing to apply either. */
        this._appliedConfig = null;
        this._settleTimer = 0;
        this._uvTimer = 0;

        /* What the texture row was last built from, which view of the maps
           is pressed ('none', 'all' or a map's id), and the preview-only
           corrections on the normal and roughness buttons. */
        this._textureSignature = '';
        this._textureView = 'none';
        this._mapOptions = { flipGreen: false, invertRoughness: false };

        /* The one status line shows the selected brush's help by default and
           borrows the space for a message, so both have to be remembered. */
        this._hint = this.el.statusText.textContent;
        this._message = null;
        this._isError = false;
        this._statusTimer = 0;

        /* What the Export button says, which a host page may rename, and
           whether the page it hands results to is still busy with one. */
        this._exportLabel = this.el.export.textContent.trim();
        this._exportBusy = false;

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
            const files = [...(this.el.file.files || [])];
            /* Clearing the value lets the same file be chosen twice running. */
            this.el.file.value = '';
            if (files.length) call('onOpenFiles', files);
        });
        this._bindDrop(on, call);

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
        on(this.el.symmetry, 'change', () => this._settleConfig());

        /* Every one of these four is baked in by preprocess, so each is a
           rebuild -- including Force Quads, which decides what the target
           vertex count has to be aimed at. */
        for (const button of document.querySelectorAll('button.toggle')) {
            on(button, 'click', () => {
                this.setOption(button.dataset.opt, !this.option(button.dataset.opt));
                this._settleConfig();
            });
        }

        for (const button of document.querySelectorAll('button.tool')) {
            on(button, 'click', () => call('onSelectTool', button.dataset.tool));
        }
        on(this.el.clear, 'click', () => call('onClearStrokes'));

        for (const button of document.querySelectorAll('button.seg[data-surface]')) {
            on(button, 'click', () => call('onSurfaceChange', button.dataset.surface));
        }

        /* Delegated, because these buttons are built when a model arrives and
           replaced when the next one does. */
        on(this.el.textureSlots, 'click', (event) => {
            const toggle = event.target.closest('button.texture-option');
            if (toggle) {
                const name = toggle.dataset.option;
                this._mapOptions = { ...this._mapOptions, [name]: !this._mapOptions[name] };
                toggle.setAttribute('aria-pressed', String(this._mapOptions[name]));
                call('onMapOptionsChange', this.mapOptions());
                return;
            }
            const button = event.target.closest('button.texture');
            if (button) call('onTextureChange', button.dataset.view);
        });

        /* Read at extraction time, so changing it makes the result on screen
           stale without needing a rebuild. `change` rather than `input`, so a
           typed count is not re-extracted once per keystroke. */
        on(this.el.smoothing, 'change', () =>
            call('onExtractOptionsChange', this.extractOptions())
        );

        /* The chart size travels with the export as well as with a preview:
           moving the slider and pressing Export without ever hovering would
           otherwise write the layout from whatever was last looked at. */
        on(this.el.export, 'click', () =>
            call('onExport', {
                format: this.el.format.value,
                leniency: this.uvLeniency(),
                ...this.extractOptions(),
            })
        );

        for (const input of document.querySelectorAll('input[data-layer]')) {
            on(input, 'change', () =>
                call('onLayerToggle', input.dataset.layer, input.checked)
            );
        }

        /* A pointer press must not leave the keyboard behind on the control it
           pressed. The view keys belong to the viewport, and a button or a
           slider that keeps the focus both swallows them and holds a focus
           ring nobody asked for. Tabbing is untouched: it is not a press, and
           the text boxes are left alone so typing can continue. */
        on(document, 'pointerup', (event) => {
            const control =
                event.target.closest && event.target.closest('button, input[type="range"]');
            if (control && control === document.activeElement) control.blur();
        });

        this._bindUv(on, call);
    }

    /**
     * The UV control, which selects the view that shows what it does.
     *
     * A flattened mesh cannot be judged from a number, so touching the slider
     * switches to Result UV: reaching for this control is the request to see
     * a layout. Merely passing over it is not, which is why this is a press
     * and not a hover.
     */
    _bindUv(on, call) {
        on(this.el.uvZone, 'pointerdown', () => call('onSurfaceChange', 'uv'));
        on(this.el.uvLeniency, 'input', () => {
            clearTimeout(this._uvTimer);
            this._uvTimer = setTimeout(
                () => call('onUvChange', this.uvLeniency()),
                UV_SETTLE_MS
            );
        });
    }

    /**
     * Take a model dropped anywhere on the viewer: files, or a whole folder.
     *
     * A folder is how a model usually arrives -- the FBX or OBJ beside its
     * textures, often in a subfolder of their own -- so it is walked and
     * everything the import can use is sent.  The entries have to be taken
     * while the drop event is still running, which is why they are read
     * before the first await.
     */
    _bindDrop(on, call) {
        const stage = byId('stage');
        const carriesFiles = (event) =>
            Boolean(event.dataTransfer) && [...event.dataTransfer.types].includes('Files');

        on(document, 'dragover', (event) => {
            if (!carriesFiles(event)) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = 'copy';
            stage.classList.add('dropping');
        });
        on(document, 'dragleave', (event) => {
            if (!event.relatedTarget) stage.classList.remove('dropping');
        });
        on(document, 'drop', async (event) => {
            if (!carriesFiles(event)) return;
            event.preventDefault();
            stage.classList.remove('dropping');

            const entries = [...event.dataTransfer.items]
                .map((item) => item.webkitGetAsEntry && item.webkitGetAsEntry())
                .filter(Boolean);
            const loose = [...event.dataTransfer.files];
            try {
                const files = entries.length
                    ? await importableFiles(entries)
                    : loose.filter((file) => IMPORT_SUFFIXES.has(suffixOf(file.name)));
                if (files.length) {
                    call('onOpenFiles', files);
                } else {
                    this.setStatus('Nothing in that drop can be imported: drop a 3D model ' +
                        '(FBX, OBJ, glTF, GLB...) with its textures, or a folder or zip ' +
                        'holding one', true);
                }
            } catch (err) {
                this.setStatus(`Could not read the dropped files: ${err.message}`, true);
            }
        });
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
            this._settleTimer = 0;
            const config = this.readConfig();
            if (!this._appliedConfig || sameConfig(config, this._appliedConfig)) return;
            this._appliedConfig = config;
            if (this.handlers.onConfigChange) this.handlers.onConfigChange(config);
        }, CONFIG_SETTLE_MS);
    }

    /**
     * Whether an edited setting is still waiting to be sent.
     *
     * The viewport asks before building a result: a mesh half a second from
     * being rebuilt is not worth extracting, and the work would land just in
     * time to be thrown away.
     */
    configPending() {
        return Boolean(this._settleTimer);
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
            extrinsic: this.option('extrinsic'),
            align_to_boundaries: this.option('boundaries'),
            crease_angle: this.option('creases') ? CREASE_ANGLE : -1.0,
            // Goes through preprocess rather than through extract, because the
            // pure quad step subdivides afterwards: the target vertex count
            // has to be aimed at a quarter of itself to still come true.
            pure_quad: this.option('pure-quad'),
        };
    }

    /**
     * Every setting in the panel that shapes the output, for a host page that
     * takes the result from here: it can keep them with the mesh, and open
     * the viewer with them next time.
     *
     * Named for what the controls say rather than for the solver's Config,
     * and read from the controls rather than from the server, because this
     * is what the user pressed the button on. The one derived value is
     * `crease_angle`, which is what "Sharp Creases" means to the solver.
     */
    exportSettings() {
        const config = this.readConfig();
        return {
            target_vertices: config.vertex_count,
            pure_quad: config.pure_quad,
            symmetry: { rosy: config.rosy, posy: config.posy },
            smoothing: this.extractOptions().smooth_iter,
            extrinsic: config.extrinsic,
            boundaries: config.align_to_boundaries,
            creases: this.option('creases'),
            crease_angle: config.crease_angle,
            format: this.el.format.value,
            uv_leniency: this.uvLeniency(),
        };
    }

    /** Which view the segmented control currently has selected. */
    surface() {
        const chosen = document.querySelector('button.seg[aria-checked="true"]');
        return chosen ? chosen.dataset.surface : 'mesh';
    }

    /** The one setting extract() reads, which needs no rebuild. */
    extractOptions() {
        return { smooth_iter: Math.max(0, Number(this.el.smoothing.value) || 0) };
    }

    _toggle(name) {
        return document.querySelector(`button.toggle[data-opt="${name}"]`);
    }

    option(name) {
        const button = this._toggle(name);
        return Boolean(button && button.getAttribute('aria-pressed') === 'true');
    }

    setOption(name, on) {
        const button = this._toggle(name);
        if (button) button.setAttribute('aria-pressed', String(Boolean(on)));
    }

    layerState(name) {
        const input = document.querySelector(`input[data-layer="${name}"]`);
        return Boolean(input && input.checked);
    }

    /** Chart-size leniency in [0, 1], which is what the server's mapping takes. */
    uvLeniency() {
        const raw = Number(this.el.uvLeniency.value);
        const span = Number(this.el.uvLeniency.max) || 100;
        return Math.min(1, Math.max(0, (Number.isFinite(raw) ? raw : span / 2) / span));
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
     *
     * So is every control, while an edit is still waiting to be sent. A status
     * frame carries the config in force, which during those few hundred
     * milliseconds is the one the edit is replacing: writing it back reverted
     * the new value and then found nothing to apply, so a target typed while
     * the solver happened to be busy silently did nothing at all.
     */
    showConfig(config) {
        if (!config || this.configPending()) return;
        const editing = document.activeElement;
        const settable = (element) => element !== editing;

        const key = `${config.rosy},${config.posy}`;
        if (key in SYMMETRIES && settable(this.el.symmetry)) this.el.symmetry.value = key;
        if (config.vertex_count > 0 && settable(this.el.target)
            && settable(this.el.targetRange)) {
            this.el.target.value = String(clampTarget(config.vertex_count));
            this.el.targetRange.value = targetToSlider(config.vertex_count);
        }
        this.setOption('extrinsic', config.extrinsic);
        this.setOption('boundaries', config.align_to_boundaries);
        this.setOption('creases', Number(config.crease_angle) >= 0);
        this.setOption('pure-quad', config.pure_quad);

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

    /** How many markers are currently on the model, or null for none known. */
    showSingularities(count) {
        const line = this.el.singularityCount;
        if (count === null || count === undefined) {
            line.hidden = true;
            return;
        }
        line.hidden = false;
        line.textContent = `${count.toLocaleString()} singularities`;
    }

    /** Hide the UV control outright where xatlas is not installed. */
    setUvAvailable(available) {
        this.el.uvZone.hidden = !available;
    }

    /**
     * Offer the ways of looking at the maps the imported file carried.
     *
     * None, All, then one button per kind of map -- Base colour, Normal,
     * Roughness... -- whichever material it came from; a packed map gives one
     * button per channel it packs.  Rebuilt whenever the list changes, not
     * just its length, because it is a property of the model: an OBJ with no
     * maps has no row at all.  None starts pressed -- the model arrives shaded
     * the way every other model in here is, and a map is something you ask for.
     *
     * @param {Array<{id: string, label: string, slot: number, channel: string}>} buttons
     *        the /source header's `buttons`
     * @param {Array<object>} materials  the header's `materials`, whose
     *        guesses say which file each map came from and why
     */
    showTextures(buttons, materials = []) {
        const described = describeButtons(buttons || [], materials);
        const signature = JSON.stringify(described);
        if (signature === this._textureSignature) return;
        this._textureSignature = signature;

        const row = this.el.textureSlots;
        row.textContent = '';
        row.hidden = described.length === 0;
        this._textureView = 'none';
        this._mapOptions = { flipGreen: false, invertRoughness: false };
        if (described.length === 0) return;

        row.append(
            this._textureButton('none', 'None', NONE_HINT),
            this._textureButton('all', 'All', ALL_HINT)
        );
        for (const map of described) {
            const button = this._textureButton(map.id, map.label, map.hint);
            if (map.unsure) {
                const mark = document.createElement('span');
                mark.className = 'unsure';
                mark.textContent = '?';
                button.append(mark);
                button.setAttribute('aria-label', `${map.label} (uncertain)`);
            }
            const extra = MAP_OPTIONS[map.id];
            if (!extra) {
                row.append(button);
                continue;
            }
            /* The correction sits against its map, not in a menu: the moment
               a normal map looks inside out is the moment to press it. */
            const toggle = document.createElement('button');
            toggle.className = 'texture-option';
            toggle.dataset.option = extra.option;
            toggle.setAttribute('aria-pressed', 'false');
            toggle.textContent = extra.label;
            toggle.dataset.hint = extra.hint;
            const pair = document.createElement('span');
            pair.className = 'texture-pair';
            pair.append(button, toggle);
            row.append(pair);
        }
    }

    _textureButton(view, label, hint) {
        const button = document.createElement('button');
        button.className = 'texture';
        button.dataset.view = view;
        button.setAttribute('role', 'radio');
        button.setAttribute('aria-checked', String(view === this._textureView));
        button.textContent = label;
        button.dataset.hint = hint;
        return button;
    }

    /** Which view of the maps is pressed: 'none', 'all' or a map's id. */
    textureView() {
        return this._textureView;
    }

    /** Press one view of the maps. @returns {boolean} true on a change */
    setTextureView(view) {
        const wanted = view || 'none';
        if (wanted === this._textureView) return false;
        this._textureView = wanted;
        for (const button of this.el.textureSlots.querySelectorAll('button.texture')) {
            button.setAttribute('aria-checked', String(button.dataset.view === wanted));
        }
        return true;
    }

    /** The preview-only corrections: {flipGreen, invertRoughness}. */
    mapOptions() {
        return { ...this._mapOptions };
    }

    /**
     * Report a layout the server has cut.
     *
     * The chart count is the number the slider is really setting -- "fewer,
     * larger chunks" is a count -- so it reads back beside it.
     */
    showUv(layout) {
        if (!layout) {
            this.el.uvCharts.textContent = '';
            return;
        }
        const charts = Number(layout.charts) || 0;
        this.el.uvCharts.textContent = `${charts.toLocaleString()}`;
        if (layout.leniency !== undefined && this.el.uvLeniency !== document.activeElement) {
            const span = Number(this.el.uvLeniency.max) || 100;
            this.el.uvLeniency.value = String(Math.round(layout.leniency * span));
        }
    }

    /**
     * How far along the unwrapper is, or null when it is not running.
     *
     * It borrows the control's own label rather than the status line, which
     * belongs to the brush: the wait is this slider's, and so is the place
     * that reports it.
     */
    showUnwrapping(progress) {
        this.el.uvLabel.textContent =
            progress === null || progress === undefined
                ? 'UV chunks'
                : `UV chunks (unwrapping ${Math.round(progress * 100)}%)`;
    }

    /** The extracted mesh's size, or null once it is stale.
     *
     * Vertices only: that is the number the target above is expressed in, so
     * it is the one worth checking the result against.
     */
    showOutput(counts) {
        const line = this.el.outputStats;
        if (!counts) {
            line.hidden = true;
            return;
        }
        line.hidden = false;
        line.textContent = `${counts.vertices.toLocaleString()} vertexes`;
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

    /**
     * Rename the Export button, and say what it does in host mode.
     *
     * @param {{exportLabel: string, exportAction: string}} options  from
     *        options.js; the defaults leave the button exactly as it was
     */
    useExportOptions({ exportLabel, exportAction }) {
        this._exportLabel = exportLabel;
        this.el.export.textContent = exportLabel;
        if (exportAction === 'host') this.el.export.dataset.hint = HOST_EXPORT_HINT;
    }

    /**
     * Show what the host page is doing with a result it was handed.
     *
     * busy: the button spins, says "<label>...", and cannot be pressed again
     * until the page is done. done: the button comes back and the status line
     * confirms for a moment. error: the button comes back and the page's
     * message goes on the status line as an error, which stays until read.
     *
     * @param {'busy'|'done'|'error'} state
     * @param {string} [message]  the page's own words, if it sent any
     */
    setExportState(state, message) {
        const busy = state === 'busy';
        this._exportBusy = busy;
        this.el.export.classList.toggle('busy', busy);
        this.el.export.setAttribute('aria-busy', String(busy));
        this.el.export.textContent = busy ? `${this._exportLabel}…` : this._exportLabel;
        this.setSolving(Boolean(this._solving));

        if (state === 'done') {
            this.setStatus(message || `${this._exportLabel}: done`, false);
            /* A confirmation is not worth keeping the brush's help away for.
               Anything said after it cancels this along with it. */
            this._statusTimer = setTimeout(() => this.setStatus(null), NOTE_LINGER_MS);
        } else if (state === 'error') {
            this.setStatus(message || `${this._exportLabel} failed`, true);
        } else if (busy && message) {
            this.setStatus(message, false);
        }
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
        this.el.export.disabled = active || !this._ready || this._exportBusy;
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
