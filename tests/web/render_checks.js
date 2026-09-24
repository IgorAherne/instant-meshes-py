/*
    render_checks.js: pixel-exact checks of the viewer's texture views.

    Everything runs through the shipped renderer.js -- its loadTexture, its
    inspect and lit materials, its environment and key light -- on a flat
    square facing the camera, and reads the result back with readPixels.
    Test images are encoded here as PNG bytes, so every stored value is
    known exactly and no image file has to be kept in the repository.
*/

import * as THREE from 'three';
import { Viewer, loadTexture } from '../../python/instant_meshes_brush/static/renderer.js';

/* ------------------------------------------------------------------ */
/*  PNG bytes, uncompressed                                            */
/* ------------------------------------------------------------------ */

const CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; ++n) {
        let c = n;
        for (let k = 0; k < 8; ++k) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        table[n] = c >>> 0;
    }
    return table;
})();

function crc32(bytes) {
    let c = 0xffffffff;
    for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
}

function adler32(bytes) {
    let a = 1;
    let b = 0;
    for (const x of bytes) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    return ((b << 16) | a) >>> 0;
}

function chunk(type, data) {
    const out = new Uint8Array(12 + data.length);
    const view = new DataView(out.buffer);
    view.setUint32(0, data.length);
    out.set([...type].map((ch) => ch.charCodeAt(0)), 4);
    out.set(data, 8);
    view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
    return out;
}

/** An 8-bit RGBA PNG of `rows` (top row first), each row [r,g,b,a, ...]. */
function png(width, rows) {
    const raw = new Uint8Array(rows.length * (1 + width * 4));
    rows.forEach((row, r) => raw.set(row, r * (1 + width * 4) + 1));
    /* zlib, one stored (uncompressed) deflate block: small images only. */
    const z = new Uint8Array(2 + 5 + raw.length + 4);
    const zv = new DataView(z.buffer);
    z.set([0x78, 0x01, 0x01, raw.length & 0xff, raw.length >> 8,
        ~raw.length & 0xff, (~raw.length >> 8) & 0xff]);
    z.set(raw, 7);
    zv.setUint32(7 + raw.length, adler32(raw));

    const header = new Uint8Array(13);
    const hv = new DataView(header.buffer);
    hv.setUint32(0, width);
    hv.setUint32(4, rows.length);
    header.set([8, 6, 0, 0, 0], 8);
    const signature = [137, 80, 78, 71, 13, 10, 26, 10];
    return new Blob([new Uint8Array(signature), chunk('IHDR', header), chunk('IDAT', z),
        chunk('IEND', new Uint8Array(0))], { type: 'image/png' });
}

/** A 4x4 PNG of one colour. */
function flat(r, g, b, a = 255) {
    const row = Array.from({ length: 4 }, () => [r, g, b, a]).flat();
    return png(4, [row, row, row, row]);
}

/* ------------------------------------------------------------------ */
/*  Stage                                                              */
/* ------------------------------------------------------------------ */

const viewer = new Viewer(document.getElementById('view'), document.getElementById('overlay'));
/* Frames are drawn here, one at a time, so nothing may draw in between. */
cancelAnimationFrame(viewer._raf);
viewer._applyResize();
const gl = viewer.renderer.getContext();

const shaderErrors = [];
viewer.renderer.debug.onShaderError = (glContext, program, vertex, fragment) => {
    shaderErrors.push(glContext.getProgramInfoLog(program) || 'shader error');
};

/* The unit square facing +Z, UVs v-up like every model the server sends. */
viewer.setSourceMesh({
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uv: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    groups: [[0, 0, 2]],
    materials: 1,
});
viewer.setSourceVisible(true);
viewer.camera.position.set(0, 0, 3);
viewer.camera.lookAt(0, 0, 0);
viewer.camera.updateMatrixWorld();

/** Render, then the RGB under a point of the square (x, y in [-1, 1]). */
function readAt(x = 0, y = 0) {
    viewer.renderer.render(viewer.scene, viewer.camera);
    const p = new THREE.Vector3(x, y, 0).project(viewer.camera);
    const px = Math.round((p.x * 0.5 + 0.5) * gl.drawingBufferWidth);
    const py = Math.round((p.y * 0.5 + 0.5) * gl.drawingBufferHeight);
    const out = new Uint8Array(4);
    gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
    return [out[0], out[1], out[2]];
}

const INSPECT = { channel: 'rgb', flipGreen: false, invert: false };
const LIT = { flipGreen: false, invertRoughness: false, occlusion: false };
const PLAIN = { base_color_factor: [1, 1, 1, 1], metallic: 0, roughness: 1 };
const NO_MAPS = { basecolor: null, normal: null, orm: null, emissive: null };

async function inspect(blob, view) {
    const texture = await loadTexture(blob, 'linear');
    viewer.setSourceInspect([texture], { ...INSPECT, ...view });
    const rgb = readAt();
    texture.dispose();
    return rgb;
}

/* ------------------------------------------------------------------ */
/*  Checks                                                             */
/* ------------------------------------------------------------------ */

const checks = [];

function check(name, measured, expected, pass) {
    checks.push({ name, measured, expected, pass: Boolean(pass) });
}

const near = (a, b, tolerance) => Math.abs(a - b) <= tolerance;

/** The GPU's own sRGB transfer, as three's sRGBTransferOETF computes it. */
function srgb(linear) {
    return linear < 0.0031308 ? linear * 12.92 : 1.055 * Math.pow(linear, 1 / 2.4) - 0.055;
}

/** Khronos PBR Neutral, as renderer.js inlines it. */
function neutral([r, g, b]) {
    const x = Math.min(r, g, b);
    const offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    let c = [r - offset, g - offset, b - offset];
    const peak = Math.max(...c);
    if (peak < 0.76) return c;
    const d = 1 - 0.76;
    const newPeak = 1 - (d * d) / (peak + d - 0.76);
    c = c.map((v) => (v * newPeak) / peak);
    const t = 1 - 1 / (0.15 * (peak - newPeak) + 1);
    return c.map((v) => v + (newPeak - v) * t);
}

async function run() {
    /* 1. Inspect shows the stored bytes: 128 stays 128. Tagging the map sRGB
          instead is what the viewer used to do, and shows it as 55. */
    const grey = await inspect(flat(128, 128, 128), {});
    check('inspect: 128 grey reads 128', grey.join(','), '128 +-1',
        grey.every((v) => near(v, 128, 1)));
    const srgbTagged = await loadTexture(flat(128, 128, 128), 'srgb');
    viewer.setSourceInspect([srgbTagged], INSPECT);
    const darkened = readAt();
    srgbTagged.dispose();
    check('reference: the same map decoded as sRGB', darkened.join(','), '55 (the old bug)',
        near(darkened[0], 55, 2));

    /* 2. Each channel of a packed map, as grey, plus the two corrections. */
    const packed = flat(10, 128, 200, 60);
    for (const [channel, want] of [['r', 10], ['g', 128], ['b', 200], ['a', 60]]) {
        const got = await inspect(packed, { channel });
        check(`inspect: channel ${channel} of (10,128,200,60)`, got.join(','), `${want} grey +-1`,
            got.every((v) => near(v, want, 1)));
    }
    const flipped = await inspect(packed, { flipGreen: true });
    check('inspect: flip green', flipped.join(','), '10,127,200 +-1',
        near(flipped[0], 10, 1) && near(flipped[1], 127, 1) && near(flipped[2], 200, 1));
    const inverted = await inspect(packed, { channel: 'g', invert: true });
    check('inspect: invert (gloss read as roughness)', inverted.join(','), '127 grey +-1',
        inverted.every((v) => near(v, 127, 1)));

    /* 3. Orientation: the file's top row lands at the top of the model (v = 1). */
    const white = Array(16).fill(255);
    const black = [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255];
    const topWhite = await loadTexture(png(4, [white, white, black, black]), 'linear');
    viewer.setSourceInspect([topWhite], INSPECT);
    const top = readAt(0, 0.75)[0];
    const bottom = readAt(0, -0.75)[0];
    topWhite.dispose();
    check('inspect: image top at v = 1', `top ${top}, bottom ${bottom}`, 'top > 250, bottom < 5',
        top > 250 && bottom < 5);

    /* 4. The lit material compiles, with every map and every variant. */
    const maps = {
        basecolor: await loadTexture(flat(200, 120, 90, 255), 'srgb'),
        normal: await loadTexture(flat(128, 128, 255), 'linear'),
        orm: await loadTexture(flat(255, 128, 0), 'linear'),
        emissive: await loadTexture(flat(40, 40, 40), 'srgb'),
    };
    const variants = [
        ['OPAQUE', {}, LIT],
        ['BLEND, double sided', { alpha_mode: 'BLEND', double_sided: true },
            { ...LIT, occlusion: true }],
        ['MASK, inverted roughness', { alpha_mode: 'MASK', alpha_cutoff: 0.3 },
            { ...LIT, invertRoughness: true }],
        ['object-space normals, flipped', { normal_space: 'object' }, { ...LIT, flipGreen: true }],
    ];
    for (const [label, spec, options] of variants) {
        viewer.setSourceLit([{ ...PLAIN, ...spec }], [maps], options);
        readAt();
        const error = gl.getError();
        check(`lit: ${label} compiles`, `gl error ${error}, shader errors ${shaderErrors.length}`,
            '0, 0', error === 0 && shaderErrors.length === 0);
    }
    for (const texture of Object.values(maps)) texture.dispose();

    /* 5. Normal map convention (OpenGL, green = +v) under the key light only,
          which sits up and to the left of the view. */
    const key = viewer.camera.children.find((child) => child.isDirectionalLight);
    const room = viewer.scene.environment;
    viewer.scene.environment = null;
    const lit = async (encoded, options = {}) => {
        const normal = await loadTexture(flat(...encoded), 'linear');
        viewer.setSourceLit([PLAIN], [{ ...NO_MAPS, normal }], { ...LIT, ...options });
        const value = readAt()[0];
        normal.dispose();
        return value;
    };
    const facing = await lit([128, 128, 255]);
    const up = await lit([128, 192, 238]);
    const down = await lit([128, 64, 238]);
    const right = await lit([192, 128, 238]);
    const upFlipped = await lit([128, 192, 238], { flipGreen: true });
    check('lit: a map tilted to +v faces the upper key light',
        `up ${up}, flat ${facing}, down ${down}`, 'up > flat + 8, flat > down + 8',
        up > facing + 8 && facing > down + 8);
    check('lit: +u is to the right (away from the key)', `right ${right}, flat ${facing}`,
        'right < flat - 8', right < facing - 8);
    check('lit: flip green turns +v into -v', `flipped ${upFlipped}, down ${down}`, 'equal +-2',
        near(upFlipped, down, 2));

    /* 6. White diffuse under a uniform environment of 0.5, key light off. */
    key.visible = false;
    const grey50 = new THREE.Scene();
    grey50.background = new THREE.Color(0.5, 0.5, 0.5);
    const generator = new THREE.PMREMGenerator(viewer.renderer);
    const uniform = generator.fromScene(grey50, 0).texture;
    generator.dispose();
    viewer.scene.environment = uniform;

    /* Unpatched first: three's own shading, linear out, is the radiance. */
    const reference = new THREE.MeshStandardMaterial({
        color: 0xffffff, roughness: 1, metalness: 0,
    });
    const litMaterials = viewer.sourceMesh.material;
    viewer.sourceMesh.material = [reference];
    const radiance = readAt()[0];
    viewer.sourceMesh.material = litMaterials;
    reference.dispose();
    check('environment: white diffuse, linear radiance', `${radiance}`, '128 +-3 (0.5)',
        near(radiance, 128, 3));

    viewer.setSourceLit([PLAIN], [NO_MAPS], LIT);
    const shown = readAt()[0];
    const linear = radiance / 255;
    const expected = Math.round(255 * srgb(neutral([linear, linear, linear])[0]));
    check('lit: white diffuse under 0.5, PBR Neutral + sRGB', `${shown}`,
        `${expected} +-3 (187 without tone mapping)`, near(shown, expected, 3));

    viewer.scene.environment = room;
    key.visible = true;
    uniform.dispose();

    check('no GL error at the end', `${gl.getError()}`, '0', gl.getError() === 0);
}

function report(error) {
    if (error) {
        check('the checks ran to the end', String(error.stack || error), 'no exception', false);
    }
    const rows = document.getElementById('rows');
    for (const { name, measured, expected, pass } of checks) {
        const row = rows.insertRow();
        for (const text of [name, measured, expected]) row.insertCell().textContent = text;
        const verdict = row.insertCell();
        verdict.textContent = pass ? 'pass' : 'FAIL';
        verdict.className = pass ? 'pass' : 'fail';
    }
    const pass = checks.length > 0 && checks.every((c) => c.pass);
    const verdict = JSON.stringify({ pass, checks });
    document.getElementById('result').textContent = verdict;
    document.title = pass ? 'PASS' : 'FAIL';
    /* run_checks.py opens the page with ?report and waits for this. */
    if (new URLSearchParams(location.search).has('report')) {
        fetch('result', { method: 'POST', body: verdict });
    }
}

run().then(() => report(null), report);
