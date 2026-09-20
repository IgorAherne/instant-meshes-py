/*
    renderer.js: the three.js half of the brushing viewer.

    Every layer here mirrors one draw call of the native application, and the
    numeric constants are the ones quoted in build_test/recon/shaders.md so a
    stroke, a singularity marker or an extracted wireframe lands where the
    native viewer would put it.

    The mesh object is deliberately never transformed.  Camera framing moves the
    camera instead, which keeps world space identical to mesh space -- the space
    the server's BVH works in -- so screenRay() and pickSurface() can hand their
    results straight to the solver.  assertMeshSpaceIsWorldSpace() enforces it.
*/

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { FieldMaterial } from './field_material.js';

/* Stroke ribbons: viewer.cpp:2692-2693 and the 0.85 alpha of viewer.cpp:2810. */
const STROKE_THICKNESS_DIVISOR = 1.5;
const STROKE_NORMAL_OFFSET_DIVISOR = 3.0;
const STROKE_ALPHA = 0.85;
const STROKE_KIND_EDGE = 1;

/* Delete handles: viewer.cpp:2864-2879 lifts the icon by avgEdge/10. */
const HANDLE_LIFT_DIVISOR = 10.0;
const HANDLE_PIXELS = 11;

/* Singularity markers: viewer.cpp:2513 -- mRes.scale() * 0.4 at slider default. */
const SINGULARITY_SCALE = 0.4;
const SINGULARITY_MIN_PIXELS = 4;
const SINGULARITY_MAX_PIXELS = 72;

/* Native mesh shading constant, shaders.md section 1.2. */
const BASE_COLOR = [0.4, 0.5, 0.7];

/* NanoVG Color(255, 100) with nvgStrokeWidth 4, viewer.cpp:2837-2846. */
const PREVIEW_STYLE = 'rgba(255, 255, 255, 0.392)';
const PREVIEW_WIDTH = 4;

const IDENTITY = new THREE.Matrix4();

/* An arbitrary direction with no symmetry with respect to the axes; the native
   singularity geometry shader uses the same one to build a tangent frame. */
const ARBITRARY_DIRECTION = new THREE.Vector3(1.0, 2.0, 4.5);

/* ------------------------------------------------------------------ */
/*  Native stroke colours                                              */
/* ------------------------------------------------------------------ */

const PCG32_MULT = 0x5851f42d4c957f2dn;

/** pcg32 as shipped with Instant Meshes, so stroke hues match the native app. */
function pcg32Stream(initstate, initseq) {
    let state = 0n;
    const inc = BigInt.asUintN(64, (initseq << 1n) | 1n);
    const next = () => {
        const old = state;
        state = BigInt.asUintN(64, old * PCG32_MULT + inc);
        const xorshifted = Number(BigInt.asUintN(32, ((old >> 18n) ^ old) >> 27n));
        const rot = Number((old >> 59n) & 31n);
        return ((xorshifted >>> rot) | (xorshifted << ((32 - rot) & 31))) >>> 0;
    };
    next();
    state = BigInt.asUintN(64, state + initstate);
    next();
    return next;
}

/** pcg32::nextUInt(bound): rejection sampling, unbiased. */
function nextBounded(next, bound) {
    const threshold = ((~bound + 1) >>> 0) % bound;
    for (;;) {
        const r = next();
        if (r >= threshold) return r % bound;
    }
}

const FLOAT_BITS = new DataView(new ArrayBuffer(4));

function float32Bits(value) {
    FLOAT_BITS.setFloat32(0, value);
    return FLOAT_BITS.getUint32(0);
}

/** common.h:367-381, truncating to bytes exactly as the native cast does. */
function hsvToRgb255(h, s, v) {
    const scaled = h * 6;
    const sector = Math.floor(scaled);
    const f = scaled - sector;
    const p = v * (1 - s);
    const q = v * (1 - s * f);
    const t = v * (1 - s * (1 - f));
    let rgb;
    switch (sector) {
        case 0: rgb = [v, t, p]; break;
        case 1: rgb = [q, v, p]; break;
        case 2: rgb = [p, v, t]; break;
        case 3: rgb = [p, q, v]; break;
        case 4: rgb = [t, p, v]; break;
        default: rgb = [v, p, q]; break;
    }
    return rgb.map((c) => Math.min(255, Math.floor(c * 255)));
}

/**
 * viewer.cpp:2710-2713: seed pcg32 with the bit pattern of the first point's
 * coordinate sum, then walk the golden ratio around the hue circle.  The sum is
 * accumulated in float32 because that is what Eigen's redux does on a Vector3f.
 */
function strokeColor(x, y, z) {
    const sum = Math.fround(Math.fround(x + y) + z);
    const next = pcg32Stream(1n, BigInt(float32Bits(sum)));
    const hue = (nextBounded(next, 100) * 0.61803398) % 1;
    return hsvToRgb255(hue, 1.0, 1.0);
}

/* ------------------------------------------------------------------ */
/*  Shared geometry helpers                                            */
/* ------------------------------------------------------------------ */

function averageEdgeLength(positions, indices) {
    let total = 0;
    for (let i = 0; i < indices.length; i += 3) {
        const a = indices[i] * 3;
        const b = indices[i + 1] * 3;
        const c = indices[i + 2] * 3;
        total += edgeLength(positions, a, b);
        total += edgeLength(positions, b, c);
        total += edgeLength(positions, c, a);
    }
    return indices.length > 0 ? total / indices.length : 0;
}

function edgeLength(positions, a, b) {
    const dx = positions[a] - positions[b];
    const dy = positions[a + 1] - positions[b + 1];
    const dz = positions[a + 2] - positions[b + 2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function boundingSphere(positions) {
    const box = new THREE.Box3();
    const point = new THREE.Vector3();
    for (let i = 0; i < positions.length; i += 3) {
        box.expandByPoint(point.set(positions[i], positions[i + 1], positions[i + 2]));
    }
    const center = box.getCenter(new THREE.Vector3());
    let radiusSq = 0;
    for (let i = 0; i < positions.length; i += 3) {
        point.set(positions[i], positions[i + 1], positions[i + 2]);
        radiusSq = Math.max(radiusSq, point.distanceToSquared(center));
    }
    return { center, radius: Math.max(Math.sqrt(radiusSq), 1e-6) };
}

/** A placeholder an overlay object can carry until its first real update. */
function emptyGeometry() {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    return geometry;
}

function hasVertices(object) {
    const position = object.geometry.getAttribute('position');
    return Boolean(position) && position.count > 0;
}

function replaceGeometry(object, geometry) {
    if (object.geometry) object.geometry.dispose();
    object.geometry = geometry;
    object.visible = hasVertices(object);
}

/** Both endpoints pure black: the phantom edge of a triangle stored as a quad. */
function isBlackSegment(colors, offset) {
    for (let k = 0; k < 6; ++k) {
        if (colors[offset + k] !== 0) return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/*  Stroke ribbons -- viewer.cpp:2689-2740                             */
/* ------------------------------------------------------------------ */

/**
 * Build the ribbon mesh for every stroke.
 *
 * The ribbon carries its signal in the vertex alpha rather than in its shape:
 * one edge is 0x00 and, for edge-brush strokes only, the other is 0xFF, so the
 * interpolated alpha crosses 0.5 exactly on the centreline.  The fragment
 * shader darkens that band, and the resulting dark core *is* the edge the brush
 * prescribes.  Comb strokes are 0x00 on both edges and stay a flat translucent
 * ribbon.  Output alpha is the uniform 0.85 in both cases.
 */
function buildStrokeRibbons(strokes, thickness, eps) {
    let vertexCount = 0;
    let triangleCount = 0;
    for (const stroke of strokes) {
        const count = stroke.positions.length / 3;
        if (count < 2) continue;
        vertexCount += count * 2;
        triangleCount += (count - 1) * 2;
    }

    const position = new Float32Array(vertexCount * 3);
    const color = new Uint8Array(vertexCount * 4);
    const index = new Uint32Array(triangleCount * 3);

    const tangent = new THREE.Vector3();
    const normal = new THREE.Vector3();
    const axis = new THREE.Vector3();
    const lastAxis = new THREE.Vector3();

    let v = 0;
    let t = 0;

    const writeVertex = (px, py, pz, rgb, alpha) => {
        position[v * 3 + 0] = px;
        position[v * 3 + 1] = py;
        position[v * 3 + 2] = pz;
        color[v * 4 + 0] = rgb[0];
        color[v * 4 + 1] = rgb[1];
        color[v * 4 + 2] = rgb[2];
        color[v * 4 + 3] = alpha;
        v += 1;
    };

    for (const stroke of strokes) {
        const p = stroke.positions;
        const n = stroke.normals;
        const count = p.length / 3;
        if (count < 2) continue;

        const rgb = strokeColor(p[0], p[1], p[2]);
        const coreAlpha = stroke.kind === STROKE_KIND_EDGE ? 0xff : 0x00;

        /* The native start cap sits on the surface while the rest of the ribbon
           is lifted; lifting it too removes the z-fighting at the stroke head
           without changing the silhouette, because the pair is degenerate. */
        normal.set(n[0], n[1], n[2]);
        const headX = p[0] + normal.x * eps;
        const headY = p[1] + normal.y * eps;
        const headZ = p[2] + normal.z * eps;
        writeVertex(headX, headY, headZ, rgb, 0x00);
        writeVertex(headX, headY, headZ, rgb, coreAlpha);

        lastAxis.set(0, 0, 0);

        for (let i = 1; i < count; ++i) {
            const base = v - 2;
            index[t++] = base + 2;
            index[t++] = base + 0;
            index[t++] = base + 1;
            index[t++] = base + 2;
            index[t++] = base + 1;
            index[t++] = base + 3;

            const o = i * 3;
            normal.set(n[o], n[o + 1], n[o + 2]);
            tangent.set(p[o] - p[o - 3], p[o + 1] - p[o - 2], p[o + 2] - p[o - 1]);
            ribbonAxis(tangent, normal, lastAxis, axis);
            lastAxis.copy(axis);

            /* Parabolic taper: the ribbon pinches to nothing at both ends. */
            const taper = 1 - Math.pow(2 * ((i + 1) / count) - 1, 2);
            const half = taper * thickness;

            const cx = p[o] + normal.x * eps;
            const cy = p[o + 1] + normal.y * eps;
            const cz = p[o + 2] + normal.z * eps;
            writeVertex(cx + axis.x * half, cy + axis.y * half, cz + axis.z * half, rgb, 0x00);
            writeVertex(cx - axis.x * half, cy - axis.y * half, cz - axis.z * half, rgb, coreAlpha);
        }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(color, 4, true));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    return geometry;
}

/**
 * The ribbon widens along `normal x tangent`, exactly as viewer.cpp:2727 does.
 * A stroke that doubles back on itself can produce a tangent parallel to the
 * normal; reusing the previous axis there keeps the ribbon from collapsing.
 */
function ribbonAxis(tangent, normal, lastAxis, out) {
    out.copy(tangent).addScaledVector(normal, -normal.dot(tangent));
    if (out.lengthSq() < 1e-20) {
        if (lastAxis.lengthSq() > 0) {
            out.copy(lastAxis);
            return;
        }
        out.copy(ARBITRARY_DIRECTION).addScaledVector(normal, -normal.dot(ARBITRARY_DIRECTION));
    }
    out.normalize();
    out.crossVectors(normal, out);
}

function buildHandleGeometry(strokes, lift) {
    const usable = strokes.filter((stroke) => stroke.positions.length >= 3);
    const position = new Float32Array(usable.length * 3);
    const color = new Float32Array(usable.length * 3);
    const handles = [];

    usable.forEach((stroke, i) => {
        const p = stroke.positions;
        const n = stroke.normals;
        const point = new THREE.Vector3(p[0] + n[0] * lift, p[1] + n[1] * lift, p[2] + n[2] * lift);
        position[i * 3 + 0] = point.x;
        position[i * 3 + 1] = point.y;
        position[i * 3 + 2] = point.z;

        const rgb = strokeColor(p[0], p[1], p[2]);
        color[i * 3 + 0] = rgb[0] / 255;
        color[i * 3 + 1] = rgb[1] / 255;
        color[i * 3 + 2] = rgb[2] / 255;

        handles.push({ id: stroke.id, position: point });
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(color, 3));
    return { geometry, handles };
}

/* ------------------------------------------------------------------ */
/*  Materials                                                          */
/* ------------------------------------------------------------------ */

/** shader_flowline.vert/.frag, ported to GLSL ES 3.00. */
function createStrokeMaterial() {
    return new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        uniforms: { alpha: { value: STROKE_ALPHA } },
        vertexShader: `
            precision highp float;
            uniform mat4 projectionMatrix;
            uniform mat4 modelViewMatrix;
            in vec3 position;
            in vec4 color;
            out vec4 vColor;
            void main() {
                vColor = color;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            precision highp float;
            uniform float alpha;
            in vec4 vColor;
            out vec4 outColor;
            void main() {
                vec3 result = vColor.rgb;
                float d = abs(vColor.a - 0.5);
                if (d < 0.1)
                    result *= smoothstep(0.0, 1.0, d / 0.1) * 0.5 + 0.5;
                outColor = vec4(result, alpha);
            }
        `,
        transparent: true,
        blending: THREE.NormalBlending,
        depthTest: true,
        depthWrite: true,
        depthFunc: THREE.LessEqualDepth,
        side: THREE.DoubleSide,
    });
}

/**
 * Camera-facing round markers whose diameter is a world length.
 *
 * `worldSize` of zero with min == max pins a marker to a constant pixel size,
 * which is what a UI affordance like a delete handle wants.
 */
function createPointMaterial({ worldSize, minPixels, maxPixels }) {
    return new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        uniforms: {
            worldSize: { value: worldSize },
            minPixels: { value: minPixels },
            maxPixels: { value: maxPixels },
            halfHeight: { value: 1 },
        },
        vertexShader: `
            precision highp float;
            uniform mat4 projectionMatrix;
            uniform mat4 modelViewMatrix;
            uniform float worldSize;
            uniform float minPixels;
            uniform float maxPixels;
            uniform float halfHeight;
            in vec3 position;
            in vec3 color;
            out vec3 vColor;
            void main() {
                vColor = color;
                vec4 eyePos = modelViewMatrix * vec4(position, 1.0);
                gl_Position = projectionMatrix * eyePos;
                float pixelsPerUnit = projectionMatrix[1][1] * halfHeight / max(-eyePos.z, 1e-6);
                gl_PointSize = clamp(worldSize * pixelsPerUnit, minPixels, maxPixels);
            }
        `,
        fragmentShader: `
            precision highp float;
            in vec3 vColor;
            out vec4 outColor;
            void main() {
                vec2 d = gl_PointCoord * 2.0 - 1.0;
                float r2 = dot(d, d);
                if (r2 > 1.0) discard;
                float rim = smoothstep(0.55, 1.0, r2);
                outColor = vec4(vColor * (1.0 - 0.7 * rim), 1.0);
            }
        `,
        transparent: false,
        depthTest: true,
        depthWrite: true,
    });
}

/* ------------------------------------------------------------------ */
/*  Viewer                                                             */
/* ------------------------------------------------------------------ */

export class Viewer {
    /**
     * @param {HTMLCanvasElement} canvas   WebGL2 target
     * @param {HTMLCanvasElement} overlay  2D canvas for the in-progress stroke
     */
    constructor(canvas, overlay) {
        const context = canvas.getContext('webgl2', {
            antialias: true,
            alpha: false,
            depth: true,
            powerPreference: 'high-performance',
        });
        if (!context) {
            throw new Error('WebGL2 is required; this browser did not provide a webgl2 context');
        }

        /* The native application writes raw colour values into an 8-bit buffer
           with no transfer function, so colour management is switched off here
           to keep every literal in these shaders landing on screen unchanged. */
        THREE.ColorManagement.enabled = false;

        this.canvas = canvas;
        this.overlay = overlay;
        this._overlayCtx = overlay.getContext('2d');

        this.renderer = new THREE.WebGLRenderer({ canvas, context, antialias: true });
        this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
        this.renderer.setClearColor(0x2d2d30, 1);
        this._pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        this.renderer.setPixelRatio(this._pixelRatio);

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
        this.camera.position.set(0, 0, 5);

        this.controls = new OrbitControls(this.camera, canvas);
        this.controls.screenSpacePanning = true;
        this.controls.zoomToCursor = true;

        /* Called once per animation frame, before anything is drawn, so a
           caller can apply network updates at most once per displayed frame. */
        this.onFrame = null;

        this._field = null;
        this.mesh = null;
        this._scale = 1;
        this._avgEdge = 1;

        this._layers = {
            mesh: true,
            grid: true,
            strokes: true,
            singularities: true,
            output: false,
        };

        this._buildLayers();

        this._preview = null;
        this._previewDirty = false;
        this._needsResize = true;
        this._size = { width: 0, height: 0 };

        this._resizeObserver = new ResizeObserver(() => {
            this._needsResize = true;
        });
        this._resizeObserver.observe(canvas.parentElement || canvas);

        this._tick = this._tick.bind(this);
        this._raf = requestAnimationFrame(this._tick);
    }

    _buildLayers() {
        this.meshGroup = new THREE.Group();
        this.scene.add(this.meshGroup);

        this.strokeGroup = new THREE.Group();
        this.strokeGroup.visible = this._layers.strokes;
        this.scene.add(this.strokeGroup);

        this.strokeRibbons = new THREE.Mesh(emptyGeometry(), createStrokeMaterial());
        this.strokeRibbons.frustumCulled = false;
        this.strokeRibbons.visible = false;
        /* Drawn after everything else, exactly like the native drawOverlay. */
        this.strokeRibbons.renderOrder = 10;
        this.strokeGroup.add(this.strokeRibbons);

        this.strokeHandleMarkers = new THREE.Points(
            emptyGeometry(),
            createPointMaterial({ worldSize: 0, minPixels: 0, maxPixels: 0 })
        );
        this.strokeHandleMarkers.frustumCulled = false;
        this.strokeHandleMarkers.visible = false;
        this.strokeGroup.add(this.strokeHandleMarkers);

        this.singularityMarkers = new THREE.Points(
            emptyGeometry(),
            createPointMaterial({
                worldSize: SINGULARITY_SCALE,
                minPixels: SINGULARITY_MIN_PIXELS,
                maxPixels: SINGULARITY_MAX_PIXELS,
            })
        );
        this.singularityMarkers.frustumCulled = false;
        this.singularityMarkers.visible = false;
        this.singularityMarkers.name = 'singularities';
        this.scene.add(this.singularityMarkers);

        this.outputWireframe = new THREE.LineSegments(
            emptyGeometry(),
            new THREE.LineBasicMaterial({ vertexColors: true })
        );
        this.outputWireframe.frustumCulled = false;
        this.outputWireframe.visible = false;
        this.outputWireframe.name = 'output';
        this.scene.add(this.outputWireframe);

        this._strokeHandles = [];
        this._hasOutput = false;
    }

    /* -------------------------------------------------------------- */
    /*  Geometry                                                       */
    /* -------------------------------------------------------------- */

    /**
     * Replace the working mesh and frame the camera on it.
     *
     * @param {{positions: Float32Array, indices: Uint32Array,
     *          normals: Float32Array, scale: number,
     *          rosy: number, posy: number}} data
     */
    setGeometry({ positions, indices, normals, scale, rosy, posy }) {
        this._disposeMesh();

        /* Every overlay was built from the mesh that just went away: its
           singularities sit on faces that no longer exist and its extraction
           describes a different surface.  Drop them rather than leave them
           floating until the solver happens to publish replacements. */
        this.setStrokes([]);
        this.setSingularities(null, null);
        this.setExtracted(null, null);

        this._field = new FieldMaterial({ positions, normals, indices, scale, rosy, posy });
        this._field.setBaseColor(BASE_COLOR[0], BASE_COLOR[1], BASE_COLOR[2]);
        this._field.setShowGrid(this._layers.grid);

        this.mesh = new THREE.Mesh(this._field.geometry, this._field.material);
        this.mesh.name = 'mesh';
        this.mesh.visible = this._layers.mesh;
        this.meshGroup.add(this.mesh);

        this._scale = scale > 0 ? scale : 1;
        this._avgEdge = averageEdgeLength(positions, indices) || this._scale;

        const handlePixels = HANDLE_PIXELS * this._pixelRatio;
        const handleUniforms = this.strokeHandleMarkers.material.uniforms;
        handleUniforms.minPixels.value = handlePixels;
        handleUniforms.maxPixels.value = handlePixels;
        this.singularityMarkers.material.uniforms.worldSize.value = this._scale * SINGULARITY_SCALE;

        const sphere = boundingSphere(positions);
        this._frameCamera(sphere.center, sphere.radius);
        this.assertMeshSpaceIsWorldSpace();
    }

    /** Distance that fits the bounding sphere in the vertical field of view. */
    _frameCamera(center, radius) {
        const halfFov = THREE.MathUtils.degToRad(this.camera.fov) / 2;
        const distance = (radius / Math.sin(halfFov)) * 1.05;

        this.camera.near = radius * 0.01;
        this.camera.far = distance + radius * 6;
        this.camera.up.set(0, 1, 0);
        this.camera.position.set(center.x, center.y, center.z + distance);
        this.camera.updateProjectionMatrix();

        this.controls.target.copy(center);
        this.controls.minDistance = radius * 0.02;
        this.controls.maxDistance = distance * 8;
        this.controls.update();
    }

    /** Forward a solver update to the field shader. */
    setField(q, o) {
        if (this._field) this._field.updateField(q, o);
    }

    /* -------------------------------------------------------------- */
    /*  Strokes                                                        */
    /* -------------------------------------------------------------- */

    /**
     * @param {Array<{id: number, kind: number,
     *                positions: Float32Array, normals: Float32Array}>} strokes
     */
    setStrokes(strokes) {
        const list = strokes || [];
        const thickness = this._avgEdge / STROKE_THICKNESS_DIVISOR;
        const eps = this._avgEdge / STROKE_NORMAL_OFFSET_DIVISOR;

        replaceGeometry(this.strokeRibbons, buildStrokeRibbons(list, thickness, eps));

        /* The native delete icon is a 2D sprite drawn on top of the ribbon.
           Lifting the marker just past the ribbon plane reproduces that
           ordering while still letting the mesh occlude it. */
        const lift = eps + this._avgEdge / HANDLE_LIFT_DIVISOR;
        const { geometry, handles } = buildHandleGeometry(list, lift);
        replaceGeometry(this.strokeHandleMarkers, geometry);
        this._strokeHandles = handles;
    }

    /** Stroke delete handles in mesh space, for hit-testing a click. */
    strokeHandles() {
        return this._strokeHandles;
    }

    /**
     * @param {Array<{x: number, y: number}>|null} points
     *        canvas-relative CSS pixels of the stroke being drawn
     */
    setPreviewStroke(points) {
        this._preview = points && points.length > 1 ? points : null;
        this._previewDirty = true;
    }

    /* -------------------------------------------------------------- */
    /*  Solver overlays                                                */
    /* -------------------------------------------------------------- */

    /**
     * @param {Float32Array} positions  (n, 3) marker centres in mesh space
     * @param {Float32Array} colors     (n, 3) linear RGB in [0, 1]
     */
    setSingularities(positions, colors) {
        const geometry = new THREE.BufferGeometry();
        const count = positions ? Math.floor(positions.length / 3) : 0;
        geometry.setAttribute(
            'position',
            new THREE.BufferAttribute(count ? positions : new Float32Array(0), 3)
        );
        geometry.setAttribute(
            'color',
            new THREE.BufferAttribute(count ? colors : new Float32Array(0), 3)
        );
        replaceGeometry(this.singularityMarkers, geometry);
        this.singularityMarkers.visible = count > 0 && this._layers.singularities;
    }

    /**
     * Extracted quad wireframe.  A triangle stored as a degenerate quad carries
     * a phantom edge that the native fragment shader discards by testing for
     * pure black; dropping those segments here is the same thing done on the
     * CPU, and it keeps them out of the vertex buffer entirely.
     *
     * @param {Float32Array} wireframe       (2 * nF * posy, 3) endpoints
     * @param {Float32Array} wireframeColor  (2 * nF * posy, 3) endpoint colours
     */
    setExtracted(wireframe, wireframeColor) {
        const segments = wireframe ? Math.floor(wireframe.length / 6) : 0;
        const keep = new Uint8Array(segments);
        let kept = 0;
        for (let s = 0; s < segments; ++s) {
            if (!isBlackSegment(wireframeColor, s * 6)) {
                keep[s] = 1;
                kept += 1;
            }
        }

        const positions = new Float32Array(kept * 6);
        const colors = new Float32Array(kept * 6);
        let out = 0;
        for (let s = 0; s < segments; ++s) {
            if (!keep[s]) continue;
            positions.set(wireframe.subarray(s * 6, s * 6 + 6), out);
            colors.set(wireframeColor.subarray(s * 6, s * 6 + 6), out);
            out += 6;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        replaceGeometry(this.outputWireframe, geometry);

        this._hasOutput = kept > 0;
        this.outputWireframe.visible = this._hasOutput && this._layers.output;
    }

    /* -------------------------------------------------------------- */
    /*  Layers and interaction state                                   */
    /* -------------------------------------------------------------- */

    /** @param {'mesh'|'grid'|'strokes'|'singularities'|'output'} name */
    setLayerVisible(name, visible) {
        if (!(name in this._layers)) {
            throw new Error(`unknown layer "${name}"`);
        }
        this._layers[name] = visible;

        switch (name) {
            case 'mesh':
                if (this.mesh) this.mesh.visible = visible;
                break;
            case 'grid':
                if (this._field) this._field.setShowGrid(visible);
                break;
            case 'strokes':
                this.strokeGroup.visible = visible;
                break;
            case 'singularities':
                this.singularityMarkers.visible = visible && hasVertices(this.singularityMarkers);
                break;
            default:
                this.outputWireframe.visible = visible && this._hasOutput;
                break;
        }
    }

    /**
     * Hand the primary drag to a brush, or give it back to the camera.
     *
     * Only the left button and the one-finger gesture change hands: the native
     * viewer keeps the wheel and the right button working while a tool is
     * selected, so navigating never requires putting the brush down.
     */
    setControlsEnabled(enabled) {
        this.controls.mouseButtons.LEFT = enabled ? THREE.MOUSE.ROTATE : null;
        this.controls.touches.ONE = enabled ? THREE.TOUCH.ROTATE : null;
        this.canvas.classList.toggle('drawing', !enabled);
    }

    get scale() {
        return this._scale;
    }

    get averageEdgeLength() {
        return this._avgEdge;
    }

    /** Camera position in mesh space, which the erase query needs as its eye. */
    eye() {
        return new Float32Array([
            this.camera.position.x,
            this.camera.position.y,
            this.camera.position.z,
        ]);
    }

    /* -------------------------------------------------------------- */
    /*  Picking                                                        */
    /* -------------------------------------------------------------- */

    assertMeshSpaceIsWorldSpace() {
        if (this.mesh && !this.mesh.matrixWorld.equals(IDENTITY)) {
            throw new Error(
                'the working mesh must stay untransformed: screenRay() and pickSurface() ' +
                    'only return mesh-space results while world space equals mesh space'
            );
        }
    }

    /**
     * Ray through a client point, in MESH space.
     *
     * This is the explicit inverse-MVP route of viewer.cpp:2969-2973 rather
     * than Raycaster.setFromCamera, so the near/far unprojection and the y flip
     * match the native code exactly.
     *
     * @returns {{origin: Float32Array, direction: Float32Array}}
     */
    screenRay(clientX, clientY) {
        const origin = new THREE.Vector3();
        const direction = new THREE.Vector3();
        this._screenRayInto(clientX, clientY, origin, direction);
        return {
            origin: new Float32Array([origin.x, origin.y, origin.z]),
            direction: new Float32Array([direction.x, direction.y, direction.z]),
        };
    }

    _screenRayInto(clientX, clientY, origin, direction) {
        this.assertMeshSpaceIsWorldSpace();
        /* Pointer events arrive between frames, so refresh the camera matrices
           rather than unprojecting through the previous frame's view. */
        this.camera.updateMatrixWorld();

        const rect = this.canvas.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        const ndcX = (x / rect.width) * 2 - 1;
        const ndcY = ((rect.height - y) / rect.height) * 2 - 1;

        const inverse = new THREE.Matrix4()
            .multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse)
            .invert();

        origin.set(ndcX, ndcY, -1).applyMatrix4(inverse);
        direction.set(ndcX, ndcY, 1).applyMatrix4(inverse).sub(origin).normalize();
    }

    /**
     * @returns {{point: Float32Array, faceIndex: number}|null}
     *          faceIndex indexes the de-indexed triangle list, which
     *          FieldMaterial builds in input face order.
     */
    pickSurface(clientX, clientY) {
        if (!this.mesh) return null;

        const origin = new THREE.Vector3();
        const direction = new THREE.Vector3();
        this._screenRayInto(clientX, clientY, origin, direction);

        const raycaster = new THREE.Raycaster(origin, direction);
        const hits = raycaster.intersectObject(this.mesh, false);
        if (hits.length === 0) return null;

        const hit = hits[0];
        return {
            point: new Float32Array([hit.point.x, hit.point.y, hit.point.z]),
            faceIndex: hit.faceIndex,
        };
    }

    /**
     * @returns {{x: number, y: number}|null} canvas-relative CSS pixels, or
     *          null when the point sits behind the near plane.
     */
    projectToScreen(point) {
        this.camera.updateMatrixWorld();
        const eyeSpace = point.clone().applyMatrix4(this.camera.matrixWorldInverse);
        if (eyeSpace.z > -this.camera.near) return null;

        const ndc = eyeSpace.applyMatrix4(this.camera.projectionMatrix);
        return {
            x: (ndc.x * 0.5 + 0.5) * this._size.width,
            y: (1 - (ndc.y * 0.5 + 0.5)) * this._size.height,
        };
    }

    /* -------------------------------------------------------------- */
    /*  Frame loop                                                     */
    /* -------------------------------------------------------------- */

    _tick() {
        this._raf = requestAnimationFrame(this._tick);

        if (this._needsResize) this._applyResize();
        if (this.onFrame) this.onFrame();

        this.controls.update();
        if (this._field) this._field.update(this.camera, this.mesh);
        this.renderer.render(this.scene, this.camera);

        if (this._previewDirty) this._drawPreview();
    }

    _applyResize() {
        this._needsResize = false;

        const host = this.canvas.parentElement || this.canvas;
        const width = Math.max(host.clientWidth, 1);
        const height = Math.max(host.clientHeight, 1);
        if (width === this._size.width && height === this._size.height) return;

        this._size = { width, height };
        this.renderer.setSize(width, height, false);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.overlay.width = Math.round(width * this._pixelRatio);
        this.overlay.height = Math.round(height * this._pixelRatio);
        this._overlayCtx.setTransform(this._pixelRatio, 0, 0, this._pixelRatio, 0, 0);
        this._previewDirty = true;

        /* gl_PointSize is expressed in device pixels. */
        const halfHeight = (height * this._pixelRatio) / 2;
        this.strokeHandleMarkers.material.uniforms.halfHeight.value = halfHeight;
        this.singularityMarkers.material.uniforms.halfHeight.value = halfHeight;
    }

    _drawPreview() {
        this._previewDirty = false;

        const ctx = this._overlayCtx;
        ctx.clearRect(0, 0, this._size.width, this._size.height);
        if (!this._preview) return;

        ctx.lineWidth = PREVIEW_WIDTH;
        ctx.strokeStyle = PREVIEW_STYLE;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(this._preview[0].x, this._preview[0].y);
        for (let i = 1; i < this._preview.length; ++i) {
            ctx.lineTo(this._preview[i].x, this._preview[i].y);
        }
        ctx.stroke();
    }

    /* -------------------------------------------------------------- */
    /*  Teardown                                                       */
    /* -------------------------------------------------------------- */

    _disposeMesh() {
        if (this.mesh) {
            this.meshGroup.remove(this.mesh);
            this.mesh = null;
        }
        if (this._field) {
            this._field.dispose();
            this._field = null;
        }
    }

    dispose() {
        cancelAnimationFrame(this._raf);
        this._resizeObserver.disconnect();
        this.controls.dispose();
        this._disposeMesh();

        for (const object of [
            this.strokeRibbons,
            this.strokeHandleMarkers,
            this.singularityMarkers,
            this.outputWireframe,
        ]) {
            object.geometry.dispose();
            object.material.dispose();
        }
        this.renderer.dispose();
    }
}
