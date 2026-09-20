/**
 * The red field-aligned grid over the input mesh -- a WebGL2 port of the native
 * preview (resources/shader_mesh.vert + shader_mesh.geo + shader_mesh.frag).
 *
 * The native effect comes out of a geometry shader that needs all three corners
 * of a triangle at once, and WebGL2 has no geometry shaders.  So the triangles
 * are de-indexed and every per-vertex quantity lives in an RGBA32F data texture
 * that the vertex shader reads with texelFetch(): each of the three invocations
 * of a triangle recomputes all three corners and keeps its own.  The chain is
 * deterministic and side-effect free, so the three invocations agree bit for
 * bit and the interpolated texcoord is exactly what EmitVertex() produced.
 *
 * The point of the data textures is the refresh cost.  A solve iteration only
 * re-uploads Q and O -- 32 bytes per vertex -- while topology, positions and
 * normals stay resident on the GPU.
 */

import * as THREE from 'three';

/* Uniform values the native viewer uses, src/viewer.cpp:40-44 and 2440-2461. */
const BASE_COLOR = [0.4, 0.5, 0.7];
const SPECULAR_COLOR = [1.0, 1.0, 1.0];
const INTERIOR_FACTOR = [0.5, 0.5, 0.5];
const EDGE_FACTOR_0 = [0.5, 1.0, 1.0];
const EDGE_FACTOR_1 = [1.0, 1.0, 0.5];
const EDGE_FACTOR_2 = [1.0, 1.0, 1.0];

/**
 * The native camera never moves: the arcball spins the model underneath a light
 * fixed at world (0, 0.3, 5) while the view matrix stays T(0, 0, -5), which
 * leaves the light 0.3 above the eye in camera space.  A browser camera orbits
 * instead, so update() re-derives the world position from the eye each frame to
 * preserve that relationship.  For a camera at (0, 0, 5) looking at the origin
 * -- the native starting pose -- this reproduces the constant exactly.
 */
const LIGHT_IN_EYE_SPACE = new THREE.Vector3(0.0, 0.3, 0.0);

/** WebGL2 guarantees at least this, and a power of two keeps id%w / id/w cheap. */
const MIN_TEXTURE_WIDTH = 2048;

const VERTEX_SHADER = `
precision highp float;
precision highp int;
/* ESSL 3.00 defaults sampler2D to lowp in BOTH stages, which is free to
   truncate an RGBA32F fetch. Every data-texture read below depends on this. */
precision highp sampler2D;

uniform mat4 projectionMatrix;   /* three.js fills these three; native proj, */
uniform mat4 viewMatrix;         /* view and model respectively.             */
uniform mat4 modelMatrix;

uniform vec3 light_position;     /* world space */
uniform vec3 camera_local;       /* eye in object space, viewer.cpp:2408 */
uniform float scale;             /* target edge length */
uniform float inv_scale;
uniform float show_uvs;

uniform sampler2D tex_face;      /* (i0, i1, i2, 0) per triangle, ids as floats */
uniform sampler2D tex_position;
uniform sampler2D tex_normal;    /* shading normal and field normal at once */
uniform sampler2D tex_q;         /* orientation field, level 0 */
uniform sampler2D tex_o;         /* position field, level 0 */

uniform int vertex_tex_width;
uniform int face_tex_width;

out vec3 v_to_eye;
out vec3 v_to_light;
out vec3 v_normal;
out vec2 v_texcoord;

vec4 fetch_vertex(sampler2D tex, int id) {
    return texelFetch(tex, ivec2(id % vertex_tex_width, id / vertex_tex_width), 0);
}

vec4 fetch_face(sampler2D tex, int id) {
    return texelFetch(tex, ivec2(id % face_tex_width, id / face_tex_width), 0);
}

/* Natively this sits inside the ROSY == 6 branch, which is safe only because
   the app builds nothing but the 2/4, 4/4 and 6/3 combinations. This API allows
   POSY == 3 with any ROSY, and the POSY == 3 rounding needs it as well. */
#if ROSY == 6 || POSY == 3
    vec3 rotate60(vec3 d, vec3 n) {
        return 0.8660254037 * cross(n, d) + 0.5 * (d + n * dot(n, d));
    }
#endif

#if ROSY == 2
    vec3 compat_orientation(vec3 q, vec3 ref, vec3 n) {
        return q * sign(dot(q, ref));
    }
#elif ROSY == 4
    vec3 compat_orientation(vec3 q, vec3 ref, vec3 n) {
        vec3 t = cross(n, q);
        float dp0 = dot(q, ref), dp1 = dot(t, ref);
        if (abs(dp0) > abs(dp1))
            return q * sign(dp0);
        else
            return t * sign(dp1);
    }
#else
    vec3 compat_orientation(vec3 q, vec3 ref, vec3 n) {
        vec3 t[3] = vec3[3](rotate60(q, -n), q, rotate60(q, n));
        float dp[3] = float[3](dot(t[0], ref), dot(t[1], ref), dot(t[2], ref));
        float abs_dp[3] = float[3](abs(dp[0]), abs(dp[1]), abs(dp[2]));

        if (abs_dp[0] >= abs_dp[1] && abs_dp[0] >= abs_dp[2])
            return t[0] * sign(dp[0]);
        else if (abs_dp[1] >= abs_dp[0] && abs_dp[1] >= abs_dp[2])
            return t[1] * sign(dp[1]);
        else
            return t[2] * sign(dp[2]);
    }
#endif

#if POSY == 4
    vec3 compat_position(vec3 o, vec3 ref, vec3 q, vec3 t, vec3 n) {
        vec3 d = ref - o;
        return o +
            q * round(dot(q, d) * inv_scale) * scale +
            t * round(dot(t, d) * inv_scale) * scale;
    }
#else
    vec3 compat_position(vec3 o, vec3 ref, vec3 q, vec3 t, vec3 n) {
        vec3 d = ref - o;
        t = rotate60(q, n);

        float dpq = dot(q, d), dpt = dot(t, d);
        float u = floor(( 4.0 * dpq - 2.0 * dpt) * (1.0 / 3.0) * inv_scale);
        float v = floor((-2.0 * dpq + 4.0 * dpt) * (1.0 / 3.0) * inv_scale);

        q *= scale; t *= scale;
        o = o + q * u + t * v - ref;

        vec3 candidates[4] = vec3[4](o, o + q, o + t, o + q + t);

        float best_length = 1e20;
        /* Natively -1, which indexes out of bounds if every length is NaN. */
        int best_index = 0;
        for (int i = 0; i < 4; ++i) {
            float len = dot(candidates[i], candidates[i]);
            if (len < best_length) {
                best_length = len;
                best_index = i;
            }
        }
        return candidates[best_index] + ref;
    }
#endif

void main() {
    int corner = gl_VertexID % 3;
    ivec3 vid = ivec3(fetch_face(tex_face, gl_VertexID / 3).xyz);

    vec3 p[3];
    for (int i = 0; i < 3; ++i)
        p[i] = fetch_vertex(tex_position, vid[i]).xyz;

    vec3 face_normal = normalize(cross(p[1] - p[0], p[2] - p[0]));

    /* shader_mesh.geo:97-98, the only culling the native app performs. All
       three corners take this branch together, so a shared degenerate position
       past the far plane (z > w) drops the primitive; a merely degenerate one
       is not reliably discarded when the same shader draws GL_LINES. */
    if (dot(p[0] - camera_local, face_normal) > 0.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        v_to_eye = vec3(0.0);
        v_to_light = vec3(0.0);
        v_normal = vec3(0.0, 0.0, 1.0);
        v_texcoord = vec2(0.0);
        return;
    }

    vec2 texcoord = vec2(0.0);

    if (show_uvs == 1.0) {
        /* Step 1: rotate everything into the triangle plane */
        vec3 tangents[3], uv[3];
        for (int i = 0; i < 3; ++i) {
            vec3 normal_data = fetch_vertex(tex_normal, vid[i]).xyz;
            vec3 q = normalize(fetch_vertex(tex_q, vid[i]).xyz);
            vec3 o = fetch_vertex(tex_o, vid[i]).xyz;

            float cos_theta = dot(normal_data, face_normal);

            if (cos_theta < 0.9999) {
                vec3 axis = cross(normal_data, face_normal);
                float sin_theta2 = dot(axis, axis);
                float factor = (1.0 - cos_theta) / sin_theta2;

                tangents[i] = q * cos_theta + cross(axis, q)
                    + axis * dot(axis, q) * factor;

                vec3 to_o = o - p[i];
                uv[i] = to_o * cos_theta + cross(axis, to_o)
                    + axis * dot(axis, to_o) * factor + p[i];
            } else {
                tangents[i] = q;
                uv[i] = o;
            }
        }

        /* Step 2: search orientation field quotient space */
        for (int i = 1; i < 3; ++i)
            tangents[i] = compat_orientation(tangents[i], tangents[0], face_normal);

        vec3 bitangents[3];
        for (int i = 0; i < 3; ++i)
            bitangents[i] = cross(face_normal, tangents[i]);

        /* Step 3: search position field quotient space */
        for (int i = 1; i < 3; ++i)
            uv[i] = compat_position(uv[i], uv[0], tangents[i], bitangents[i], face_normal);

        /* Step 4: keep only this invocation's corner. The rasterizer
           interpolates the three of them exactly as it did after EmitVertex(),
           which is why v_texcoord must not be flat. */
        vec3 rel = p[corner] - uv[corner];
        texcoord = vec2(
            dot(rel, tangents[corner]) * inv_scale,
            dot(rel, bitangents[corner]) * inv_scale
        );
    }

    vec4 pos_camera = viewMatrix * (modelMatrix * vec4(p[corner], 1.0));
    vec3 normal = fetch_vertex(tex_normal, vid[corner]).xyz;

    gl_Position = projectionMatrix * pos_camera;
    v_to_light = (viewMatrix * vec4(light_position, 1.0)).xyz - pos_camera.xyz;
    v_to_eye = -pos_camera.xyz;
    /* shader_mesh.geo:154 writes model * (view * n), which survives only because
       the native view matrix is a pure translation. An orbiting browser camera
       makes that visibly wrong, so the operands are in the order the lighting
       needs: to_eye and to_light are camera space, so the normal must be too. */
    v_normal = (viewMatrix * (modelMatrix * vec4(normal, 0.0))).xyz;
    v_texcoord = texcoord;
}
`;

const FRAGMENT_SHADER = `
/* The original says lowp, which desktop GL ignores but ES honours: lowp would
   collapse fract(texcoord + 0.5) to noise once the texcoord reaches the tens. */
precision highp float;
precision highp int;

uniform float show_uvs;
uniform vec3 specular_color;
uniform vec3 base_color;
uniform vec3 edge_factor_0;
uniform vec3 edge_factor_1;
uniform vec3 edge_factor_2;
uniform vec3 interior_factor;

in vec3 v_to_eye;
in vec3 v_to_light;
in vec3 v_normal;
in vec2 v_texcoord;

out vec4 outColor;

void main() {
    vec3 Kd = base_color;
    vec3 Ks = specular_color;
    vec3 Ka = Kd * 0.2;

    vec3 to_light = normalize(v_to_light);
    vec3 to_eye = normalize(v_to_eye);
    vec3 normal = normalize(v_normal);
    vec3 refl = reflect(-to_light, normal);

    float diffuse_factor = max(0.0, dot(to_light, normal));
    float specular_factor = pow(max(dot(to_eye, refl), 0.0), 10.0);

    vec3 finalColor = Ka + Kd * diffuse_factor + Ks * specular_factor;

    if (show_uvs == 1.0) {
        #if POSY == 4
            bool a = abs(fract(v_texcoord.x + 0.5) - 0.5) < 0.1;
            bool b = abs(fract(v_texcoord.y + 0.5) - 0.5) < 0.1;
            #if ROSY == 2
                if (a && !b)
                    finalColor *= edge_factor_0;
                else if (b && !a)
                    finalColor *= edge_factor_1;
                else if (a && b)
                    finalColor *= edge_factor_2;
                else
                    finalColor *= interior_factor;
            #else
                if (a || b)
                    finalColor *= edge_factor_2;
                else
                    finalColor *= interior_factor;
            #endif
        #else
            const float inv_sqrt3 = 0.577350269189626;
            vec3 tx = vec3(
                v_texcoord.x + inv_sqrt3 * v_texcoord.y,
                v_texcoord.x - inv_sqrt3 * v_texcoord.y,
                2.0 * inv_sqrt3 * v_texcoord.y
            );

            bool a = abs(fract(tx.x + 0.5) - 0.5) < 0.1;
            bool b = abs(fract(tx.y + 0.5) - 0.5) < 0.1;
            bool c = abs(fract(tx.z + 0.5) - 0.5) < 0.1;
            if (a || b || c)
                finalColor *= edge_factor_2;
            else
                finalColor *= interior_factor;
        #endif
    }

    outColor = vec4(finalColor, 1.0);
}
`;

let probedMaxTextureSize = 0;

/**
 * The real GL_MAX_TEXTURE_SIZE, probed once from a throwaway context that is
 * released immediately.  FieldMaterial is built before any renderer exists, and
 * assuming the 2048 floor would cap a data texture at 4.2M vertices on hardware
 * that allows sixteen times that.
 *
 * @returns {number}
 */
function maxTextureSize() {
  if (probedMaxTextureSize) return probedMaxTextureSize;

  probedMaxTextureSize = MIN_TEXTURE_WIDTH;
  const gl = document.createElement('canvas').getContext('webgl2');
  if (gl) {
    probedMaxTextureSize = Math.max(
      MIN_TEXTURE_WIDTH,
      gl.getParameter(gl.MAX_TEXTURE_SIZE)
    );
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
  return probedMaxTextureSize;
}

/**
 * Smallest width from 2048 up, doubling, that fits `count` texels within the
 * hardware row limit.  The guaranteed 2048 floor already holds 4.2M entries, so
 * every mesh this tool realistically sees takes the first branch and the probe
 * -- which costs a throwaway WebGL context -- never runs at all.
 *
 * @param {number} count
 * @param {string} label  used in the out-of-range message
 * @returns {{width: number, height: number}}
 */
function texelGrid(count, label) {
  if (count <= MIN_TEXTURE_WIDTH * MIN_TEXTURE_WIDTH) {
    return {
      width: MIN_TEXTURE_WIDTH,
      height: Math.max(1, Math.ceil(count / MIN_TEXTURE_WIDTH)),
    };
  }

  const limit = maxTextureSize();
  let width = MIN_TEXTURE_WIDTH;
  while (Math.ceil(count / width) > limit && width * 2 <= limit) width *= 2;

  const height = Math.max(1, Math.ceil(count / width));
  if (height > limit) {
    throw new RangeError(
      `${label}: ${count} entries do not fit a ${limit}x${limit} data texture`
    );
  }
  return { width, height };
}

/**
 * An RGBA32F data texture. NEAREST filtering, CLAMP_TO_EDGE and no mipmaps are
 * mandatory: any other combination leaves the texture incomplete and every
 * texelFetch silently returns (0, 0, 0, 1).
 *
 * @param {Float32Array} data  width * height * 4 floats
 * @param {number} width
 * @param {number} height
 * @returns {THREE.DataTexture}
 */
function floatTexture(data, width, height) {
  const texture = new THREE.DataTexture(
    data,
    width,
    height,
    THREE.RGBAFormat,
    THREE.FloatType
  );
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Scatter a tightly packed (n, 3) array into the xyz of an RGBA texel buffer.
 *
 * @param {Float32Array} dst  texel storage, 4 floats per entry
 * @param {ArrayLike<number>} src  3 values per entry
 * @param {number} count  number of entries to write
 */
function writeVec3(dst, src, count) {
  for (let i = 0; i < count; i++) {
    const s = i * 3;
    const d = i * 4;
    dst[d] = src[s];
    dst[d + 1] = src[s + 1];
    dst[d + 2] = src[s + 2];
  }
}

/**
 * Expand indexed triangles into 3 * nF standalone vertices, in the vertex order
 * the shader assumes (`gl_VertexID / 3` is the triangle, `% 3` the corner).
 *
 * @param {Float32Array} positions  (nV * 3)
 * @param {Uint32Array} indices  (nF * 3)
 * @returns {Float32Array} (nF * 9)
 */
function deIndexPositions(positions, indices) {
  const vertexCount = positions.length / 3;
  const out = new Float32Array(indices.length * 3);
  for (let i = 0; i < indices.length; i++) {
    const v = indices[i];
    /* Reading past the end yields undefined, which stores as NaN and poisons
       the bounding sphere -- the mesh then fails frustum culling and simply
       never appears, with nothing in the console to say why. */
    if (!(v >= 0 && v < vertexCount)) {
      throw new RangeError(
        `triangle index ${v} at position ${i} is outside ${vertexCount} vertices`
      );
    }
    const s = v * 3;
    const d = i * 3;
    out[d] = positions[s];
    out[d + 1] = positions[s + 1];
    out[d + 2] = positions[s + 2];
  }
  return out;
}

const _view = /* @__PURE__ */ new THREE.Matrix4();
const _viewModel = /* @__PURE__ */ new THREE.Matrix4();

export class FieldMaterial {
  #geometry;
  #material;
  #textures;
  #vertexCount;
  #showGrid = true;
  #haveQ = false;
  #haveO = false;

  /**
   * @param {object} spec
   * @param {Float32Array} spec.positions  (nV*3) vertex positions
   * @param {Float32Array} spec.normals    (nV*3) vertex normals
   * @param {Uint32Array}  spec.indices    (nF*3) triangle vertex indices
   * @param {number} spec.scale            target edge length
   * @param {number} spec.rosy             2, 4 or 6
   * @param {number} spec.posy             3 or 4
   */
  constructor({ positions, normals, indices, scale, rosy, posy }) {
    if (!positions || positions.length === 0 || positions.length % 3 !== 0) {
      throw new RangeError('positions must be a non-empty (nV, 3) Float32Array');
    }
    if (!normals || normals.length !== positions.length) {
      throw new RangeError('normals must have the same length as positions');
    }
    if (!indices || indices.length === 0 || indices.length % 3 !== 0) {
      throw new RangeError('indices must be a non-empty (nF, 3) Uint32Array');
    }
    if (rosy !== 2 && rosy !== 4 && rosy !== 6) {
      throw new RangeError(`rosy must be 2, 4 or 6, got ${rosy}`);
    }
    if (posy !== 3 && posy !== 4) {
      throw new RangeError(`posy must be 3 or 4, got ${posy}`);
    }
    if (!(scale > 0)) {
      throw new RangeError(`scale must be positive, got ${scale}`);
    }

    const vertexCount = positions.length / 3;
    const faceCount = indices.length / 3;
    this.#vertexCount = vertexCount;

    const vertexGrid = texelGrid(vertexCount, 'vertex data texture');
    const faceGrid = texelGrid(faceCount, 'triangle data texture');
    const vertexTexels = vertexGrid.width * vertexGrid.height;

    /* Ids go into a float texture rather than a usampler2D: float32 is exact
       well past any mesh this tool handles, and usampler2D has no default
       precision in ESSL 3.00 and is weakly supported on mobile drivers. */
    const faceData = new Float32Array(faceGrid.width * faceGrid.height * 4);
    writeVec3(faceData, indices, faceCount);

    const positionData = new Float32Array(vertexTexels * 4);
    writeVec3(positionData, positions, vertexCount);

    /* The headless session never splits creases, so the shading normal and the
       normal Q and O are defined against are the same array -- the native app's
       no-crease path, where normal_data simply aliases normal. */
    const normalData = new Float32Array(vertexTexels * 4);
    writeVec3(normalData, normals, vertexCount);

    const perVertex = (data) => floatTexture(data, vertexGrid.width, vertexGrid.height);
    const blank = () => new Float32Array(vertexTexels * 4);

    this.#textures = {
      tex_face: floatTexture(faceData, faceGrid.width, faceGrid.height),
      tex_position: perVertex(positionData),
      tex_normal: perVertex(normalData),
      tex_q: perVertex(blank()),
      tex_o: perVertex(blank()),
    };

    this.#material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      /* Surfaces as #define SHADER_NAME, which is what a WebGL debugger shows. */
      name: 'InstantMeshesField',
      /* The browser twin of nanogui's .define("ROSY", "4"); three.js keys its
         program cache on these, so the three symmetry variants coexist. */
      defines: { ROSY: rosy, POSY: posy },
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms: {
        light_position: { value: new THREE.Vector3(0.0, 0.3, 5.0) },
        camera_local: { value: new THREE.Vector3() },
        scale: { value: scale },
        inv_scale: { value: 1.0 / scale },
        show_uvs: { value: 0.0 },
        base_color: { value: new THREE.Vector3(...BASE_COLOR) },
        specular_color: { value: new THREE.Vector3(...SPECULAR_COLOR) },
        interior_factor: { value: new THREE.Vector3(...INTERIOR_FACTOR) },
        edge_factor_0: { value: new THREE.Vector3(...EDGE_FACTOR_0) },
        edge_factor_1: { value: new THREE.Vector3(...EDGE_FACTOR_1) },
        edge_factor_2: { value: new THREE.Vector3(...EDGE_FACTOR_2) },
        vertex_tex_width: { value: vertexGrid.width },
        face_tex_width: { value: faceGrid.width },
        tex_face: { value: this.#textures.tex_face },
        tex_position: { value: this.#textures.tex_position },
        tex_normal: { value: this.#textures.tex_normal },
        tex_q: { value: this.#textures.tex_q },
        tex_o: { value: this.#textures.tex_o },
      },
      /* The vertex shader reproduces the geometry shader's back-face test, and
         the native app never enables GL_CULL_FACE. */
      side: THREE.DoubleSide,
      depthTest: true,
      depthWrite: true,
      depthFunc: THREE.LessEqualDepth,
      /* viewer.cpp:2462-2463, so a wireframe overlay wins the depth test. */
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });

    /* The shader reads positions from tex_position, but three.js takes the draw
       count from this attribute, and real values also give a correct bounding
       sphere and a working Raycaster. */
    this.#geometry = new THREE.BufferGeometry();
    this.#geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(deIndexPositions(positions, indices), 3)
    );
  }

  /** THREE.BufferGeometry of de-indexed triangles, ready for a THREE.Mesh. */
  get geometry() {
    return this.#geometry;
  }

  /** THREE.RawShaderMaterial to pair with that geometry. */
  get material() {
    return this.#material;
  }

  /**
   * Upload new fields. Either may be null to leave it unchanged.
   *
   * @param {Float32Array|null} q (nV*3)
   * @param {Float32Array|null} o (nV*3)
   */
  updateField(q, o) {
    /* Both lengths are checked before either upload: rejecting the second half
       of an update after the first has landed would pair a fresh Q with a stale
       O and keep drawing that as if it were a solution. */
    if (q) this.#checkFieldLength(q, 'q');
    if (o) this.#checkFieldLength(o, 'o');

    if (q) {
      this.#uploadField(this.#textures.tex_q, q);
      this.#haveQ = true;
    }
    if (o) {
      this.#uploadField(this.#textures.tex_o, o);
      this.#haveO = true;
    }
    this.#refreshShowUvs();
  }

  /** Toggle the grid overlay; when false the mesh renders plain-shaded. */
  setShowGrid(enabled) {
    this.#showGrid = Boolean(enabled);
    this.#refreshShowUvs();
  }

  /** Base surface colour, three floats 0..1. */
  setBaseColor(r, g, b) {
    this.#material.uniforms.base_color.value.set(r, g, b);
  }

  /**
   * Must be called every frame before rendering, with the camera, so the shader
   * gets camera_local and the light position the native app uses.
   *
   * @param {THREE.Camera} camera
   * @param {THREE.Object3D} object  the mesh carrying this geometry
   */
  update(camera, object) {
    camera.updateWorldMatrix(true, false);
    object.updateWorldMatrix(true, false);

    /* viewer.cpp:2408-2409: the eye in the mesh's object space, which is the
       frame tex_position lives in and therefore the frame the back-face test
       must be evaluated in. */
    _view.copy(camera.matrixWorld).invert();
    _viewModel.multiplyMatrices(_view, object.matrixWorld).invert();

    const uniforms = this.#material.uniforms;
    uniforms.camera_local.value.setFromMatrixPosition(_viewModel);
    uniforms.light_position.value
      .copy(LIGHT_IN_EYE_SPACE)
      .applyMatrix4(camera.matrixWorld);
  }

  /** Free all GPU resources. */
  dispose() {
    for (const texture of Object.values(this.#textures)) texture.dispose();
    this.#material.dispose();
    this.#geometry.dispose();
  }

  /**
   * @param {Float32Array} field  (nV*3)
   * @param {string} label
   */
  #checkFieldLength(field, label) {
    if (field.length !== this.#vertexCount * 3) {
      throw new RangeError(
        `field ${label} has ${field.length} values, expected ${this.#vertexCount * 3}`
      );
    }
  }

  /**
   * @param {THREE.DataTexture} texture
   * @param {Float32Array} field  (nV*3), already length-checked
   */
  #uploadField(texture, field) {
    writeVec3(texture.image.data, field, this.#vertexCount);
    texture.needsUpdate = true;
  }

  /**
   * The native gate is `attribVersion("uv") > 0 && visualization == 1`: no grid
   * until a field has actually arrived, so a zeroed texture is never drawn as
   * if it were a solution.  Q is included because the vertex shader normalizes
   * it, and normalizing a zero vector yields NaN.
   */
  #refreshShowUvs() {
    const on = this.#showGrid && this.#haveQ && this.#haveO;
    this.#material.uniforms.show_uvs.value = on ? 1.0 : 0.0;
  }
}

/**
 * Line geometry for the wireframe of the same de-indexed triangles, so the
 * input mesh can be drawn as an overlay of lines.
 *
 * Every triangle contributes its own three edges and shared edges are drawn
 * twice, which is what glPolygonMode(GL_LINE) did natively.  Vertex ids match
 * FieldMaterial's geometry, so FieldMaterial's vertex shader would also cull
 * the hidden lines correctly; pairing it with a plain line material and letting
 * the depth test hide them is equivalent for closed meshes.  The native colour
 * is vec4(0.1, 0.1, 0.2, 1.0) (viewer.cpp:2482).
 *
 * @param {Float32Array} positions  (nV*3)
 * @param {Uint32Array} indices  (nF*3)
 * @returns {THREE.BufferGeometry}
 */
export function buildWireframeGeometry(positions, indices) {
  const faceCount = indices.length / 3;
  const edges = new Uint32Array(faceCount * 6);
  for (let f = 0; f < faceCount; f++) {
    const v = f * 3;
    const e = f * 6;
    edges[e] = v;
    edges[e + 1] = v + 1;
    edges[e + 2] = v + 1;
    edges[e + 3] = v + 2;
    edges[e + 4] = v + 2;
    edges[e + 5] = v;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(deIndexPositions(positions, indices), 3)
  );
  geometry.setIndex(new THREE.Uint32BufferAttribute(edges, 1));
  return geometry;
}
