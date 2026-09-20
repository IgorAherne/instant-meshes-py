/*
    session.h: headless, GUI-free driver for the Instant Meshes pipeline.

    Viewer (src/viewer.cpp) couples the algorithm to NanoGUI, GLFW and OpenGL.
    This class reproduces the parts of that state machine that matter -- loading,
    preprocessing, brush strokes as field constraints, singularity attractors,
    interruptible solving and extraction -- with no windowing dependency, so it
    can drive a browser-based UI through the Python bindings.

    Everything that touches the hierarchy takes mRes.mutex(), because the
    Optimizer runs its sweeps on a background thread that holds the same lock.

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"
#include "hierarchy.h"
#include "field.h"
#include "meshstats.h"
#include "smoothcurve.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

class BVH;

namespace instant_meshes {

/// Brush flavours that leave a persistent constraint behind.
enum StrokeKind {
    /** Orientation comb: aligns the cross field with the stroke tangent. */
    StrokeOrientation = 0,
    /** Edge brush: additionally pins the position field onto the stroke. */
    StrokeEdge = 1
};

struct Stroke {
    uint32_t id;
    int kind;
    std::vector<CurvePoint> curve;
};

/// Snapshot of the solver's state, cheap enough to poll at display rate.
struct SolveStatus {
    bool active;          ///< a solve is in flight
    float progress;       ///< [0,1]; 1 while running non-hierarchically
    int level;            ///< current hierarchy level
    int iterationsQ;      ///< bumped whenever Q(0) changes -- use as a version
    int iterationsO;      ///< bumped whenever O(0) changes -- use as a version
    /// Message from the last solver-thread failure, or empty. Reading clears it.
    std::string error;
};

/// Result of a full extraction, in the layout the viewer uploads to the GPU.
struct ExtractedMesh {
    MatrixXf V;    ///< 3 x nV  extracted vertex positions
    MatrixXu F;    ///< posy x nF face indices (degenerate quads repeat the last index)
    MatrixXf Nf;   ///< 3 x nF  face normals
    /// 3 x (2 * nF * posy) line-segment endpoints, ready to draw as GL_LINES.
    MatrixXf wireframe;
    /// 3 x (2 * nF * posy) per-endpoint colour; black marks a triangle's fake edge.
    MatrixXf wireframeColor;
};

struct Config {
    int rosy = 4;                  ///< 2, 4 or 6
    int posy = 4;                  ///< 3, 4 (6 is expressed as posy == 3)
    Float scale = -1;              ///< target edge length; < 0 to derive it
    int faceCount = -1;            ///< target face count; < 0 to derive it
    int vertexCount = -1;          ///< target vertex count; < 0 to derive it
    Float creaseAngle = -1;        ///< degrees; < 0 disables crease detection
    bool extrinsic = true;
    bool alignToBoundaries = false;
    bool deterministic = false;    ///< reproducible at the cost of some speed
    int smoothIter = 2;            ///< reprojection passes during extraction
    bool pureQuad = false;         ///< subdivide to eliminate triangles
};

/**
 * One editing session: a mesh, its multiresolution hierarchy, a BVH, a solver
 * thread and the set of brush strokes currently constraining the field.
 *
 * Typical lifecycle:
 *   setMesh(V, F) / loadFile(path)
 *   preprocess(cfg)
 *   solveOrientations(-1); waitSolve(); solvePositions(-1); waitSolve();
 *   ... projectStroke() / addStroke() / solve again ...
 *   extract()
 */
class Session {
public:
    Session();
    ~Session();

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;

    /* ------------------------------------------------------------------ */
    /*  Input                                                             */
    /* ------------------------------------------------------------------ */

    /// Set the input triangle mesh. V is 3 x nV, F is 3 x nF.
    void setMesh(const MatrixXf &V, const MatrixXu &F);

    /// Load an OBJ/PLY/OFF file from disk.
    void loadFile(const std::string &path);

    /* ------------------------------------------------------------------ */
    /*  Preprocessing                                                     */
    /* ------------------------------------------------------------------ */

    /**
     * Subdivide (when the input is too coarse), build the directed-edge and
     * adjacency structures, normals, dual areas, the multiresolution hierarchy
     * and the BVH.  Mirrors Viewer::loadInput / batch_process.
     *
     * Safe to call again with a different Config to re-target the resolution.
     * Brush strokes are carried over: they are paths in space, so rebuilding
     * only invalidates the face indices inside them, which reprojectStrokes()
     * restores.  The solved fields are not -- they are discarded and have to
     * be solved again, which also means an attractor's effect does not
     * survive: it moved a singularity in a field that no longer exists.
     */
    void preprocess(const Config &cfg, const ProgressCallback &progress = ProgressCallback());

    bool ready() const { return mReady; }
    const Config &config() const { return mConfig; }

    /**
     * Change the two options only extract() reads.
     *
     * Everything else in Config is baked into the hierarchy by preprocess(),
     * so changing it means rebuilding and losing every stroke.  These two are
     * consumed at extraction time, so a caller that wants to re-extract with,
     * say, pure quads should not have to pay for -- or be punished by -- a
     * rebuild it does not need.
     */
    void setExtractionOptions(int smoothIter, bool pureQuad);

    /* ------------------------------------------------------------------ */
    /*  Geometry of the working (post-subdivision) mesh                   */
    /* ------------------------------------------------------------------ */

    const MatrixXf &vertices() const;      ///< 3 x nV
    const MatrixXu &faces() const;         ///< 3 x nF
    const MatrixXf &normals() const;       ///< 3 x nV
    Float scale() const;                   ///< target edge length
    const MeshStats &stats() const { return mStats; }

    /// Vertices in the working mesh; the length of every per-vertex array.
    uint32_t vertexCount() const;

    /// Copy of the orientation field at level 0 (3 x nV). Takes the lock.
    MatrixXf orientationField() const;
    /// Copy of the position field at level 0 (3 x nV). Takes the lock.
    MatrixXf positionField() const;

    /**
     * Copy a field straight into caller-owned storage of 3 * vertexCount()
     * floats, laid out x,y,z per vertex.
     *
     * A browser session polls these many times a second; going through
     * orientationField() would copy the whole array twice (once out of the
     * hierarchy, once into the destination buffer).
     */
    void copyOrientationField(float *destination) const;
    void copyPositionField(float *destination) const;

    /**
     * Vertices duplicated along sharp creases: duplicate index -> original.
     * Empty unless Config::creaseAngle >= 0.  A renderer that wants the same
     * faceting as the desktop app expands its arrays with this map.
     */
    const std::map<uint32_t, uint32_t> &creaseMap() const { return mCreaseMap; }

    /* ------------------------------------------------------------------ */
    /*  Brush strokes                                                     */
    /* ------------------------------------------------------------------ */

    /**
     * Ray-cast a polyline onto the surface and smooth it along the mesh,
     * reproducing Viewer::mouseButtonEvent's release path.
     *
     * Rays are given in the mesh's own coordinate system (3 x N each), which
     * keeps every projection convention on the client side where the camera
     * actually lives.  Returns false if any sample misses the surface or the
     * smoothing fails, in which case the stroke should be discarded.
     *
     * `attractor` selects the geodesic mode used for singularity dragging.
     */
    bool projectStroke(const MatrixXf &rayOrigins, const MatrixXf &rayDirections,
                       bool attractor, std::vector<CurvePoint> &out) const;

    /// Add a stroke and re-apply all field constraints. Returns its id.
    uint32_t addStroke(int kind, const std::vector<CurvePoint> &curve);

    /// Remove a stroke by id and re-apply constraints. False if unknown.
    bool eraseStroke(uint32_t id);

    /**
     * Remove the stroke whose start marker is closest to `point` and visible
     * from `eye`, within `radius` of `point`.  This is the headless equivalent
     * of clicking a stroke's delete handle.  Returns the erased id or 0.
     */
    uint32_t eraseStrokeNear(const Vector3f &point, const Vector3f &eye, Float radius);

    void clearStrokes();
    const std::vector<Stroke> &strokes() const { return mStrokes; }

    /**
     * Drag a singularity along `curve` (an attractor stroke).
     *
     * Starts a level-0 solve with the integer variables left frozen -- the mode
     * the GUI's attractor tools use -- and queues the move, which the solver
     * applies one face per sweep.  The solve keeps running until stopSolve();
     * poll status() to watch the singularity travel.
     *
     * Leaving the integers frozen is what makes the move a local edit instead
     * of a full re-solve, so this deliberately does not honour brush strokes.
     */
    void applyAttractor(const std::vector<CurvePoint> &curve, bool orientation);

    /* ------------------------------------------------------------------ */
    /*  Solving                                                           */
    /* ------------------------------------------------------------------ */

    /**
     * Start smoothing the orientation field.
     *
     * level < 0 runs the full hierarchical schedule (coarse to fine, six sweeps
     * per level) and stops on its own.  level == 0 refines in place at full
     * resolution and runs until stopSolve(), which is far cheaper and is what
     * to use for live feedback while brushing.
     *
     * Both modes thaw the integer variables first, because the frozen sweep in
     * field.cpp deliberately ignores CQ/CQw -- leaving them frozen would make a
     * brush stroke have no effect at all.  Only applyAttractor() wants the
     * frozen behaviour, and it manages that itself.
     */
    void solveOrientations(int level = -1);
    void solvePositions(int level = -1);

    void stopSolve();                 ///< ask the solver to finish its sweep
    void waitSolve();                 ///< block until the solve finishes
    SolveStatus status() const;

    /**
     * Run both fields to completion synchronously: orientations, then
     * positions. Equivalent to what batch_process does.
     */
    void solveAll();

    /* ------------------------------------------------------------------ */
    /*  Results                                                           */
    /* ------------------------------------------------------------------ */

    /// Face index -> singularity index, as drawn by the GUI.
    std::map<uint32_t, uint32_t> orientationSingularities() const;
    std::map<uint32_t, Vector2i> positionSingularities() const;

    /// Extract the quad/triangle mesh from the current fields.
    ExtractedMesh extract();

    /**
     * Write an extracted mesh to disk; the format follows the extension
     * (.obj or .ply).
     *
     * `faceNormals` is ignored for PLY, where writing them produces a file many
     * readers reject -- see the implementation for why.
     */
    void writeMesh(const std::string &path, const ExtractedMesh &mesh,
                   bool faceNormals = true) const;

    /* ------------------------------------------------------------------ */
    /*  Tuning                                                            */
    /* ------------------------------------------------------------------ */

    /// How often (ms) a hierarchical solve propagates its state down to level 0
    /// so the client can display it.  Lower is more responsive, slightly slower.
    void setPreviewInterval(int ms);

private:
    /// Rebuild CQ/CO/CQw/COw from mStrokes. Mirrors Viewer::refreshStrokes.
    void applyConstraints();

    /**
     * Carry strokes across a rebuild.
     *
     * The curve is a path in space and preprocess() only refines the surface
     * it was drawn on, so re-projecting it costs one short ray per point --
     * far better than the alternative, which is losing the user's work every
     * time they nudge the target resolution.  A point whose surface has gone
     * is dropped, and a stroke left with fewer than two is.
     */
    void reprojectStrokes(const std::vector<Stroke> &strokes);

    /**
     * Re-root an attractor stroke on the singularity it was aimed at.
     *
     * A singularity occupies one triangle of the working mesh, while the
     * marker drawn over it is many times that size, so a drag that starts on
     * the dot usually does not start on the face -- and the move then does
     * nothing.  Returns `curve` unchanged when it already starts on one, or
     * when nothing is near enough to have been meant.
     */
    std::vector<CurvePoint> snapToSingularity(const std::vector<CurvePoint> &curve,
                                              bool orientation) const;

    void requireReady() const;

    MatrixXf mV;                 ///< input vertices, consumed by preprocess()
    MatrixXu mF;                 ///< input faces, consumed by preprocess()

    MeshStats mStats;
    MultiResolutionHierarchy mRes;
    std::unique_ptr<Optimizer> mOptimizer;
    BVH *mBVH = nullptr;

    VectorXb mBoundaryVertices, mNonmanifoldVertices;
    std::set<uint32_t> mCreaseSet;
    std::map<uint32_t, uint32_t> mCreaseMap;

    std::vector<Stroke> mStrokes;
    uint32_t mNextStrokeId = 1;

    Config mConfig;
    bool mReady = false;
};

/// Silence the algorithm's chatty std::cout progress reports.
void setVerbose(bool verbose);
bool verbose();

/// Number of worker threads used by the internal pool (-1 = all cores).
void setThreadCount(int n);

} // namespace instant_meshes
