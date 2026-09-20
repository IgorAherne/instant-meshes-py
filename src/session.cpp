/*
    session.cpp: headless driver for the Instant Meshes pipeline.

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "session.h"

#include "adjacency.h"
#include "bvh.h"
#include "dedge.h"
#include "extract.h"
#include "meshio.h"
#include "normal.h"
#include "subdivide.h"

#include <stdexcept>

/* Sized by setThreadCount(); defined in globals.cpp alongside the core. */
extern int nprocs;

namespace instant_meshes {

namespace {

/// Swallows everything written to it, used by setVerbose(false).
class NullBuffer : public std::streambuf {
protected:
    int overflow(int c) override { return c; }
};

NullBuffer g_nullBuffer;
std::streambuf *g_coutBuffer = nullptr;
bool g_verbose = true;

} // namespace

void setVerbose(bool value) {
    if (value == g_verbose)
        return;
    g_verbose = value;
    if (value) {
        if (g_coutBuffer)
            std::cout.rdbuf(g_coutBuffer);
        g_coutBuffer = nullptr;
    } else {
        g_coutBuffer = std::cout.rdbuf(&g_nullBuffer);
    }
}

bool verbose() { return g_verbose; }

void setThreadCount(int n) {
    nprocs = n;
#if defined(INSTANT_MESHES_TBB_SHIM)
    /* Applies immediately; the solver thread's task_scheduler_init would only
       take effect the next time it is constructed. */
    tbb::detail::Scheduler::get().setConcurrency(n);
#endif
}

// ---------------------------------------------------------------------------

Session::Session() { }

Session::~Session() {
    if (mOptimizer) {
        {
            /* stop() calls propagateSolution(), which rewrites the hierarchy;
               doing that while the solver thread is mid-sweep would be a race,
               so take the lock exactly as the desktop app does. */
            std::lock_guard<ordered_lock> lock(mRes.mutex());
            mOptimizer->stop();
        }
        /* shutdown() joins the solver thread, which first has to re-acquire the
           lock released above. */
        mOptimizer->shutdown();
        mOptimizer.reset();
    }
    /* The BVH holds raw pointers into the hierarchy, so it has to go before the
       members are destroyed. */
    delete mBVH;
    mBVH = nullptr;
}

void Session::requireReady() const {
    if (!mReady)
        throw std::runtime_error("Session: preprocess() must be called first");
}

// ---------------------------------------------------------------------------
//  Input
// ---------------------------------------------------------------------------

void Session::setMesh(const MatrixXf &V, const MatrixXu &F) {
    if (V.rows() != 3)
        throw std::runtime_error("Session::setMesh: vertices must be a 3 x N matrix");
    if (F.rows() != 3)
        throw std::runtime_error("Session::setMesh: faces must be a 3 x N matrix");
    if (V.cols() == 0 || F.cols() == 0)
        throw std::runtime_error("Session::setMesh: empty mesh");
    if (F.maxCoeff() >= (uint32_t) V.cols())
        throw std::runtime_error("Session::setMesh: face index out of range");

    mV = V;
    mF = F;
    mReady = false;
    /* A different input mesh is a different subject: preprocess() now carries
       strokes across a rebuild, and without this they would be re-projected
       onto whatever happened to occupy the same space. */
    mStrokes.clear();
}

void Session::loadFile(const std::string &path) {
    MatrixXf V, N;
    MatrixXu F;
    load_mesh_or_pointcloud(path, F, V, N);
    if (F.size() == 0)
        throw std::runtime_error("Session::loadFile: point clouds are not supported");
    mV = std::move(V);
    mF = std::move(F);
    mReady = false;
    /* A different input mesh is a different subject: preprocess() now carries
       strokes across a rebuild, and without this they would be re-projected
       onto whatever happened to occupy the same space. */
    mStrokes.clear();
}

// ---------------------------------------------------------------------------
//  Preprocessing
// ---------------------------------------------------------------------------

void Session::preprocess(const Config &cfg, const ProgressCallback &progress) {
    if (mV.cols() == 0)
        throw std::runtime_error("Session::preprocess: no mesh loaded");
    if (cfg.rosy != 2 && cfg.rosy != 4 && cfg.rosy != 6)
        throw std::runtime_error("Session::preprocess: rosy must be 2, 4 or 6");
    if (cfg.posy != 3 && cfg.posy != 4)
        throw std::runtime_error("Session::preprocess: posy must be 3 or 4");

    /* Tear down any previous run before the hierarchy is rebuilt under it. */
    if (mOptimizer) {
        {
            std::lock_guard<ordered_lock> lock(mRes.mutex());
            mOptimizer->stop();
        }
        mOptimizer->shutdown();
        mOptimizer.reset();
    }
    delete mBVH;
    mBVH = nullptr;
    /* Strokes survive: they are re-projected onto the rebuilt mesh at the end
       of this function. Only the face indices inside them go stale -- the
       curve itself is a path in space, and subdivision refines the surface it
       was drawn on rather than moving it. */
    std::vector<Stroke> carried;
    carried.swap(mStrokes);
    mCreaseMap.clear();
    mCreaseSet.clear();
    mReady = false;

    mConfig = cfg;

    /* Work on copies: subdivision rewrites V and F in place, and the caller
       should be able to re-target the resolution without reloading. */
    MatrixXf V = mV;
    MatrixXu F = mF;

    mStats = compute_mesh_stats(F, V, cfg.deterministic, progress);

    Float scale = cfg.scale;
    int faceCount = cfg.faceCount;
    int vertexCount = cfg.vertexCount;

    if (scale < 0 && vertexCount < 0 && faceCount < 0)
        vertexCount = (int) (V.cols() / 16);

    /* Identical derivation to batch.cpp:78-90 and Viewer::loadInput. */
    if (scale > 0) {
        Float faceArea = cfg.posy == 4 ? (scale * scale)
                                       : (std::sqrt(3.f) / 4.f * scale * scale);
        faceCount = (int) (mStats.mSurfaceArea / faceArea);
        vertexCount = cfg.posy == 4 ? faceCount : (faceCount / 2);
    } else if (faceCount > 0) {
        Float faceArea = mStats.mSurfaceArea / faceCount;
        vertexCount = cfg.posy == 4 ? faceCount : (faceCount / 2);
        scale = cfg.posy == 4 ? std::sqrt(faceArea)
                              : (2 * std::sqrt(faceArea * std::sqrt(1.f / 3.f)));
    } else if (vertexCount > 0) {
        faceCount = cfg.posy == 4 ? vertexCount : (vertexCount * 2);
        Float faceArea = mStats.mSurfaceArea / faceCount;
        scale = cfg.posy == 4 ? std::sqrt(faceArea)
                              : (2 * std::sqrt(faceArea * std::sqrt(1.f / 3.f)));
    }
    mConfig.scale = scale;
    mConfig.faceCount = faceCount;
    mConfig.vertexCount = vertexCount;

    /* Subdivide when the input cannot represent the requested edge length. */
    if (mStats.mMaximumEdgeLength * 2 > scale ||
        mStats.mMaximumEdgeLength > mStats.mAverageEdgeLength * 2) {
        VectorXu V2E, E2E;
        build_dedge(F, V, V2E, E2E, mBoundaryVertices, mNonmanifoldVertices, progress);
        subdivide(F, V, V2E, E2E, mBoundaryVertices, mNonmanifoldVertices,
                  std::min(scale / 2, (Float) mStats.mAverageEdgeLength * 2),
                  cfg.deterministic, progress);
        mStats = compute_mesh_stats(F, V, cfg.deterministic, progress);
    }

    mRes.free();
    mRes.setF(std::move(F));
    mRes.setV(std::move(V));

    VectorXu V2E, E2E;
    build_dedge(mRes.F(), mRes.V(), V2E, E2E, mBoundaryVertices,
                mNonmanifoldVertices, progress);

    AdjacencyMatrix adj = generate_adjacency_matrix_uniform(
        mRes.F(), V2E, E2E, mNonmanifoldVertices, progress);

    MatrixXf N;
    if (cfg.creaseAngle >= 0) {
        /* generate_crease_normals duplicates vertices along sharp edges; the
           hierarchy keeps the original vertex count and mCreaseMap records
           duplicate -> original so a renderer can expand the arrays again. */
        MatrixXf V_crease = mRes.V();
        MatrixXu F_crease = mRes.F();
        MatrixXf N_crease;
        generate_crease_normals(F_crease, V_crease, V2E, E2E, mBoundaryVertices,
                                mNonmanifoldVertices, cfg.creaseAngle, N_crease,
                                mCreaseMap, progress);
        N = N_crease.topLeftCorner(3, mRes.V().cols());
        for (auto const &kv : mCreaseMap)
            mCreaseSet.insert(kv.second);
    } else {
        generate_smooth_normals(mRes.F(), mRes.V(), V2E, E2E,
                                mNonmanifoldVertices, N, progress);
    }

    VectorXf A;
    compute_dual_vertex_areas(mRes.F(), mRes.V(), V2E, E2E, mNonmanifoldVertices, A);

    mRes.setE2E(std::move(E2E));
    mRes.setAdj(std::move(adj));
    mRes.setN(std::move(N));
    mRes.setA(std::move(A));
    mRes.setScale(scale);
    mRes.build(cfg.deterministic, progress);
    mRes.resetSolution();

    mBVH = new BVH(&mRes.F(), &mRes.V(), &mRes.N(), mStats.mAABB);
    mBVH->build(progress);

    mOptimizer.reset(new Optimizer(mRes, true));
    mOptimizer->setRoSy(cfg.rosy);
    mOptimizer->setPoSy(cfg.posy);
    mOptimizer->setExtrinsic(cfg.extrinsic);

    mReady = true;

    reprojectStrokes(carried);

    /* Boundary alignment is expressed as constraints, so it goes through the
       same path as brush strokes. */
    applyConstraints();
}

void Session::reprojectStrokes(const std::vector<Stroke> &strokes) {
    mStrokes.clear();
    if (strokes.empty())
        return;

    /* Far enough above the surface to clear the floating-point noise of the
       point that was projected onto it, near enough not to reach the other
       side of a thin wall. */
    const Float lift = mStats.mAverageEdgeLength;

    std::vector<CurvePoint> curve;
    for (const Stroke &stroke : strokes) {
        curve.clear();
        curve.reserve(stroke.curve.size());

        for (const CurvePoint &old : stroke.curve) {
            /* Straight down the stored normal, from just above and then, if
               that found nothing, from just below: a normal can flip when the
               crease setting changes, and the point is on the surface either
               way. */
            Vector2f uv;
            uint32_t f;
            Float t;
            bool hit = mBVH->rayIntersect(
                Ray(old.p + old.n * lift, -old.n, 0, 2 * lift), f, t, &uv);
            if (!hit) {
                hit = mBVH->rayIntersect(
                    Ray(old.p - old.n * lift, old.n, 0, 2 * lift), f, t, &uv);
                if (!hit)
                    continue;
            }

            CurvePoint pt;
            pt.p = old.p;  /* the curve itself, not the point the ray landed on */
            pt.n = ((1 - uv.sum()) * mRes.N().col(mRes.F()(0, f)) +
                    uv.x() * mRes.N().col(mRes.F()(1, f)) +
                    uv.y() * mRes.N().col(mRes.F()(2, f))).normalized();
            pt.f = f;
            curve.push_back(pt);
        }

        /* Re-smoothing walks the new mesh between the points, which is what
           makes the face sequence contiguous again -- the constraint builder
           needs that, and a curve carried over from a coarser mesh will have
           gaps in it. */
        if (curve.size() < 2 || !smooth_curve(mBVH, mRes.E2E(), curve, false))
            continue;

        Stroke moved;
        moved.id = stroke.id;
        moved.kind = stroke.kind;
        moved.curve = curve;
        mStrokes.push_back(std::move(moved));
    }
}

// ---------------------------------------------------------------------------
//  Geometry accessors
// ---------------------------------------------------------------------------

const MatrixXf &Session::vertices() const { requireReady(); return mRes.V(); }
const MatrixXu &Session::faces() const { requireReady(); return mRes.F(); }
const MatrixXf &Session::normals() const { requireReady(); return mRes.N(); }
Float Session::scale() const { requireReady(); return mRes.scale(); }

MatrixXf Session::orientationField() const {
    requireReady();
    std::lock_guard<ordered_lock> lock(const_cast<MultiResolutionHierarchy &>(mRes).mutex());
    return mRes.Q(0);
}

MatrixXf Session::positionField() const {
    requireReady();
    std::lock_guard<ordered_lock> lock(const_cast<MultiResolutionHierarchy &>(mRes).mutex());
    return mRes.O(0);
}

uint32_t Session::vertexCount() const {
    requireReady();
    return (uint32_t) mRes.V().cols();
}

void Session::copyOrientationField(float *destination) const {
    requireReady();
    std::lock_guard<ordered_lock> lock(const_cast<MultiResolutionHierarchy &>(mRes).mutex());
    const MatrixXf &Q = mRes.Q(0);
    std::memcpy(destination, Q.data(), sizeof(float) * (size_t) Q.size());
}

void Session::copyPositionField(float *destination) const {
    requireReady();
    std::lock_guard<ordered_lock> lock(const_cast<MultiResolutionHierarchy &>(mRes).mutex());
    const MatrixXf &O = mRes.O(0);
    std::memcpy(destination, O.data(), sizeof(float) * (size_t) O.size());
}

// ---------------------------------------------------------------------------
//  Strokes
// ---------------------------------------------------------------------------

namespace {
/**
 * How squarely the surface has to face the camera for a stroke to *end* there.
 *
 * |cos| between the interpolated normal and the view ray: 1 head-on, 0 on the
 * silhouette.  0.3 is about 17 degrees around the outline of a smooth shape,
 * which on a screen is a few percent of its radius -- just inside the edge,
 * which is what makes the endpoint land on surface the user could see rather
 * than on a sliver pointing away from them.
 */
constexpr Float MIN_STROKE_FACING = (Float) 0.3;
}  // namespace

bool Session::projectStroke(const MatrixXf &rayOrigins, const MatrixXf &rayDirections,
                            bool attractor, std::vector<CurvePoint> &out) const {
    requireReady();
    if (rayOrigins.rows() != 3 || rayDirections.rows() != 3)
        throw std::runtime_error("Session::projectStroke: rays must be 3 x N matrices");
    if (rayOrigins.cols() != rayDirections.cols())
        throw std::runtime_error("Session::projectStroke: origin/direction count mismatch");

    out.clear();
    const uint32_t count = (uint32_t) rayOrigins.cols();
    if (count == 0)
        return false;

    const MatrixXf &N = mRes.N();
    const MatrixXu &F = mRes.F();

    std::vector<CurvePoint> points((size_t) count);
    std::vector<Float> facing((size_t) count, (Float) 0);
    std::vector<uint8_t> hit((size_t) count, 0);

    for (uint32_t i = 0; i < count; ++i) {
        Ray ray(rayOrigins.col(i), rayDirections.col(i).normalized());
        Vector2f uv;
        uint32_t f;
        Float t;

        if (!mBVH->rayIntersect(ray, f, t, &uv))
            continue;

        CurvePoint &pt = points[i];
        pt.p = ray(t);
        pt.n = ((1 - uv.sum()) * N.col(F(0, f)) + uv.x() * N.col(F(1, f)) +
                uv.y() * N.col(F(2, f))).normalized();
        pt.f = f;
        /* 1 where the surface faces the camera squarely, 0 on the silhouette. */
        facing[i] = std::abs(pt.n.dot(ray.d));
        hit[i] = 1;
    }

    /* The longest uninterrupted run of hits.
       A drag that begins beside the model, or crosses a hole and comes back,
       is still a perfectly clear instruction, so the rays that miss are
       dropped rather than rejecting the whole stroke -- which is what the GUI
       does, because there a stroke cannot start anywhere but on the surface.
       Taking the longest run rather than every hit is what stops a stroke that
       left the model and returned from being stitched together across the gap
       with a straight line through empty space. */
    uint32_t begin = 0, length = 0, runStart = 0, runLength = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!hit[i]) {
            runLength = 0;
            continue;
        }
        if (runLength++ == 0)
            runStart = i;
        if (runLength > length) {
            begin = runStart;
            length = runLength;
        }
    }
    if (length < 2)
        return false;

    uint32_t first = begin, last = begin + length - 1;
    if (!attractor) {
        /* Step the ends inside the silhouette.
           Whatever survived above now starts exactly where the surface turns
           away from the camera, and there a comb stroke lands on whichever
           side-facing sliver happens to be under the cursor -- combing a
           direction the user cannot see, which shows up as a kink in the flow
           along the outline.  Walking in past the grazing samples puts the
           endpoints on surface they were actually looking at.

           An attractor is exempt: its first point has to stay on the singular
           face the user started the drag from. */
        uint32_t f = first, l = last;
        while (f < l && facing[f] < MIN_STROKE_FACING) ++f;
        while (l > f && facing[l] < MIN_STROKE_FACING) --l;
        /* A stroke drawn entirely across a grazing band is still what was
           asked for; only take the eroded ends if something is left. */
        if (l > f) {
            first = f;
            last = l;
        }
    }

    out.reserve((size_t) (last - first + 1));
    for (uint32_t i = first; i <= last; ++i)
        out.push_back(points[i]);

    if (!smooth_curve(mBVH, mRes.E2E(), out, attractor)) {
        out.clear();
        return false;
    }
    return true;
}

uint32_t Session::addStroke(int kind, const std::vector<CurvePoint> &curve) {
    requireReady();
    if (kind != StrokeOrientation && kind != StrokeEdge)
        throw std::runtime_error("Session::addStroke: kind must be 0 (comb) or 1 (edge)");
    if (curve.size() < 2)
        throw std::runtime_error("Session::addStroke: a stroke needs at least two points");

    Stroke stroke;
    stroke.id = mNextStrokeId++;
    stroke.kind = kind;
    stroke.curve = curve;
    mStrokes.push_back(std::move(stroke));
    applyConstraints();
    return mStrokes.back().id;
}

bool Session::eraseStroke(uint32_t id) {
    requireReady();
    for (auto it = mStrokes.begin(); it != mStrokes.end(); ++it) {
        if (it->id == id) {
            mStrokes.erase(it);
            applyConstraints();
            return true;
        }
    }
    return false;
}

uint32_t Session::eraseStrokeNear(const Vector3f &point, const Vector3f &eye, Float radius) {
    requireReady();
    const Float lift = mStats.mAverageEdgeLength / 10;
    Float bestDistance = radius;
    uint32_t bestId = 0;

    for (auto const &stroke : mStrokes) {
        /* The GUI anchors a stroke's delete handle slightly above its first
           point so the marker is not buried inside the surface. */
        Vector3f anchor = stroke.curve[0].p + stroke.curve[0].n * lift;
        Float distance = (anchor - point).norm();
        if (distance > bestDistance)
            continue;
        /* Ignore handles hidden behind the model, as the GUI does. */
        if (mBVH->rayIntersect(Ray(anchor, eye - anchor, 0.0f, 1.0f)))
            continue;
        bestDistance = distance;
        bestId = stroke.id;
    }

    if (bestId != 0)
        eraseStroke(bestId);
    return bestId;
}

void Session::clearStrokes() {
    requireReady();
    if (mStrokes.empty())
        return;
    mStrokes.clear();
    applyConstraints();
}

std::vector<CurvePoint> Session::snapToSingularity(const std::vector<CurvePoint> &curve,
                                                   bool orientation) const {
    if (curve.empty())
        return curve;

    /* Collect the singular faces of the field this attractor edits. */
    std::set<uint32_t> singular;
    if (orientation) {
        for (auto const &kv : orientationSingularities())
            singular.insert(kv.first);
    } else {
        for (auto const &kv : positionSingularities())
            singular.insert(kv.first);
    }
    if (singular.empty() || singular.count(curve.front().f))
        return curve;

    /* The nearest one the user could plausibly have been aiming at. The marker
       is drawn at mRes.scale() * 0.4 across, so a couple of edge lengths is
       several times its own width -- wide enough to forgive the aim, narrow
       enough that it cannot reach a different singularity. */
    const Float reach = mRes.scale() * 2;
    const MatrixXu &F = mRes.F();
    const MatrixXf &V = mRes.V();
    const MatrixXf &N = mRes.N();

    uint32_t best = (uint32_t) -1;
    Float bestDistance = reach;
    for (uint32_t f : singular) {
        Vector3f centre =
            (V.col(F(0, f)) + V.col(F(1, f)) + V.col(F(2, f))) / 3.f;
        Float distance = (centre - curve.front().p).norm();
        if (distance < bestDistance) {
            bestDistance = distance;
            best = f;
        }
    }
    if (best == (uint32_t) -1)
        return curve;

    CurvePoint start;
    start.p = (V.col(F(0, best)) + V.col(F(1, best)) + V.col(F(2, best))) / 3.f;
    start.n = (N.col(F(0, best)) + N.col(F(1, best)) + N.col(F(2, best))).normalized();
    start.f = best;

    std::vector<CurvePoint> snapped;
    snapped.reserve(curve.size() + 1);
    snapped.push_back(start);
    snapped.insert(snapped.end(), curve.begin(), curve.end());

    /* Re-routing is what fills in the faces between the marker and where the
       drag actually began; without it the two would not share an edge. */
    if (!smooth_curve(mBVH, mRes.E2E(), snapped, true))
        return curve;
    return snapped;
}

void Session::applyAttractor(const std::vector<CurvePoint> &drawn, bool orientation) {
    requireReady();
    if (drawn.size() < 2)
        return;

    /* A singularity is one triangle of the working mesh under a marker drawn
       many times its size, so a drag that starts on the dot usually does not
       start on the face. Forgiving that is the difference between the tool
       working and appearing to do nothing at all. */
    const std::vector<CurvePoint> curve = snapToSingularity(drawn, orientation);

    /* Optimizer::run walks the path from the back, so hand it the faces in
       reverse: the singularity is dragged from the stroke's end to its start. */
    std::vector<uint32_t> faces;
    faces.reserve(curve.size());
    for (auto it = curve.rbegin(); it != curve.rend(); ++it) {
        /* The solver steps one face at a time, so repeats would ask it to move
           a singularity onto the face it already occupies. */
        if (faces.empty() || faces.back() != it->f)
            faces.push_back(it->f);
    }

    if (faces.size() < 2)
        return;

    /* move_*_singularity requires each consecutive pair to share an edge and
       throws otherwise. Catch that here, where the message can still name the
       cause, instead of on the solver thread. */
    const MatrixXu &F = mRes.F();
    for (size_t i = 1; i < faces.size(); ++i) {
        int shared = 0;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                if (F(a, faces[i - 1]) == F(b, faces[i]))
                    ++shared;
        if (shared != 2)
            throw std::runtime_error(
                "Session::applyAttractor: the stroke is not a chain of "
                "edge-adjacent faces. Project it with attractor=true so that "
                "smooth_curve routes it along the surface.");
    }

    {
        std::lock_guard<ordered_lock> lock(mRes.mutex());
        /* Start the level-0 solve WITHOUT thawing: moving a singularity is a
           local edit within a fixed integer topology, which is exactly what the
           frozen sweep provides. optimizeOrientations(0) leaves the flags
           alone, so nothing else is needed here. */
        if (orientation)
            mOptimizer->optimizeOrientations(0);
        else
            mOptimizer->optimizePositions(0);
    }
    mOptimizer->notify();

    /* moveSingularity takes the hierarchy lock itself, so it must come after
       the block above releases it. */
    mOptimizer->moveSingularity(faces, orientation);
    mOptimizer->notify();
}

void Session::applyConstraints() {
    if (!mReady)
        return;

    std::lock_guard<ordered_lock> lock(mRes.mutex());

    const MatrixXu &F = mRes.F();
    const MatrixXf &N = mRes.N(), &V = mRes.V();
    const VectorXu &E2E = mRes.E2E();

    mRes.clearConstraints();

    if (mConfig.alignToBoundaries) {
        for (uint32_t i = 0; i < 3 * (uint32_t) F.cols(); ++i) {
            if (E2E[i] != INVALID)
                continue;
            uint32_t i0 = F(i % 3, i / 3);
            uint32_t i1 = F((i + 1) % 3, i / 3);
            Vector3f p0 = V.col(i0), p1 = V.col(i1);
            Vector3f edge = p1 - p0;
            if (edge.squaredNorm() == 0)
                continue;
            edge.normalize();
            mRes.CO().col(i0) = p0;
            mRes.CO().col(i1) = p1;
            mRes.CQ().col(i0) = mRes.CQ().col(i1) = edge;
            mRes.CQw()[i0] = mRes.CQw()[i1] = mRes.COw()[i0] = mRes.COw()[i1] = 1.0f;
        }
    }

    for (auto const &stroke : mStrokes) {
        auto const &curve = stroke.curve;
        for (uint32_t i = 0; i < (uint32_t) curve.size(); ++i) {
            Vector3f tangent;
            if (i == 0)
                tangent = curve[1].p - curve[0].p;
            else if (i == curve.size() - 1)
                tangent = curve[curve.size() - 1].p - curve[curve.size() - 2].p;
            else
                tangent = curve[i + 1].p - curve[i - 1].p;
            tangent.normalize();

            for (int j = 0; j < 3; ++j) {
                uint32_t v = F(j, curve[i].f);
                Vector3f tlocal = tangent;
                tlocal -= tlocal.dot(N.col(v)) * N.col(v);
                tlocal.normalize();

                mRes.CQ().col(v) = tlocal;
                mRes.CQw()[v] = 1.0f;

                /* The edge brush additionally pins the position field, which is
                   what makes an output edge run along the stroke. */
                if (stroke.kind == StrokeEdge) {
                    mRes.CO().col(v) = curve[i].p;
                    mRes.COw()[v] = 1.0f;
                }
            }
        }
    }

    mRes.propagateConstraints(mOptimizer->rosy(), mOptimizer->posy());
}

// ---------------------------------------------------------------------------
//  Solving
// ---------------------------------------------------------------------------

void Session::solveOrientations(int level) {
    requireReady();
    {
        std::lock_guard<ordered_lock> lock(mRes.mutex());
        /* optimizeOrientations() only thaws for level != 0, but the frozen
           sweep ignores the constraint arrays, so a level-0 solve would quietly
           discard every brush stroke. Thaw explicitly. */
        mRes.setFrozenQ(false);
        mOptimizer->optimizeOrientations(level);
    }
    mOptimizer->notify();
}

void Session::solvePositions(int level) {
    requireReady();
    {
        std::lock_guard<ordered_lock> lock(mRes.mutex());
        mRes.setFrozenO(false);
        mOptimizer->optimizePositions(level);
    }
    mOptimizer->notify();
}

void Session::stopSolve() {
    if (!mReady)
        return;
    std::lock_guard<ordered_lock> lock(mRes.mutex());
    mOptimizer->stop();
}

void Session::waitSolve() {
    requireReady();
    mOptimizer->wait();
}

void Session::solveAll() {
    requireReady();
    solveOrientations(-1);
    waitSolve();
    solvePositions(-1);
    waitSolve();
}

SolveStatus Session::status() const {
    SolveStatus s;
    if (!mReady) {
        s.active = false;
        s.progress = 0.f;
        s.level = -1;
        s.iterationsQ = -1;
        s.iterationsO = -1;
        return s;
    }
    Optimizer &opt = *const_cast<Session *>(this)->mOptimizer;
    MultiResolutionHierarchy &res = const_cast<Session *>(this)->mRes;
    std::lock_guard<ordered_lock> lock(res.mutex());
    s.active = opt.active();
    s.progress = opt.progress();
    s.level = opt.level();
    s.iterationsQ = res.iterationsQ();
    s.iterationsO = res.iterationsO();
    s.error = opt.takeLastError();
    return s;
}

void Session::setPreviewInterval(int ms) {
    requireReady();
    mOptimizer->setPreviewInterval(ms);
}

// ---------------------------------------------------------------------------
//  Results
// ---------------------------------------------------------------------------

std::map<uint32_t, uint32_t> Session::orientationSingularities() const {
    requireReady();
    MultiResolutionHierarchy &res = const_cast<Session *>(this)->mRes;
    std::lock_guard<ordered_lock> lock(res.mutex());
    std::map<uint32_t, uint32_t> sing;
    compute_orientation_singularities(mRes, sing, mConfig.extrinsic, mConfig.rosy);
    return sing;
}

std::map<uint32_t, Vector2i> Session::positionSingularities() const {
    requireReady();
    MultiResolutionHierarchy &res = const_cast<Session *>(this)->mRes;
    std::lock_guard<ordered_lock> lock(res.mutex());
    std::map<uint32_t, uint32_t> orientSing;
    compute_orientation_singularities(mRes, orientSing, mConfig.extrinsic, mConfig.rosy);
    std::map<uint32_t, Vector2i> posSing;
    compute_position_singularities(mRes, orientSing, posSing, mConfig.extrinsic,
                                   mConfig.rosy, mConfig.posy);
    return posSing;
}

void Session::setExtractionOptions(int smoothIter, bool pureQuad) {
    /* Both are read by extract() on the calling thread and by nothing else, so
       this needs no lock -- unlike everything that touches the hierarchy. */
    mConfig.smoothIter = std::max(0, smoothIter);
    mConfig.pureQuad = pureQuad;
}

ExtractedMesh Session::extract() {
    requireReady();
    std::lock_guard<ordered_lock> lock(mRes.mutex());

    const int rosy = mConfig.rosy, posy = mConfig.posy;

    std::vector<std::vector<TaggedLink>> adj;
    std::set<uint32_t> creaseOut;
    ExtractedMesh result;
    MatrixXf N;

    extract_graph(mRes, mConfig.extrinsic, rosy, posy, adj, result.V, N,
                  mCreaseSet, creaseOut, mConfig.deterministic);

    extract_faces(adj, result.V, N, result.Nf, result.F, posy, mRes.scale(),
                  creaseOut, true, mConfig.pureQuad, mBVH, mConfig.smoothIter);

    /* Line list for the preview overlay, matching Viewer::extractMesh:1496-1513.
       A quad whose last two indices coincide is really a triangle; its phantom
       edge is coloured black so it disappears against the wireframe. */
    const uint32_t nF = (uint32_t) result.F.cols();
    result.wireframe.resize(3, (Eigen::Index) nF * posy * 2);
    result.wireframeColor.resize(3, (Eigen::Index) nF * posy * 2);

    const Vector3f red = Vector3f::UnitX();
    for (uint32_t i = 0; i < nF; ++i) {
        bool irregular = posy == 4 && result.F(2, i) == result.F(3, i);
        for (int j = 0; j < posy; ++j) {
            uint32_t k = result.F(j, i), kn = result.F((j + 1) % posy, i);
            Vector3f col = (irregular && j >= 1) ? Vector3f::Zero() : red;
            Eigen::Index base = (Eigen::Index) i * 2 * posy + j * 2;
            result.wireframe.col(base + 0) = result.V.col(k);
            result.wireframe.col(base + 1) = result.V.col(kn);
            result.wireframeColor.col(base + 0) = col;
            result.wireframeColor.col(base + 1) = col;
        }
    }

    return result;
}

void Session::writeMesh(const std::string &path, const ExtractedMesh &mesh,
                        bool faceNormals) const {
    /* write_ply emits per-face normals as scalar properties *after* the
       variable-length vertex_indices list. That is legal PLY, but it makes the
       record size non-constant, and readers that parse an element in one
       vectorised pass (trimesh among them) reject the file. OBJ has no such
       problem, so normals stay on there and are dropped for PLY. */
    std::string extension;
    if (path.size() > 4)
        extension = str_tolower(path.substr(path.size() - 4));
    const bool keepNormals = faceNormals && extension != ".ply";

    write_mesh(path, mesh.F, mesh.V, MatrixXf(),
               keepNormals ? mesh.Nf : MatrixXf());
}

} // namespace instant_meshes
