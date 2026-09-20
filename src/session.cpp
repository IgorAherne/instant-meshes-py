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
    mStrokes.clear();
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

    /* Boundary alignment is expressed as constraints, so it goes through the
       same path as brush strokes. */
    applyConstraints();
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

bool Session::projectStroke(const MatrixXf &rayOrigins, const MatrixXf &rayDirections,
                            bool attractor, std::vector<CurvePoint> &out) const {
    requireReady();
    if (rayOrigins.rows() != 3 || rayDirections.rows() != 3)
        throw std::runtime_error("Session::projectStroke: rays must be 3 x N matrices");
    if (rayOrigins.cols() != rayDirections.cols())
        throw std::runtime_error("Session::projectStroke: origin/direction count mismatch");

    out.clear();
    if (rayOrigins.cols() == 0)
        return false;

    const MatrixXf &N = mRes.N();
    const MatrixXu &F = mRes.F();

    out.reserve((size_t) rayOrigins.cols());
    for (uint32_t i = 0; i < (uint32_t) rayOrigins.cols(); ++i) {
        Ray ray(rayOrigins.col(i), rayDirections.col(i).normalized());
        Vector2f uv;
        uint32_t f;
        Float t;

        /* A single miss aborts the whole stroke, exactly as the GUI does: a
           stroke that leaves the surface has no meaningful projection. */
        if (!mBVH->rayIntersect(ray, f, t, &uv)) {
            out.clear();
            return false;
        }

        CurvePoint pt;
        pt.p = ray(t);
        pt.n = ((1 - uv.sum()) * N.col(F(0, f)) + uv.x() * N.col(F(1, f)) +
                uv.y() * N.col(F(2, f))).normalized();
        pt.f = f;
        out.push_back(pt);
    }

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

void Session::applyAttractor(const std::vector<CurvePoint> &curve, bool orientation) {
    requireReady();
    if (curve.size() < 2)
        return;
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
