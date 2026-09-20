/*
    python_bindings.cpp: pybind11 interface to the headless Instant Meshes session.

    Array convention
    ----------------
    Internally everything is a column-major Eigen matrix of shape 3 x N, whose
    memory layout is byte-identical to a C-contiguous (N, 3) numpy array.  Every
    conversion here exploits that, so crossing the language boundary costs one
    memcpy and never a transpose.

    Threading
    ---------
    Every call that can block or that takes the hierarchy lock releases the GIL,
    so a web server can keep serving requests while a solve runs.

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "session.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <sstream>

namespace py = pybind11;
using namespace instant_meshes;

namespace {

typedef py::array_t<float, py::array::c_style | py::array::forcecast> FloatArray;
typedef py::array_t<uint32_t, py::array::c_style | py::array::forcecast> UIntArray;

/// Copy a k x N column-major matrix into an (N, k) C-contiguous numpy array.
template <typename Scalar, int Rows>
py::array_t<Scalar> rows_from(const Eigen::Matrix<Scalar, Rows, Eigen::Dynamic> &m) {
    const py::ssize_t n = (py::ssize_t) m.cols();
    const py::ssize_t k = (py::ssize_t) m.rows();
    py::array_t<Scalar> out(std::vector<py::ssize_t>{n, k});
    if (n > 0 && k > 0)
        std::memcpy(out.mutable_data(), m.data(), sizeof(Scalar) * (size_t) (n * k));
    return out;
}

/// Read a length-3 array (either (3,) or (1, 3)) into a Vector3f.
Vector3f vec3_from(const FloatArray &a, const char *what) {
    auto info = a.request();
    size_t total = 1;
    for (py::ssize_t i = 0; i < info.ndim; ++i)
        total *= (size_t) info.shape[i];
    if (total != 3)
        throw std::runtime_error(std::string(what) + ": expected 3 values");
    const float *p = static_cast<const float *>(info.ptr);
    return Vector3f(p[0], p[1], p[2]);
}

/// Reverse of rows_from(): validate an (N, k) array and copy it into k x N.
template <typename Scalar, typename Array>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
cols_from(const Array &a, py::ssize_t k, const char *what) {
    auto info = a.request();
    if (info.ndim != 2 || info.shape[1] != k) {
        std::ostringstream oss;
        oss << what << ": expected an (N, " << k << ") array, got shape (";
        for (py::ssize_t i = 0; i < info.ndim; ++i)
            oss << info.shape[i] << (i + 1 < info.ndim ? ", " : "");
        oss << ")";
        throw std::runtime_error(oss.str());
    }
    const py::ssize_t n = info.shape[0];
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> m((Eigen::Index) k, (Eigen::Index) n);
    if (n > 0)
        std::memcpy(m.data(), info.ptr, sizeof(Scalar) * (size_t) (n * k));
    return m;
}

/// A projected, mesh-aligned brush stroke.
struct Curve {
    std::vector<CurvePoint> points;

    py::array_t<float> positions() const {
        py::array_t<float> out(std::vector<py::ssize_t>{(py::ssize_t) points.size(), 3});
        auto r = out.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < r.shape(0); ++i)
            for (int j = 0; j < 3; ++j)
                r(i, j) = points[(size_t) i].p[j];
        return out;
    }

    py::array_t<float> normals() const {
        py::array_t<float> out(std::vector<py::ssize_t>{(py::ssize_t) points.size(), 3});
        auto r = out.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < r.shape(0); ++i)
            for (int j = 0; j < 3; ++j)
                r(i, j) = points[(size_t) i].n[j];
        return out;
    }

    py::array_t<uint32_t> faces() const {
        py::array_t<uint32_t> out((py::ssize_t) points.size());
        auto r = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < r.shape(0); ++i)
            r(i) = points[(size_t) i].f;
        return out;
    }

    static Curve from_arrays(const FloatArray &positions, const FloatArray &normals,
                             const UIntArray &faces) {
        auto p = positions.unchecked<2>();
        auto n = normals.unchecked<2>();
        auto f = faces.unchecked<1>();
        if (p.shape(1) != 3 || n.shape(1) != 3)
            throw std::runtime_error("Curve: positions and normals must be (N, 3)");
        if (p.shape(0) != n.shape(0) || p.shape(0) != f.shape(0))
            throw std::runtime_error("Curve: positions, normals and faces must agree in length");

        Curve c;
        c.points.resize((size_t) p.shape(0));
        for (py::ssize_t i = 0; i < p.shape(0); ++i) {
            CurvePoint &pt = c.points[(size_t) i];
            pt.p = Vector3f(p(i, 0), p(i, 1), p(i, 2));
            pt.n = Vector3f(n(i, 0), n(i, 1), n(i, 2));
            pt.f = f(i);
        }
        return c;
    }
};

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Headless Instant Meshes: field-aligned remeshing with brush-stroke guidance";

#if defined(INSTANT_MESHES_VERSION_INFO)
    m.attr("__version__") = INSTANT_MESHES_VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

    m.def("set_verbose", &setVerbose, py::arg("verbose"),
          "Enable or suppress the algorithm's progress output on stdout.");
    m.def("get_verbose", &verbose);
    m.def("set_thread_count", &setThreadCount, py::arg("threads"),
          "Worker threads for the internal pool; -1 uses every core.");

    py::enum_<StrokeKind>(m, "StrokeKind")
        .value("ORIENTATION", StrokeOrientation,
               "Orientation comb: steer the cross field along the stroke")
        .value("EDGE", StrokeEdge,
               "Edge brush: additionally place an output edge on the stroke")
        .export_values();

    py::class_<Curve>(m, "Curve",
                      "A brush stroke projected and smoothed onto the mesh surface")
        .def(py::init<>())
        .def_static("from_arrays", &Curve::from_arrays,
                    py::arg("positions"), py::arg("normals"), py::arg("faces"))
        .def_property_readonly("positions", &Curve::positions, "(N, 3) float32")
        .def_property_readonly("normals", &Curve::normals, "(N, 3) float32")
        .def_property_readonly("faces", &Curve::faces, "(N,) uint32 triangle indices")
        .def("__len__", [](const Curve &c) { return c.points.size(); })
        .def("__repr__", [](const Curve &c) {
            return "<Curve with " + std::to_string(c.points.size()) + " points>";
        });

    py::class_<Config>(m, "Config", "Remeshing parameters")
        .def(py::init<>())
        .def(py::init([](int rosy, int posy, float scale, int face_count,
                         int vertex_count, float crease_angle, bool extrinsic,
                         bool align_to_boundaries, bool deterministic,
                         int smooth_iter, bool pure_quad) {
                 Config c;
                 c.rosy = rosy;
                 c.posy = posy;
                 c.scale = scale;
                 c.faceCount = face_count;
                 c.vertexCount = vertex_count;
                 c.creaseAngle = crease_angle;
                 c.extrinsic = extrinsic;
                 c.alignToBoundaries = align_to_boundaries;
                 c.deterministic = deterministic;
                 c.smoothIter = smooth_iter;
                 c.pureQuad = pure_quad;
                 return c;
             }),
             py::arg("rosy") = 4, py::arg("posy") = 4, py::arg("scale") = -1.f,
             py::arg("face_count") = -1, py::arg("vertex_count") = -1,
             py::arg("crease_angle") = -1.f, py::arg("extrinsic") = true,
             py::arg("align_to_boundaries") = false, py::arg("deterministic") = false,
             py::arg("smooth_iter") = 2, py::arg("pure_quad") = false)
        .def_readwrite("rosy", &Config::rosy, "Rotational symmetry: 2, 4 or 6")
        .def_readwrite("posy", &Config::posy, "Positional symmetry: 3 or 4")
        .def_readwrite("scale", &Config::scale, "Target edge length; <0 derives it")
        .def_readwrite("face_count", &Config::faceCount)
        .def_readwrite("vertex_count", &Config::vertexCount)
        .def_readwrite("crease_angle", &Config::creaseAngle, "Degrees; <0 disables creases")
        .def_readwrite("extrinsic", &Config::extrinsic)
        .def_readwrite("align_to_boundaries", &Config::alignToBoundaries)
        .def_readwrite("deterministic", &Config::deterministic)
        .def_readwrite("smooth_iter", &Config::smoothIter)
        .def_readwrite("pure_quad", &Config::pureQuad)
        .def("__repr__", [](const Config &c) {
            std::ostringstream oss;
            oss << "<Config rosy=" << c.rosy << " posy=" << c.posy
                << " scale=" << c.scale << " vertex_count=" << c.vertexCount << ">";
            return oss.str();
        });

    py::class_<MeshStats>(m, "MeshStats",
                          "Geometric summary of the working mesh; a viewer needs\n"
                          "these to frame the camera and size its overlays.")
        .def_property_readonly("average_edge_length",
            [](const MeshStats &s) { return s.mAverageEdgeLength; })
        .def_property_readonly("maximum_edge_length",
            [](const MeshStats &s) { return s.mMaximumEdgeLength; })
        .def_property_readonly("surface_area",
            [](const MeshStats &s) { return s.mSurfaceArea; })
        .def_property_readonly("weighted_center",
            [](const MeshStats &s) {
                return py::make_tuple(s.mWeightedCenter.x(), s.mWeightedCenter.y(),
                                      s.mWeightedCenter.z());
            },
            "Area-weighted centroid, what the desktop app orbits around")
        .def_property_readonly("aabb_min",
            [](const MeshStats &s) {
                return py::make_tuple(s.mAABB.min.x(), s.mAABB.min.y(), s.mAABB.min.z());
            })
        .def_property_readonly("aabb_max",
            [](const MeshStats &s) {
                return py::make_tuple(s.mAABB.max.x(), s.mAABB.max.y(), s.mAABB.max.z());
            })
        .def("__repr__", [](const MeshStats &s) {
            std::ostringstream oss;
            oss << "<MeshStats area=" << s.mSurfaceArea
                << " avg_edge=" << s.mAverageEdgeLength << ">";
            return oss.str();
        });

    py::class_<SolveStatus>(m, "SolveStatus")
        .def_readonly("active", &SolveStatus::active)
        .def_readonly("progress", &SolveStatus::progress)
        .def_readonly("level", &SolveStatus::level)
        .def_readonly("iterations_q", &SolveStatus::iterationsQ,
                      "Version counter for the orientation field")
        .def_readonly("iterations_o", &SolveStatus::iterationsO,
                      "Version counter for the position field")
        .def_readonly("error", &SolveStatus::error,
                      "Message from the last solver failure, or an empty string. "
                      "Reading the status clears it, so surface it the first time.")
        .def("__repr__", [](const SolveStatus &s) {
            std::ostringstream oss;
            oss << "<SolveStatus active=" << (s.active ? "True" : "False")
                << " progress=" << s.progress << " level=" << s.level;
            if (!s.error.empty())
                oss << " error=" << s.error;
            oss << ">";
            return oss.str();
        });

    py::class_<ExtractedMesh>(m, "ExtractedMesh")
        .def_property_readonly("vertices",
            [](const ExtractedMesh &e) { return rows_from(e.V); }, "(nV, 3) float32")
        .def_property_readonly("faces",
            [](const ExtractedMesh &e) { return rows_from(e.F); },
            "(nF, posy) uint32; a quad with a repeated last index is a triangle")
        .def_property_readonly("face_normals",
            [](const ExtractedMesh &e) { return rows_from(e.Nf); }, "(nF, 3) float32")
        .def_property_readonly("wireframe",
            [](const ExtractedMesh &e) { return rows_from(e.wireframe); },
            "(2 * nF * posy, 3) float32 line-segment endpoints")
        .def_property_readonly("wireframe_color",
            [](const ExtractedMesh &e) { return rows_from(e.wireframeColor); },
            "(2 * nF * posy, 3) float32; black marks a triangle's phantom edge")
        .def("__repr__", [](const ExtractedMesh &e) {
            std::ostringstream oss;
            oss << "<ExtractedMesh V=" << e.V.cols() << " F=" << e.F.cols() << ">";
            return oss.str();
        });

    py::class_<Session>(m, "Session", "One editing session over a single mesh")
        .def(py::init<>())

        .def("set_mesh",
             [](Session &s, const FloatArray &vertices, const UIntArray &faces) {
                 MatrixXf V = cols_from<float>(vertices, 3, "set_mesh(vertices)");
                 MatrixXu F = cols_from<uint32_t>(faces, 3, "set_mesh(faces)");
                 py::gil_scoped_release release;
                 s.setMesh(V, F);
             },
             py::arg("vertices"), py::arg("faces"),
             "Set the input triangle mesh from (nV, 3) and (nF, 3) arrays.")

        .def("load_file", &Session::loadFile, py::arg("path"),
             py::call_guard<py::gil_scoped_release>(),
             "Load an OBJ/PLY mesh from disk.")

        .def("preprocess",
             [](Session &s, const Config &cfg, py::object progress) {
                 if (progress.is_none()) {
                     py::gil_scoped_release release;
                     s.preprocess(cfg);
                     return;
                 }
                 /* The callback re-enters Python from whichever worker thread
                    reaches it, so it must reacquire the GIL each time. */
                 auto cb = [&progress](const std::string &text, Float pct) {
                     py::gil_scoped_acquire acquire;
                     progress(text, pct);
                 };
                 py::gil_scoped_release release;
                 s.preprocess(cfg, cb);
             },
             py::arg("config") = Config(), py::arg("progress") = py::none(),
             "Build the hierarchy and acceleration structures. Drops strokes.")

        .def_property_readonly("ready", &Session::ready)
        .def_property_readonly("config", &Session::config)
        .def_property_readonly("scale", &Session::scale)
        .def_property_readonly("stats", &Session::stats,
                               py::return_value_policy::reference_internal)

        .def_property_readonly("vertices",
            [](const Session &s) { return rows_from(s.vertices()); },
            "(nV, 3) float32 vertices of the working mesh (after subdivision)")
        .def_property_readonly("faces",
            [](const Session &s) { return rows_from(s.faces()); },
            "(nF, 3) uint32 triangles of the working mesh")
        .def_property_readonly("normals",
            [](const Session &s) { return rows_from(s.normals()); },
            "(nV, 3) float32 vertex normals")
        /* These are polled many times a second while brushing, so the field is
           copied straight into the numpy buffer instead of through a temporary
           Eigen matrix. */
        .def_property_readonly("orientation_field",
            [](const Session &s) {
                py::array_t<float> out(
                    std::vector<py::ssize_t>{(py::ssize_t) s.vertexCount(), 3});
                float *dst = out.mutable_data();
                {
                    py::gil_scoped_release release;
                    s.copyOrientationField(dst);
                }
                return out;
            },
            "(nV, 3) float32 cross-field tangents")
        .def_property_readonly("position_field",
            [](const Session &s) {
                py::array_t<float> out(
                    std::vector<py::ssize_t>{(py::ssize_t) s.vertexCount(), 3});
                float *dst = out.mutable_data();
                {
                    py::gil_scoped_release release;
                    s.copyPositionField(dst);
                }
                return out;
            },
            "(nV, 3) float32 position-field samples")
        .def_property_readonly("vertex_count",
            [](const Session &s) { return s.vertexCount(); },
            "Vertices in the working mesh")
        .def_property_readonly("crease_map",
            [](const Session &s) { return s.creaseMap(); },
            "{duplicate vertex: original vertex} along sharp creases, or {}")

        .def("project_stroke",
             [](const Session &s, const FloatArray &origins, const FloatArray &directions,
                bool attractor) -> py::object {
                 MatrixXf O = cols_from<float>(origins, 3, "project_stroke(origins)");
                 MatrixXf D = cols_from<float>(directions, 3, "project_stroke(directions)");
                 Curve curve;
                 bool ok;
                 {
                     py::gil_scoped_release release;
                     ok = s.projectStroke(O, D, attractor, curve.points);
                 }
                 if (!ok)
                     return py::none();
                 return py::cast(curve);
             },
             py::arg("ray_origins"), py::arg("ray_directions"), py::arg("attractor") = false,
             "Ray-cast an (N, 3) set of rays onto the surface and smooth the hit\n"
             "points into a surface curve. Returns None if any ray misses.")

        .def("add_stroke",
             [](Session &s, int kind, const Curve &curve) {
                 py::gil_scoped_release release;
                 return s.addStroke(kind, curve.points);
             },
             py::arg("kind"), py::arg("curve"),
             "Add a stroke and re-apply the field constraints. Returns its id.")

        .def("erase_stroke",
             [](Session &s, uint32_t id) {
                 py::gil_scoped_release release;
                 return s.eraseStroke(id);
             },
             py::arg("stroke_id"))

        .def("erase_stroke_near",
             [](Session &s, const FloatArray &point, const FloatArray &eye, float radius) {
                 Vector3f p = vec3_from(point, "erase_stroke_near(point)");
                 Vector3f e = vec3_from(eye, "erase_stroke_near(eye)");
                 py::gil_scoped_release release;
                 return s.eraseStrokeNear(p, e, radius);
             },
             py::arg("point"), py::arg("eye"), py::arg("radius"),
             "Erase the stroke handle nearest to `point` that is visible from\n"
             "`eye`. Returns the erased id, or 0 if nothing was close enough.")

        .def("clear_strokes", &Session::clearStrokes,
             py::call_guard<py::gil_scoped_release>())

        .def_property_readonly("strokes",
            [](const Session &s) {
                py::list out;
                for (auto const &stroke : s.strokes()) {
                    Curve c;
                    c.points = stroke.curve;
                    py::dict d;
                    d["id"] = stroke.id;
                    d["kind"] = stroke.kind;
                    d["curve"] = py::cast(c);
                    out.append(d);
                }
                return out;
            },
            "List of {'id', 'kind', 'curve'} dictionaries")

        .def("apply_attractor",
             [](Session &s, const Curve &curve, bool orientation) {
                 py::gil_scoped_release release;
                 s.applyAttractor(curve.points, orientation);
             },
             py::arg("curve"), py::arg("orientation") = true,
             "Drag a singularity along the stroke. Starts the level-0 frozen\n"
             "solve the move needs; call stop_solve() when it has travelled far\n"
             "enough, or wait for status().active to clear.")

        .def("solve_orientations", &Session::solveOrientations, py::arg("level") = -1,
             py::call_guard<py::gil_scoped_release>(),
             "Start smoothing the orientation field. level=-1 runs the full\n"
             "hierarchical schedule and stops by itself; level=0 refines in place\n"
             "until stop_solve() and is the cheap mode for live feedback.\n"
             "Both honour brush strokes.")
        .def("solve_positions", &Session::solvePositions, py::arg("level") = -1,
             py::call_guard<py::gil_scoped_release>())
        .def("stop_solve", &Session::stopSolve, py::call_guard<py::gil_scoped_release>())
        .def("wait_solve", &Session::waitSolve, py::call_guard<py::gil_scoped_release>())
        .def("solve_all", &Session::solveAll, py::call_guard<py::gil_scoped_release>(),
             "Run orientations then positions to completion, synchronously.")
        /* def_property_* cannot take a call_guard, so these release the GIL by
           hand around the part that can block on the hierarchy lock. */
        .def_property_readonly("status",
            [](const Session &s) {
                py::gil_scoped_release release;
                return s.status();
            })
        .def("set_preview_interval", &Session::setPreviewInterval, py::arg("milliseconds"),
             "How often an interactive solve publishes an intermediate result.")

        .def_property_readonly("orientation_singularities",
            [](const Session &s) {
                std::map<uint32_t, uint32_t> sing;
                {
                    py::gil_scoped_release release;
                    sing = s.orientationSingularities();
                }
                return sing;
            },
            "{face index: singularity index}")
        .def_property_readonly("position_singularities",
            [](const Session &s) {
                std::map<uint32_t, Vector2i> sing;
                {
                    py::gil_scoped_release release;
                    sing = s.positionSingularities();
                }
                py::dict out;
                for (auto const &kv : sing)
                    out[py::cast(kv.first)] = py::make_tuple(kv.second.x(), kv.second.y());
                return out;
            },
            "{face index: (i, j) integer shift}")

        .def("extract", &Session::extract, py::call_guard<py::gil_scoped_release>(),
             "Extract the output mesh from the current fields.")
        .def("write_mesh", &Session::writeMesh, py::arg("path"), py::arg("mesh"),
             py::arg("face_normals") = true,
             py::call_guard<py::gil_scoped_release>(),
             "Write an extracted mesh to OBJ or PLY, chosen by file extension.\n"
             "face_normals is ignored for PLY, where including them produces a\n"
             "file many readers reject.");
}
