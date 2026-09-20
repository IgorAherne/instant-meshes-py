cmake_minimum_required(VERSION 3.18)
project(InstantMeshes LANGUAGES C CXX)

# ---------------------------------------------------------------------------
#  Options
#
#  The default build is the Python extension, which needs nothing beyond a C++
#  compiler and Eigen headers.  The original NanoGUI/GLFW desktop application is
#  still buildable, but it is opt-in so that the heavy submodules stay optional.
# ---------------------------------------------------------------------------

option(INSTANT_MESHES_BUILD_PYTHON "Build the Python extension module" ON)
option(INSTANT_MESHES_BUILD_GUI    "Build the native NanoGUI application"  OFF)
option(INSTANT_MESHES_DEV_MODE     "Instant meshes developer mode"         OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_GENERATOR MATCHES "Visual Studio")
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

# ---------------------------------------------------------------------------
#  Eigen
#
#  Tried in order of decreasing explicitness so that packagers, submodule users
#  and plain `pip install .` all work.  Eigen is header-only; we never link it.
# ---------------------------------------------------------------------------

set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Path to an Eigen 3 installation")

if(EIGEN3_INCLUDE_DIR AND EXISTS "${EIGEN3_INCLUDE_DIR}/signature_of_eigen3_matrix_library")
  message(STATUS "Eigen: using EIGEN3_INCLUDE_DIR (${EIGEN3_INCLUDE_DIR})")
  set(IM_EIGEN_INCLUDE "${EIGEN3_INCLUDE_DIR}")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ext/eigen/signature_of_eigen3_matrix_library")
  message(STATUS "Eigen: using the vendored copy in ext/eigen")
  set(IM_EIGEN_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/ext/eigen")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ext/nanogui/ext/eigen/signature_of_eigen3_matrix_library")
  message(STATUS "Eigen: using the copy inside the NanoGUI submodule")
  set(IM_EIGEN_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/ext/nanogui/ext/eigen")
else()
  find_package(Eigen3 3.3 NO_MODULE QUIET)
  if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen: using the system installation found by find_package")
    get_target_property(IM_EIGEN_INCLUDE Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  else()
    message(STATUS "Eigen: not found locally, downloading 3.4.0")
    include(FetchContent)
    FetchContent_Declare(eigen
      URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
      URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72)
    FetchContent_Populate(eigen)
    set(IM_EIGEN_INCLUDE "${eigen_SOURCE_DIR}")
  endif()
endif()

# ---------------------------------------------------------------------------
#  Bundled header-only dependencies
# ---------------------------------------------------------------------------

foreach(_dep dset pcg32)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ext/${_dep}/README.md")
    message(FATAL_ERROR
      "The submodule ext/${_dep} is missing. Recover with:\n"
      "  git submodule update --init ext/dset ext/pcg32")
  endif()
endforeach()

set(IM_CORE_INCLUDES
  ${IM_EIGEN_INCLUDE}
  # std::thread-based stand-ins for TBB and pss, placed ahead of the real ones
  # so no algorithm source needs to know which implementation it gets.
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/tbb_shim
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/pss_shim
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/dset
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/pcg32
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/rply
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/half
  ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---------------------------------------------------------------------------
#  Core algorithm library (no windowing, no OpenGL)
# ---------------------------------------------------------------------------

set(IM_CORE_SOURCES
  src/adjacency.cpp
  src/bvh.cpp
  src/cleanup.cpp
  src/dedge.cpp
  src/extract.cpp
  src/field.cpp
  src/hierarchy.cpp
  src/meshio.cpp
  src/meshstats.cpp
  src/normal.cpp
  src/reorder.cpp
  src/serializer.cpp
  src/smoothcurve.cpp
  src/subdivide.cpp
  src/batch.cpp
  src/globals.cpp
  ext/rply/rply.c)

add_library(im_core STATIC ${IM_CORE_SOURCES})
target_include_directories(im_core PUBLIC ${IM_CORE_INCLUDES})

if(MSVC)
  target_compile_definitions(im_core PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX _USE_MATH_DEFINES)
  # /bigobj: extract.cpp and field.cpp exceed the default section limit once
  # Eigen's templates are instantiated at -O2.
  target_compile_options(im_core PRIVATE /MP /bigobj /EHsc /W3)
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    # Eigen's vectorised paths assume 16-byte alignment that 32-bit MSVC does
    # not guarantee for heap allocations.
    target_compile_definitions(im_core PUBLIC EIGEN_DONT_ALIGN)
    target_compile_options(im_core PRIVATE /arch:SSE2)
  endif()
else()
  target_compile_options(im_core PRIVATE -Wall -Wextra -Wno-unused-parameter
                                         -Wno-deprecated-declarations)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(im_core PRIVATE -Wno-gnu-anonymous-struct
                                           -Wno-c99-extensions
                                           -Wno-nested-anon-types)
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(im_core PUBLIC Threads::Threads)

if(INSTANT_MESHES_DEV_MODE)
  target_compile_definitions(im_core PUBLIC DEV_MODE)
endif()

set_source_files_properties(ext/rply/rply.c PROPERTIES LANGUAGE C)
if(MSVC)
  set_source_files_properties(ext/rply/rply.c PROPERTIES COMPILE_FLAGS "/wd4127")
endif()

# ---------------------------------------------------------------------------
#  Python extension
# ---------------------------------------------------------------------------

if(INSTANT_MESHES_BUILD_PYTHON)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_core MODULE
    src/session.cpp
    src/python_bindings.cpp)

  target_link_libraries(_core PRIVATE im_core)
  target_compile_definitions(_core PRIVATE
    INSTANT_MESHES_VERSION_INFO="${SKBUILD_PROJECT_VERSION}")

  if(MSVC)
    target_compile_options(_core PRIVATE /bigobj /EHsc)
  endif()

  # Link-time optimisation across the extension and the algorithm core, where
  # the toolchain supports it.
  include(CheckIPOSupported)
  check_ipo_supported(RESULT IM_IPO_OK OUTPUT IM_IPO_MSG)
  if(IM_IPO_OK AND U_CMAKE_BUILD_TYPE MATCHES "RELEASE|RELWITHDEBINFO|MINSIZEREL")
    set_property(TARGET _core im_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()

  if(DEFINED SKBUILD_PROJECT_NAME)
    install(TARGETS _core DESTINATION instant_meshes_brush)
  endif()
endif()

# ---------------------------------------------------------------------------
#  Optional native GUI (requires the NanoGUI and TBB submodules)
# ---------------------------------------------------------------------------

if(INSTANT_MESHES_BUILD_GUI)
  if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext/nanogui/ext/glfw")
    message(FATAL_ERROR
      "INSTANT_MESHES_BUILD_GUI requires the NanoGUI submodule. Recover with:\n"
      "  git submodule update --init --recursive ext/nanogui")
  endif()

  set(NANOGUI_BUILD_EXAMPLE OFF CACHE BOOL " " FORCE)
  set(NANOGUI_BUILD_SHARED  OFF CACHE BOOL " " FORCE)
  set(NANOGUI_BUILD_PYTHON  OFF CACHE BOOL " " FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanogui ext_build/nanogui)
  set_property(TARGET nanogui glfw glfw_objects bin2c PROPERTY FOLDER "dependencies")

  set(PNG_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/comb.png
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/delete_stroke.png
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/loadmesh.png
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/scare.png
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/sing_dir.png
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/sing_pos.png
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/instantmeshes.png)
  file(GLOB FRAG_FILES "${CMAKE_CURRENT_SOURCE_DIR}/resources/*.frag")
  file(GLOB VERT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/resources/*.vert")
  file(GLOB GEO_FILES  "${CMAKE_CURRENT_SOURCE_DIR}/resources/*.geo")
  set(IM_RESOURCES ${PNG_FILES} ${FRAG_FILES} ${VERT_FILES} ${GEO_FILES})

  add_custom_command(
    OUTPUT resources.cpp
    COMMAND bin2c ARGS resources.cpp resources.h ${IM_RESOURCES}
    DEPENDS bin2c ${IM_RESOURCES}
    COMMENT "Running bin2c"
    VERBATIM)

  set(EXTRA_SOURCE "")
  if(APPLE)
    set(EXTRA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/resources/im.icns")
  elseif(WIN32)
    set(EXTRA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/resources/im.rc")
  endif()

  add_executable(InstantMeshes MACOSX_BUNDLE
    ${CMAKE_CURRENT_BINARY_DIR}/resources.cpp
    src/glutil.cpp
    src/widgets.cpp
    src/main.cpp
    src/viewer.cpp
    ${EXTRA_SOURCE})

  target_include_directories(InstantMeshes PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/nanogui/include
    ${NANOGUI_EXTRA_INCS}
    ${CMAKE_CURRENT_BINARY_DIR})
  target_compile_definitions(InstantMeshes PRIVATE ${NANOGUI_EXTRA_DEFS})
  target_link_libraries(InstantMeshes PRIVATE im_core nanogui ${NANOGUI_EXTRA_LIBS})
  set_target_properties(InstantMeshes PROPERTIES OUTPUT_NAME "Instant Meshes")

  if(APPLE)
    set_target_properties(InstantMeshes PROPERTIES
      MACOSX_BUNDLE_BUNDLE_NAME "Instant Meshes"
      MACOSX_BUNDLE_BUNDLE_GUI_IDENTIFIER "com.im.InstantMeshes"
      MACOSX_BUNDLE_ICON_FILE im.icns
      MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/resources/MacOSXBundleInfo.plist.in)
    set_source_files_properties(resources/im.icns PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
  endif()
endif()
