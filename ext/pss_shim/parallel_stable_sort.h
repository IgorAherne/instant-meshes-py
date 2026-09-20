/*
    parallel_stable_sort.h -- dependency-free stand-in for ext/pss.

    The upstream ext/pss submodule is built on the legacy `tbb::task` API, which
    was removed in oneTBB and is not provided by our TBB shim.  Instant Meshes
    calls exactly one function from it (src/extract.cpp), so we reimplement that
    function on top of the shim's stable parallel merge sort instead of dragging
    the old task scheduler along.

    Placing ext/pss_shim ahead of ext/pss on the include path is enough to swap
    the implementation; no source file needs to change.

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include <tbb/tbb.h>

#include <functional>
#include <iterator>

namespace pss {

/// Stable sort of [xs, xe) using `comp`, parallelised across all cores.
template <typename RandomAccessIterator, typename Compare>
void parallel_stable_sort(RandomAccessIterator xs, RandomAccessIterator xe,
                          Compare comp) {
    tbb::parallel_sort(xs, xe, comp);
}

template <typename RandomAccessIterator>
void parallel_stable_sort(RandomAccessIterator xs, RandomAccessIterator xe) {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
    tbb::parallel_sort(xs, xe, std::less<T>());
}

} // namespace pss
