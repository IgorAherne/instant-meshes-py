/*
    tbb/tbb.h -- header-only, dependency-free stand-in for the subset of Intel
    Threading Building Blocks that Instant Meshes actually uses.

    Why this exists
    ---------------
    Instant Meshes historically vendored wjakob's fork of TBB 2017 as a git
    submodule.  That fork no longer builds cleanly on current toolchains, and
    modern oneTBB is not a drop-in replacement (the legacy `tbb::task` API used
    by src/bvh.cpp and ext/pss was removed in oneTBB 2021).  Shipping a Python
    extension that links against an external TBB would also force us to bundle
    and relocate a shared library into every wheel.

    This header implements the ~12 TBB entry points the algorithm core touches
    on top of <thread>/<atomic>/<mutex> only.  It has no build step, no runtime
    dependency, and behaves identically on Windows, Linux and macOS.

    Deliberate behavioural notes
    ----------------------------
    * Nested parallelism is safe.  A thread that blocks waiting for its own
      parallel region keeps executing pending work items from other regions
      instead of idling, so a `parallel_for` invoked from inside another
      `parallel_for` (src/bvh.cpp does exactly this) can never deadlock.
    * `parallel_deterministic_reduce` partitions the range using only
      (size, grainsize) and folds partial results strictly left-to-right, so it
      returns bit-identical results on every machine regardless of core count.
      Plain `parallel_reduce` caps the chunk count at 4x the pool size to keep
      dispatch overhead down, so -- exactly as with real TBB -- its floating
      point result depends on how many cores the machine has.  Call sites that
      need reproducibility already use the deterministic variant.
    * `parallel_sort` is implemented with a *stable* parallel merge sort.  TBB's
      is unstable; guaranteeing stability is a strictly stronger contract and
      removes a source of run-to-run variation in the extraction stage.
    * The reduction value type is deduced from the body's return type rather
      than from `identity`.  TBB deduces it from `identity`, which silently
      truncates `parallel_deterministic_reduce(range, 0, map, reduce)` in
      src/bvh.cpp (an `int` identity accumulating `double` radii) to integers.

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

/// Lets callers detect the shim and reach its extensions (e.g. setConcurrency).
#define INSTANT_MESHES_TBB_SHIM 1

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace tbb {

// ---------------------------------------------------------------------------
//  blocked_range
// ---------------------------------------------------------------------------

template <typename Index> class blocked_range {
public:
    typedef Index const_iterator;
    typedef std::size_t size_type;

    blocked_range() : mBegin(Index()), mEnd(Index()), mGrainSize(1) { }

    blocked_range(Index begin, Index end, size_type grainsize = 1)
        : mBegin(begin), mEnd(end), mGrainSize(grainsize ? grainsize : 1) { }

    Index begin() const { return mBegin; }
    Index end() const { return mEnd; }
    size_type size() const { return mEnd > mBegin ? size_type(mEnd - mBegin) : 0; }
    size_type grainsize() const { return mGrainSize; }
    bool empty() const { return !(mBegin < mEnd); }
    bool is_divisible() const { return mGrainSize < size(); }

private:
    Index mBegin, mEnd;
    size_type mGrainSize;
};

// ---------------------------------------------------------------------------
//  spin_mutex
// ---------------------------------------------------------------------------

/// Test-and-test-and-set lock. Non-recursive, like TBB's.
class spin_mutex {
public:
    spin_mutex() : mFlag(false) { }

    /* TBB's spin_mutex is neither copyable nor movable, but Instant Meshes
       stores them in `std::vector<tbb::spin_mutex>` (src/extract.cpp:875,
       src/hierarchy.cpp:289).  That only ever value-initialises the elements,
       so defining the copy ctor to produce a fresh unlocked mutex keeps the
       vector constructible without ever duplicating lock state. */
    spin_mutex(const spin_mutex &) : mFlag(false) { }
    spin_mutex &operator=(const spin_mutex &) { return *this; }

    void lock() {
        while (mFlag.exchange(true, std::memory_order_acquire)) {
            do {
                std::this_thread::yield();
            } while (mFlag.load(std::memory_order_relaxed));
        }
    }

    bool try_lock() {
        return !mFlag.load(std::memory_order_relaxed) &&
               !mFlag.exchange(true, std::memory_order_acquire);
    }

    void unlock() { mFlag.store(false, std::memory_order_release); }

    class scoped_lock {
    public:
        scoped_lock() : mMutex(nullptr) { }
        explicit scoped_lock(spin_mutex &m) : mMutex(&m) { mMutex->lock(); }
        ~scoped_lock() { release(); }

        void acquire(spin_mutex &m) { release(); mMutex = &m; mMutex->lock(); }
        bool try_acquire(spin_mutex &m) {
            release();
            if (m.try_lock()) { mMutex = &m; return true; }
            return false;
        }
        void release() { if (mMutex) { mMutex->unlock(); mMutex = nullptr; } }

        scoped_lock(const scoped_lock &) = delete;
        scoped_lock &operator=(const scoped_lock &) = delete;

    private:
        spin_mutex *mMutex;
    };

private:
    std::atomic<bool> mFlag;
};

typedef spin_mutex mutex;

// ---------------------------------------------------------------------------
//  Scheduler
// ---------------------------------------------------------------------------

namespace detail {

/// One fan-out of `count` indexed work items sharing a single body.
struct Job {
    const std::function<void(std::size_t)> *body;
    std::size_t count;
    std::atomic<std::size_t> next;      ///< next index to hand out
    std::atomic<std::size_t> remaining; ///< items not yet finished
    std::mutex errorMutex;
    std::exception_ptr error;

    Job(const std::function<void(std::size_t)> *body_, std::size_t count_)
        : body(body_), count(count_), next(0), remaining(count_) { }
};

typedef std::shared_ptr<Job> JobPtr;

/**
 * Global fixed-size worker pool with caller participation.
 *
 * Work items live in per-job atomic counters rather than in a queue, so
 * dispatching a chunk costs one `fetch_add` plus one short critical section
 * when a thread has to switch jobs.
 *
 * Deadlock freedom: a thread waiting on its own job always tries to execute
 * pending items from *any* live job first, and only ever sleeps on a timed
 * wait.  Jobs are held by `shared_ptr`, so a worker can never observe a job
 * that its owner has already destroyed.
 */
class Scheduler {
public:
    static Scheduler &get() {
        /* Intentionally leaked: joining worker threads from a static destructor
           deadlocks against the Windows loader lock when a Python extension is
           unloaded.  The workers exit on `mShutdown` and own no OS resources
           that outlive the process. */
        static Scheduler *instance = new Scheduler();
        return *instance;
    }

    int concurrency() const { return mConcurrency.load(std::memory_order_relaxed); }

    /**
     * Request a worker count.
     *
     * Ignored while a parallel region is in flight, because resizing under load
     * would mean joining threads that are mid-item.  The pool only ever grows;
     * lowering the count simply leaves the surplus workers parked on the
     * condition variable, and `chunk_count` immediately stops handing them work.
     */
    void setConcurrency(int n) {
        if (n <= 0)
            n = defaultConcurrency();
        std::lock_guard<std::mutex> lock(mMutex);
        if (mLiveJobs != 0 || n == concurrency())
            return;
        mConcurrency.store(n, std::memory_order_relaxed);
        spawnWorkersLocked();
    }

    static int defaultConcurrency() {
        unsigned int hc = std::thread::hardware_concurrency();
        return hc == 0 ? 1 : (int) hc;
    }

    /// Run `fn(0..count-1)`, blocking until every index has completed.
    void run(std::size_t count, const std::function<void(std::size_t)> &fn) {
        if (count == 0)
            return;
        /* Single-threaded pool, or nothing to hand out: run every item here.
           Looping over all of them (rather than just index 0) is what makes the
           serial path equivalent to the parallel one. */
        if (count == 1 || concurrency() <= 1) {
            for (std::size_t i = 0; i < count; ++i)
                fn(i);
            return;
        }

        JobPtr job = std::make_shared<Job>(&fn, count);
        {
            std::lock_guard<std::mutex> lock(mMutex);
            ensureWorkersLocked();
            mJobs.push_back(job);
            ++mLiveJobs;
        }
        mCond.notify_all();

        drain(job.get());

        {
            std::lock_guard<std::mutex> lock(mMutex);
            for (std::size_t i = mJobs.size(); i-- > 0; ) {
                if (mJobs[i].get() == job.get()) {
                    mJobs.erase(mJobs.begin() + (std::ptrdiff_t) i);
                    break;
                }
            }
            --mLiveJobs;
        }

        if (job->error)
            std::rethrow_exception(job->error);
    }

private:
    Scheduler()
        : mConcurrency(defaultConcurrency()), mLiveJobs(0), mShutdown(false) { }

    /// Execute pending items until `target` is complete (or forever if null).
    void drain(Job *target) {
        JobPtr current;
        for (;;) {
            if (current) {
                std::size_t idx = current->next.fetch_add(1, std::memory_order_relaxed);
                if (idx < current->count) {
                    execute(*current, idx);
                    continue;
                }
                current.reset();
            }

            if (target && target->remaining.load(std::memory_order_acquire) == 0)
                return;

            std::unique_lock<std::mutex> lock(mMutex);
            if (!target && mShutdown)
                return;

            /* Deepest job first: finishing inner regions first bounds the
               number of simultaneously open regions and keeps caches warm. */
            for (std::size_t i = mJobs.size(); i-- > 0; ) {
                if (mJobs[i]->next.load(std::memory_order_relaxed) < mJobs[i]->count) {
                    current = mJobs[i];
                    break;
                }
            }
            if (current)
                continue;

            if (target) {
                if (target->remaining.load(std::memory_order_acquire) == 0)
                    return;
                /* Every item is claimed but not all have finished. Sleep with a
                   timeout so a missed notify can never wedge the region. */
                mCond.wait_for(lock, std::chrono::microseconds(200));
            } else {
                mCond.wait(lock);
            }
        }
    }

    void execute(Job &job, std::size_t index) {
        try {
            (*job.body)(index);
        } catch (...) {
            std::lock_guard<std::mutex> lock(job.errorMutex);
            if (!job.error)
                job.error = std::current_exception();
        }
        if (job.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(mMutex);
            mCond.notify_all();
        }
    }

    void ensureWorkersLocked() {
        if (mWorkers.empty())
            spawnWorkersLocked();
    }

    void spawnWorkersLocked() {
        const int n = concurrency();
        std::size_t want = n > 1 ? (std::size_t) (n - 1) : 0;
        /* The pool only ever grows: shrinking would require joining threads
           that may be mid-item, and over-subscription is harmless because idle
           workers block on the condition variable. */
        for (std::size_t i = mWorkers.size(); i < want; ++i)
            mWorkers.emplace_back([this] { drain(nullptr); });
    }

    std::mutex mMutex;
    std::condition_variable mCond;
    std::vector<JobPtr> mJobs;
    std::vector<std::thread> mWorkers;
    std::atomic<int> mConcurrency;
    int mLiveJobs;      ///< parallel regions currently open, guarded by mMutex
    bool mShutdown;     ///< set only by tests; the pool normally outlives main()
};

/// Split [0, size) into chunks no larger than `grainsize`, capped so that tiny
/// bodies do not pay more dispatch overhead than they save.
inline std::size_t chunk_count(std::size_t size, std::size_t grainsize, bool deterministic) {
    if (size == 0)
        return 0;
    if (grainsize == 0)
        grainsize = 1;
    std::size_t byGrain = (size + grainsize - 1) / grainsize;
    if (deterministic)
        return byGrain;
    std::size_t cap = (std::size_t) Scheduler::get().concurrency() * 4;
    if (cap < 1)
        cap = 1;
    return byGrain < cap ? byGrain : cap;
}

template <typename Index>
inline blocked_range<Index> sub_range(const blocked_range<Index> &range,
                                      std::size_t chunk, std::size_t chunks) {
    std::size_t size = range.size();
    std::size_t begin = (size * chunk) / chunks;
    std::size_t end = (size * (chunk + 1)) / chunks;
    return blocked_range<Index>(Index(range.begin() + begin),
                                Index(range.begin() + end),
                                range.grainsize());
}

} // namespace detail

// ---------------------------------------------------------------------------
//  task_scheduler_init
// ---------------------------------------------------------------------------

class task_scheduler_init {
public:
    enum { automatic = -1, deferred = -2 };

    explicit task_scheduler_init(int max_threads = automatic) {
        if (max_threads != deferred)
            detail::Scheduler::get().setConcurrency(max_threads);
    }
    ~task_scheduler_init() { }

    task_scheduler_init(const task_scheduler_init &) = delete;
    task_scheduler_init &operator=(const task_scheduler_init &) = delete;

    static int default_num_threads() { return detail::Scheduler::defaultConcurrency(); }
};

// ---------------------------------------------------------------------------
//  parallel_for
// ---------------------------------------------------------------------------

/// Range form: `body(const blocked_range<Index> &)`.
template <typename Index, typename Body>
void parallel_for(const blocked_range<Index> &range, const Body &body) {
    std::size_t chunks = detail::chunk_count(range.size(), range.grainsize(), false);
    if (chunks <= 1) {
        if (!range.empty())
            body(range);
        return;
    }
    std::function<void(std::size_t)> fn = [&](std::size_t chunk) {
        body(detail::sub_range(range, chunk, chunks));
    };
    detail::Scheduler::get().run(chunks, fn);
}

/// Index form: `body(Index)`, one call per element.
template <typename Index, typename Body>
void parallel_for(Index first, Index last, const Body &body) {
    if (!(first < last))
        return;
    blocked_range<Index> range(first, last, 1);
    std::size_t size = range.size();
    std::size_t chunks = detail::chunk_count(size, 1, false);
    if (chunks <= 1) {
        for (Index i = first; i < last; ++i)
            body(i);
        return;
    }
    std::function<void(std::size_t)> fn = [&](std::size_t chunk) {
        blocked_range<Index> sub = detail::sub_range(range, chunk, chunks);
        for (Index i = sub.begin(); i < sub.end(); ++i)
            body(i);
    };
    detail::Scheduler::get().run(chunks, fn);
}

// ---------------------------------------------------------------------------
//  parallel_reduce / parallel_deterministic_reduce
// ---------------------------------------------------------------------------

namespace detail {

template <typename Index, typename Value, typename RealBody, typename Reduction>
Value reduce_impl(const blocked_range<Index> &range, const Value &identity,
                  const RealBody &real_body, const Reduction &reduction,
                  bool deterministic) {
    std::size_t chunks = chunk_count(range.size(), range.grainsize(), deterministic);
    if (chunks == 0)
        return identity;
    if (chunks == 1)
        return real_body(range, identity);

    std::vector<Value> partial(chunks, identity);
    std::function<void(std::size_t)> fn = [&](std::size_t chunk) {
        partial[chunk] = real_body(sub_range(range, chunk, chunks), identity);
    };
    Scheduler::get().run(chunks, fn);

    /* Strict left-to-right fold: identical on every machine and core count. */
    Value result = partial[0];
    for (std::size_t i = 1; i < chunks; ++i)
        result = reduction(result, partial[i]);
    return result;
}

} // namespace detail

template <typename Index, typename Identity, typename RealBody, typename Reduction>
auto parallel_reduce(const blocked_range<Index> &range, const Identity &identity,
                     const RealBody &real_body, const Reduction &reduction)
    -> decltype(real_body(range, identity)) {
    typedef decltype(real_body(range, identity)) Value;
    return detail::reduce_impl<Index, Value, RealBody, Reduction>(
        range, Value(identity), real_body, reduction, false);
}

template <typename Index, typename Identity, typename RealBody, typename Reduction>
auto parallel_deterministic_reduce(const blocked_range<Index> &range,
                                   const Identity &identity,
                                   const RealBody &real_body,
                                   const Reduction &reduction)
    -> decltype(real_body(range, identity)) {
    typedef decltype(real_body(range, identity)) Value;
    return detail::reduce_impl<Index, Value, RealBody, Reduction>(
        range, Value(identity), real_body, reduction, true);
}

// ---------------------------------------------------------------------------
//  parallel_invoke
// ---------------------------------------------------------------------------

template <typename F0, typename F1>
void parallel_invoke(const F0 &f0, const F1 &f1) {
    std::function<void(std::size_t)> fn = [&](std::size_t i) {
        if (i == 0) f0(); else f1();
    };
    detail::Scheduler::get().run(2, fn);
}

template <typename F0, typename F1, typename F2>
void parallel_invoke(const F0 &f0, const F1 &f1, const F2 &f2) {
    std::function<void(std::size_t)> fn = [&](std::size_t i) {
        if (i == 0) f0(); else if (i == 1) f1(); else f2();
    };
    detail::Scheduler::get().run(3, fn);
}

// ---------------------------------------------------------------------------
//  parallel_sort  (stable parallel merge sort)
// ---------------------------------------------------------------------------

namespace detail {

template <typename It, typename Compare>
void merge_sort(It begin, It end,
                typename std::iterator_traits<It>::value_type *buffer,
                Compare comp, int depth) {
    typedef typename std::iterator_traits<It>::difference_type diff_t;
    diff_t size = end - begin;
    /* 4096 keeps the serial base case comfortably inside L2 while still giving
       every worker thread something to do on realistic inputs. */
    if (size < 4096 || depth <= 0) {
        std::stable_sort(begin, end, comp);
        return;
    }

    It mid = begin + size / 2;
    auto left = [&] { merge_sort(begin, mid, buffer, comp, depth - 1); };
    auto right = [&] { merge_sort(mid, end, buffer + (mid - begin), comp, depth - 1); };
    parallel_invoke(left, right);

    std::merge(std::make_move_iterator(begin), std::make_move_iterator(mid),
               std::make_move_iterator(mid), std::make_move_iterator(end),
               buffer, comp);
    std::move(buffer, buffer + size, begin);
}

} // namespace detail

template <typename It, typename Compare>
void parallel_sort(It begin, It end, const Compare &comp) {
    typedef typename std::iterator_traits<It>::value_type T;
    typename std::iterator_traits<It>::difference_type size = end - begin;
    if (size < 4096 || detail::Scheduler::get().concurrency() <= 1) {
        std::stable_sort(begin, end, comp);
        return;
    }
    int depth = 0;
    for (int n = detail::Scheduler::get().concurrency(); n > 1; n >>= 1)
        ++depth;
    depth += 2; /* a little oversubscription helps load balancing */

    std::vector<T> buffer((std::size_t) size);
    detail::merge_sort(begin, end, buffer.data(), comp, depth);
}

template <typename It>
void parallel_sort(It begin, It end) {
    typedef typename std::iterator_traits<It>::value_type T;
    parallel_sort(begin, end, std::less<T>());
}

// ---------------------------------------------------------------------------
//  concurrent_vector
// ---------------------------------------------------------------------------

/**
 * Append-only vector safe for concurrent `push_back`.
 *
 * Contract (matches every use in Instant Meshes): pushes happen in one parallel
 * phase, reads (`operator[]`, `size`, iteration, sorting) in a later one.  Reads
 * are therefore unsynchronised and as fast as `std::vector`'s.  Reading while
 * another thread pushes is undefined, exactly as it would be for a
 * `std::vector`.
 */
template <typename T, typename Allocator = std::allocator<T>>
class concurrent_vector {
public:
    typedef typename std::vector<T, Allocator>::iterator iterator;
    typedef typename std::vector<T, Allocator>::const_iterator const_iterator;
    typedef typename std::vector<T, Allocator>::value_type value_type;
    typedef typename std::vector<T, Allocator>::reference reference;
    typedef typename std::vector<T, Allocator>::const_reference const_reference;
    typedef std::size_t size_type;

    concurrent_vector() { }

    void reserve(size_type n) {
        spin_mutex::scoped_lock lock(mMutex);
        mData.reserve(n);
    }

    iterator push_back(const T &value) {
        spin_mutex::scoped_lock lock(mMutex);
        mData.push_back(value);
        return mData.begin() + (std::ptrdiff_t) (mData.size() - 1);
    }

    iterator push_back(T &&value) {
        spin_mutex::scoped_lock lock(mMutex);
        mData.push_back(std::move(value));
        return mData.begin() + (std::ptrdiff_t) (mData.size() - 1);
    }

    reference operator[](size_type i) { return mData[i]; }
    const_reference operator[](size_type i) const { return mData[i]; }

    size_type size() const { return mData.size(); }
    bool empty() const { return mData.empty(); }
    void clear() { mData.clear(); }

    iterator begin() { return mData.begin(); }
    iterator end() { return mData.end(); }
    const_iterator begin() const { return mData.begin(); }
    const_iterator end() const { return mData.end(); }

private:
    std::vector<T, Allocator> mData;
    spin_mutex mMutex;
};

// ---------------------------------------------------------------------------
//  concurrent_priority_queue
// ---------------------------------------------------------------------------

template <typename T, typename Compare = std::less<T>>
class concurrent_priority_queue {
public:
    typedef T value_type;
    typedef std::size_t size_type;

    concurrent_priority_queue() { }

    void push(const T &value) {
        spin_mutex::scoped_lock lock(mMutex);
        mQueue.push(value);
    }

    bool try_pop(T &result) {
        spin_mutex::scoped_lock lock(mMutex);
        if (mQueue.empty())
            return false;
        result = mQueue.top();
        mQueue.pop();
        return true;
    }

    bool empty() const {
        spin_mutex::scoped_lock lock(mMutex);
        return mQueue.empty();
    }

    size_type size() const {
        spin_mutex::scoped_lock lock(mMutex);
        return mQueue.size();
    }

    void clear() {
        spin_mutex::scoped_lock lock(mMutex);
        mQueue = std::priority_queue<T, std::vector<T>, Compare>();
    }

private:
    std::priority_queue<T, std::vector<T>, Compare> mQueue;
    mutable spin_mutex mMutex;
};

} // namespace tbb
