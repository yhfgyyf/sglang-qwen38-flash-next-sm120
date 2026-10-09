// Copyright 2026 Qwen38 native contributors. SPDX-License-Identifier: Apache-2.0
#include "executor.h"
#include <cuda_runtime.h>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

extern "C" int q38_ple_read(void*, const int64_t*, size_t, uint8_t*, size_t);
extern "C" const char* q38_ple_last_error();
cudaError_t q38_launch_decode(const uint8_t*, void*, size_t, cudaStream_t);

namespace {
thread_local std::string last_error;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
void require_eager_stream(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  check(cudaStreamIsCapturing(stream, &status));
  require(status == cudaStreamCaptureStatusNone,
          "native I/O/replay is forbidden on a capturing CUDA stream");
}
size_t checked_bytes(size_t count, size_t width) {
  require(width && count <= std::numeric_limits<size_t>::max() / width,
          "buffer size overflows size_t");
  return count * width;
}
template<class F> int protect(F&& fn) noexcept {
  try { last_error.clear(); fn(); return 0; }
  catch (const std::exception& error) { last_error = error.what(); return -1; }
  catch (...) { last_error = "unknown native exception"; return -1; }
}

// The main thread queues IDs and immediately continues independent GPU work.
// Only the worker waits for those IDs. collect() waits for storage, then queues
// copy+decode; it does NOT synchronize the consuming CUDA stream on the host.
class Pipe {
 public:
  enum class State { Idle, Pending, Reading, Ready };
  void* store;
  size_t row_bytes, max_rows;
  int device;
  int64_t* ids = nullptr;
  uint8_t* rows = nullptr;
  uint8_t* device_rows = nullptr;
  cudaEvent_t ids_ready = nullptr, copy_done = nullptr;
  bool copy_recorded = false, stopping = false, poisoned = false;
  State state = State::Idle;
  int64_t generation = 0;
  size_t count = 0;
  std::string failure;
  std::mutex mutex;
  std::condition_variable cv;
  std::thread worker;

  Pipe(void* s, size_t width, size_t capacity, int dev)
      : store(s), row_bytes(width), max_rows(capacity), device(dev) {
    require(store && row_bytes == 160 && max_rows, "invalid Qwen38 PLE pipe geometry");
    const auto bytes = checked_bytes(max_rows, row_bytes);
    const auto id_bytes = checked_bytes(max_rows, sizeof(int64_t));
    check(cudaSetDevice(device));
    try {
      check(cudaHostAlloc(reinterpret_cast<void**>(&ids), id_bytes, cudaHostAllocDefault));
      check(cudaHostAlloc(reinterpret_cast<void**>(&rows), bytes, cudaHostAllocDefault));
      check(cudaMalloc(reinterpret_cast<void**>(&device_rows), bytes));
      check(cudaEventCreateWithFlags(&ids_ready, cudaEventDisableTiming));
      check(cudaEventCreateWithFlags(&copy_done, cudaEventDisableTiming));
      worker = std::thread(&Pipe::run, this);
    } catch (...) { release(); throw; }
  }
  ~Pipe() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      cv.notify_all();
    }
    if (worker.joinable()) worker.join();
    // All storage writes have finished. Only this pipe's transfer can still
    // use the staging bytes; no device-wide synchronize is needed.
    cudaSetDevice(device);
    if (copy_recorded) cudaEventSynchronize(copy_done);
    release();
  }
  void release() noexcept {
    if (ids_ready) cudaEventDestroy(ids_ready);
    if (copy_done) cudaEventDestroy(copy_done);
    if (device_rows) cudaFree(device_rows);
    if (rows) cudaFreeHost(rows);
    if (ids) cudaFreeHost(ids);
    ids_ready = copy_done = nullptr;
    device_rows = rows = nullptr;
    ids = nullptr;
  }
  void run() noexcept {
    std::unique_lock<std::mutex> lock(mutex);
    while (true) {
      cv.wait(lock, [&] { return stopping || state == State::Pending; });
      if (stopping && state != State::Pending) return;
      state = State::Reading;
      const size_t n = count;
      const bool wait_copy = copy_recorded;
      lock.unlock();
      std::string error;
      try {
        check(cudaSetDevice(device));
        check(cudaEventSynchronize(ids_ready));
        if (wait_copy) check(cudaEventSynchronize(copy_done));
        if (q38_ple_read(store, ids, n, rows, n * row_bytes) != 0)
          throw std::runtime_error(q38_ple_last_error());
      } catch (const std::exception& e) { error = e.what(); }
      catch (...) { error = "native PLE worker failed"; }
      lock.lock();
      failure = std::move(error);
      state = State::Ready;
      cv.notify_all();
      if (stopping) return;
    }
  }
  int64_t issue(const int64_t* source, size_t n, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex);
    require(state == State::Idle && !stopping && !poisoned,
            "PLE gather already in flight or pipe is poisoned");
    require(n && n <= max_rows && source, "invalid PLE ID buffer/count");
    require(generation < std::numeric_limits<int64_t>::max(), "PLE generation exhausted");
    check(cudaSetDevice(device));
    require_eager_stream(stream);
    try {
      check(cudaMemcpyAsync(ids, source, n * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
      check(cudaEventRecord(ids_ready, stream));
    } catch (...) {
      // A successful memcpy followed by a failed event must not leave host
      // memory unprotected. Quiesce this stream before allowing destruction.
      cudaStreamSynchronize(stream);
      poisoned = true;
      throw;
    }
    ++generation;
    count = n;
    failure.clear();
    state = State::Pending;
    cv.notify_all();
    return generation;
  }
  void collect(int64_t ticket, void* output, size_t n, cudaStream_t stream) {
    std::unique_lock<std::mutex> lock(mutex);
    require(state != State::Idle && ticket == generation, "stale or missing PLE ticket");
    require(output && n == count, "PLE output count does not match issue");
    check(cudaSetDevice(device));
    require_eager_stream(stream);
    cv.wait(lock, [&] { return state == State::Ready; });
    if (!failure.empty()) {
      const std::string error = failure;
      state = State::Idle;
      throw std::runtime_error(error);
    }
    check(cudaSetDevice(device));
    const size_t bytes = n * row_bytes;
    try {
      check(cudaMemcpyAsync(device_rows, rows, bytes, cudaMemcpyHostToDevice, stream));
      check(q38_launch_decode(device_rows, output, bytes, stream));
      check(cudaEventRecord(copy_done, stream));
    } catch (...) {
      cudaStreamSynchronize(stream);
      poisoned = true;
      state = State::Idle;
      throw;
    }
    copy_recorded = true;
    state = State::Idle;
  }
  void abandon() noexcept {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return state == State::Idle || state == State::Ready; });
    state = State::Idle;
  }
};

enum class Kind { Graph, Issue, Collect };
struct Operation {
  Kind kind;
  uintptr_t graph = 0;
  Pipe* pipe = nullptr;
  void* buffer = nullptr;
  size_t count = 0;
};
class Plan {
 public:
  int device;
  bool sealed = false, poisoned = false;
  std::vector<Operation> ops;
  std::mutex mutex;
  cudaEvent_t finished = nullptr;
  bool replayed = false;
  explicit Plan(int dev) : device(dev) {
    check(cudaSetDevice(device));
    check(cudaEventCreateWithFlags(&finished, cudaEventDisableTiming));
  }
  ~Plan() {
    cudaSetDevice(device);
    if (replayed) cudaEventSynchronize(finished);
    cudaEventDestroy(finished);
  }
  void add(Operation op) {
    require(!sealed, "cannot mutate a sealed native plan");
    ops.push_back(op);
  }
  void seal() {
    require(!sealed && !ops.empty(), "native plan is empty or already sealed");
    std::unordered_map<Pipe*, size_t> pending;
    for (const auto& op : ops) {
      if (op.kind == Kind::Graph) require(op.graph, "null graph executable");
      else {
        require(op.pipe && op.buffer && op.count && op.count <= op.pipe->max_rows,
                "invalid PLE plan operation");
        require(op.pipe->device == device, "plan/pipe CUDA device mismatch");
        if (op.kind == Kind::Issue) {
          require(!pending.count(op.pipe), "duplicate PLE issue without collect");
          pending.emplace(op.pipe, op.count);
        } else {
          auto it = pending.find(op.pipe);
          require(it != pending.end() && it->second == op.count, "unmatched PLE collect");
          pending.erase(it);
        }
      }
    }
    require(pending.empty(), "native plan leaves an outstanding PLE gather");
    sealed = true;
  }
  void replay(cudaStream_t stream) {
    std::unique_lock<std::mutex> lock(mutex, std::try_to_lock);
    require(lock.owns_lock(), "concurrent replay of one native plan is unsupported");
    require(sealed && !poisoned, "native plan is not sealed or is poisoned");
    check(cudaSetDevice(device));
    require_eager_stream(stream);
    // This also orders graph-pool reuse if the caller changes streams.
    if (replayed) check(cudaStreamWaitEvent(stream, finished, 0));
    std::unordered_map<Pipe*, int64_t> pending;
    try {
      for (const auto& op : ops) {
        switch (op.kind) {
          case Kind::Graph:
            check(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(op.graph), stream));
            break;
          case Kind::Issue:
            pending[op.pipe] = op.pipe->issue(static_cast<int64_t*>(op.buffer), op.count, stream);
            break;
          case Kind::Collect:
            // Same stream as consumer graph: no Python event/host synchronization
            // is needed, even if the original eager path used a side stream.
            op.pipe->collect(pending.at(op.pipe), op.buffer, op.count, stream);
            pending.erase(op.pipe);
            break;
        }
      }
      check(cudaEventRecord(finished, stream));
      replayed = true;
    } catch (...) {
      poisoned = true;
      // A failed plan can already have enqueued kernels using its borrowed
      // graph pool/output. Drain this stream AND this plan's storage jobs
      // before Python is allowed to release any associated allocations.
      cudaStreamSynchronize(stream);
      for (const auto& item : pending) item.first->abandon();
      throw;
    }
  }
};
}

extern "C" const char* q38_native_last_error() { return last_error.c_str(); }
extern "C" void* q38_pipe_create(void* store, size_t width, size_t capacity, int device) {
  Pipe* pipe = nullptr;
  if (protect([&] { pipe = new Pipe(store, width, capacity, device); })) return nullptr;
  return pipe;
}
extern "C" int64_t q38_pipe_issue(void* p, const int64_t* ids, size_t n, uintptr_t stream) {
  int64_t ticket = -1;
  protect([&] { require(p, "null pipe"); ticket = static_cast<Pipe*>(p)->issue(ids, n, reinterpret_cast<cudaStream_t>(stream)); });
  return ticket;
}
extern "C" int q38_pipe_collect(void* p, int64_t ticket, void* out, size_t n, uintptr_t stream) {
  return protect([&] { require(p, "null pipe"); static_cast<Pipe*>(p)->collect(ticket, out, n, reinterpret_cast<cudaStream_t>(stream)); });
}
extern "C" int q38_pipe_pending(void* p) {
  int pending = -1;
  protect([&] {
    require(p, "null pipe");
    auto* pipe = static_cast<Pipe*>(p);
    std::lock_guard<std::mutex> lock(pipe->mutex);
    pending = pipe->state == Pipe::State::Idle ? 0 : 1;
  });
  return pending;
}
extern "C" int q38_pipe_close(void* p) { return protect([&] { delete static_cast<Pipe*>(p); }); }
extern "C" void* q38_plan_create(int device) {
  Plan* plan = nullptr;
  if (protect([&] { plan = new Plan(device); })) return nullptr;
  return plan;
}
extern "C" int q38_plan_add_graph(void* p, uintptr_t graph) {
  return protect([&] { require(p, "null plan"); static_cast<Plan*>(p)->add({Kind::Graph, graph}); });
}
extern "C" int q38_plan_add_issue(void* p, void* pipe, const int64_t* ids, size_t n) {
  return protect([&] { require(p, "null plan"); static_cast<Plan*>(p)->add({Kind::Issue, 0, static_cast<Pipe*>(pipe), const_cast<int64_t*>(ids), n}); });
}
extern "C" int q38_plan_add_collect(void* p, void* pipe, void* output, size_t n) {
  return protect([&] { require(p, "null plan"); static_cast<Plan*>(p)->add({Kind::Collect, 0, static_cast<Pipe*>(pipe), output, n}); });
}
extern "C" int q38_plan_seal(void* p) {
  return protect([&] { require(p, "null plan"); static_cast<Plan*>(p)->seal(); });
}
extern "C" int q38_plan_replay(void* p, uintptr_t stream) {
  return protect([&] { require(p, "null plan"); static_cast<Plan*>(p)->replay(reinterpret_cast<cudaStream_t>(stream)); });
}
extern "C" int q38_plan_close(void* p) { return protect([&] { delete static_cast<Plan*>(p); }); }
extern "C" int q38_decode_fp8(const uint8_t* in, void* out, size_t n, uintptr_t stream) {
  return protect([&] { require((in && out) || !n, "null FP8 conversion buffer"); check(q38_launch_decode(in, out, n, reinterpret_cast<cudaStream_t>(stream))); });
}
