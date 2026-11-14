#include "nsys.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "common.hpp"

namespace cta {
namespace {

class Db {
 public:
  explicit Db(const std::string& path) {
    if (sqlite3_open(path.c_str(), &db_) != SQLITE_OK) throw err("open");
  }
  ~Db() { sqlite3_close(db_); }
  Db(const Db&) = delete;
  Db& operator=(const Db&) = delete;
  void exec(const char* sql) {
    char* msg = nullptr;
    if (sqlite3_exec(db_, sql, nullptr, nullptr, &msg) != SQLITE_OK) {
      std::string m = msg ? msg : "?";
      sqlite3_free(msg);
      throw std::runtime_error("sqlite: " + m);
    }
  }
  sqlite3* get() { return db_; }
  std::runtime_error err(const char* what) {
    return std::runtime_error(std::string("sqlite ") + what + ": " + sqlite3_errmsg(db_));
  }

 private:
  sqlite3* db_ = nullptr;
};

class Stmt {
 public:
  Stmt(Db& db, const char* sql) : db_(db) {
    if (sqlite3_prepare_v2(db.get(), sql, -1, &s_, nullptr) != SQLITE_OK) throw db.err("prepare");
  }
  ~Stmt() { sqlite3_finalize(s_); }
  Stmt(const Stmt&) = delete;
  Stmt& operator=(const Stmt&) = delete;
  Stmt& i(int idx, std::int64_t v) { sqlite3_bind_int64(s_, idx, v); return *this; }
  Stmt& d(int idx, double v) { sqlite3_bind_double(s_, idx, v); return *this; }
  Stmt& t(int idx, const std::string& v) {
    sqlite3_bind_text(s_, idx, v.c_str(), static_cast<int>(v.size()), SQLITE_TRANSIENT);
    return *this;
  }
  void run() {
    if (sqlite3_step(s_) != SQLITE_DONE) throw db_.err("step");
    sqlite3_reset(s_);
    sqlite3_clear_bindings(s_);
  }

 private:
  Db& db_;
  sqlite3_stmt* s_ = nullptr;
};

}  // namespace

void write_nsys_sqlite(const std::string& path, int steps, int gemm_us, int comm_us,
                       int comm_overlap_us, int warmup_penalty_us, int lock_wait_us,
                       double tensor_active_pct, bool legacy_gpu_metrics) {
  std::remove(path.c_str());
  Db db(path);
  db.exec("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;");
  db.exec(R"(
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                demangledName INTEGER, shortName INTEGER, start INTEGER, end INTEGER, streamId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                copyKind INTEGER, start INTEGER, end INTEGER, bytes INTEGER, streamId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                nameId INTEGER, start INTEGER, end INTEGER
            );
            CREATE TABLE NVTX_EVENTS (
                textId INTEGER, text TEXT, start INTEGER, end INTEGER
            );
            CREATE TABLE OSRT_API (
                nameId INTEGER, start INTEGER, end INTEGER
            );
            )");
  db.exec("BEGIN");
  {
    Stmt s(db, "INSERT INTO StringIds VALUES (?, ?)");
    const char* names[] = {"void cutlass::Kernel<fp16>(half const*, int)",
                           "ncclDevKernel_AllReduce_Sum_f16_RING_LL", "cudaLaunchKernel",
                           "pthread_mutex_lock", "poll", "TransformerBlock_0"};
    for (int k = 0; k < 6; ++k) s.i(1, k + 1).t(2, names[k]).run();
  }
  Stmt nvtx_step(db, "INSERT INTO NVTX_EVENTS VALUES (NULL, ?, ?, ?)");
  Stmt nvtx_block(db, "INSERT INTO NVTX_EVENTS VALUES (6, NULL, ?, ?)");
  Stmt k_gemm(db, "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 1, ?, ?, 7)");
  Stmt k_comm(db, "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2, 2, ?, ?, 8)");
  Stmt memcpy_(db, "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (1, ?, ?, 65536, 9)");
  Stmt runtime(db, "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (3, ?, ?)");
  Stmt osrt(db, "INSERT INTO OSRT_API VALUES (4, ?, ?)");

  std::int64_t step_start = 1000000;
  for (int step = 0; step < steps; ++step) {
    const std::int64_t warmup_extra = step == 0 ? static_cast<std::int64_t>(warmup_penalty_us) * 1000 : 0;
    const std::int64_t gemm_dur = static_cast<std::int64_t>(gemm_us) * 1000 + warmup_extra;
    const std::int64_t comm_dur = static_cast<std::int64_t>(comm_us) * 1000;
    const std::int64_t overlap = std::min<std::int64_t>(static_cast<std::int64_t>(comm_overlap_us) * 1000, gemm_dur);
    const std::int64_t gemm_start = step_start;
    const std::int64_t gemm_end = gemm_start + gemm_dur;
    const std::int64_t comm_start = gemm_end - overlap;
    const std::int64_t comm_end = comm_start + comm_dur;
    const std::int64_t step_end = std::max(gemm_end, comm_end) + 50000;

    nvtx_step.t(1, "ProfilerStep#" + std::to_string(step)).i(2, step_start).i(3, step_end).run();
    nvtx_block.i(1, gemm_start).i(2, gemm_end).run();
    k_gemm.i(1, gemm_start).i(2, gemm_end).run();
    k_comm.i(1, comm_start).i(2, comm_end).run();
    memcpy_.i(1, gemm_start).i(2, gemm_start + 20000).run();
    runtime.i(1, gemm_start - 5000).i(2, gemm_start).run();
    osrt.i(1, gemm_start).i(2, gemm_start + static_cast<std::int64_t>(lock_wait_us) * 1000).run();
    step_start = step_end;
  }

  if (legacy_gpu_metrics) {
    db.exec("CREATE TABLE GPU_METRICS (rawTimestamp INTEGER, value TEXT)");
    auto str = [](double v) { std::string s; append_double(s, v); return s; };
    // Python's str(dict) form, which the legacy reader parses.
    const std::string payload = "{'SMs Active [Throughput %]': '" + str(tensor_active_pct + 10.0) +
                                "', 'Tensor Active [Throughput %]': '" + str(tensor_active_pct) +
                                "', 'SM Issue [Throughput %]': '" + str(tensor_active_pct - 5.0) + "'}";
    Stmt s(db, "INSERT INTO GPU_METRICS VALUES (?, ?)");
    s.i(1, 1000).t(2, payload).run();
    s.i(1, 2000).t(2, payload).run();
  } else {
    db.exec(R"(
                CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER PRIMARY KEY, metricName TEXT);
                CREATE TABLE GPU_METRICS (metricId INTEGER, timestamp INTEGER, value REAL);
                )");
    Stmt names(db, "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?)");
    names.i(1, 1).t(2, "SMs Active [Throughput %]").run();
    names.i(1, 2).t(2, "Tensor Active [Throughput %]").run();
    names.i(1, 3).t(2, "SM Issue [Throughput %]").run();
    Stmt m(db, "INSERT INTO GPU_METRICS VALUES (?, ?, ?)");
    m.i(1, 1).i(2, 1000).d(3, tensor_active_pct + 10.0).run();
    m.i(1, 2).i(2, 1000).d(3, tensor_active_pct).run();
    m.i(1, 3).i(2, 1000).d(3, tensor_active_pct - 5.0).run();
  }
  db.exec("COMMIT");
}

}  // namespace cta
