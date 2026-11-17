#include "readers/nsys/reader.hpp"

#include <exception>
#include <limits>
#include <set>
#include <string>
#include <thread>

#include "util/util.hpp"
#include "util/interner.hpp"
#include "util/launch_map.hpp"
#include "util/scoped_join.hpp"
#include "util/sqlite.hpp"

namespace cta::readers {
namespace {

using std::int64_t;

// CUPTI memcpy kinds: 1 HtoD, 2 DtoH, 8 DtoD, 9 HtoH, 10 PtoP; others are rare
// array or unified-memory copies.
const char* memcpy_name(int64_t kind) {
  switch (kind) {
    case 1: return "Memcpy_H2D";
    case 2: return "Memcpy_D2H";
    case 8: return "Memcpy_D2D";
    case 9: return "Memcpy_H2H";
    case 10: return "Memcpy_P2P";
    default: return "Memcpy_Other";
  }
}

// "t.<name>" when the table has the column, else a default SQL expression.
std::string column_or(const std::set<std::string>& columns, const char* name, const char* fallback) {
  return columns.count(name) ? std::string("t.") + name : std::string(fallback);
}

struct Events {
  Interner names;
  std::vector<std::int32_t> name;
  std::vector<std::int8_t> category;
  std::vector<int64_t> start, end, bytes, stream, device, correlation;
};

void add_event(Events& e, std::string_view name, std::int8_t category, int64_t start, int64_t end, int64_t bytes,
               int64_t stream, int64_t device, int64_t correlation) {
  e.name.push_back(e.names.intern(name));
  e.category.push_back(category);
  e.start.push_back(start);
  e.end.push_back(end);
  e.bytes.push_back(bytes);
  e.stream.push_back(stream);
  e.device.push_back(device);
  e.correlation.push_back(correlation);
}

// A NULL correlation id, which matches no launch, and a NULL launch time.
constexpr int64_t kNoCorrelation = std::numeric_limits<int64_t>::min();
constexpr int64_t kNoStart = std::numeric_limits<int64_t>::min();

int64_t correlation_of(const SqliteQuery& q, int column) { return q.is_null(column) ? kNoCorrelation : q.integer(column); }

// The kernels, memcpys and memsets of the export, in that order.
void read_events(const std::string& path, const std::set<std::string>& tables, Events& e) {
  SqliteReader db(path);
  const bool has_string_ids = tables.count("StringIds") > 0;

  if (tables.count("CUPTI_ACTIVITY_KIND_KERNEL")) {
    const auto columns = sqlite_columns(db, "CUPTI_ACTIVITY_KIND_KERNEL");
    const std::string name_column = columns.count("demangledName") ? "demangledName" : "shortName";
    const std::string name_expr =
        has_string_ids ? "COALESCE(s.value, CAST(t." + name_column + " AS TEXT))" : "CAST(t." + name_column + " AS TEXT)";
    const std::string join = has_string_ids ? "LEFT JOIN StringIds s ON t." + name_column + " = s.id" : "";
    SqliteQuery q(db, "SELECT " + name_expr + ", t.start, t.end, " + column_or(columns, "deviceId", "0") + ", " +
                          column_or(columns, "streamId", "0") + ", " + column_or(columns, "correlationId", "-1") +
                          " FROM CUPTI_ACTIVITY_KIND_KERNEL t " + join + " WHERE t.end > t.start");
    while (q.step()) {
      const std::string_view name = q.is_null(0) ? std::string_view("None") : q.text(0);
      add_event(e, name, contains_nccl(name) ? 1 : 0, q.integer(1), q.integer(2), 0, q.integer(4), q.integer(3),
                correlation_of(q, 5));
    }
  }

  if (tables.count("CUPTI_ACTIVITY_KIND_MEMCPY")) {
    const auto columns = sqlite_columns(db, "CUPTI_ACTIVITY_KIND_MEMCPY");
    SqliteQuery q(db, "SELECT " + column_or(columns, "copyKind", "0") + ", t.start, t.end, " +
                          column_or(columns, "bytes", "0") + ", " + column_or(columns, "deviceId", "0") + ", " +
                          column_or(columns, "streamId", "0") + ", " + column_or(columns, "correlationId", "-1") +
                          " FROM CUPTI_ACTIVITY_KIND_MEMCPY t WHERE t.end > t.start");
    while (q.step()) {
      add_event(e, memcpy_name(q.integer(0)), 2, q.integer(1), q.integer(2), q.integer(3), q.integer(5), q.integer(4),
                correlation_of(q, 6));
    }
  }

  if (tables.count("CUPTI_ACTIVITY_KIND_MEMSET")) {
    const auto columns = sqlite_columns(db, "CUPTI_ACTIVITY_KIND_MEMSET");
    SqliteQuery q(db, "SELECT t.start, t.end, " + column_or(columns, "bytes", "0") + ", " +
                          column_or(columns, "deviceId", "0") + ", " + column_or(columns, "streamId", "0") + ", " +
                          column_or(columns, "correlationId", "-1") +
                          " FROM CUPTI_ACTIVITY_KIND_MEMSET t WHERE t.end > t.start");
    while (q.step()) {
      add_event(e, "Memset", 2, q.integer(0), q.integer(1), q.integer(2), q.integer(4), q.integer(3),
                correlation_of(q, 5));
    }
  }
}

// The launching thread and time of each correlation id; the first row wins.
void read_launches(const std::string& path, const std::set<std::string>& tables, LaunchMap& launches) {
  if (!tables.count("CUPTI_ACTIVITY_KIND_RUNTIME")) return;
  SqliteReader db(path);
  const auto columns = sqlite_columns(db, "CUPTI_ACTIVITY_KIND_RUNTIME");
  if (!columns.count("correlationId") || !columns.count("globalTid")) return;
  SqliteQuery q(db, "SELECT correlationId, globalTid, start FROM CUPTI_ACTIVITY_KIND_RUNTIME");
  while (q.step()) {
    if (q.is_null(0)) continue;
    launches.put_if_absent(q.integer(0), q.is_null(2) ? kNoStart : q.integer(2), q.is_null(1) ? -1 : q.integer(1));
  }
}

// Every runtime call with its name, time, thread and correlation id.
void read_runtime_calls(const std::string& path, const std::set<std::string>& tables, NsysCalls& out) {
  if (!tables.count("CUPTI_ACTIVITY_KIND_RUNTIME")) return;
  SqliteReader db(path);
  const auto columns = sqlite_columns(db, "CUPTI_ACTIVITY_KIND_RUNTIME");
  const bool has_names = tables.count("StringIds") > 0 && columns.count("nameId") > 0;
  const std::string name_expr = has_names ? "COALESCE(s.value, 'unknown')" : "'unknown'";
  const std::string join = has_names ? "LEFT JOIN StringIds s ON r.nameId = s.id" : "";
  const std::string thread_expr = columns.count("globalTid") ? "r.globalTid" : "-1";
  const std::string correlation_expr = columns.count("correlationId") ? "r.correlationId" : "-1";
  SqliteQuery q(db, "SELECT " + name_expr + ", r.start, r.end, " + thread_expr + ", " + correlation_expr +
                        " FROM CUPTI_ACTIVITY_KIND_RUNTIME r " + join + " WHERE r.end > r.start");
  Interner names;
  while (q.step()) {
    out.name.push_back(names.intern(q.text(0)));
    out.start.push_back(q.integer(1));
    out.end.push_back(q.integer(2));
    out.thread.push_back(q.is_null(3) ? -1 : q.integer(3));
    out.correlation.push_back(q.is_null(4) ? -1 : q.integer(4));
  }
  out.names.assign(names.items().begin(), names.items().end());
}

struct Ranges {
  Interner texts;
  std::vector<std::int32_t> text;
  std::vector<int64_t> start, end, thread;
};

// The closed NVTX ranges. Only push/pop and start/end ranges are read when
// the event type is known; the text comes from the range or from StringIds.
void read_nvtx(const std::string& path, const std::set<std::string>& tables, Ranges& r) {
  if (!tables.count("NVTX_EVENTS")) return;
  SqliteReader db(path);
  const auto columns = sqlite_columns(db, "NVTX_EVENTS");
  const std::string thread_expr = columns.count("globalTid") ? "n.globalTid" : "0";
  std::string where = "WHERE n.end IS NOT NULL AND n.end > n.start";
  if (columns.count("eventType")) where += " AND n.eventType IN (59, 60)";
  std::string sql;
  if (columns.count("textId") && tables.count("StringIds")) {
    const std::string text_expr = columns.count("text") ? "COALESCE(n.text, s.value, '')" : "COALESCE(s.value, '')";
    sql = "SELECT " + text_expr + ", n.start, n.end, " + thread_expr +
          " FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id " + where;
  } else if (columns.count("text")) {
    sql = "SELECT COALESCE(n.text, ''), n.start, n.end, " + thread_expr + " FROM NVTX_EVENTS n " + where;
  } else {
    return;
  }
  SqliteQuery q(db, sql);
  while (q.step()) {
    r.text.push_back(r.texts.intern(q.text(0)));
    r.start.push_back(q.integer(1));
    r.end.push_back(q.integer(2));
    r.thread.push_back(q.is_null(3) ? 0 : q.integer(3));
  }
}

struct Metrics {
  bool present = false;
  Interner names;
  std::vector<std::int32_t> name;
  std::vector<int64_t> timestamp;
  std::vector<double> value;
};

void read_metrics(const std::string& path, const std::set<std::string>& tables, Metrics& m) {
  if (!tables.count("GPU_METRICS") || !tables.count("TARGET_INFO_GPU_METRICS")) return;
  SqliteReader db(path);
  const auto columns = sqlite_columns(db, "GPU_METRICS");
  if (!columns.count("metricId")) return;
  m.present = true;
  const std::string timestamp = columns.count("timestamp") ? "timestamp" : "rawTimestamp";
  SqliteQuery q(db, "SELECT t.metricName, g." + timestamp +
                        ", g.value FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS t ON g.metricId = t.metricId");
  while (q.step()) {
    m.name.push_back(m.names.intern(q.is_null(0) ? std::string_view("None") : q.text(0)));
    m.timestamp.push_back(q.is_null(1) ? 0 : q.integer(1));
    m.value.push_back(q.real(2));
  }
}

// The total duration and number of calls per API name.
void read_summary(const std::string& path, const std::set<std::string>& tables, const char* table, NsysSummary& out) {
  if (!tables.count(table) || !tables.count("StringIds")) return;
  SqliteReader db(path);
  SqliteQuery q(db, std::string("SELECT COALESCE(s.value, 'unknown'), SUM(r.end - r.start), COUNT(*) FROM ") + table +
                        " r LEFT JOIN StringIds s ON r.nameId = s.id WHERE r.end > r.start GROUP BY 1");
  while (q.step()) {
    out.names.emplace_back(q.text(0));
    out.duration_ns.push_back(q.integer(1));
    out.calls.push_back(q.integer(2));
  }
}

}  // namespace

NsysColumns read_nsys(const std::string& path) {
  std::set<std::string> tables;
  {
    SqliteReader db(path);
    tables = sqlite_tables(db);
  }

  Events events;
  LaunchMap launches;
  Ranges ranges;
  Metrics metrics;
  NsysSummary runtime, osrt;
  NsysCalls runtime_calls;
  std::exception_ptr errors[7];
  auto run = [&](int slot, auto&& body) {
    try {
      body();
    } catch (...) {
      errors[slot] = std::current_exception();
    }
  };
  {
    ScopedJoin pool;
    pool.emplace([&] { run(0, [&] { read_launches(path, tables, launches); }); });
    pool.emplace([&] { run(1, [&] { read_nvtx(path, tables, ranges); }); });
    pool.emplace([&] { run(2, [&] { read_metrics(path, tables, metrics); }); });
    pool.emplace([&] { run(4, [&] { read_summary(path, tables, "CUPTI_ACTIVITY_KIND_RUNTIME", runtime); }); });
    pool.emplace([&] { run(5, [&] { read_summary(path, tables, "OSRT_API", osrt); }); });
    pool.emplace([&] { run(6, [&] { read_runtime_calls(path, tables, runtime_calls); }); });
    run(3, [&] { read_events(path, tables, events); });
    pool.join_all();
  }
  for (const auto& e : errors)
    if (e) std::rethrow_exception(e);

  NsysColumns out;
  const std::size_t n = events.name.size();
  out.gpu_launch_ns.resize(n);
  out.gpu_launch_thread.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    LaunchMap::Hit hit{};
    const bool found = launches.find(events.correlation[i], hit);
    out.gpu_launch_ns[i] = (found && hit.start != kNoStart) ? static_cast<double>(hit.start)
                                                              : std::numeric_limits<double>::quiet_NaN();
    out.gpu_launch_thread[i] = found ? hit.thread : -1;
  }
  out.names.assign(events.names.items().begin(), events.names.items().end());
  out.gpu_name = std::move(events.name);
  out.gpu_category = std::move(events.category);
  out.gpu_start = std::move(events.start);
  out.gpu_end = std::move(events.end);
  out.gpu_bytes = std::move(events.bytes);
  out.gpu_stream = std::move(events.stream);
  out.gpu_device = std::move(events.device);

  out.nvtx_texts.assign(ranges.texts.items().begin(), ranges.texts.items().end());
  out.nvtx_text = std::move(ranges.text);
  out.nvtx_start = std::move(ranges.start);
  out.nvtx_end = std::move(ranges.end);
  out.nvtx_thread = std::move(ranges.thread);

  out.runtime = std::move(runtime);
  out.osrt = std::move(osrt);
  out.runtime_calls = std::move(runtime_calls);
  out.has_metrics = metrics.present;
  out.metric_names.assign(metrics.names.items().begin(), metrics.names.items().end());
  out.metric_name = std::move(metrics.name);
  out.metric_timestamp = std::move(metrics.timestamp);
  out.metric_value = std::move(metrics.value);
  return out;
}

}  // namespace cta::readers
