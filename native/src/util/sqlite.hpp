// Small RAII wrappers over the SQLite C API, for reading.
#pragma once

#include <sqlite3.h>

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cta {

class SqliteReader {
 public:
  // Opens the database read-only. Throws std::runtime_error.
  explicit SqliteReader(const std::string& path) {
    if (sqlite3_open_v2(path.c_str(), &db_, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
      const std::string why = db_ ? sqlite3_errmsg(db_) : "out of memory";
      sqlite3_close(db_);
      throw std::runtime_error("cannot open " + path + ": " + why);
    }
  }
  ~SqliteReader() { sqlite3_close(db_); }
  SqliteReader(const SqliteReader&) = delete;
  SqliteReader& operator=(const SqliteReader&) = delete;
  sqlite3* get() const { return db_; }

 private:
  sqlite3* db_ = nullptr;
};

// One prepared statement, stepped row by row.
class SqliteQuery {
 public:
  SqliteQuery(const SqliteReader& db, const std::string& sql) : db_(db.get()) {
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt_, nullptr) != SQLITE_OK)
      throw std::runtime_error(std::string("sqlite: ") + sqlite3_errmsg(db_) + " in: " + sql);
  }
  ~SqliteQuery() { sqlite3_finalize(stmt_); }
  SqliteQuery(const SqliteQuery&) = delete;
  SqliteQuery& operator=(const SqliteQuery&) = delete;

  // Moves to the next row; false when there is none.
  bool step() {
    const int rc = sqlite3_step(stmt_);
    if (rc == SQLITE_ROW) return true;
    if (rc == SQLITE_DONE) return false;
    throw std::runtime_error(std::string("sqlite: ") + sqlite3_errmsg(db_));
  }
  bool is_null(int column) const { return sqlite3_column_type(stmt_, column) == SQLITE_NULL; }
  std::int64_t integer(int column) const { return sqlite3_column_int64(stmt_, column); }
  double real(int column) const { return sqlite3_column_double(stmt_, column); }
  std::string_view text(int column) const {
    const unsigned char* p = sqlite3_column_text(stmt_, column);
    return p ? std::string_view(reinterpret_cast<const char*>(p), static_cast<std::size_t>(sqlite3_column_bytes(stmt_, column)))
             : std::string_view();
  }

 private:
  sqlite3* db_;
  sqlite3_stmt* stmt_ = nullptr;
};

// Names of the tables of a database.
inline std::set<std::string> sqlite_tables(const SqliteReader& db) {
  std::set<std::string> out;
  SqliteQuery q(db, "SELECT name FROM sqlite_master WHERE type='table'");
  while (q.step()) out.emplace(q.text(0));
  return out;
}

// Names of the columns of a table.
inline std::set<std::string> sqlite_columns(const SqliteReader& db, const std::string& table) {
  std::set<std::string> out;
  SqliteQuery q(db, "PRAGMA table_info(" + table + ")");
  while (q.step()) out.emplace(q.text(1));
  return out;
}

}  // namespace cta
