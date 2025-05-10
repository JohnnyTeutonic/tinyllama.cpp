#include "logger.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <vector>

namespace {

void ensure_log_truncated() {
  static bool truncated = false;
  if (!truncated) {
    std::ofstream log("debugging.log", std::ios::trunc);
    truncated = true;
  }
}
}  // namespace

void Logger::info(const std::string& message) {
  ensure_log_truncated();
  std::ofstream log("debugging.log", std::ios::app);
  log << "[INFO] " << message << std::endl;
}

void Logger::error(const std::string& message) {
  ensure_log_truncated();
  std::ofstream log("debugging.log", std::ios::app);
  log << "[ERROR] " << message << std::endl;
}

void Logger::warning(const std::string& message) {
  ensure_log_truncated();
  std::ofstream log("debugging.log", std::ios::app);
  log << "[WARNING] " << message << std::endl;
}

void Logger::debug(const std::string& message) {
  ensure_log_truncated();
  std::ofstream log("debugging.log", std::ios::app);
  log << "[DEBUG] " << message << std::endl;
}

void Logger::fatal(const std::string& message) {
  ensure_log_truncated();

  std::ofstream log("debugging.log", std::ios::app);
  log << "[FATAL] " << message << std::endl;
  log.close();

  std::cerr << "[FATAL] " << message << std::endl;

  std::exit(EXIT_FAILURE);
}

void Logger::log_vector_stats(const std::string& name,
                              const std::vector<float>& v, int n_show) {
  if (v.empty()) {
    info(name + ": (empty)");
    return;
  }
  float minv = *std::min_element(v.begin(), v.end());
  float maxv = *std::max_element(v.begin(), v.end());
  float mean = std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
  bool all_finite =
      std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
  std::string first_vals;
  for (int i = 0; i < n_show && i < v.size(); ++i)
    first_vals += std::to_string(v[i]) + " ";
  info(name + ": min=" + std::to_string(minv) +
       ", max=" + std::to_string(maxv) + ", mean=" + std::to_string(mean) +
       ", all_finite=" + (all_finite ? "yes" : "no") +
       ", first_vals=" + first_vals);
}