#include "logger.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace {

void ensure_log_truncated() {
  static bool truncated = false;
  if (!truncated) {
    std::ofstream log("debugging.log", std::ios::trunc);
    truncated = true;
  }
}
}  // namespace

// Initialize static members
Logger::Level Logger::current_level_ = Logger::Level::INFO; // Default level
std::ofstream Logger::log_file_stream_;
std::string Logger::log_file_path_ = "debugging.log"; // Default log file
bool Logger::console_enabled_ = false; // MODIFIED: Default to console logging DISABLED
bool Logger::log_file_truncated_ = false;
std::mutex logger_mutex; // Global mutex for logger operations

void Logger::set_level(Level new_level) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    current_level_ = new_level;
}

Logger::Level Logger::get_level() {
    std::lock_guard<std::mutex> lock(logger_mutex);
    return current_level_;
}

void Logger::set_logfile(const std::string& filename) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (log_file_stream_.is_open()) {
        log_file_stream_.close();
    }
    log_file_path_ = filename;
    log_file_truncated_ = false; // Force truncation for new file
    ensure_logfile_open_and_truncated(); // Open new file
}

void Logger::enable_console(bool enabled) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    console_enabled_ = enabled;
}

std::string Logger::level_to_string(Level level) {
    switch (level) {
        case Level::DEBUG:    return "DEBUG";
        case Level::INFO:     return "INFO";
        case Level::WARNING:  return "WARNING";
        case Level::ERROR:    return "ERROR";
        case Level::CRITICAL: return "CRITICAL";
        case Level::OFF:      return "OFF"; // Should not happen if check is done before
        default:              return "UNKNOWN";
    }
}

void Logger::ensure_logfile_open_and_truncated() {
    if (!log_file_stream_.is_open() || !log_file_truncated_) {
        if (log_file_stream_.is_open()) {
            log_file_stream_.close();
        }
        if (!log_file_truncated_) {
            log_file_stream_.open(log_file_path_, std::ios::out | std::ios::trunc);
            log_file_truncated_ = true;
        } else {
            log_file_stream_.open(log_file_path_, std::ios::out | std::ios::app);
        }
        
        if (!log_file_stream_.is_open()) {
            // Fallback to cerr if file cannot be opened
            std::cerr << "[LOGGER_ERROR] Failed to open log file: " << log_file_path_ << std::endl;
        }
    } else if (log_file_stream_.is_open() && log_file_stream_.tellp() == 0 && !log_file_truncated_) {
        // File was opened by another instance/call but not truncated by this instance yet.
        // This case is tricky with static loggers if not managed carefully.
        // For simplicity, we assume first successful open truncates.
        // Re-opening in trunc mode if it's empty and we haven't truncated.
         log_file_stream_.close();
         log_file_stream_.open(log_file_path_, std::ios::out | std::ios::trunc);
         log_file_truncated_ = true;
         if (!log_file_stream_.is_open()) {
            std::cerr << "[LOGGER_ERROR] Failed to re-open/truncate log file: " << log_file_path_ << std::endl;
        }
    }
}

void Logger::log_internal(Level level, const std::string& message) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (level < current_level_ || level == Level::OFF) {
        return;
    }

    ensure_logfile_open_and_truncated();

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream log_line_stream;
    log_line_stream << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") 
                    << " [" << level_to_string(level) << "] " << message;
    std::string log_line = log_line_stream.str();

    if (console_enabled_) {
        if (level == Level::ERROR || level == Level::CRITICAL || level == Level::WARNING) {
            std::cerr << log_line << std::endl;
        } else {
            std::cout << log_line << std::endl;
        }
    }

    if (log_file_stream_.is_open()) {
        log_file_stream_ << log_line << std::endl;
        log_file_stream_.flush(); // Ensure it's written immediately
    }
}

void Logger::debug(const std::string& message) {
    log_internal(Level::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log_internal(Level::INFO, message);
}

void Logger::warning(const std::string& message) {
    log_internal(Level::WARNING, message);
}

void Logger::error(const std::string& message) {
    log_internal(Level::ERROR, message);
}

void Logger::critical(const std::string& message) {
    log_internal(Level::CRITICAL, message);
}

void Logger::fatal(const std::string& message) {
    log_internal(Level::CRITICAL, "[FATAL] " + message); // Log as critical
    if (log_file_stream_.is_open()) {
        log_file_stream_.close();
    }
    std::cerr << "[FATAL] " << message << std::endl; // Ensure it goes to cerr
  std::exit(EXIT_FAILURE);
}

void Logger::log_vector_stats(const std::string& name,
                              const std::vector<float>& v, int n_show) {
  if (v.empty()) {
    log_internal(Level::INFO, name + ": (empty vector)");
    return;
  }
  float min_val = v[0];
  float max_val = v[0];
  double sum_val = 0.0;
  bool all_finite_vals = true;

  for (float val : v) {
    if (std::isnan(val) || std::isinf(val)) {
      all_finite_vals = false;
    }
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
    sum_val += static_cast<double>(val);
  }
  float mean_val = static_cast<float>(sum_val / v.size());

  std::ostringstream oss;
  oss << name << ": size=" << v.size()
      << ", min=" << min_val
      << ", max=" << max_val
      << ", mean=" << mean_val
      << ", all_finite=" << (all_finite_vals ? "yes" : "no")
      << ", first_vals=[";
  for (int i = 0; i < std::min((int)v.size(), n_show); ++i) {
    oss << v[i] << (i == std::min((int)v.size(), n_show) - 1 ? "" : ", ");
  }
  oss << "]";
  log_internal(Level::INFO, oss.str());
}

void Logger::log_vector_stats_int8(const std::string& name, const std::vector<int8_t>& v, int n_show) {
  if (v.empty()) {
    log_internal(Level::INFO, name + ": (empty int8_t vector)");
    return;
  }
  int8_t min_val = v[0];
  int8_t max_val = v[0];
  long long sum_val = 0; // Use long long for sum to avoid overflow with many int8_t values

  for (int8_t val : v) {
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
    sum_val += static_cast<long long>(val);
  }
  // Cast sum to double for mean calculation to preserve precision before casting to float
  float mean_val = static_cast<float>(static_cast<double>(sum_val) / v.size());

  std::ostringstream oss;
  oss << name << ": size=" << v.size()
      << ", min=" << static_cast<int>(min_val) // Cast to int for printing
      << ", max=" << static_cast<int>(max_val) // Cast to int for printing
      << ", mean=" << mean_val
      << ", first_vals=[";
  for (int i = 0; i < std::min((int)v.size(), n_show); ++i) {
    oss << static_cast<int>(v[i]) << (i == std::min((int)v.size(), n_show) - 1 ? "" : ", ");
  }
  oss << "]";
  log_internal(Level::INFO, oss.str());
}

std::string Logger::ptrToString(const void* ptr) {
    std::ostringstream oss;
    oss << ptr;
    return oss.str();
}

// Templated helper to convert unsigned integral types to hex string
template <typename T>
std::string Logger::to_hex(T val) {
    std::ostringstream oss;
    oss << "0x" << std::hex << static_cast<unsigned long long>(val); // Cast to larger type if necessary
    return oss.str();
}

// Explicit template instantiation for uint16_t if needed, or ensure it's defined in a way that uint16ToHex can find it.
// Or, make uint16ToHex call the templated version directly.

std::string Logger::uint16ToHex(uint16_t val) {
    return to_hex(val); // Now calls the templated version
}