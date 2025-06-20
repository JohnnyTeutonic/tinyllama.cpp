#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>

// Windows header protection
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// Undefine common Windows macros that might conflict
#ifdef ERROR
#undef ERROR
#endif
#ifdef DEBUG
#undef DEBUG
#endif
#ifdef INFO
#undef INFO
#endif
#ifdef WARNING
#undef WARNING
#endif
#ifdef CRITICAL
#undef CRITICAL
#endif
#endif // _WIN32

/**
 * @file logger.h
 * @brief Logging utilities for the TinyLlama implementation
 *
 * This file provides a static Logger class that handles different types of log
 * messages with various severity levels. It also includes utilities for logging
 * vector statistics, which is useful for debugging model operations.
 */

/**
 * @brief Static logging class for application-wide logging
 * 
 * Provides methods for logging messages at different severity levels (info,
 * error, warning, debug, fatal) and utilities for logging vector statistics.
 * All methods are static and can be called from anywhere in the application.
 */
class Logger {

 public:
  // Explicitly scoped enumeration to avoid conflicts with Windows macros
  enum class Level : int { 
    DEBUG = 0, 
    INFO = 1, 
    WARNING = 2, 
    ERROR = 3, 
    CRITICAL = 4, 
    OFF = 5 
  };

  static void set_level(Level new_level);
  static Level get_level();
  static void set_logfile(const std::string& filename);
  static void enable_console(bool enabled);

  static void debug(const std::string& message);
  static void info(const std::string& message);
  static void warning(const std::string& message);
  static void error(const std::string& message);
  static void critical(const std::string& message);
  static void fatal(const std::string& message);

  // Helper to convert pointer to string for logging
  static std::string ptrToString(const void* ptr);

  // Helper to convert uint16_t to hex string for logging
  static std::string uint16ToHex(uint16_t val);

  // Templated helper to convert unsigned integral types to hex string
  template <typename T>
  static std::string to_hex(T val);

  // Helper for vector stats (if it's part of public API, otherwise private)
  static void log_vector_stats(const std::string& name, const std::vector<float>& v, int n_show = 5);
  static void log_vector_stats_int8(const std::string& name, const std::vector<int8_t>& v, int n_show = 5);

 private:
  static Level current_level_;
  static std::ofstream log_file_stream_;
  static std::string log_file_path_;
  static bool console_enabled_;
  static bool log_file_truncated_;

  static void log_internal(Level level, const std::string& message);
  static std::string level_to_string(Level level);
  static void ensure_logfile_open_and_truncated();
};