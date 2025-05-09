#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <vector>

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
  /**
   * @brief Logs an informational message
   * @param message The message to log
   */
  static void info(const std::string& message);

  /**
   * @brief Logs an error message
   * @param message The error message to log
   */
  static void error(const std::string& message);

  /**
   * @brief Logs a warning message
   * @param message The warning message to log
   */
  static void warning(const std::string& message);

  /**
   * @brief Logs a debug message
   * @param message The debug message to log
   */
  static void debug(const std::string& message);

  /**
   * @brief Logs a fatal error message
   * @param message The fatal error message to log
   * @note This method may terminate the application depending on configuration
   */
  static void fatal(const std::string& message);

  /**
   * @brief Logs statistics about a float vector
   * @param name Name of the vector for identification
   * @param v Vector to analyze
   * @param n_show Number of elements to show in the log (default: 10)
   * @note Logs min, max, mean, and first n_show elements of the vector
   */
  static void log_vector_stats(const std::string& name,
                               const std::vector<float>& v, int n_show = 10);
};

#endif