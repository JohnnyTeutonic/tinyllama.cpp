#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <vector>

class Logger {
 public:
  static void info(const std::string& message);

  static void error(const std::string& message);

  static void warning(const std::string& message);

  static void debug(const std::string& message);

  static void fatal(const std::string& message);

  static void log_vector_stats(const std::string& name,
                               const std::vector<float>& v, int n_show = 10);
};

#endif