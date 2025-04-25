#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <vector>

class Logger {
public:
    // Log an info message (appends to debugging.log)
    static void info(const std::string& message);
    // Log an error message (appends to debugging.log)
    static void error(const std::string& message);
    // Log vector stats: min, max, mean, all_finite, and first n_show values
    static void log_vector_stats(const std::string& name, const std::vector<float>& v, int n_show = 10);
};

#endif // LOGGER_H 