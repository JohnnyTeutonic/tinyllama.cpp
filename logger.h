#ifndef LOGGER_H
#define LOGGER_H

#include <string>

class Logger {
public:
    // Log an info message (appends to debugging.log)
    static void info(const std::string& message);
    // Log an error message (appends to debugging.log)
    static void error(const std::string& message);
};

#endif // LOGGER_H 