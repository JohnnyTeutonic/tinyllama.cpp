#include "logger.h"
#include <fstream>

void Logger::info(const std::string& message) {
    std::ofstream log("debugging.log", std::ios::app);
    log << "[INFO] " << message << std::endl;
}

void Logger::error(const std::string& message) {
    std::ofstream log("debugging.log", std::ios::app);
    log << "[ERROR] " << message << std::endl;
} 