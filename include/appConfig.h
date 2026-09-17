#pragma once
#include <filesystem>

/**
 * @file appConfig.h
 * @brief Public API for appConfig.
 */

/**
 * @brief Configuration settings for the core application.
 *
 * This struct holds all the parameters parsed from the command line. 
 * It is passed by const reference to the application during initialization.
 */
struct AppConfig {
    /// The absolute path to the onnx model weights used.
    std::filesystem::path modelFile;

    /// The absolute path to the jpg image used.
    std::filesystem::path imageFile;
};