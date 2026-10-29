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

    /// The absolute path to the input given with --input (a .jpg image for now).
    std::filesystem::path inputFile;

    /// The absolute path where the image with drawn boxes is saved (--output). Empty if not given.
    std::filesystem::path outputFile;

    /// The absolute path where the detections are saved as csv (--detections). Empty if not given.
    std::filesystem::path detectionsFile;
};