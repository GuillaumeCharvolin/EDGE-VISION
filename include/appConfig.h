#pragma once
#include <filesystem>
#include <optional>

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

    /// The absolute path to the input given with --input (a .jpg image, or a .mp4/.avi video).
    /// Empty when the input is a camera.
    std::filesystem::path inputFile;

    /// The camera index given with --input (e.g. 0 for /dev/video0). Empty when the input is a file.
    std::optional<int> cameraIndex;

    /// The absolute path where the image (.jpg) or video (.mp4) with drawn boxes is saved (--output). Empty if not given.
    std::filesystem::path outputFile;

    /// The absolute path where the detections are saved as csv (--detections). Empty if not given.
    std::filesystem::path detectionsFile;

    /// The absolute path where per-frame stage timings are saved as csv (--timings). Empty if not given.
    std::filesystem::path timingsFile;

    /// Number of first frames left out of the timings csv (--warmup), because they are slower.
    int warmupFrames = 10;
};