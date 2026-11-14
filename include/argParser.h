#pragma once
#include "appConfig.h"
#include <optional>

/**
 * @file argParser.h
 * @brief Command-line options of the program.
 */

/**
 * @brief Read and check the command-line options.
 *
 * - --model (required): an existing .onnx file.
 * - --input (required): a .jpg image, a .mp4/.avi video, or a camera index (e.g. 0).
 * - --output: .jpg for an image, .mp4 for a video or camera.
 * - --detections: .csv, image input only (for now).
 * - --timings: .csv, video or camera input only. --warmup: frames left out of it.
 *
 * On error, a message is printed and nullopt is returned. All paths in the result are absolute.
 *
 * @param argc Number of arguments, as given to main().
 * @param argv The arguments, as given to main().
 * @return The checked configuration, or nullopt if an option is missing or invalid.
 */
std::optional<AppConfig> parseArguments(int argc, char** argv);

/**
 * @brief Tell whether a file is a video the program can read, from its extension.
 * @param file Path of the file.
 * @return true for .mp4 and .avi files.
 */
bool isVideoFile(const std::filesystem::path& file);
