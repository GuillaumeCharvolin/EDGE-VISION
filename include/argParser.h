#pragma once
#include "appConfig.h"
#include <optional>

/**
 * @file argParser.h
 * @brief Public API for argParser.
 */

/**
 * @brief Parsing of arguments given to the program file with --args
 *
 * --input can be a .jpg image or a .mp4/.avi video. --output must match it (.jpg for an image,
 * .mp4 for a video), and --detections is refused for a video.
 *
 * This function ensure the required arguments are given to the program, verify their compliance (file exist, correct format)
 * and make these arguments ready to be used.
 *
 * @param argc number of argumentes given to the program (separated by spaces)
 * @param argv pointer to the list of string given to the program.
 * 
 * @return An object AppConfig if the parsing succeed, nullopt if not.
 */
std::optional<AppConfig> parseArguments(int argc, char** argv);

/**
 * @brief Tell whether a file is a video the program can read, from its extension.
 * @param file Path of the file.
 * @return true for .mp4 and .avi files.
 */
bool isVideoFile(const std::filesystem::path& file);
