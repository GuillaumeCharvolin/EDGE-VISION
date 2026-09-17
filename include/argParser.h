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
 * This function ensure the required arguments are given to the program, verify their compliance (file exist, correct format)
 * and make these arguments ready to be used.
 *
 * @param argc number of argumentes given to the program (separated by spaces)
 * @param argv pointer to the list of string given to the program.
 * 
 * @return An object AppConfig if the parsing succeed, nullopt if not.
 */
std::optional<AppConfig> parseArguments(int argc, char** argv);