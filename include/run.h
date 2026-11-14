#pragma once
#include "appConfig.h"
#include "model.h"

/**
 * @file run.h
 * @brief Run the detector on an image, a video file or a camera.
 *
 * Each run function returns a process exit code: 0 on success, 1 on error.
 */

/**
 * @brief Make Ctrl+C stop the frame loop cleanly instead of killing the program,
 * so the camera is released and the output video is finalised.
 */
void stopOnCtrlC();

/**
 * @brief Detect objects in one image, print them, and save the --output / --detections files if asked.
 */
int runImage(Model& model, const AppConfig& config);

/**
 * @brief Detect objects in every frame of a video file, without a window.
 */
int runVideoFile(Model& model, const AppConfig& config);

/**
 * @brief Detect objects live from a camera and show them in a window. 'q', Esc or Ctrl+C stops.
 */
int runCamera(Model& model, const AppConfig& config);
