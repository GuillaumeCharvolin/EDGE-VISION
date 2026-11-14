#pragma once
#include <chrono>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "pipeline.h"

/**
 * @file stream.h
 * @brief Small helpers for the video / camera frame loop: FPS, timings CSV, output video.
 */

using Clock = std::chrono::steady_clock;

/**
 * @brief Milliseconds elapsed since start.
 */
double millisecondsSince(Clock::time_point start);

/**
 * @brief Measures frames per second: a smoothed value for the overlay and a mean for the summary.
 */
class FpsMeter {
public:
    /** @param start When the loop starts (default: now). */
    explicit FpsMeter(Clock::time_point start = Clock::now());

    /**
     * @brief Call once per frame.
     * @param now The current time (default: now; tests pass a fixed time).
     * @return The FPS smoothed over the last frames (90% previous value, 10% new frame),
     * so the number on screen does not jump at every frame.
     */
    double tick(Clock::time_point now = Clock::now());

    /** @return Frames counted so far divided by the time since start. */
    double meanFps(Clock::time_point now = Clock::now()) const;

private:
    Clock::time_point start_;
    Clock::time_point previous_;
    int frames_ = 0;
    double smoothedFps_ = 0.0;
};

/**
 * @brief Writes one CSV row of stage timings per frame, skipping the warm-up frames.
 * Does nothing until open() succeeds.
 */
class TimingsLog {
public:
    /**
     * @brief Create the file and write the header.
     * @param file The CSV file to write.
     * @param warmupFrames Frames 0 to warmupFrames - 1 are not recorded: they are slower
     * (memory allocation, cold caches) and would distort the statistics.
     * @return false (with an error message) if the file cannot be created.
     */
    bool open(const std::filesystem::path& file, int warmupFrames);

    /** @brief Write the row for this frame, unless it is a warm-up frame or the log is not open. */
    void record(int frameIndex, const StageTimings& timings);

    bool isOpen() const { return csv_.is_open(); }

private:
    std::ofstream csv_;
    int warmupFrames_ = 0;
};

/**
 * @brief The output video. Opened on the first frame, once the frame size is known.
 */
class VideoOutput {
public:
    /**
     * @param file The .mp4 file to write, or an empty path to write nothing.
     * @param fps Playback rate written in the file.
     */
    VideoOutput(std::filesystem::path file, double fps);

    /** @return false (with an error message) if the file cannot be opened; true otherwise. */
    bool write(const cv::Mat& frame);

private:
    std::filesystem::path file_;
    double fps_;
    cv::VideoWriter writer_;
};
