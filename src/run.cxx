#include "run.h"
#include "draw.h"
#include "pipeline.h"
#include "preprocess.h"
#include "serialize.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>

namespace {

// Set by Ctrl+C (SIGINT). The frame loop checks it and stops cleanly, so the camera is released
// and the output video is finalised. A signal handler may only write to a lock-free atomic.
std::atomic<bool> stopRequested{false};

void onSigint(int) {
    stopRequested = true;
}

// Frame loop shared by video files and cameras. With showWindow, frames are displayed live
// and 'q' or Esc stops the loop.
int runStream(Model& model, const AppConfig& config, cv::VideoCapture& capture, bool showWindow) {
    using Clock = std::chrono::steady_clock;
    const std::string windowName = "edge_vision";

    // Write the output at the same rate as the input, so a video file plays at normal speed.
    // For a camera this is the camera rate, higher than what we process: the recording plays too fast.
    double inputFps = capture.get(cv::CAP_PROP_FPS);
    if (inputFps <= 0) {
        inputFps = 30.0;
    }
    cv::VideoWriter writer;

    std::ofstream timingsCsv;
    if (!config.timingsFile.empty()) {
        timingsCsv.open(config.timingsFile);
        if (!timingsCsv) {
            std::cerr << "Error: Could not open the timings file: " << config.timingsFile.string() << std::endl;
            return 1;
        }
        writeTimingsCsvHeader(timingsCsv);
    }

    cv::Mat frame;
    int frameCount = 0;
    double smoothedFps = 0.0;
    const Clock::time_point start = Clock::now();
    Clock::time_point previous = start;

    while (!stopRequested) {
        StageTimings timings;
        const Clock::time_point frameStart = Clock::now();
        if (!capture.read(frame)) {
            break;  // end of the video file, or camera disconnected
        }
        timings.captureMs = std::chrono::duration<double, std::milli>(Clock::now() - frameStart).count();

        std::optional<std::vector<Detection>> detectionsOpt = detect(model, frame, &timings);
        if (!detectionsOpt) {
            return 1;
        }
        drawDetections(frame, *detectionsOpt);

        // FPS from the time between two frames (read + detect + draw + write),
        // smoothed so the number does not jump at every frame.
        const Clock::time_point now = Clock::now();
        const double frameSeconds = std::chrono::duration<double>(now - previous).count();
        previous = now;
        const double instantFps = frameSeconds > 0 ? 1.0 / frameSeconds : 0.0;
        smoothedFps = frameCount == 0 ? instantFps : 0.9 * smoothedFps + 0.1 * instantFps;
        drawFps(frame, smoothedFps);

        if (!config.outputFile.empty()) {
            // Opened on the first frame, once the frame size is known.
            if (!writer.isOpened()) {
                writer.open(config.outputFile.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            inputFps, frame.size());
                if (!writer.isOpened()) {
                    std::cerr << "Error: Could not write the output video to: " << config.outputFile.string() << std::endl;
                    return 1;
                }
            }
            writer.write(frame);
        }

        if (showWindow) {
            cv::imshow(windowName, frame);
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {  // 27 = Esc
                stopRequested = true;
            }
        }

        // Warm-up frames are slower (memory allocation, cold caches) and left out of the csv.
        if (timingsCsv.is_open() && frameCount >= config.warmupFrames) {
            timings.totalMs = std::chrono::duration<double, std::milli>(Clock::now() - frameStart).count();
            writeTimingsCsvRow(timingsCsv, frameCount, timings);
        }

        ++frameCount;
        if (frameCount % 100 == 0) {
            std::cout << "  " << frameCount << " frames, FPS " << smoothedFps << '\n';
        }
    }

    if (showWindow) {
        cv::destroyWindow(windowName);
    }
    if (stopRequested) {
        std::cout << "Stopped by user\n";
    }

    if (frameCount == 0 && !stopRequested) {
        std::cerr << "Error: No frame could be read from the input" << std::endl;
        return 1;
    }

    const double totalSeconds = std::chrono::duration<double>(Clock::now() - start).count();
    std::cout << "Processed " << frameCount << " frames, mean FPS "
              << (totalSeconds > 0 ? frameCount / totalSeconds : 0.0) << '\n';
    if (!config.outputFile.empty()) {
        std::cout << "Saved " << config.outputFile.string() << '\n';
    }
    if (timingsCsv.is_open()) {
        std::cout << "Saved " << config.timingsFile.string() << '\n';
    }
    return 0;
}

}  // namespace

void stopOnCtrlC() {
    std::signal(SIGINT, onSigint);
}

int runImage(Model& model, const AppConfig& config) {
    std::optional<cv::Mat> matOpt = loadImage(config.inputFile);
    if (!matOpt) {
        return 1;
    }
    cv::Mat mat = std::move(*matOpt);

    std::optional<std::vector<Detection>> detectionsOpt = detect(model, mat);
    if (!detectionsOpt) {
        return 1;
    }
    const std::vector<Detection>& detections = *detectionsOpt;

    std::cout << "Detections (" << detections.size() << "):\n";
    for (const Detection& det : detections) {
        std::cout << "  class=" << cocoClassName(det.classId) << " score=" << det.score
                  << " x=" << det.box.x << " y=" << det.box.y
                  << " w=" << det.box.width << " h=" << det.box.height << '\n';
    }

    if (!config.outputFile.empty()) {
        drawDetections(mat, detections);
        if (!cv::imwrite(config.outputFile.string(), mat)) {
            std::cerr << "Error: Could not write the output image to: " << config.outputFile.string() << std::endl;
            return 1;
        }
        std::cout << "Saved " << config.outputFile.string() << '\n';
    }

    if (!config.detectionsFile.empty()) {
        std::ofstream csv(config.detectionsFile);
        if (!csv) {
            std::cerr << "Error: Could not open the detections file: " << config.detectionsFile.string() << std::endl;
            return 1;
        }
        writeDetectionsCsv(csv, detections);
        std::cout << "Saved " << config.detectionsFile.string() << '\n';
    }

    return 0;
}

int runVideoFile(Model& model, const AppConfig& config) {
    cv::VideoCapture capture(config.inputFile.string());
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open the video at: " << config.inputFile.string() << std::endl;
        return 1;
    }
    return runStream(model, config, capture, /*showWindow=*/false);
}

int runCamera(Model& model, const AppConfig& config) {
    cv::VideoCapture capture(*config.cameraIndex);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open camera " << *config.cameraIndex << std::endl;
        return 1;
    }
    // Keep at most one frame waiting, so we always process a recent frame instead of an old one
    // when inference is slower than the camera. Some backends ignore this setting.
    capture.set(cv::CAP_PROP_BUFFERSIZE, 1);

    std::cout << "Camera " << *config.cameraIndex << " opened. Press q, Esc or Ctrl+C to stop.\n";
    return runStream(model, config, capture, /*showWindow=*/true);
}
