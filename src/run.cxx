#include "run.h"
#include "draw.h"
#include "pipeline.h"
#include "preprocess.h"
#include "serialize.h"
#include "stream.h"

#include <atomic>
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

const char* const windowName = "edge_vision";

// Rate written in the output video: the input's rate, so a video file plays at normal speed.
// For a camera this is the camera rate, higher than what we process: the recording plays too fast.
double inputFps(cv::VideoCapture& capture) {
    const double fps = capture.get(cv::CAP_PROP_FPS);
    return fps > 0 ? fps : 30.0;
}

// Show the frame in a window. Returns true if the user pressed 'q' or Esc.
bool showAndCheckQuit(const cv::Mat& frame) {
    cv::imshow(windowName, frame);
    const int key = cv::waitKey(1);
    return key == 'q' || key == 27;  // 27 = Esc
}

// Frame loop shared by video files and cameras. With showWindow, frames are displayed live
// and 'q' or Esc stops the loop.
int runStream(Model& model, const AppConfig& config, cv::VideoCapture& capture, bool showWindow) {
    TimingsLog timingsLog;
    if (!config.timingsFile.empty() && !timingsLog.open(config.timingsFile, config.warmupFrames)) {
        return 1;
    }
    VideoOutput video(config.outputFile, inputFps(capture));
    FpsMeter fps;

    cv::Mat frame;
    int frameCount = 0;
    while (!stopRequested) {
        StageTimings timings;
        const Clock::time_point frameStart = Clock::now();
        if (!capture.read(frame)) {
            break;  // end of the video file, or camera disconnected
        }
        timings.captureMs = millisecondsSince(frameStart);

        std::optional<std::vector<Detection>> detections = detect(model, frame, &timings);
        if (!detections) {
            return 1;
        }
        drawDetections(frame, *detections);
        const double currentFps = fps.tick();
        drawFps(frame, currentFps);

        if (!video.write(frame)) {
            return 1;
        }
        if (showWindow && showAndCheckQuit(frame)) {
            stopRequested = true;
        }

        timings.totalMs = millisecondsSince(frameStart);
        timingsLog.record(frameCount, timings);

        ++frameCount;
        if (frameCount % 100 == 0) {
            std::cout << "  " << frameCount << " frames, FPS " << currentFps << '\n';
        }
    }

    if (showWindow) {
        cv::destroyWindow(windowName);
    }
    if (stopRequested) {
        std::cout << "Stopped by user\n";
    } else if (frameCount == 0) {
        std::cerr << "Error: No frame could be read from the input" << std::endl;
        return 1;
    }

    std::cout << "Processed " << frameCount << " frames, mean FPS " << fps.meanFps() << '\n';
    if (!config.outputFile.empty()) {
        std::cout << "Saved " << config.outputFile.string() << '\n';
    }
    if (timingsLog.isOpen()) {
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
