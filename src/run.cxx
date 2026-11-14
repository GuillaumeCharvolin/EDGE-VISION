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

// Draw the boxes on the image and save it. Returns false (with an error message) on failure.
bool saveImageWithBoxes(cv::Mat& image, const std::vector<Detection>& detections,
                        const std::filesystem::path& file) {
    drawDetections(image, detections);
    if (!cv::imwrite(file.string(), image)) {
        std::cerr << "Error: Could not write the output image to: " << file.string() << std::endl;
        return false;
    }
    std::cout << "Saved " << file.string() << '\n';
    return true;
}

// Save the detections as CSV. Returns false (with an error message) on failure.
bool saveDetectionsCsv(const std::vector<Detection>& detections, const std::filesystem::path& file) {
    std::ofstream csv(file);
    if (!csv) {
        std::cerr << "Error: Could not open the detections file: " << file.string() << std::endl;
        return false;
    }
    writeDetectionsCsv(csv, detections);
    std::cout << "Saved " << file.string() << '\n';
    return true;
}

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
    std::optional<cv::Mat> image = loadImage(config.inputFile);
    if (!image) {
        return 1;
    }

    std::optional<std::vector<Detection>> detections = detect(model, *image);
    if (!detections) {
        return 1;
    }
    writeDetectionsText(std::cout, *detections);

    if (!config.outputFile.empty() && !saveImageWithBoxes(*image, *detections, config.outputFile)) {
        return 1;
    }
    if (!config.detectionsFile.empty() && !saveDetectionsCsv(*detections, config.detectionsFile)) {
        return 1;
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
