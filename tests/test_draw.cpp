#include <gtest/gtest.h>

#include <vector>

#include "draw.h"

TEST(CocoClassNameTest, KnownAndUnknownIds) {
    EXPECT_EQ(cocoClassName(0), "person");
    EXPECT_EQ(cocoClassName(2), "car");
    EXPECT_EQ(cocoClassName(79), "toothbrush");
    EXPECT_EQ(cocoClassName(80), "unknown");
    EXPECT_EQ(cocoClassName(-1), "unknown");
}

TEST(DrawDetectionsTest, DrawsBoxEdgeOnly) {
    cv::Mat mat(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<Detection> dets = {{2, 0.9f, cv::Rect2f(50, 50, 100, 100)}};

    drawDetections(mat, dets);

    const cv::Vec3b black(0, 0, 0);
    EXPECT_NE(mat.at<cv::Vec3b>(100, 50), black);   // left edge (row, col)
    EXPECT_NE(mat.at<cv::Vec3b>(149, 100), black);  // bottom edge
    EXPECT_EQ(mat.at<cv::Vec3b>(100, 100), black);  // box center untouched
    EXPECT_EQ(mat.at<cv::Vec3b>(190, 190), black);  // far corner untouched
}

TEST(DrawDetectionsTest, NoDetectionLeavesImageUnchanged) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::Mat before = mat.clone();

    drawDetections(mat, {});

    EXPECT_EQ(cv::norm(mat, before, cv::NORM_INF), 0.0);
}
