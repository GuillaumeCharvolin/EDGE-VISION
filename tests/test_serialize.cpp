#include <gtest/gtest.h>

#include <sstream>
#include <vector>

#include "serialize.h"

TEST(WriteDetectionsTextTest, CountThenOneLinePerDetection) {
    std::ostringstream out;
    writeDetectionsText(out, {{2, 0.758534f, cv::Rect2f(441.131f, 543.643f, 59.3481f, 47.5384f)},
                              {0, 0.5f, cv::Rect2f(1, 2, 3, 4)}});

    EXPECT_EQ(out.str(),
              "Detections (2):\n"
              "  class=car score=0.758534 x=441.131 y=543.643 w=59.3481 h=47.5384\n"
              "  class=person score=0.5 x=1 y=2 w=3 h=4\n");
}

TEST(WriteDetectionsCsvTest, NoDetectionWritesHeaderOnly) {
    std::ostringstream out;
    writeDetectionsCsv(out, {});

    EXPECT_EQ(out.str(), "class_id,class_name,score,x,y,w,h\n");
}

TEST(WriteDetectionsCsvTest, OneDetection) {
    std::ostringstream out;
    writeDetectionsCsv(out, {{2, 0.758534f, cv::Rect2f(441.131f, 543.643f, 59.3481f, 47.5384f)}});

    EXPECT_EQ(out.str(),
              "class_id,class_name,score,x,y,w,h\n"
              "2,car,0.7585,441.13,543.64,59.35,47.54\n");
}

TEST(WriteDetectionsCsvTest, KeepsOrder) {
    std::ostringstream out;
    writeDetectionsCsv(out, {
        {0, 0.9f, cv::Rect2f(1, 2, 3, 4)},
        {16, 0.5f, cv::Rect2f(10, 20, 30, 40)},
    });

    EXPECT_EQ(out.str(),
              "class_id,class_name,score,x,y,w,h\n"
              "0,person,0.9000,1.00,2.00,3.00,4.00\n"
              "16,dog,0.5000,10.00,20.00,30.00,40.00\n");
}

TEST(WriteTimingsCsvTest, HeaderOnly) {
    std::ostringstream out;
    writeTimingsCsvHeader(out);

    EXPECT_EQ(out.str(), "frame,capture_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms\n");
}

TEST(WriteTimingsCsvTest, TwoRows) {
    std::ostringstream out;
    writeTimingsCsvHeader(out);
    writeTimingsCsvRow(out, 10, {1.5, 4.25, 62.0, 0.1234, 75.0});
    writeTimingsCsvRow(out, 11, {2.0, 4.0, 61.5, 0.1, 73.9999});

    EXPECT_EQ(out.str(),
              "frame,capture_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms\n"
              "10,1.500,4.250,62.000,0.123,75.000\n"
              "11,2.000,4.000,61.500,0.100,74.000\n");
}
