#include <gtest/gtest.h>

#include <sstream>
#include <vector>

#include "serialize.h"

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
