#include <gtest/gtest.h>
#include <iostream>
#include <sstream>

#include "hello.h"

TEST(OutputTest, PrintsHelloWorld) {
    std::stringstream buffer;
    std::streambuf* old_buffer = std::cout.rdbuf();

    std::cout.rdbuf(buffer.rdbuf());

    hello_function();

    std::cout.rdbuf(old_buffer);

    EXPECT_EQ(buffer.str(), "Hello world!\n");
}