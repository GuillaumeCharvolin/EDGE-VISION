cmake_minimum_required(VERSION 3.23)
project(edge_vision)

include(FetchContent)

############################################################################
# DEPENDENCIES
FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.4.2
)
FetchContent_MakeAvailable(cli11)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

############################################################################
# EDGE-VISION
add_library(edge_vision_lib src/argParser.cxx)
target_include_directories(edge_vision_lib PUBLIC include)
target_link_libraries(edge_vision_lib PUBLIC CLI11::CLI11)

add_executable(edge_vision src/main.cxx)
target_link_libraries(edge_vision PRIVATE edge_vision_lib)

############################################################################
# TESTS
enable_testing()

add_executable(my_tests tests/test_argParser.cpp)
target_link_libraries(my_tests PRIVATE edge_vision_lib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(my_tests)

############################################################################
# DOCS
find_package(Doxygen)
if (DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
        COMMENT "Generating API documentation with Doxygen into docs/api/"
        VERBATIM
    )
else()
    message(WARNING "Doxygen needs to be installed to generate the doxygen documentation")
endif()
