cmake_minimum_required(VERSION 3.23)


############################################################################
# EDGE VISION
project(edge_vision)

add_library(edge_vision_lib src/hello.cxx)          # defines hello_function
target_include_directories(edge_vision_lib PUBLIC include)

add_executable(edge_vision src/main.cxx)
target_link_libraries(edge_vision PRIVATE edge_vision_lib)


############################################################################
# DOXYGEN
find_package(Doxygen)

if (DOXYGEN_FOUND)
    # Run from docs/ so OUTPUT_DIRECTORY=api lands in docs/api/
    # (Doxygen resolves OUTPUT_DIRECTORY relative to the working directory).
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
        COMMENT "Generating API documentation with Doxygen into docs/api/"
        VERBATIM
    )
else()
    message(WARNING "Doxygen needs to be installed to generate the doxygen documentation")
endif()


############################################################################
# GTEST
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0 # Use a specific version for stability
)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(my_tests tests/test_main.cpp)
target_link_libraries(my_tests PRIVATE edge_vision_lib GTest::gtest_main)

include(GoogleTest)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/Testing)

gtest_discover_tests(my_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/Testing
)