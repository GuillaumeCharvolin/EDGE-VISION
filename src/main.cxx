#include "argParser.h"

// entrypoint function
int main(int argc, char** argv) {

    std::optional<AppConfig> configOpt = parseArguments(argc, argv);
    if (!configOpt) {
        return 1;
    }

    AppConfig config = configOpt.value();

    return 0;
}
