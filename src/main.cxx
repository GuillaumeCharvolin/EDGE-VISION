#include "argParser.h"
#include "model.h"

int main(int argc, char** argv) {
    std::optional<AppConfig> configOpt = parseArguments(argc, argv);
    if (!configOpt) {
        return 1;
    }
    const AppConfig& config = *configOpt;

    std::optional<Model> modelOpt = Model::load(config.modelFile);
    if (!modelOpt) {
        return 1;
    }
    Model model = std::move(*modelOpt);

    return 0;
}
