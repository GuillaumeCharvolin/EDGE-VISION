#include "appConfig.h"
#include "argParser.h"
#include "model.h"
#include "run.h"

#include <optional>

int main(int argc, char** argv) {
    std::optional<AppConfig> config = parseArguments(argc, argv);
    if (!config) {
        return 1;
    }

    std::optional<Model> model = Model::load(config->modelFile);
    if (!model) {
        return 1;
    }

    stopOnCtrlC();

    if (config->cameraIndex) {
        return runCamera(*model, *config);
    }
    if (isVideoFile(config->inputFile)) {
        return runVideoFile(*model, *config);
    }
    return runImage(*model, *config);
}
