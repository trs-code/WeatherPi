#include <stdio.h>
#include <unistd.h>
#include "src/include/cml.h"
#include "src/include/sensor/sensor.h"

int main()
{
    float vals[3] = {0.0, 0.0, 0.0};
    layer** modelLayers = NULL;
    model* wModel = load_model("weathrModelContextBest", modelLayers;
    if(success != 0) exit(EXIT_FAILURE);

    success = getWeatherInfo(vals);
    if(success != 0) exit(EXIT_FAILURE);

    for(int i = 0; i < 3; i++) printf("%d\n", vals[i]);

    hakai_model(&wModel);
    free(modelLayers);
    modelLayers = NULL;
    return 0;
}