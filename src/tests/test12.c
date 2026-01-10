#include <stdio.h>
#include "../include/cml.h"

int main()
{
    char filename[] = "THPOneHot.csv";
    layer** windowLayers = NULL;
    layer** modelLayers = NULL;
    float** inArrays = (float**)NULL;
    float** outArrays = (float**)NULL; 
    int numIns = 3;
    int numOuts = 2;
    int numSamples = 48;
    int windowSize = 0;

    model *wethrModel = load_context_model("weathrModelContextBest.cml", &modelLayers, &windowLayers, &windowSize);
    if(wethrModel == NULL)
    {
        printf("Failed to load model\n");
        goto error1;
    }

    if(read_csv(filename, numSamples, numIns, numOuts, &inArrays, &outArrays) != 0) goto error2;

    train_context_model_sgd_fast(wethrModel, windowLayers, 100, numSamples, inArrays, outArrays, 0, windowSize);

    hakai_matrix(&inArrays, numSamples);
    hakai_matrix(&outArrays, numSamples);
    hakai_model(&wethrModel);
    free(modelLayers);
    modelLayers = NULL;
    free(windowLayers);
    windowLayers = NULL;
    printf("\nEnd\n");
    return 0;

error2:
    hakai_matrix(&inArrays, numSamples);
    hakai_matrix(&outArrays, numSamples);
    hakai_model(&wethrModel);
    free(modelLayers);
    modelLayers = NULL;
    free(windowLayers);
    windowLayers = NULL;
error1:
    exit(EXIT_FAILURE);
}