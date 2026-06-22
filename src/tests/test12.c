#include <stdio.h>
#include "../include/cml.h"

int main()
{
    char filename[] = "THPOneHotFull.csv";
    layer** modelLayers = NULL;
    float** inArrays = (float**)NULL;
    float** outArrays = (float**)NULL; 
    int numIns = 3;
    int numOuts = 2;
    int numSamples = 67764;

    model *wethrModel = load_model("weathrModelContextBest.cml", &modelLayers);
    if(wethrModel == NULL)
    {
        printf("Failed to load model\n");
        goto error1;
    }

    if(read_csv(filename, numSamples, numIns, numOuts, &inArrays, &outArrays) != 0) goto error2;

    train_rnn_sgd_fast(wethrModel, 20, numSamples, inArrays, outArrays, 0.0, 24, 0.0);

    hakai_matrix(&inArrays, numSamples);
    hakai_matrix(&outArrays, numSamples);
    hakai_model(&wethrModel);
    free(modelLayers);
    modelLayers = NULL;
    printf("\nEnd\n");
    return 0;

error2:
    hakai_matrix(&inArrays, numSamples);
    hakai_matrix(&outArrays, numSamples);
    hakai_model(&wethrModel);
    free(modelLayers);
    modelLayers = NULL;
error1:
    exit(EXIT_FAILURE);
}