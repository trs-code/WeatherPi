#include <stdio.h>
#include "../include/cml.h"

int main()
{
    char filename[] = "THPDiffsFull.csv";
    layer** modelLayers = NULL;
    float** inArrays = (float**)NULL;
    float** outArrays = (float**)NULL; 
    int numIns = 9;
    int numOuts = 1;
    int numSamples = 67758;

    model *wethrModel = load_model("../../weathrModel.cml", &modelLayers);
    if(wethrModel == NULL)
    {
        printf("Failed to load model\n");
        goto error1;
    }

    if(read_csv(filename, numSamples, numIns, numOuts, &inArrays, &outArrays) != 0) goto error2;

    train_model_sgd(wethrModel, 2, numSamples, inArrays, outArrays, 1.0, 0.2);

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
error1:
    if(modelLayers != NULL)
    {
        free(modelLayers);
        modelLayers = NULL;
    }
    exit(EXIT_FAILURE);
}