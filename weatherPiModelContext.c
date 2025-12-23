#include <stdio.h>
#include "nn_include/nn.h"

int main()
{
    char filename[] = "dataset/THP.csv";
    layer** windowLayers = NULL;
    float** inArrays = (float**)NULL;
    float** outArrays = (float**)NULL; 
    int numIns = 3;
    int numOuts = 1;
    int numSamples = 67764;
    int windowSize = 24;

    layer* inLayer0 = make_input_layer(3);
    if(inLayer0 == NULL)
    {
        printf("Memory allocation failed at inLayer\n");
        goto error1;
    }
    
    layer* hiddenLayer0 = make_hidden_layer((layer**[]){&inLayer0}, 1, 'h');
    if(hiddenLayer0 == NULL)
    {
        printf("Memory allocation failed at hiddenLayer0\n");
        goto error2;
    }

    extend_context(&hiddenLayer0, windowSize, &windowLayers);
    if(windowLayers == NULL) goto error3;

    layer* outLayer = make_output_layer((layer**[]){&windowLayers[(2 * windowSize) - 1]}, 1, 'g');
    if(outLayer == NULL)
    {
        printf("Memory allocation failed at outLayer\n");
        goto error4;
    }

    model *wethrModel = construct_model((layer**[]){&windowLayers[(2 * windowSize) - 2]}, &outLayer, 3 + (2 * windowSize), 1, 0.00001f, 'q');
    if(wethrModel == NULL)
    {
        printf("Memory allocation failed at model\n");
        goto error5;
    }

    if(read_csv(filename, numSamples, numIns, numOuts, &inArrays, &outArrays) != 0) goto error6;

    train_context_model_sgd(wethrModel, windowLayers, 500, numSamples, inArrays, outArrays, 0.8, windowSize);

    save_model(&wethrModel, "weathrModelContext.cml");
    hakai_matrix(&inArrays, numSamples);
    hakai_matrix(&outArrays, numSamples);
    hakai_model(&wethrModel);
    free(windowLayers);
    windowLayers = NULL;
    printf("\nEnd\n");
    return 0;

error6:
    hakai_matrix(&inArrays, numSamples);
    hakai_matrix(&outArrays, numSamples);
    hakai_model(&wethrModel);
error5:
    hakai_layer_mfree(&outLayer);
error4:
    hakai_context_window(&windowLayers, windowSize);
error3:
    hakai_layer_mfree(&hiddenLayer0);
error2:
    hakai_layer_mfree(&inLayer0);
error1:
    exit(EXIT_FAILURE);
}