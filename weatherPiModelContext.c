#include <stdio.h>
#include "src/include/cml.h"

int main()
{
    char filename[] = "dataset/THPOneHot.csv";
    layer** windowLayers = NULL;
    float** inArrays = (float**)NULL;
    float** outArrays = (float**)NULL; 
    int numIns = 3;
    int numOuts = 2;
    int numSamples = 67764;
    int windowSize = 12;

    layer* inLayer0 = make_input_layer(3);
    if(inLayer0 == NULL)goto error1;
    
    layer* hiddenLayer0 = make_hidden_layer((layer**[]){&inLayer0}, 64, 1, 'h');
    if(hiddenLayer0 == NULL) goto error2;

    extend_context(hiddenLayer0, windowSize, &windowLayers);
    if(windowLayers == NULL) goto error3;

    layer* hiddenLayer1 = make_hidden_layer((layer**[]){&hiddenLayer0}, 32, 1, 'r');
    if(hiddenLayer1 == NULL) goto error3;

    layer* outLayer = make_output_layer((layer**[]){&hiddenLayer1}, 2, 1, 'f');
    if(outLayer == NULL) goto error4;

    model *wethrModel = construct_model((layer**[]){&inLayer0}, &outLayer, 4 + (2 * windowSize), 1, 0.000000002, 'x');
    if(wethrModel == NULL) goto error5;

    if(read_csv(filename, numSamples, numIns, numOuts, &inArrays, &outArrays) != 0) goto error6;

    train_rnn_sgd_fast(wethrModel, 5, numSamples, inArrays, outArrays, 0.85, windowSize, 0.3);

    //save_model(wethrModel, "weathrModelContext.cml");
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
    hakai_layer(&outLayer);
error4:
    hakai_context_window(&windowLayers, windowSize);
error3:
    hakai_layer(&hiddenLayer0);
error2:
    hakai_layer(&inLayer0);
error1:
    printf("Error Occurred!\nEXITING");
    exit(EXIT_FAILURE);
}