#include <stdio.h>
#include "src/include/cml.h"
#include "src/include/layer_construct.h"

int main()
{
    char filename[] = "dataset/THPOneHot.csv";
    layer** windowLayers = NULL;
    float** inArrays = (float**)NULL;
    float** outArrays = (float**)NULL; 
    int numIns = 3;
    int numOuts = 2;
    int numSamples = 67764;
    int windowSize = 3;

    layer* inLayer0 = make_input_layer(3);
    if(inLayer0 == NULL)goto error1;
    
    layer* hiddenLayer0 = make_hidden_layer((layer**[]){&inLayer0}, 64, 1, 'h');
    if(hiddenLayer0 == NULL) goto error2;

    extend_context(hiddenLayer0, windowSize, &windowLayers);
    if(windowLayers == NULL) goto error3;

    //layer* attnLayer0 = make_attention_layer(&hiddenLayer0, windowSize, 'x');

    layer* outLayer = make_output_layer((layer**[]){&hiddenLayer0}, 2, 1, 'x');
    if(outLayer == NULL) goto error4;

    model *wethrModel = construct_model(&outLayer, 0.00001, 'x');
    if(wethrModel == NULL) goto error5;

    if(read_csv(filename, numSamples, numIns, numOuts, &inArrays, &outArrays) != 0) goto error6;

    train_rnn_sgd(wethrModel, 10, numSamples, inArrays, outArrays, 0.85, windowSize, 0.25);

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