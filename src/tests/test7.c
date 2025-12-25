#include <stdio.h>
#include "../include/cml.h"

/*  
Model Structure
        inLayer0 - denseLayer0 - 
                                \
        inLayer1 - denseLayer1 -- denseLayer3 - denseLayer4 - outLayer
                                /
        inLayer2 - denseLayer2 
*/

int main()
{
    char filename[] = "THPDiffs.csv";
    int numIns = 9;
    int numOuts = 1;
    int numSamples = 20;
    float** inArrays = (float**)NULL;
    float** outArrays = (float**)NULL; 

    layer* inLayer0 = make_input_layer(3);
    if(inLayer0 == NULL) goto error1;

    layer* inLayer1 = make_input_layer(3);
    if(inLayer0 == NULL) goto error2;

    layer* inLayer2 = make_input_layer(3);
    if(inLayer0 == NULL) goto error3;

    layer* denseLayer0 = make_hidden_layer((layer**[]){&inLayer0}, 3, 1, 'u');
    if(denseLayer0 == NULL) goto error4;

    layer* denseLayer1 = make_hidden_layer((layer**[]){&inLayer1}, 3, 1, 'u');
    if(denseLayer1 == NULL) goto error5;

    layer* denseLayer2 = make_hidden_layer((layer**[]){&inLayer2}, 3, 1, 'u');
    if(denseLayer2 == NULL) goto error6;

    layer* denseLayer3 = make_hidden_layer((layer**[]){&denseLayer0, &denseLayer1, &denseLayer2}, 9, 3, 'u');
    if(denseLayer3 == NULL) goto error7;

    layer* denseLayer4 = make_hidden_layer((layer**[]){&denseLayer3}, 3, 1, 'u');
    if(denseLayer4 == NULL) goto error8;

    layer* outLayer = make_output_layer((layer**[]){&denseLayer4}, 1, 1, 'g');
    if(denseLayer2 == NULL) goto error9;

    layer* inLayers[] = {inLayer0, inLayer1, inLayer2};
    model *wethrModel = construct_model((layer**[]){&inLayer0, &inLayer1, &inLayer2}, &outLayer, 9, 3, 0.1f, 'q');
    if(wethrModel == NULL) goto error10;

    read_csv(filename, numSamples, numIns, numOuts, &inArrays, &outArrays);

    train_model_sgd(wethrModel, 500, numSamples, inArrays, outArrays, 0.5);

    save_model(&wethrModel, "testWeatherModel.cml");
    
    hakai_matrix(&inArrays, numSamples);
    hakai_matrix(&outArrays, numSamples);
    hakai_model(&wethrModel);
    printf("\nSuccess\n");
    return 0;

error10:
    hakai_layer_mfree(&outLayer);
error9:
    hakai_layer_mfree(&denseLayer4);
error8:
    hakai_layer_mfree(&denseLayer3);
error7:
    hakai_layer_mfree(&denseLayer2);
error6:
    hakai_layer_mfree(&denseLayer1);
error5:
    hakai_layer_mfree(&denseLayer0);
error4:
    hakai_layer_mfree(&inLayer2);
error3:
    hakai_layer_mfree(&inLayer1);
error2:
    hakai_layer_mfree(&inLayer0);
error1:    
    exit(EXIT_FAILURE);
}