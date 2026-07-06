#include <stdio.h>
#include "src/include/cml.h"

int main()
{
    char filename[] = "dataset/THPDiffs.csv";
    int numIns = 9;
    int numOuts = 1;
    int numSamples = 67758;
    float** inArrays = (float**)NULL;
    float** outArrays = (float**)NULL; 

    layer* inLayer0 = make_input_layer(3);
    if(inLayer0 == NULL) goto error1;

    layer* inLayer1 = make_input_layer(3);
    if(inLayer0 == NULL) goto error2;

    layer* inLayer2 = make_input_layer(3);
    if(inLayer0 == NULL) goto error3;

    layer* denseLayer0 = make_hidden_layer((layer**[]){&inLayer0}, 9, 1, 'h');
    if(denseLayer0 == NULL) goto error4;

    layer* denseLayer1 = make_hidden_layer((layer**[]){&inLayer1}, 9, 1, 'h');
    if(denseLayer1 == NULL) goto error5;

    layer* denseLayer2 = make_hidden_layer((layer**[]){&inLayer2}, 9, 1, 'h');
    if(denseLayer2 == NULL) goto error6;

    layer* denseLayer3 = make_hidden_layer((layer**[]){&denseLayer0, &denseLayer1, &denseLayer2}, 36, 3, 'h');
    if(denseLayer3 == NULL) goto error7;

    layer* denseLayer4 = make_hidden_layer((layer**[]){&denseLayer3}, 18, 1, 'h');
    if(denseLayer4 == NULL) goto error8;

    layer* outLayer = make_output_layer((layer**[]){&denseLayer4}, 1, 1, 's');
    if(denseLayer2 == NULL) goto error9;

    model *wethrModel = construct_model((layer**[]){&inLayer0, &inLayer1, &inLayer2}, &outLayer, 9, 3, 0.0000005f, 'n');
    if(wethrModel == NULL) goto error10;

    if(read_csv(filename, numSamples, numIns, numOuts, &inArrays, &outArrays) != 0) goto error11;

    train_model_sgd_fast(wethrModel, 10, numSamples, inArrays, outArrays, 0.8, 0.5);

    //save_model(wethrModel, "weathrModel.cml");
    hakai_matrix(&inArrays, numSamples);
    hakai_matrix(&outArrays, numSamples);
    hakai_model(&wethrModel);
    printf("\nEnd\n");
    return 0;

error11:
    hakai_matrix(&inArrays, numSamples);
    hakai_matrix(&outArrays, numSamples);
    hakai_model(&wethrModel);
error10:
    hakai_layer(&outLayer);
error9:
    hakai_layer(&denseLayer4);
error8:
    hakai_layer(&denseLayer3);
error7:
    hakai_layer(&denseLayer2);
error6:
    hakai_layer(&denseLayer1);
error5:
    hakai_layer(&denseLayer0);
error4:
    hakai_layer(&inLayer2);
error3:
    hakai_layer(&inLayer1);
error2:
    hakai_layer(&inLayer0);
error1:    
    exit(EXIT_FAILURE);
}