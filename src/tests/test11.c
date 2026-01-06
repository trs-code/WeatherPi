#include <stdio.h>
#include "../include/cml.h"

/*  
Model Structure
        inLayer - layer0 - outLayer

*/

int main()
{
    int upperLimit = 10000;
    float** inArr = (float**)malloc(sizeof(float*));
    float** outArr = (float**)malloc(sizeof(float*));

    inArr[0] = (float*)malloc(sizeof(float) * upperLimit);
    outArr[0] = (float*)malloc(sizeof(float));

    for(int i = 0; i < upperLimit; i++) inArr[0][i] = (float)i / upperLimit;
    outArr[0][0] = 0.905405;

    layer* inLayer = make_input_layer(upperLimit);
    if(inLayer == NULL)
    {
        printf("Memory allocation failed at inLayer\n");
        goto error1;
    }
    
    layer* layer0 = make_hidden_layer((layer**[]){&inLayer}, upperLimit, 1, 'u');
    if(layer0 == NULL)
    {
        printf("Memory allocation failed at layer0\n");
        goto error2;
    }

    layer* outLayer = make_output_layer((layer**[]){&layer0}, 1, 1, 'g');
    if(outLayer == NULL)
    {
        printf("Memory allocation failed at outLayer\n");
        goto error3;
    }

    model* myModel = construct_model((layer**[]){&inLayer}, &outLayer, 3, 1, 0.1f, 'q');
    if(myModel == NULL)
    {
        printf("Memory allocation failed at model\n");
        goto error4;
    }

    printf("Model creation successful\n\n");

    train_model_sgd(myModel, 5, 1, inArr, outArr, 1.0);

    hakai_model(&myModel);
    free(inArr[0]);
    inArr[0] = NULL;
    free(outArr[0]);
    outArr[0] = NULL;
    free(inArr);
    inArr = NULL;
    free(outArr);
    outArr = NULL;

    return 0;

error4:
    hakai_layer_mfree(&outLayer);
error3:
    hakai_layer_mfree(&layer0);
error2:
    hakai_layer_mfree(&inLayer);
error1:
    exit(EXIT_FAILURE);
}