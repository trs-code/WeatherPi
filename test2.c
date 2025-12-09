#include <stdio.h>
#include "nn_include/nn.h"

int main()
{   
    layer* inLayer0 = make_input_layer(2, 1);
    if(inLayer0 == NULL)
    {
        printf("Memory allocation failed at inLayer0\n");
        goto error1;
    }

    layer* inLayer1 = make_input_layer(1, 1);
    if(inLayer1 == NULL)
    {
        printf("Memory allocation failed at inLayer1\n");
        goto error2;
    }
    
    layer* layer0_in[] = {inLayer0, inLayer1};
    layer* layer0 = make_dense_layer(layer0_in, 3, 2, 1);
    if(layer0 == NULL)
    {
        printf("Memory allocation failed at layer0\n");
        goto error3;
    }

    layer* outLayer_in[] = {layer0};
    layer* outLayer = make_output_layer(outLayer_in, 1, 1);
    if(outLayer == NULL)
    {
        printf("Memory allocation failed at outLayer\n");
        goto error4;
    }

    layer* inLayers[] = {inLayer0, inLayer1};
    model *myModel = construct_model(inLayers, outLayer, 4, 2, 1.0f, 1);
    if(myModel == NULL)
    {
        printf("Memory allocation failed at model\n");
        goto error5;
    }

    printf("Model creation successful\n\n");

    float values0[] = {0.35, 0.35};
    float values1[] = {0.39};
    memcpy(myModel->inLayers[0]->outputs, values0, 2*sizeof(float));
    memcpy(myModel->inLayers[1]->outputs, values1, sizeof(float));

    float target[] = {0.905405};
    memcpy(myModel->targets, target, sizeof(float));

    forward_out(myModel->outLayer);
    sgd_backprop(myModel->outLayer, myModel);
    for(int i = 0; i < 3; i++) printf("Layer 1 Activation[%d] is: %f\n", i, layer0->outputs[i]);
    printf("\noutLayer Activation is: %f\n", outLayer->outputs[0]);
    
    for(int i = 0; i < 3; i++) printf("Layer 1 Backerror[%d] is: %f\n", i, layer0->backErrors[i]);
    
    calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
    //
    // for(int i = 0; i < 10000; i++)
    // {
    //     zero_everything(myModel->outLayer);
    //     forward_out(myModel->outLayer);
    //     sgd_backprop(myModel->outLayer, myModel);
    //     calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
    //     //printf("%d\n", i);
    // }
    
    printf("\nModel output is: %f\nTarget is : %f", myModel->outLayer->outputs[0], target[0]);
    printf("\noutLayer Weights:\n");
    for(int i = 0; i < 3; i++) printf("[%f]\n", outLayer->weights[0][i]);
    printf("\nLayer 1 Weights:\n");
    for(int i = 0; i < 3; i++) printf("[%f] [%f] [%f]\n", layer0->weights[0][i], layer0->weights[1][i], layer0->weights[2][i]);

    printf("\noutLayer Backerror is: %f\n", outLayer->backErrors[0]);
    for(int i = 0; i < 3; i++) printf("Layer 1 Backerror[%d] is: %f\n", i, layer0->backErrors[i]);
    //
    printf("\noutLayer pre-activation is: %f\n", outLayer->preActivations[0]);
    for(int i = 0; i < 3; i++) printf("Layer 1 pre-activation[%d] is: %f\n", i, layer0->preActivations[i]);

    hakai_model(myModel);
    
    printf("Test Successful\n");
    return 0;

error5:
    hakai_layer_mfree(outLayer);
error4:
    hakai_layer_mfree(layer0);
error3:
    hakai_layer_mfree(inLayer1);
error2:
    hakai_layer_mfree(inLayer0);
error1:
    exit(EXIT_FAILURE);
}