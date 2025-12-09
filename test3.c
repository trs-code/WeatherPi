#include <stdio.h>
#include "nn_include/nn.h"

int main()
{
    layer* inLayer = make_input_layer(3, 1, 0);
    if(inLayer == NULL)
    {
        printf("Memory allocation failed at inLayer\n");
        goto error1;
    }
    layer* layer_in[] = {inLayer};

    layer* layer0 = make_dense_layer(layer_in, 2, 1, 1, 1);
    if(layer0 == NULL)
    {
        printf("Memory allocation failed at layer0\n");
        goto error2;
    }
    
    
    layer* layer1 = make_dense_layer(layer_in, 1, 1, 1, 2);
    if(layer1 == NULL)
    {
        printf("Memory allocation failed at layer1\n");
        goto error3;
    }

    layer* outLayer_in[] = {layer0, layer1};
    layer* outLayer = make_output_layer(outLayer_in, 1, 2, 3);
    if(layer1 == NULL)
    {
        printf("Memory allocation failed at outLayer\n");
        goto error4;
    }

    layer* inLayers[] = {inLayer};
    model *myModel = construct_model(inLayers, outLayer, 4, 1, 1.0f, 1);
    if(myModel == NULL)
    {
        printf("Memory allocation failed at model\n");
        goto error5;
    }

    printf("Model creation successful\n\n");

    float values[] = {0.05, 0.10, 0.15};
    memcpy(myModel->inLayers[0]->outputs, values, 3*sizeof(float));

    float target[] = {0.905405};
    memcpy(myModel->targets, target, sizeof(float));

    forward_out(myModel->outLayer);
    sgd_backprop(myModel->outLayer, myModel);
    printf("Layer 0 Activation[0] is: %f\n", layer0->outputs[0]);
    printf("Layer 0 Activation[1] is: %f\n", layer0->outputs[1]);
    printf("Layer 1 Activation[0] is: %f\n", layer0->outputs[0]);
    printf("\noutLayer Activation is: %f\n", outLayer->outputs[0]);

    calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
    
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
    printf("\nMid Layer Weights:\n");

    printf("[%f] [%f] [%f]\n", layer0->weights[0][0], layer0->weights[0][1], layer0->weights[0][2]);
    printf("[%f] [%f] [%f]\n", layer0->weights[1][0], layer0->weights[1][1], layer0->weights[1][2]);
    printf("[%f] [%f] [%f]\n", layer1->weights[0][0], layer1->weights[0][1], layer1->weights[0][2]);

    printf("Layer 0 Backerror[0] is: %f\n", layer0->backErrors[0]);
    printf("Layer 0 Backerror[1] is: %f\n", layer0->backErrors[1]);
    printf("Layer 1 Backerror[0] is: %f\n", layer1->backErrors[0]);

    printf("\noutLayer pre-activation is: %f\n", outLayer->preActivations[0]);
    printf("Layer 0 Backerror[0] is: %f\n", layer0->preActivations[0]);
    printf("Layer 0 Backerror[1] is: %f\n", layer0->preActivations[1]);
    printf("Layer 1 Backerror[0] is: %f\n", layer1->preActivations[0]);;

    hakai_model(myModel);
    
    printf("Test Successful\n");
    return 0;

error5:
    hakai_layer_mfree(outLayer);
error4:
    hakai_layer_mfree(layer1);
error3:
    hakai_layer_mfree(layer0);
error2:
    hakai_layer_mfree(inLayer);
error1:
    exit(EXIT_FAILURE);
}