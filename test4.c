#include <stdio.h>
#include "nn_include/nn.h"

int main()
{
    struct model *myModel = construct_model(6, 2, 1.0f, 1);
    if(myModel == NULL)
    {
        printf("Memory allocation failed at model\n");
        goto error1;
    }
    
    struct layer* inLayer0 = make_input_layer(2, 1, 0);
    if(inLayer0 == NULL)
    {
        printf("Memory allocation failed at inLayer0\n");
        goto error2;
    }

    struct layer* inLayer1 = make_input_layer(1, 1, 1);
    if(inLayer1 == NULL)
    {
        printf("Memory allocation failed at inLayer1\n");
        goto error4;
    }

    struct layer* layer_in0[] = {inLayer0};
    struct layer* layer_in1[] = {inLayer1};

    struct layer* layer0 = make_dense_layer(layer_in0, 2, 1, 1, 2);
    if(layer0 == NULL)
    {
        printf("Memory allocation failed at layer0\n");
        goto error3;
    }
    
    struct layer* layer1 = make_dense_layer(layer_in1, 1, 1, 1, 3);
    if(layer1 == NULL)
    {
        printf("Memory allocation failed at layer1\n");
        goto error5;
    }

    struct layer* layer_in2[] = {layer0, layer1};
    struct layer* layer2 = make_dense_layer(layer_in2, 3, 2, 1, 4);
    if(layer2 == NULL)
    {
        printf("Memory allocation failed at layer2\n");
        goto error6;
    }

    struct layer* outLayer_in[] = {layer2};
    struct layer* outLayer = make_output_layer(outLayer_in, 1, 1, 5);
    if(outLayer == NULL)
    {
        printf("Memory allocation failed at outLayer\n");
        goto error7;
    }

    myModel->inLayers[0] = inLayer0;
    myModel->inLayers[1] = inLayer1;
    myModel->outLayer = outLayer;

    printf("Model creation successful\n\n");

    float values0[] = {0.04, 0.05};
    float values1[] = {0.1};
    memcpy(myModel->inLayers[0]->outputs, values0, 2*sizeof(float));
    memcpy(myModel->inLayers[1]->outputs, values1, sizeof(float));

    float target[] = {0.905405};
    memcpy(myModel->targets, target, sizeof(float));

    forward_out(myModel->outLayer);
    sgd_backprop(myModel->outLayer, myModel);
    // printf("Layer 0 Activation[0] is: %f\n", layer0->outputs[0]);
    // printf("Layer 0 Activation[1] is: %f\n", layer0->outputs[1]);
    // printf("Layer 1 Activation[0] is: %f\n", layer0->outputs[0]);
    // printf("\noutLayer Activation is: %f\n", outLayer->outputs[0]);

    calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
    
    // for(int i = 0; i < 10000; i++)
    // {
    //     zero_everything(myModel->outLayer);
    //     forward_out(myModel->outLayer);
    //     sgd_backprop(myModel->outLayer, myModel);
    //     calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
    //     //printf("%d\n", i);
    // }
    
    // printf("\nModel output is: %f\nTarget is : %f", myModel->outLayer->outputs[0], target[0]);
    // printf("\noutLayer Weights:\n");
    // for(int i = 0; i < 3; i++) printf("[%f]\n", outLayer->weights[0][i]);
    // printf("\nMid Layer Weights:\n");

    // printf("[%f] [%f] [%f]\n", layer0->weights[0][0], layer0->weights[0][1], layer0->weights[0][2]);
    // printf("[%f] [%f] [%f]\n", layer0->weights[1][0], layer0->weights[1][1], layer0->weights[1][2]);
    // printf("[%f] [%f] [%f]\n", layer1->weights[0][0], layer1->weights[0][1], layer1->weights[0][2]);

    // printf("Layer 0 Backerror[0] is: %f\n", layer0->backErrors[0]);
    // printf("Layer 0 Backerror[1] is: %f\n", layer0->backErrors[1]);
    // printf("Layer 1 Backerror[0] is: %f\n", layer1->backErrors[0]);

    // printf("\noutLayer pre-activation is: %f\n", outLayer->preActivations[0]);
    // printf("Layer 0 Backerror[0] is: %f\n", layer0->preActivations[0]);
    // printf("Layer 0 Backerror[1] is: %f\n", layer0->preActivations[1]);
    // printf("Layer 1 Backerror[0] is: %f\n", layer1->preActivations[0]);;

    hakai_model(myModel);
    
    printf("Test Successful\n");
    return 0;

error7:
    hakai_layer_mfree(layer2);
error6:
    hakai_layer_mfree(layer1);
error5:
    hakai_layer_mfree(inLayer1);
error4:
    hakai_layer_mfree(layer0);
error3:
    hakai_layer_mfree(inLayer0);
error2:
    hakai_model_mfree(myModel);
error1:
    exit(EXIT_FAILURE);
}