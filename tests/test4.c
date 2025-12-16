#include <stdio.h>
#include "../nn_include/nn.h"

int main()
{   
    layer* inLayer0 = make_input_layer(2);
    if(inLayer0 == NULL)
    {
        printf("Memory allocation failed at inLayer0\n");
        goto error1;
    }

    layer* inLayer1 = make_input_layer(1);
    if(inLayer1 == NULL)
    {
        printf("Memory allocation failed at inLayer1\n");
        goto error2;
    }


    layer* layer0 = make_dense_layer((layer**[]){&inLayer0}, 2, 1, 1, 'u');
    if(layer0 == NULL)
    {
        printf("Memory allocation failed at layer0\n");
        goto error3;
    }
    
    layer* layer1 = make_dense_layer((layer**[]){&inLayer1}, 1, 1, 1, 'u');
    if(layer1 == NULL)
    {
        printf("Memory allocation failed at layer1\n");
        goto error4;
    }

    layer* layer2 = make_dense_layer((layer**[]){&layer0, &layer1}, 3, 2, 1, 'u');
    if(layer2 == NULL)
    {
        printf("Memory allocation failed at layer2\n");
        goto error5;
    }

    layer* outLayer = make_output_layer((layer**[]){&layer2}, 1, 1, 'g');
    if(outLayer == NULL)
    {
        printf("Memory allocation failed at outLayer\n");
        goto error6;
    }

    model *myModel = construct_model((layer**[]){&inLayer0, &inLayer1}, &outLayer, 6, 2, 1.0f, 'q');
    if(myModel == NULL)
    {
        printf("Memory allocation failed at model\n");
        goto error7;
    }

    printf("Model creation successful\n\n");

    memcpy((*myModel->inLayers[0])->outputs, (float[]){0.04, 0.05}, 2*sizeof(float));
    memcpy((*myModel->inLayers[1])->outputs, (float[]){0.1}, sizeof(float));

    memcpy(myModel->targets, (float[]){0.905405}, sizeof(float));

    forward_out(myModel->outLayer);
    sgd_backprop(myModel->outLayer, &myModel);
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

    hakai_model(&myModel);
    
    printf("Test Successful\n");
    return 0;

error7:
    hakai_layer_mfree(&outLayer);
error6:
    hakai_layer_mfree(&layer2);
error5:
    hakai_layer_mfree(&layer1);
error4:
    hakai_layer_mfree(&layer0);
error3:
    hakai_layer_mfree(&inLayer1);
error2:
    hakai_layer_mfree(&inLayer0);
error1:
    exit(EXIT_FAILURE);
}