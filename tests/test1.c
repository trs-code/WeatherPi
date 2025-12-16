#include <stdio.h>
#include <math.h>
#include "../nn_include/nn.h"

int main()
{
    layer* inLayer = make_input_layer(3);

    if(inLayer == NULL)
    {
        printf("Memory allocation failed at inLayer\n");
        goto error1;
    }
    
    layer* layer0 = make_dense_layer((layer**[]){&inLayer}, 3, 1, 1, 'u');
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

    model *myModel = construct_model((layer**[]){&inLayer}, &outLayer, 3, 1, 1.0f, 'q');
    if(myModel == NULL)
    {
        printf("Memory allocation failed at model\n");
        goto error4;
    }

    printf("Model creation successful\n\n");

    memcpy((*myModel->inLayers[0])->outputs, (float[]){0.05, 0.10, 0.15}, 3*sizeof(float));
    memcpy(myModel->targets, (float[]){0.905405}, sizeof(float));

    // forward_out(*(myModel->outLayer));
    // sgd_backprop(*(myModel->outLayer), &myModel);
    // for(int i = 0; i < 3; i++) printf("Layer 1 Activation[%d] is: %f\n", i, layer0->outputs[i]);
    // printf("\nLayer 2 Activation is: %f\n", outLayer->outputs[0]);

    // printf("\nModel output is: %f\n", (*myModel->outLayer)->outputs[0]);
    // printf("\nLayer 2 Backerror is: %f\n", outLayer->backErrors[0]);
    // for(int i = 0; i < 3; i++) printf("Layer 1 Backerror[%d] is: %f\n", i, layer0->backErrors[i]);
    
    // calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
    
    for(int i = 0; i < 300; i++)
    {
        zero_everything(myModel->outLayer);
        forward_out(myModel->outLayer);
        sgd_backprop(myModel->outLayer, &myModel);
        calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
        //printf("%d\n", i);
    }

    printf("\nModel output is: %f\nTarget is : %f", (*myModel->outLayer)->outputs[0], myModel->targets[0]);
    printf("\nLayer 2 Weights:\n");
    for(int i = 0; i < 3; i++) printf("[%f]\n", outLayer->weights[0][i]);
    printf("\nLayer 1 Weights:\n");
    for(int i = 0; i < 3; i++) printf("[%f] [%f] [%f]\n", layer0->weights[0][i], layer0->weights[1][i], layer0->weights[2][i]);

    printf("\nLayer 2 Backerror is: %f\n", outLayer->backErrors[0]);
    for(int i = 0; i < 3; i++) printf("Layer 1 Backerror[%d] is: %f\n", i, layer0->backErrors[i]);

    printf("\nLayer 2 pre-activation is: %f\n", outLayer->preActivations[0]);
    for(int i = 0; i < 3; i++) printf("Layer 1 pre-activation[%d] is: %f\n", i, layer0->preActivations[i]);

    save_model(&myModel, "testModel1.cml");

    hakai_model(&myModel);
    
    printf("Test Successful\n");
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