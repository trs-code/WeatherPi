#include <stdio.h>
#include "nn_include/nn.h"

int main()
{

    layer* inLayer0 = make_input_layer(3, 1, 0);
    if(inLayer0 == NULL)
    {
        printf("Memory allocation failed at inLayer0\n");
        goto error1;
    }

    layer* inLayer1 = make_input_layer(3, 1, 1);
    if(inLayer0 == NULL)
    {
        printf("Memory allocation failed at inLayer1\n");
        goto error2;
    }

    layer* inLayer2 = make_input_layer(3, 1, 2);
    if(inLayer0 == NULL)
    {
        printf("Memory allocation failed at inLayer2\n");
        goto error3;
    }

    layer* inLayerPrev0[] = {inLayer0}; 
    layer* denseLayer0 = make_dense_layer(inLayerPrev0, 3, 1, 1, 3);
    if(denseLayer0 == NULL)
    {
        printf("Memory allocation failed at denseLayer0\n");
        goto error4;
    }

    layer* inLayerPrev1[] = {inLayer1};
    layer* denseLayer1 = make_dense_layer(inLayerPrev1, 3, 1, 1, 4);
    if(denseLayer1 == NULL)
    {
        printf("Memory allocation failed at denseLayer1\n");
        goto error5;
    }

    layer* inLayerPrev2[] = {inLayer2};
    layer* denseLayer2 = make_dense_layer(inLayerPrev2, 3, 1, 1, 5);
    if(denseLayer2 == NULL)
    {
        printf("Memory allocation failed at denseLayer2\n");
        goto error6;
    }

    layer* denseLayer3Prev[] = {denseLayer0, denseLayer1, denseLayer2};
    layer* denseLayer3 = make_dense_layer(denseLayer3Prev, 9, 3, 1, 6);
    if(denseLayer3 == NULL)
    {
        printf("Memory allocation failed at denseLayer3\n");
        goto error7;
    }

    layer* denseLayer4Prev[] = {denseLayer3};
    layer* denseLayer4 = make_dense_layer(denseLayer4Prev, 3, 1, 1, 7);
    if(denseLayer4 == NULL)
    {
        printf("Memory allocation failed at denseLayer5\n");
        goto error8;
    }

    layer* outLayerPrev[] = {denseLayer4};
    layer* outLayer = make_output_layer(outLayerPrev, 1, 1, 8);
    if(denseLayer2 == NULL)
    {
        printf("Memory allocation failed at outLayer\n");
        goto error9;
    }

    layer* inLayers[] = {inLayer0, inLayer1, inLayer2};
    model *wethrModel = construct_model(inLayers, outLayer, 9, 3, 0.01f, 1);
    if(wethrModel == NULL)
    {
        printf("Memory allocation failed at wethrModel\n");
        goto error10;
    }

    wethrModel->inLayers[0] = inLayer0;
    wethrModel->inLayers[1] = inLayer1;
    wethrModel->inLayers[2] = inLayer2;
    wethrModel->outLayer = outLayer;
    // Line 216 - 33.08,55.83,30.27,-0.9,4.43,0.03,-0.66,0.32,0.02,0
    float testIn0[] = {33.08, 55.83, 30.27};
    float testIn1[] = {-0.9, 4.43, 0.03};
    float testIn2[] = {-0.66, 0.32, 0.02};
    float testTarg[] = {0};

    // memcpy(wethrModel->inLayers[0]->outputs, testIn0, 3*sizeof(float));
    // memcpy(wethrModel->inLayers[1]->outputs, testIn1, 3*sizeof(float));
    // memcpy(wethrModel->inLayers[2]->outputs, testIn2, 3*sizeof(float));
    // memcpy(wethrModel->targets, testTarg, sizeof(float));

    for(int i = 0; i < 10000; i++)
    {
        memcpy(wethrModel->inLayers[0]->outputs, testIn0, 3*sizeof(float));
        memcpy(wethrModel->inLayers[1]->outputs, testIn1, 3*sizeof(float));
        memcpy(wethrModel->inLayers[2]->outputs, testIn2, 3*sizeof(float));
        memcpy(wethrModel->targets, testTarg, sizeof(float));
        zero_everything(wethrModel->outLayer);
        forward_out(wethrModel->outLayer);
        sgd_backprop(wethrModel->outLayer, wethrModel);
        calculate_and_apply_grads(wethrModel->outLayer, wethrModel->learning_rate);
    }
    printf("\nModel output is: %f\nTarget is : %f", wethrModel->outLayer->outputs[0], testTarg[0]);


    hakai_model(wethrModel);
    printf("\nSuccess\n");
    return 0;

error10:
    hakai_layer_mfree(outLayer);
error9:
    hakai_layer_mfree(denseLayer4);
error8:
    hakai_layer_mfree(denseLayer3);
error7:
    hakai_layer_mfree(denseLayer2);
error6:
    hakai_layer_mfree(denseLayer1);
error5:
    hakai_layer_mfree(denseLayer0);
error4:
    hakai_layer_mfree(inLayer2);
error3:
    hakai_layer_mfree(inLayer1);
error2:
    hakai_layer_mfree(inLayer0);
error1:    
    exit(EXIT_FAILURE);
}