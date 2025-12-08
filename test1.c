#include <stdio.h>
#include "nn_include/nn.h"

int main()
{
    struct model *myModel = construct_model(3, 1, 1.0f, 1);
    if(myModel == NULL)
    {
        printf("Memory allocation failed at model\n");
        goto error1;
    }
    

    struct layer* layer0 = make_input_layer(3, 1, 0);
    if(layer0 == NULL)
    {
        printf("Memory allocation failed at layer0\n");
        goto error2;
    }
    
    struct layer* layer1_in[] = {layer0};
    struct layer* layer1 = make_dense_layer(layer1_in, 3, 1, 1, 1);
    if(layer1 == NULL)
    {
        printf("Memory allocation failed at layer1\n");
        goto error3;
    }

    struct layer* layer2_in[] = {layer1};
    struct layer* layer2 = make_output_layer(layer2_in, 1, 1, 2);
    if(layer1 == NULL)
    {
        printf("Memory allocation failed at layer0\n");
        goto error4;
    }

    myModel->inLayers[0] = layer0;
    myModel->outLayer = layer2;

    printf("Model creation successful\n\n");

    // float values[] = {0.05, 0.10, 0.15};
    // memcpy(myModel->inLayers[0]->outputs, values, 3*sizeof(float));
    // float target[] = {0.905405};
    // memcpy(myModel->targets, target, sizeof(float));

    // forward_out(myModel->outLayer);
    // sgd_backprop(myModel->outLayer, myModel);
    // for(int i = 0; i < 3; i++) printf("Layer 1 Activation[%d] is: %f\n", i, layer1->outputs[i]);
    // printf("\nLayer 2 Activation is: %f\n", layer2->outputs[0]);

    // printf("\nModel output is: %f\n", myModel->outLayer->outputs[0]);
    // printf("\nLayer 2 Backerror is: %f\n", layer2->backErrors[0]);
    // for(int i = 0; i < 3; i++) printf("Layer 1 Backerror[%d] is: %f\n", i, layer1->backErrors[i]);
    
    // calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
    
    // for(int i = 0; i < 10000; i++)
    // {
    //     zero_everything(myModel->outLayer);
    //     forward_out(myModel->outLayer);
    //     sgd_backprop(myModel->outLayer, myModel);
    //     calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
    //     //printf("%d\n", i);
    // }

    // printf("\nModel output is: %f\nTarget is : %f", myModel->outLayer->outputs[0], target[0]);
    // printf("\nLayer 2 Weights:\n");
    // for(int i = 0; i < 3; i++) printf("[%f]\n", layer2->weights[0][i]);
    // printf("\nLayer 1 Weights:\n");
    // for(int i = 0; i < 3; i++) printf("[%f] [%f] [%f]\n", layer1->weights[0][i], layer1->weights[1][i], layer1->weights[2][i]);

    // printf("\nLayer 2 Backerror is: %f\n", layer2->backErrors[0]);
    // for(int i = 0; i < 3; i++) printf("Layer 1 Backerror[%d] is: %f\n", i, layer1->backErrors[i]);

    // printf("\nLayer 2 pre-activation is: %f\n", layer2->preActivations[0]);
    // for(int i = 0; i < 3; i++) printf("Layer 1 pre-activation[%d] is: %f\n", i, layer1->preActivations[i]);

    // struct layer* layers[3] = {layer0, layer1, layer2};
    // for(int i = 0; i < 3; i++)
    // {
    //     hakai_layer_mfree(layers[i]);
    // }

    // hakai_layer_mfree(layer0);
    // hakai_layer_mfree(layer1);
    // hakai_layer_mfree(layer2);

    hakai_model(myModel);
    
    printf("Test Successful\n");
    return 0;

error4:
    hakai_layer_mfree(layer2);
error3:
    hakai_layer_mfree(layer1);
error2:
    hakai_model_mfree(myModel);
error1:
    exit(EXIT_FAILURE);
}