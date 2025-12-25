#include <stdio.h>
#include "../include/cml.h"

/*  
Model Structure
        inLayer - hiddenLayer0 -- outLayer
                       \  /     |
                        \/      |
                                |
                           hiddenLayer1
                               \  /
                                \/
*/

int main()
{
    layer* inLayer = make_input_layer(3);

    if(inLayer == NULL)
    {
        printf("Memory allocation failed at inLayer\n");
        goto error1;
    }
    
    layer* hiddenLayer0 = make_referential_layer((layer**[]){&inLayer}, 1, 1, 'h', &hiddenLayer0);
    if(hiddenLayer0 == NULL)
    {
        printf("Memory allocation failed at hiddenLayer0\n");
        goto error2;
    }

    layer* hiddenLayer1 = make_referential_layer((layer**[]){&hiddenLayer0}, 1, 1, 'h', &hiddenLayer1);
    if(hiddenLayer0 == NULL)
    {
        printf("Memory allocation failed at hiddenLayer1\n");
        goto error3;
    }


    layer* outLayer = make_output_layer((layer**[]){&hiddenLayer1}, 1, 1, 'g');
    if(outLayer == NULL)
    {
        printf("Memory allocation failed at outLayer\n");
        goto error4;
    }

    model *myModel = construct_model((layer**[]){&inLayer}, &outLayer, 4, 1, 1.0f, 'q');
    if(myModel == NULL)
    {
        printf("Memory allocation failed at model\n");
        goto error5;
    }

    printf("Model creation successful\n\n");

    memcpy((*myModel->inLayers[0])->outputs, (float[]){0.05, 0.10, 0.15}, 3*sizeof(float));
    memcpy(myModel->targets, (float[]){0.905405}, sizeof(float));

    forward_out(myModel->outLayer);
    sgd_backprop(myModel->outLayer, &myModel);
    for(int i = 0; i < 1; i++) printf("hiddenLayer0 output[%d] is: %f\n", i, hiddenLayer0->outputs[i]);

    printf("\nModel output is: %f\n", (*myModel->outLayer)->outputs[0]);
    printf("\noutLayer Backerror is: %f\n", outLayer->backErrors[0]);
    for(int i = 0; i < 1; i++) printf("hiddenLayer0 Backerror[%d] is: %f\n", i, hiddenLayer0->backErrors[i]);
    
    calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
    
    for(int i = 0; i < 500; i++)
    {
        zero_everything(myModel->outLayer);
        forward_out(myModel->outLayer);
        sgd_backprop(myModel->outLayer, &myModel);
        calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
    }

    printf("\nModel output is: %f\nTarget is : %f", (*myModel->outLayer)->outputs[0], myModel->targets[0]);
    printf("\noutLayer Weights:\n");
    for(int i = 0; i < 1; i++) printf("[%f]\n", outLayer->weights[0][i]);
    printf("\nhiddenLayer0 Weights:\n");
    for(int i = 0; i < 4; i++) printf("[%f]\n", hiddenLayer0->weights[0][i]);

    printf("\noutLayer Backerror is: %f\n", outLayer->backErrors[0]);
    for(int i = 0; i < 1; i++) printf("hiddenLayer0 Backerror[%d] is: %f\n", i, hiddenLayer0->backErrors[i]);

    printf("\noutLayer preActivation is: %f\n", outLayer->preActivations[0]);
    for(int i = 0; i < 1; i++) printf("hiddenLayer0 preActivation[%d] is: %f\n", i, hiddenLayer0->preActivations[i]);

    save_model(&myModel, "test9Model.cml");

    hakai_model(&myModel);
    
    printf("Test Successful\n");
    return 0;

error5:
    hakai_layer_mfree(&outLayer);
error4:
    hakai_layer_mfree(&hiddenLayer1);
error3:
    hakai_layer_mfree(&hiddenLayer0);
error2:
    hakai_layer_mfree(&inLayer);
error1:
    exit(EXIT_FAILURE);
}
