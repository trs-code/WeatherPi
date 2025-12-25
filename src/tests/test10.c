#include <stdio.h>
#include "../include/cml.h"

/*  
Model Structure
                    
            X(t - 1) 
        inLayer----hiddenLayer0 
            |          |
            |          |h(t-1)
            |          |        h(t)
            X(t)---hiddenLayer1 -- outLayer
                        .
                        .
                        .
                context x windowSize
*/

int main() 
{
    layer** windowLayers = NULL;
    int windowSize = 24;
    layer* layers[2 * (windowSize + 1)];

    layer* inLayer0 = make_input_layer(3);
    if(inLayer0 == NULL)
    {
        printf("Memory allocation failed at inLayer\n");
        goto error1;
    }
    
    layer* hiddenLayer0 = make_hidden_layer((layer**[]){&inLayer0}, 1, 1, 'h');
    if(hiddenLayer0 == NULL)
    {
        printf("Memory allocation failed at hiddenLayer0\n");
        goto error2;
    }

    extend_context(&hiddenLayer0, windowSize, (&windowLayers));
    if(windowLayers == NULL) goto error3;

    //layers[0] = hiddenLayer0;
    for(int i = 0; i < (2 * (windowSize + 1)); i++) layers[i] = windowLayers[i];//*layers[i - 1]->prevLayers[1];

    layer* outLayer = make_output_layer((layer**[]){&hiddenLayer0}, 1, 1, 'g');
    if(outLayer == NULL)
    {
        printf("Memory allocation failed at outLayer\n");
        goto error3;
    }

    model *myModel = construct_model((layer**[]){&inLayer0}, &outLayer, 3 + (2 * windowSize), 1, 1.0f, 'q');
    if(myModel == NULL)
    {
        printf("Memory allocation failed at model\n");
        goto error4;
    }

    printf("Model creation successful\n\n");

    load_context_window(windowLayers, (float[]){0.05, 0.10, 0.15}, windowSize);
    //memcpy((*myModel->inLayers[0])->outputs, (float[]){0.05, 0.10, 0.15}, 3*sizeof(float));
    memcpy(myModel->targets, (float[]){0.905405}, sizeof(float));

    forward_out(myModel->outLayer);
    sgd_backprop(myModel->outLayer, &myModel);
    for(int i = 0; i < 1; i++) printf("hiddenLayer0 output[%d] is: %f\n", i, hiddenLayer0->outputs[i]);

    printf("\nModel output is: %f\n", (*myModel->outLayer)->outputs[0]);
    printf("\noutLayer Backerror is: %f\n", outLayer->backErrors[0]);
    
    calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
    
    for(int i = 0; i < 200; i++)
    {
        zero_everything(myModel->outLayer);
        load_context_window(windowLayers, (float[]){0.05, 0.10, 0.15}, windowSize);
        forward_out(myModel->outLayer);
        sgd_backprop(myModel->outLayer, &myModel);
        calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);
    }

    printf("\nModel output is: %f\nTarget is : %f", (*myModel->outLayer)->outputs[0], myModel->targets[0]);
    printf("\noutLayer Weights:\n");
    for(int i = 0; i < 1; i++) printf("[%f]\n", outLayer->weights[0][i]);

    printf("\noutLayer Backerror is: %f\n", outLayer->backErrors[0]);
    for(int i = 0; i < 1; i++) printf("hiddenLayer0 Backerror[%d] is: %f\n", i, hiddenLayer0->backErrors[i]);

    printf("\noutLayer preActivation is: %f\n", outLayer->preActivations[0]);
    for(int i = 0; i < 1; i++) printf("hiddenLayer0 preActivation[%d] is: %f\n", i, hiddenLayer0->preActivations[i]);

    save_model(&myModel, "test10Model.cml");

    hakai_model(&myModel);

    free(windowLayers);
    windowLayers = NULL;
    
    printf("Test Successful\n");
    return 0;

error4:
    hakai_layer_mfree(&outLayer);
error3:
    hakai_layer_mfree(&hiddenLayer0);
error2:
    hakai_layer_mfree(&inLayer0);
error1:
    exit(EXIT_FAILURE);
}
