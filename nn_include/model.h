#ifndef NN_MODEL
#define NN_MODEL
#include "./layer.h"

// 44 Bytes for an empty model husk
// 8 extra bytes for input layer for the model - numInLayers
// 8 extra bytes for each layer in model - numLayers
struct model
{   
    struct layer **inLayers; // References to the input layers of the model - entry point for model operations
    struct layer *outLayer; // References the output layers of the model - entry point for model operations
    int *layer_ids; // Checks for the presence of a layer in the model - layer_id[id # of the layer] = 0 if absent 1 if constructed
    float *layer_outs; // Reference for output values of models - helps do DP forward pass on DAG graph
    float learning_rate; // Learning rate for the NN
    int numLayers; // Number of total layers in the NN
    int numInLayers; // Number of input layers in the NN
};

struct model* construct_model(int numLayers, int numInLayers, float learning_rate)
{
    struct model *myModel = (struct model*)malloc(sizeof(struct model));
    if(myModel == NULL)
    {
        return NULL;
    }

    myModel->layer_ids = (int *)calloc(numLayers, sizeof(int));
    if(myModel->layer_ids == NULL)
    {
        goto error1;
    }

    myModel->layer_outs = (float *)calloc(numLayers, sizeof(float));
    if(myModel->layer_ids == NULL)
    {
        goto error2;
    }

    myModel->inLayers = (struct layer **)calloc(numInLayers, sizeof(struct layer*));
    if(myModel->inLayers == NULL)
    {
        goto error3;
    }

    myModel->numLayers = numLayers;
    myModel->learning_rate = learning_rate;
    myModel->numInLayers = numInLayers;

    return myModel;

error3:
    free(myModel->layer_outs);
    myModel->layer_outs = NULL;
error2:
    free(myModel->layer_ids);
    myModel->layer_outs = NULL;
error1:
    free(myModel);
    myModel = NULL;

    return NULL;
}

// Works with multiple layers to get an output
float forward_out(struct layer* layer, struct model* myModel)
{
    // Start from out array and work backwards - DP Solution O(Number of neurons amongst all layers) time complexity
    float output = 0;

    if(myModel->layer_outs[layer->layer_id] != 0)
    {
        return myModel->layer_outs[layer->layer_id];
    }
    
    if(layer->numPrevLayers == 0)
    {
        return layer_forward(layer, 1.0f);
    }

    for(int i = 0; i < layer->numPrevLayers; i++)
    {
        output += layer_forward(layer->prevLayers[i], forward_out(layer->prevLayers[i], myModel));
    }

    myModel->layer_outs[layer->layer_id] = output;
    return output;
}

float* backward_pass(struct layer* layer, float expected, float prediction, float learning_rate)
{
    float diff = expected-prediction;
    float loss = 0.5*diff*diff;

    //finish


}

// For clearing the outputs once a forward pass is done
void clear_layer_outs(struct model* myModel)
{
    for(int i = 0; i < (myModel->numLayers); i++)
    {
        myModel->layer_outs[i] = 0.0f;
    }
}

// Enter this function with the outArray of the model and let it do its thing
// One big advantage of the doubly linked list structure is being able to exploit the convergence of the model on the input layer
void clear_model(struct layer** layerArr)
{
    if(layerArr == NULL)
    {
        return;
    }
    for(int i = 1; i < ((sizeof(layerArr))/(sizeof(struct layer*))); i++)
    {
        if(layerArr[i] == NULL)
        {
            return;
        }
        clear_model(layerArr[i]->prevLayers);
        if(i > 0) hakai_layer(layerArr[i]);
        layerArr[i] = NULL;
    }
    free(layerArr);
    layerArr = NULL;
}

void hakai_model(struct model* myModel)
{
    struct layer **outArr = (struct layer**)malloc(sizeof(struct layer*));
    outArr[0] = myModel->outLayer;
    clear_model(outArr);
    
    free(myModel->layer_outs);
    myModel->layer_outs = NULL;

    free(myModel->layer_ids);
    myModel->layer_ids = NULL;

    free(myModel);
    myModel = NULL;
}

void save_model(struct model* saveMod);

struct model* load_model(char* filename[]);


#endif