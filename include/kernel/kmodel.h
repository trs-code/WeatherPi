#ifndef NN_MODEL
#define NN_MODEL
#include <klayer.h>

// 44 Bytes for an empty model husk
// 8 extra bytes for input layer for the model - numInLayers
// 8 extra bytes for each layer in model - numLayers
struct model
{
    int *layer_ids; // Checks for the presence of a layer in the model - layer_id[id # of the layer] = 0 if absent 1 if constructed
    float *layer_outs; // Reference for output values of models - helps do DP forward pass on DAG graph
    layer **inLayers; // References to the input layers of the model - entry point for model operations
    layer *outLayer; // References the output layers of the model - entry point for model operations
    float learning_rate; // Learning rate for the NN
    int numLayers; // Number of total layers in the NN
    int numInLayers; // Number of input layers in the NN
};

model* construct_model(int numLayers, int numInLayers, float learning_rate)
{
    struct model *myModel = (struct model*)kmalloc(sizeof(struct model));
    if(myModel == NULL)
    {
        return NULL;
    }

    myModel->layer_ids = (int *)kcalloc(numLayers, sizeof(int));
    if(myModel->layer_ids == NULL)
    {
        goto error1;
    }

    myModel->layer_outs = (float *)kcalloc(numLayers, sizeof(float));
    if(myModel->layer_ids == NULL)
    {
        goto error2;
    }

    myModel->inLayers = (layer **)kcalloc(myModel->numInLayers, sizeof(layer*) * numInLayers);
    if(myModel->inLayers == NULL)
    {
        goto error3;
    }

    myModel->numLayers = numLayers;
    myModel->learning_rate = learning_rate;
    myModel->numInLayers = numInLayers;

error3:
    kfree(myModel->layer_outs);
    myModel->layer_outs = NULL;
error2:
    kfree(myModel->layer_ids);
    myModel->layer_outs = NULL;
error1:
    kfree(myModel);
    myModel = NULL;

    return NULL;
}

// Works with multiple layers to get an output
float forward_out(layer* layer, model* myModel)
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

void backward_pass(layer, float expected, float prediction, float learning_rate)
{
    float diff = expected-prediction;
    float loss = 0.5*diff*diff;


}

// Enter this function with the outArray of the model and let it do its thing
// One big advantage of the doubly linked list structure is being able to exploit the convergence of the model on the input layer
void clear_model(layer** layerArr)
{
    for(int i = 0; i < ((sizeof(layerArr))/(sizeof(layer*))); i++)
    {
        if(layerArr[i] == NULL)
        {
            return;
        }
        clear_model(layerArr[i]->prevLayers);
        hakai_layer(layerArr[i]);
        layerArr[i] = NULL;
    }
}

void clear_layer_outs(model* myModel)
{
    for(int i = 0; i < (myModel->numLayers); i++)
    {
        myModel->layer_outs[i] = 0.0f;
    }
}

void hakai_model(model* myModel)
{
    layer **outArr = (layer**)kmalloc(sizeof(layer*));
    outArr[0] = myModel->outLayer;
    clear_model(outArr);

    kfree(outArr);
    outArr = NULL;
    hakai_layer(myModel->outLayer);
    
    kfree(myModel->layer_outs);
    myModel->layer_outs = NULL;

    kfree(myModel->layer_ids);
    myModel->layer_ids = NULL;

    kfree(myModel);
    myModel = NULL;
}

void save_model(model* saveMod);

model* load_model(char* filename[]);


#endif