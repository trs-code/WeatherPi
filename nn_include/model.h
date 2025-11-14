#ifndef NN_MODEL
#define NN_MODEL
#include "./layer.h"

// 52 Bytes for an empty model husk
// 16 extra bytes for input layer for the model - numInLayers
// 16 extra bytes for each layer in model - numLayers
struct model
{   
    struct layer **inLayers; // References to the input layers of the model - entry point for model operations
    struct layer **layer_refs; // References for each hung layer while they are getting deconstructed
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

    myModel->layer_refs = (struct layer **)calloc(numInLayers, sizeof(struct layer*));
    if(myModel->layer_refs == NULL)
    {
        goto error4;
    } 

    myModel->numLayers = numLayers;
    myModel->learning_rate = learning_rate;
    myModel->numInLayers = numInLayers;

    return myModel;

error4:
    free(myModel->layer_refs);
    myModel->layer_refs = NULL;
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

//  Gets an output from the target layer
float forward_out(struct layer* layer, struct model* myModel)
{
    float layerOut = 0;
    float x = 0;

    if(myModel->layer_outs[layer->layer_id] != 0)
    {
        return myModel->layer_outs[layer->layer_id];
    }

    if(layer->numPrevLayers == 0)
    {
        x = 1.0;
    }
    else
    {
        for(int i = 0; i < layer->numPrevLayers; i++)
        {
            x += forward_out(layer->prevLayers[i], myModel);
        }
    }

    for(int i = 0; i < layer->numNodes; i++)
    {
        layerOut += layer->currLayerWeights[i] * x;
    }

    layerOut += layer->currLayerWeights[layer->numNodes];

    switch(layer->activation)
    {
        case 'r':
            layerOut = relu(layerOut);
            break;
        case 't':
            layerOut = tanh(layerOut);
            break;
        case 'i':
            break;
        default:
            return layerOut;
    }

    myModel->layer_outs[layer->layer_id] = layerOut;
    return layerOut;
}

// For clearing the outputs once a forward pass is done
void clear_layer_outs(struct model* myModel)
{
    for(int i = 0; i < (myModel->numLayers); i++)
    {
        myModel->layer_outs[i] = 0;
    }
}

// Destroy an individual layer after operations are concluded
void hakai_layer(struct layer* lay, struct model* myModel)
{
    int i = 0;
    
    if(myModel->layer_ids[lay->layer_id] == 0)
    {
        return;
    }

    free(lay->currLayerGradients);
    lay->currLayerGradients = NULL;

    free(lay->currLayerWeights);
    lay->currLayerWeights = NULL;

    free(lay->nextLayers);
    lay->nextLayers = NULL;
    
    //free(lay->prevLayers);
    lay->prevLayers = NULL;

    myModel->layer_ids[lay->layer_id] = 0;

    while(myModel->layer_refs[i] != NULL)
    {
        i += 1;
    }

    myModel->layer_refs[i] = lay;  
}

// Enter this function with the outArray of the model and let it do its thing
// One big advantage of the doubly linked list structure is being able to exploit the convergence of the model on the input layer
void clear_model(struct layer** layerArr, struct model* myModel)
{
    if(layerArr == NULL)
    {
        return;
    }
    for(int i = 0; i < ((sizeof(layerArr))/(sizeof(struct layer*))); i++)
    {
        if(myModel->layer_ids[layerArr[i]->layer_id] == 0)
        {
            continue;
        }
        clear_model(layerArr[i]->prevLayers, myModel);
        hakai_layer(layerArr[i], myModel);
        myModel->layer_ids[layerArr[i]->layer_id] = 0;
        layerArr[i] = NULL;
    }
    free(layerArr);
    layerArr = NULL;
}

void hakai_model(struct model* myModel)
{
    struct layer **outArr = (struct layer**)malloc(sizeof(struct layer*));
    if(outArr == NULL)
    {
        return;
    }

    outArr[0] = myModel->outLayer;
    clear_model(outArr, myModel);
    
    free(myModel->layer_outs);
    myModel->layer_outs = NULL;

    free(myModel->layer_ids);
    myModel->layer_ids = NULL;

    for(int i = myModel->numLayers - 1; i >= 0; i--)
    {
        free(myModel->layer_refs[i]);
        myModel->layer_refs[i] = NULL;
    }

    free(myModel->layer_refs);
    myModel->layer_refs = NULL;

    free(myModel);
    myModel = NULL;
}

void save_model(struct model* saveMod);

struct model* load_model(char* filename[]);


#endif