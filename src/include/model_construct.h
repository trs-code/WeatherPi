#pragma once

#include "layer_construct.h"
#include "layer_destruct.h"
#include "model_destruct.h"

static inline void glorot_uniform_init(layer* myLayer) 
{
    float limit = sqrt(6.0 / (myLayer->numPrevNodes + myLayer->numNodes));
    for (int i = 0; i < myLayer->numNodes; ++i) for(int j = 0; j < myLayer->numPrevNodes; j++) myLayer->weights[i][j] = ((float)rand() / RAND_MAX) * 2.0 * limit - limit;
}

// Assign layer IDs in a topological order to be able to reconstruct the network graph, also populates the layerList of the model accordingly
int assign_layer_ids(layer** currLayer, int currID, layer*** layerList)
{
    // Post order traversal so the layers can be readily identified before any of their dependencies
    layer** prevLayer;
    layer* myLayer = *currLayer;
    int myID = currID;
    if(myLayer->layerID != -1) return currID;
    myLayer->layerID = 0;

    for (int i = 0; i < myLayer->numPrevLayers; i++)
    {
        prevLayer = myLayer->prevLayers[i];
        if(prevLayer == currLayer) continue;
        myID = assign_layer_ids(prevLayer, myID, layerList);
    }
    
    myLayer->layerID = myID;
    layerList[myID] = currLayer;
    return myID + 1;    
}

// Provides an interface for the user to interact with the model without getting bogged down by little details
model* construct_model(layer*** inLayers, layer** outLayer, int numLayers, int numInLayers, float learningRate, char loss_fn)
{
    layer* currLayer;

    srand(time(NULL));
    model *myModel = (model*)malloc(sizeof(model));
    if(myModel == NULL) return NULL;

    myModel->inLayers = (layer ***)calloc(numInLayers, sizeof(layer**));
    if(myModel->inLayers == NULL) goto error1;

    myModel->layerList = (layer ***)malloc(numLayers * sizeof(layer**));
    if(myModel->layerList == NULL) goto error2;

    myModel->targets = (float *)malloc((*outLayer)->numNodes * sizeof(float));
    if(myModel->targets == NULL) goto error3;

    myModel->lossDerivatives = (float *)malloc((*outLayer)->numNodes * sizeof(float));
    if(myModel->lossDerivatives == NULL) goto error4;

    memcpy(myModel->inLayers, inLayers, sizeof(layer**) * numInLayers);
    myModel->outLayer = outLayer;
    myModel->numLayers = numLayers;
    myModel->learningRate = learningRate;
    myModel->numInLayers = numInLayers;
    myModel->loss_fn = loss_fn;

    assign_layer_ids(outLayer, 0, myModel->layerList);

    for(int i = 0; i < numLayers; i++)
    {
        currLayer = (*myModel->layerList[i]);
        if(currLayer->layerType != 'i' && currLayer->layerType != 'w') glorot_uniform_init(currLayer);
    }
    return myModel;

error4:
    free(myModel->targets);
error3:
    free(myModel->layerList);
error2:
    free(myModel->inLayers);
error1:
    free(myModel);

    return NULL;
}

// Use to automatically extend the context window of a hidden layer to enable RNN functionality
void extend_context(layer* myLayer, int windowSize, layer*** windowLayers) // reference to array of layer pointers must be provided so user can retain ownership of all created layers 
{    
    int hiddenNodes = myLayer->numNodes;
    int numInNodes = myLayer->numPrevNodes;
    char hiddenActivationFunction = myLayer->activationFunction;
    *windowLayers = NULL;

    if(windowSize < 1) return;

    myLayer->numPrevLayers += 1;
    myLayer->numPrevNodes += hiddenNodes;
    for(int i = 0; i < hiddenNodes; i++) free(myLayer->weights[i]);
    
    myLayer->prevLayers = (layer ***)realloc(myLayer->prevLayers, sizeof(layer**) * myLayer->numPrevLayers);
    if(myLayer->prevLayers == NULL) goto error1;

    for(int i = 0; i < hiddenNodes; i++)
    {
        myLayer->weights[i] = (float*)malloc(myLayer->numPrevNodes * sizeof(float));
        if(myLayer->weights[i] == NULL) goto error1;
    }

    glorot_uniform_init(myLayer);

    *windowLayers = (layer**)calloc((2 * windowSize), sizeof(layer*));
    if(*windowLayers == NULL) return;

    myLayer->prevLayers[myLayer->numPrevLayers - 1] = &(*windowLayers)[1];

    // Make last layers first so every successive timestep's hiddenLayer in the window can have the previous timestep's hiddenLayer as its prevLayer[1]
    // Meanwhile the inputs for each timestep for calculating backerrors every sequenceLength timesteps will be prevLayers[0] for each timestep's hiddenLayer
    (*windowLayers)[(2 * windowSize) - 2] = make_input_layer(numInNodes);
    if((*windowLayers)[(2 * windowSize) - 2] == NULL) goto error1;

    (*windowLayers)[(2 * windowSize) - 1] = make_window_layer((layer**[]){&(*windowLayers)[(2 * windowSize) - 2]}, hiddenNodes, 1, hiddenActivationFunction, numInNodes);
    if((*windowLayers)[(2 * windowSize) - 1] == NULL) goto error1;

    (*windowLayers)[(2 * windowSize) - 1]->weights = myLayer->weights;
    (*windowLayers)[(2 * windowSize) - 1]->biases = myLayer->biases;

    for(int i = windowSize - 1; i > 0; i--)
    {
        (*windowLayers)[(2 * i) - 2] = make_input_layer(numInNodes);
        if((*windowLayers)[(2 * i) - 2] == NULL) goto error1;
        (*windowLayers)[(2 * i) - 1] = make_window_layer((layer**[]){&(*windowLayers)[(2 * i) - 2], &(*windowLayers)[(2 * i) + 1]}, hiddenNodes, 2, hiddenActivationFunction, numInNodes + hiddenNodes);
        if((*windowLayers)[(2 * i) - 1] == NULL) goto error1;
        (*windowLayers)[(2 * i) - 1]->weights = myLayer->weights;
        (*windowLayers)[(2 * i) - 1]->biases = myLayer->biases;
    }
    
    return;

error1:
    hakai_context_window(windowLayers, windowSize);
}
