#pragma once

#include "model_destruct.h"

// Assign layer IDs in a topological order to be able to reconstruct the network graph, also populates the layerList of the model accordingly
int assign_layer_ids(layer** myLayer, int currID, layer*** layerList)
{
    // Post order traversal so the layers can be readily identified before any of their dependencies
    int myID = currID;
    if((*myLayer)->layerID != -1) return currID;

    for (int i = 0; i < (*myLayer)->numPrevLayers; i++)
    {
        if((*myLayer)->prevLayers[i] == myLayer) continue;
        myID = assign_layer_ids((*myLayer)->prevLayers[i], myID, layerList);
    }
    
    (*myLayer)->layerID = myID;
    layerList[myID] = myLayer;
    return myID + 1;    
}

// Provides an interface for the user to interact with the model without getting bogged down by little details
model* construct_model(layer*** inLayers, layer** outLayer, int numLayers, int numInLayers, float learningRate, char loss_fn)
{
    model *myModel = (model*)malloc(sizeof(model));
    if(myModel == NULL) return NULL;

    myModel->inLayers = (layer ***)calloc(numInLayers, sizeof(layer**));
    if(myModel->inLayers == NULL) goto error1;
    
    memcpy(myModel->inLayers, inLayers, sizeof(layer**) * numInLayers);

    myModel->targets = (float *)malloc((*outLayer)->numNodes * sizeof(float));
    if(myModel->targets == NULL) goto error2;

    myModel->layerList = (layer ***)malloc(numLayers * sizeof(layer**));
    if(myModel->layerList == NULL) goto error3;

    myModel->outLayer = outLayer;
    myModel->numLayers = numLayers;
    myModel->learningRate = learningRate;
    myModel->numInLayers = numInLayers;
    myModel->loss_fn = loss_fn;

    assign_layer_ids(outLayer, 0, myModel->layerList);

    return myModel;

error3:
    free(myModel->targets);
    myModel->targets = NULL;
error2:
    free(myModel->inLayers);
    myModel->inLayers = NULL;
error1:
    free(myModel);
    myModel = NULL;

    return NULL;
}

// Provides an interface for the user to interact with the model without getting bogged down by little details, takes in an already established list of layers instead of building
// the topological relations from scratch, such as when the model is being loaded from serialized version and relations are already established
model* construct_model_listed(layer*** inLayers, layer** outLayer, int numLayers, int numInLayers, float learningRate, char loss_fn, layer*** modelLayers)
{
    model* myModel = (model*)malloc(sizeof(model));
    if(myModel == NULL) return NULL;

    myModel->inLayers = (layer ***)malloc(numInLayers * sizeof(layer**));
    if(myModel->inLayers == NULL) goto error1;
    
    memcpy(myModel->inLayers, inLayers, sizeof(layer**) * numInLayers);
    
    myModel->targets = (float *)malloc((*outLayer)->numNodes * sizeof(float));
    if(myModel->targets == NULL) goto error2;

    myModel->layerList = (layer ***)malloc(numLayers * sizeof(layer**));
    if(myModel->layerList == NULL) goto error3;

    myModel->outLayer = outLayer;
    myModel->numLayers = numLayers;
    myModel->learningRate = learningRate;
    myModel->numInLayers = numInLayers;
    myModel->loss_fn = loss_fn;

    // Ownership is retained by the caller, model only has references to pointers to layer struct allocations which is ultimately owned by caller
    for(int i = 0; i < numInLayers; i++) myModel->layerList[i] = &(*modelLayers)[i];

    return myModel;

error3:
    free(myModel->targets);
    myModel->targets = NULL;
error2:
    free(myModel->inLayers);
    myModel->inLayers = NULL;
error1:
    free(myModel);
    myModel = NULL;

    return NULL;
}

// Use to automatically extend the context window of a hidden layer to enable RNN functionality
void extend_context(layer** myLayer, int windowSize, layer*** windowLayers) // reference to array of layer pointers must be provided so user can retain ownership of all created layers 
{    
    layer*** tmp0 = NULL;
    float* tmp1 = NULL;

    *windowLayers = (layer**)calloc((2 * windowSize), sizeof(layer*));
    if(*windowLayers == NULL) return;

    int hiddenNodes = (*myLayer)->numNodes;
    int numInNodes = (*myLayer)->numPrevNodes;
    char hiddenActivationFunction = (*myLayer)->activationFunction;

    // Make last layers first so every successive timestep's hiddenLayer in the window can have the previous timestep's hiddenLayer as its prevLayer[1]
    // Meanwhile the inputs for each timestep for calculating backerrors every sequenceLength timesteps will be prevLayers[0] for each timestep's hiddenLayer
    (*windowLayers)[(2 * windowSize) - 2] = make_input_layer(numInNodes);
    if((*windowLayers)[(2 * windowSize) - 2] == NULL) goto error1;

    (*windowLayers)[(2 * windowSize) - 1] = make_window_layer((layer**[]){&(*windowLayers)[(2 * windowSize) - 2]}, hiddenNodes, 1, hiddenActivationFunction, numInNodes);
    if((*windowLayers)[(2 * windowSize) - 1] == NULL) goto error1;

    for(int i = windowSize - 1; i > 0; i--)
    {
        (*windowLayers)[(2 * i) - 2] = make_input_layer(numInNodes);
        if((*windowLayers)[(2 * i) - 2] == NULL) goto error1;
        (*windowLayers)[(2 * i) - 1] = make_window_layer((layer**[]){&(*windowLayers)[(2 * i) - 2], &(*windowLayers)[(2 * i) + 1]}, hiddenNodes, 2, hiddenActivationFunction, numInNodes + hiddenNodes);
        if((*windowLayers)[(2 * i) - 1] == NULL) goto error1;
    }

    // Need to fix the current timestep's hiddenLayer so it includes the first previous timestep as a prevLayer
    (*myLayer)->numPrevLayers += 1;
    (*myLayer)->numPrevNodes += hiddenNodes;
    tmp0 = (layer ***)realloc((*myLayer)->prevLayers, sizeof(layer**) * (*myLayer)->numPrevLayers);
    if(tmp0 == NULL) goto error1;
    (*myLayer)->prevLayers = tmp0;
    tmp0 = NULL;

    (*myLayer)->prevLayers[(*myLayer)->numPrevLayers - 1] = &(*windowLayers)[1];

    for(int i = 0; i < hiddenNodes; i++)
    {
        tmp1 = (float*)realloc((*myLayer)->weights[i], sizeof(float*) * (*myLayer)->numPrevNodes);
        if(tmp1 == NULL) goto error1;
        (*myLayer)->weights[i] = tmp1;
        tmp1 = NULL;

        for(int j = (*myLayer)->numPrevNodes - hiddenNodes; j < (*myLayer)->numPrevNodes; j++) (*myLayer)->weights[i][j] = ((rand() % 100000) + 50000)/100000;
    }
    
    return;

error1:
    hakai_context_window(windowLayers, windowSize);
}
