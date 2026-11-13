#pragma once

#include "layer_construct.h"
#include "layer_destruct.h"
#include "model_destruct.h"

//Standard weight initialization technique
void glorot_uniform_weight_init(layer* myLayer) 
{
    if(!myLayer->weights) return;
    float limit = sqrt(6.0 / (myLayer->numPrevNodes + myLayer->numNodes));
    for (int i = 0; i < myLayer->numNodes; ++i) for(int j = 0; j < myLayer->numPrevNodes; j++) myLayer->weights[i][j] = ((float)rand() / RAND_MAX) * 2.0 * limit - limit;
}

// Inititialize all weights in layer to 1.0 - typically used for testing purposes
void one_weight_init(layer* myLayer) 
{
    if(!myLayer->weights) return;
    for (int i = 0; i < myLayer->numNodes; ++i) for(int j = 0; j < myLayer->numPrevNodes; j++) myLayer->weights[i][j] = 1.0f;
}

// Assign layer IDs in a topological order to be able to reconstruct the network graph, also returns the number of layers encountered by the model after successful recursion run
int assign_layer_ids(layer** currLayer, int currID, int* numIns)
{
    // Post order traversal so the layers can be readily identified before any of their dependencies
    layer** prevLayer;
    layer* myLayer = *currLayer;
    int myID = currID;

    for (int i = 0; i < myLayer->numPrevLayers; i++)
    {
        prevLayer = myLayer->prevLayers[i];
        if((*prevLayer)->layerID > -1) continue;
        myID = assign_layer_ids(prevLayer, myID, numIns);
    }
    
    if(myLayer->layerType == 'i') *numIns += 1;
    myLayer->layerID = myID;
    return myID + 1;    
}

// Do a DFS traversal of the layer graph with assigned layer ID to get a sorted list
void fill_layer_list(layer** currLayer, layer*** layerList)
{
    layer** prevLayer;
    layer* myLayer = *currLayer;
    
    layerList[myLayer->layerID] = currLayer;

    for (int i = 0; i < myLayer->numPrevLayers; i++)
    {
        prevLayer = myLayer->prevLayers[i];
        if(layerList[(*prevLayer)->layerID] != NULL) continue;
        fill_layer_list(prevLayer, layerList);
    }
    
    return;
}

// Provides an interface for the user to interact with the model without getting bogged down by little details
model* construct_model(layer** outLayer, float learningRate, char loss_fn)
{
    layer* currLayer;
    int inLayerTracker = 0;

    model *myModel = (model*)calloc(1, sizeof(model));
    if(myModel == NULL) return NULL;

    myModel->numInLayers = 0;

    myModel->numLayers = assign_layer_ids(outLayer, 0, &(myModel->numInLayers));
    if(!(myModel->numLayers && myModel->numInLayers)) goto error;

    myModel->layerList = (layer***)calloc(myModel->numLayers, sizeof(layer**));
    if(myModel->layerList == NULL) goto error;

    fill_layer_list(outLayer, myModel->layerList);

    myModel->inLayers = (layer***)malloc(myModel->numInLayers * sizeof(layer**));
    if(myModel->inLayers == NULL) goto error;

    srand(time(NULL));

    for(int i = 0; i < myModel->numLayers; i++)
    {
        currLayer = (*myModel->layerList[i]);
        
        if(currLayer->layerType == 'i')
        {
            myModel->inLayers[inLayerTracker] = myModel->layerList[i];
            inLayerTracker++;
            continue;
        }

        if(currLayer->layerType == 'w' || currLayer->layerType == 'a') continue;
        
        glorot_uniform_weight_init(currLayer);
    }

    myModel->outLayer = outLayer;

    myModel->targets = (float *)calloc((*outLayer)->numNodes, sizeof(float));
    if(myModel->targets == NULL) goto error;

    myModel->lossDerivatives = (float *)malloc((*outLayer)->numNodes * sizeof(float));
    if(myModel->lossDerivatives == NULL) goto error;

    myModel->learningRate = learningRate;
    myModel->loss_fn = loss_fn;

    // Ensure that probabalistic loss functions have pobability producing activation functions for the output layer
    switch(loss_fn)
    {
        case 'c':
            (*myModel->outLayer)->activationFunction = 'x';
            break;
        case 'n':
            (*myModel->outLayer)->activationFunction = 's';
            break;
        case 'r':
            (*myModel->outLayer)->activationFunction = 'g';
            break;
        case 'x':
            (*myModel->outLayer)->activationFunction = 'f';
    }

    // If the output layer is producing a probability function then we are going to use probabalistic loss at that point
    if((*myModel->outLayer)->activationFunction == 'x') loss_fn = 'c';
    if((*myModel->outLayer)->activationFunction == 'f') loss_fn = 'x';

    return myModel;

error:
    hakai_model(&myModel);

    return NULL;
}

// Use to automatically extend the context window of a hidden layer to enable RNN functionality
void extend_context(layer* myLayer, int windowSize, layer*** windowLayers) // reference to array of layer pointers must be provided so user can retain ownership of all created layers 
{    
    int hiddenNodes = myLayer->numNodes;
    int numPrevNodes = myLayer->numPrevNodes + hiddenNodes;
    char hiddenActivationFunction = myLayer->activationFunction;
    *windowLayers = NULL;

    if(windowSize < 1) return;

    myLayer->numPrevLayers += 1;
    myLayer->numPrevNodes = numPrevNodes;
    free(myLayer->weights[0]);
    
    myLayer->prevLayers = (layer ***)realloc(myLayer->prevLayers, sizeof(layer**) * myLayer->numPrevLayers);
    if(myLayer->prevLayers == NULL) goto error;

    myLayer->weights[0] = (float*)malloc(hiddenNodes * numPrevNodes * sizeof(float));
    if(myLayer->weights[0] == NULL) goto error;

    for(int i = 0; i < hiddenNodes; i++) myLayer->weights[i] = myLayer->weights[0] + (i * numPrevNodes);

    *windowLayers = (layer**)calloc(windowSize, sizeof(layer*)); // Might be able to make it 2 if only most second most second recent timestep input values are needed
    if(*windowLayers == NULL) return;

    myLayer->prevLayers[myLayer->numPrevLayers - 1] = &(*windowLayers)[0];

    (*windowLayers)[windowSize - 1] = make_window_layer((layer**[]){}, hiddenNodes, numPrevNodes, 0, hiddenActivationFunction);
    if((*windowLayers)[windowSize - 1] == NULL) goto error;

    for(int i = windowSize - 2; i > -1; i--)
    {
        (*windowLayers)[i] = make_window_layer((layer**[]){&(*windowLayers)[i + 1]}, hiddenNodes, numPrevNodes, 1, hiddenActivationFunction);
        if((*windowLayers)[i] == NULL) goto error;
    }
    
    return;

error:
    hakai_context_window(windowLayers, windowSize);
}
