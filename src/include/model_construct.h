#pragma once

#include "layer_construct.h"
#include "layer_destruct.h"
#include "model_destruct.h"

void glorot_uniform_init(layer* myLayer) 
{
    float limit = sqrt(6.0 / (myLayer->numPrevNodes + myLayer->numNodes));
    for (int i = 0; i < myLayer->numNodes; ++i) for(int j = 0; j < myLayer->numPrevNodes; j++) myLayer->weights[i][j] = ((float)rand() / RAND_MAX) * 2.0 * limit - limit;
}

void one_weight_init(layer* myLayer) 
{
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

    model *myModel = (model*)malloc(sizeof(model));
    if(myModel == NULL) return NULL;

    myModel->numInLayers = 0;

    myModel->numLayers = assign_layer_ids(outLayer, 0, &(myModel->numInLayers));

    myModel->layerList = (layer***)calloc(myModel->numLayers, sizeof(layer**));
    if(myModel->layerList == NULL) goto error1;

    fill_layer_list(outLayer, myModel->layerList);

    myModel->inLayers = (layer***)calloc(myModel->numInLayers, sizeof(layer**));
    if(myModel->inLayers == NULL) goto error2;

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

        if(currLayer->layerType != 'w') glorot_uniform_init(currLayer);
    }

    myModel->outLayer = outLayer;

    myModel->targets = (float *)malloc((*outLayer)->numNodes * sizeof(float));
    if(myModel->targets == NULL) goto error3;

    myModel->lossDerivatives = (float *)malloc((*outLayer)->numNodes * sizeof(float));
    if(myModel->lossDerivatives == NULL) goto error4;


    myModel->learningRate = learningRate;
    myModel->loss_fn = loss_fn;


    return myModel;

error4:
    free(myModel->targets);
error3:
    free(myModel->inLayers);
error2:
    free(myModel->layerList);
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
    (*windowLayers)[(2 * windowSize) - 2]->layerType = 't';

    (*windowLayers)[(2 * windowSize) - 1] = make_window_layer((layer**[]){&(*windowLayers)[(2 * windowSize) - 2]}, hiddenNodes, 1, hiddenActivationFunction);
    if((*windowLayers)[(2 * windowSize) - 1] == NULL) goto error1;

    (*windowLayers)[(2 * windowSize) - 1]->weights = myLayer->weights;
    (*windowLayers)[(2 * windowSize) - 1]->biases = myLayer->biases;

    for(int i = windowSize - 1; i > 0; i--)
    {
        (*windowLayers)[(2 * i) - 2] = make_input_layer(numInNodes);
        if((*windowLayers)[(2 * i) - 2] == NULL) goto error1;
        (*windowLayers)[(2 * i) - 2]->layerType = 't';

        (*windowLayers)[(2 * i) - 1] = make_window_layer((layer**[]){&(*windowLayers)[(2 * i) - 2], &(*windowLayers)[(2 * i) + 1]}, hiddenNodes, 2, hiddenActivationFunction);
        if((*windowLayers)[(2 * i) - 1] == NULL) goto error1;
        (*windowLayers)[(2 * i) - 1]->weights = myLayer->weights;
        (*windowLayers)[(2 * i) - 1]->biases = myLayer->biases;
    }
    
    return;

error1:
    hakai_context_window(windowLayers, windowSize);
}
