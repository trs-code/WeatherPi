#pragma once

#include "model_destruct.h"

// Assign layer IDs in a topological order to be able to reconstruct the network graph
int assign_layer_ids(layer** myLayer, int currID)
{
    // Post order traversal so the layers can be readily identified before any of their dependencies
    int myID = currID;
    if((*myLayer)->layerID != -1) return currID;

    for (int i = 0; i < (*myLayer)->numPrevLayers; i++)
    {
        if((*myLayer)->prevLayers[i] == myLayer) continue;
        myID = assign_layer_ids((*myLayer)->prevLayers[i], myID);
    }
    
    (*myLayer)->layerID = myID;
    return myID + 1;    
}

// Provides an interface for the user to interact with the model without getting bogged down by little details
model* construct_model( layer*** inLayers, layer** outLayer, int numLayers, int numInLayers, float learning_rate, char loss_fn)
{
    model *myModel = (model*)malloc(sizeof(model));
    if(myModel == NULL) return NULL;

    myModel->inLayers = (layer ***)malloc(numInLayers * sizeof(layer**));
    if(myModel->inLayers == NULL) goto error1;
    
    memcpy(myModel->inLayers, inLayers, sizeof(layer**) * numInLayers);
    
    myModel->outLayer = outLayer;
    myModel->numLayers = numLayers;
    myModel->learning_rate = learning_rate;
    myModel->numInLayers = numInLayers;
    myModel->loss_fn = loss_fn;

    myModel->targets = (float *)calloc((*myModel->outLayer)->numNodes, sizeof(float));
    if(myModel->targets == NULL) goto error2;

    assign_layer_ids(outLayer, 0);

    return myModel;


error2:
    free(myModel->inLayers);
    myModel->inLayers = NULL;
error1:
    free(myModel);
    myModel = NULL;

    return NULL;
}

void traverse_model_fill_layer_list(layer** myLayer, layer*** layerList)
{
    if(layerList[(*myLayer)->layerID] != NULL) return;
    layerList[(*myLayer)->layerID] = myLayer;
    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) traverse_model_fill_layer_list((*myLayer)->prevLayers[i], layerList); // if input layer then the for loop won't even initiate
}
