#pragma once

#include "layer.h"
#include <stdlib.h>

void hakai_matrix(float*** mat, int rows)
{
    if(!(*mat)) return;
    
    for(int i = 0; i < rows; i++)
    {
        free((*mat)[i]);
        (*mat)[i] = NULL;
    }

    free(*mat);
    *mat = NULL;
}

// Destroy an individual layer after operations are concluded
void hakai_layer(layer** myLayer)
{    
    layer* currLayer = (*myLayer);
    
    free(currLayer->outputs);
    currLayer->outputs = NULL;

    if(currLayer->activationFunction != 'i')
    {
        free(currLayer->backErrors);
        currLayer->backErrors = NULL;

        free(currLayer->prevLayers);
        currLayer->prevLayers = NULL;

        free(currLayer->preActivations);
        currLayer->preActivations = NULL;

        free(currLayer->biases);
        currLayer->biases = NULL;

        hakai_matrix(&currLayer->weights, currLayer->numNodes);
    }

    free(*myLayer);
    *myLayer = NULL;    
}