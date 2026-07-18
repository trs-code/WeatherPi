#pragma once

#include "layer.h"
#include <stdlib.h>

void hakai_matrix(float*** mat, int rows)
{
    if(!(*mat)) return;
    
    for(int i = 0; i < rows; i++) free((*mat)[i]);

    free(*mat);
    *mat = NULL;
}

// Destroy an individual layer after operations are concluded
void hakai_layer(layer** myLayer)
{    
    layer* currLayer = (*myLayer);
    if(!currLayer) return;
    
    free(currLayer->outputs);

    if(currLayer->activationFunction != 'i')
    {
        free(currLayer->backErrors);

        free(currLayer->prevLayers);

        free(currLayer->preActivations);

        free(currLayer->activationDerivatives);
    }

    if(currLayer->layerType != 'w') 
    {
        free(currLayer->biases);
        hakai_matrix(&currLayer->weights, currLayer->numNodes);
    }

    free(*myLayer);
    *myLayer = NULL;    
}