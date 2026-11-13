#pragma once

#include "layer.h"
#include <stdlib.h>

void hakai_matrix(float*** mat, int rows)
{
    if(!mat) return;
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

    free(currLayer->backErrors);

    free(currLayer->preActivations);

    free(currLayer->activationDerivatives);

    free(currLayer->biases);

    free(currLayer->prevLayers);
    
    if(currLayer->weights != NULL) free(currLayer->weights[0]);

    free(currLayer->weights);

    free(*myLayer);
    *myLayer = NULL;    
}