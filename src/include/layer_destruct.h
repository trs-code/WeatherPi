#pragma once

#include "layer.h"
#include <stdlib.h>

void hakai_matrix(float*** mat, int rows)
{
    for(int i = 0; i < rows; i++)
    {
        free((*mat)[i]);
        (*mat)[i] = NULL;
    }

    free(*mat);
    *mat = NULL;
}

// Destroy an individual layer after operations are concluded but a model hasn't been built yet
void hakai_layer_mfree(layer** myLayer)
{    
    free((*myLayer)->outputs);
    (*myLayer)->outputs = NULL;

    if((*myLayer)->activationFunction != 'i')
    {
        free((*myLayer)->backErrors);
        (*myLayer)->backErrors = NULL;

        free((*myLayer)->prevLayers);
        (*myLayer)->prevLayers = NULL;

        free((*myLayer)->preActivations);
        (*myLayer)->preActivations = NULL;

        free((*myLayer)->biases);
        (*myLayer)->biases = NULL;

        hakai_matrix(&(*myLayer)->weights, (*myLayer)->numNodes);
    }

    free(*myLayer);
    *myLayer = NULL;    
}

// Destroy an individual layer in the process of clearing the model
void hakai_layer(layer** myLayer)
{
    if(*myLayer == NULL) return;

    free((*myLayer)->outputs);
    (*myLayer)->outputs = NULL;

    if((*myLayer)->activationFunction != 'i')
    {
        hakai_matrix(&(*myLayer)->weights, (*myLayer)->numNodes);

        free((*myLayer)->backErrors);
        (*myLayer)->backErrors = NULL;

        free((*myLayer)->preActivations);
        (*myLayer)->preActivations = NULL;

        free((*myLayer)->biases);
        (*myLayer)->biases = NULL;
    }
    
    free(*myLayer);
    *myLayer = NULL;
}