#pragma once

#include "model.h"
#include "layer_ops.h"


// Destroy a model object independently from any layers(IMPORTANT TO DESTROY EVERY SINGLE LAYER MANUALLY IF USED)
void hakai_model_mfree(model** myModel)
{
    free((*myModel)->targets);
    (*myModel)->targets = NULL;

    free(*myModel);
    *myModel= NULL;
}

// Actual user called function to destroy the entire model which sets up an outArray to keep consistency with the clear model logic, followed by model cleanup, iterative version
void hakai_model(model** myModel)
{
    for(int i = 0; i < (*myModel)->numLayers; i++) hakai_layer((*myModel)->layerList[i]);

    free((*myModel)->layerList);
    (*myModel)->layerList = NULL;

    free((*myModel)->inLayers);
    (*myModel)->inLayers = NULL;

    free((*myModel)->targets);
    (*myModel)->targets = NULL;

    free(*myModel);
    *myModel= NULL;
}

void hakai_context_window(layer*** windowLayers, int windowSize)
{
    for(int i = 0; i < 2 * windowSize; i++) if((*windowLayers)[i] != NULL) hakai_layer(&(*windowLayers)[i]);

    free((*windowLayers));
    *windowLayers = NULL;
}