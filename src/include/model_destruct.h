#pragma once

#include "model.h"

// Actual user called function to destroy the entire model which sets up an outArray to keep consistency with the clear model logic, followed by model cleanup, iterative version
void hakai_model(model** thisModel)
{
    model* myModel = *thisModel;

    for(int i = 0; i < myModel->numLayers; i++) hakai_layer(myModel->layerList[i]);

    free(myModel->inLayers);

    free(myModel->layerList);

    free(myModel->targets);

    free(myModel->lossDerivatives);
    free(myModel);
    *thisModel= NULL;
}

void hakai_context_window(layer*** windowLayers, int windowSize)
{
    for(int i = 0; i < windowSize; i++) hakai_layer(&(*windowLayers)[i]);

    free((*windowLayers));
    *windowLayers = NULL;
}