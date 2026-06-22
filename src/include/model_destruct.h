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

// Enter this function with the outArray of the model and let it do its thing
// One big advantage of the linked list tree structure is being able to exploit the convergence of the model on the output layer
void clear_model(layer*** layerArr, int layerNums)
{
    if(layerArr == NULL) return;

    for(int i = 0; i < layerNums; i++)
    {
        if(*layerArr[i] == NULL) continue;
        
        // Check for cases of RNNs where a layer can have itself as a prevLayer but we want to call it at least once
        if((*layerArr[i])->switchVar == '4') break; 
        (*layerArr[i])->switchVar = '4';
                
        clear_model((*layerArr[i])->prevLayers, (*layerArr[i])->numPrevLayers);
        hakai_layer(layerArr[i]);
        
        *layerArr[i] = NULL;
    }
    
    free(layerArr);
    layerArr = NULL;
}

// Actual user called function to destroy the entire model which sets up an outArray to keep consistency with the clear model logic, followed by model cleanup, recursive version
void hakai_model_recur(model** myModel)
{
    layer ***outArr = (layer***)malloc(sizeof(layer**));
    if(outArr == NULL) return;

    outArr[0] = (*myModel)->outLayer;
    
    clear_model(outArr, 1);

    free((*myModel)->inLayers);
    (*myModel)->inLayers = NULL;

    free((*myModel)->targets);
    (*myModel)->targets = NULL;

    free(*myModel);
    *myModel= NULL;
}

// Actual user called function to destroy the entire model which sets up an outArray to keep consistency with the clear model logic, followed by model cleanup, iterative version
void hakai_model(model** myModel)
{
    for(int i = 0; i < (*myModel)->numLayers; i++) hakai_layer_mfree((*myModel)->layerList[i]);

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
    for(int i = 0; i < 2 * windowSize; i++) if((*windowLayers)[i] != NULL) hakai_layer_mfree(&(*windowLayers)[i]);

    free((*windowLayers));
    *windowLayers = NULL;
}