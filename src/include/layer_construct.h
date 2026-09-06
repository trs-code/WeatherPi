#pragma once

#include "layer_destruct.h"
#include <string.h>
#include <time.h>
#include <math.h>

// Solely to load input values into the model in a form where layer operations can be generalized into
layer* make_input_layer(int numNodes)
{
    // Allocate space for the input layer
    layer* inLayer = (layer*)malloc(sizeof(layer));
    if(inLayer == NULL) return NULL;

    inLayer->numPrevLayers = 0;
    inLayer->numPrevNodes = 0;
    inLayer->layerType = 'i';
    inLayer->prevLayers = NULL; // No previous layers for an input layer
    inLayer->weights = NULL;    // Input layer just accepts inputs, doesn't need actual weights, just something to facilitate forwarding values
    inLayer->biases = NULL;
    inLayer->backErrors = NULL;  // Input layer doesn't need backErrors
    inLayer->preActivations = NULL;
    inLayer->activationDerivatives = NULL;
    
    inLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(inLayer->outputs == NULL) goto error1;

    inLayer->numNodes = numNodes;
    inLayer->activationFunction = 'i';
    inLayer->layerID = -1;

    return inLayer;

error1:
    free(inLayer);
    inLayer = NULL;

    return NULL;
}

layer* make_hidden_layer(layer*** prev, int numNodes, int numPrevLayers, char activation_function)
{
    // Allocate space for the layer
    layer* hiddenLayer = (layer *)malloc(sizeof(layer));
    if(hiddenLayer == NULL) return NULL;

    // Set number of previous layers that feed into this layer and number of next layers that this layer feeds into
    // These make it easier to implement models with more complex structures than traditional NNs which would set these at 1
    // Also helps with backward passes
    hiddenLayer->numPrevLayers = numPrevLayers;
    hiddenLayer->numPrevNodes = 0;

    hiddenLayer->biases = (float *)calloc(numNodes, sizeof(float)); // Bias for each neuron
    if(hiddenLayer->biases == NULL) goto error1;
    
    hiddenLayer->backErrors = (float *)calloc((numNodes), sizeof(float));
    if(hiddenLayer->backErrors == NULL) goto error2;

    hiddenLayer->outputs = (float *)malloc(numNodes* sizeof(float));
    if(hiddenLayer->outputs == NULL) goto error3;
    
    hiddenLayer->preActivations = (float *)malloc(numNodes* sizeof(float));
    if(hiddenLayer->preActivations == NULL) goto error4;

    hiddenLayer->activationDerivatives = (float *)calloc(numNodes, sizeof(float));
    if(hiddenLayer->activationDerivatives == NULL) goto error5;

    hiddenLayer->weights = (float **)calloc(numNodes, sizeof(float*)); // Initially zeroed out so it can be freed easier later if error occurs when constructing the matrix dynamically
    if(hiddenLayer->weights == NULL) goto error6;

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING CAREFULLY
    hiddenLayer->prevLayers = (layer ***)malloc(sizeof(layer**) * numPrevLayers);
    if(hiddenLayer->prevLayers == NULL) goto error7;

    // Set the previous layers as the previous layers
    for(int i = 0; i < numPrevLayers; i++)
    {
        hiddenLayer->prevLayers[i] = prev[i]; // Could just use memcpy but this approach induces better cache hits
        hiddenLayer->numPrevNodes += (*prev[i])->numNodes;
    }

    for(int i = 0; i < numNodes; i++)
    {
        hiddenLayer->weights[i] = (float *)malloc(sizeof(float) * (hiddenLayer->numPrevNodes)); // Each column is a connection to each neuron in the previous layer
        if(hiddenLayer->weights[i] == NULL) goto error8;
    }
    
    hiddenLayer->numNodes = numNodes;
    hiddenLayer->activationFunction = activation_function;
    hiddenLayer->layerID = -1;
    hiddenLayer->layerType = 'h';

    return hiddenLayer;

error8:
    free(hiddenLayer->prevLayers);
error7:
    hakai_matrix(&(hiddenLayer->weights), numNodes);
error6:
    free(hiddenLayer->activationDerivatives);
error5:
    free(hiddenLayer->preActivations);
error4:
    free(hiddenLayer->outputs);
error3:
    free(hiddenLayer->backErrors);
error2:
    free(hiddenLayer->biases);
error1:
    free(hiddenLayer);
    hiddenLayer = NULL;

    return NULL;
}

layer* make_output_layer(layer*** prev, int numNodes, int numPrevLayers, char activation_function)
{
    layer* outLayer = make_hidden_layer(prev, numNodes, numPrevLayers, activation_function);
    if(outLayer == NULL) return NULL;
    
    outLayer->layerType = 'o';

    return outLayer;
}

layer* make_window_layer(layer*** prev, int numNodes, int numPrevLayers, char activationFunction, int numPrevNodes)
{
    // Allocate space for the input layer
    layer *windowLayer = (layer*)malloc(sizeof(layer));
    if(windowLayer == NULL) return NULL;

    windowLayer->numPrevLayers = numPrevLayers;
    windowLayer->numPrevNodes = numPrevNodes;
    windowLayer->layerType = 'w';
    windowLayer->weights = NULL; // Weights and biases are shared with base timestep layer
    windowLayer->biases = NULL;

    windowLayer->backErrors = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->backErrors == NULL) goto error1;

    windowLayer->preActivations = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->preActivations == NULL) goto error2;
    
    windowLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->outputs == NULL) goto error3;

    windowLayer->activationDerivatives = (float *)malloc(numNodes * sizeof(float));
    if(windowLayer->activationDerivatives == NULL) goto error4;

    windowLayer->prevLayers = (layer ***)calloc(numPrevLayers, sizeof(layer**)); // Possibly one prev layer, none if last in window
    if(windowLayer->prevLayers == NULL) goto error5;

    windowLayer->numNodes = numNodes;
    windowLayer->activationFunction = activationFunction;
    windowLayer->layerID = -1;

    for(int i = 0; i < numPrevLayers; i++) windowLayer->prevLayers[i] = prev[i];

    return windowLayer;

error5:
    free(windowLayer->activationDerivatives);
error4:
    free(windowLayer->outputs);
error3:
    free(windowLayer->preActivations);
error2:
    free(windowLayer->backErrors);
error1:
    free(windowLayer);
    windowLayer = NULL;

    return NULL;
}

// Not Yet Implemented 
layer** make_convolutional_layers(layer*** prevLayers, int numPrevLayers, int numFilters, int numDims, int* dims, int hasPadding);

// Not yet implemented
layer* make_attention_layer(layer*** prevLayers, int numNodes, int numPrevLayers, int windowSize, char activation_function)
{
    layer* attnLayer;
    layer*** prevWindows;
    layer** prevLayer;
    int winTrav = 0;
    int numPrevs = numPrevLayers;
    char act = (activation_function  == 'x' ? 'x' : 'f');

    for(int i = 0; i < numPrevLayers; i++)
    {
        prevLayer = prevLayers[i];
        if((*prevLayer)->layerType != 'h') return NULL;
        while(winTrav < windowSize && (*(*prevLayer)->prevLayers[(*prevLayer)->numPrevLayers - 1])->layerType == 'w')
        {
            prevLayer = (*prevLayer)->prevLayers[(*prevLayer)->numPrevLayers - 1];
            winTrav++;
            numPrevs++;
        }
        winTrav = 0;
    }

    prevWindows = malloc(numPrevs * sizeof(layer**));
    if(prevWindows == NULL) goto error1;

    for(int i = 0; i < numPrevLayers; i++)
    {
        prevLayer = prevLayers[i];
        prevWindows[i] = prevLayer;
        while(winTrav < windowSize && (*(*prevLayer)->prevLayers[(*prevLayer)->numPrevLayers - 1])->layerType == 'w')
        {
            prevLayer = (*prevLayer)->prevLayers[(*prevLayer)->numPrevLayers - 1];
            winTrav++;
            prevWindows[i + winTrav] = prevLayer;
        }
        winTrav = 0;
    }

    attnLayer = make_hidden_layer(prevWindows, numNodes, numPrevs, act);
    if(attnLayer == NULL) goto error2;

    free(prevWindows);

    attnLayer->layerType = 'a';
    return attnLayer;

error2:
    free(prevWindows);
error1:
    return NULL;
}
