#pragma once

#include "layer_destruct.h"
#include <string.h>
#include <time.h>
#include <math.h>

// Solely to load input values into the model in a form where layer operations can be generalized into
layer* make_input_layer(int numNodes)
{
    // Allocate space for the input layer
    layer* inLayer = (layer*)calloc(1, sizeof(layer));
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
    if(inLayer->outputs == NULL) goto error;

    inLayer->numNodes = numNodes;
    inLayer->activationFunction = 'i';
    inLayer->layerID = -1;

    return inLayer;

error:
    hakai_layer(&inLayer);

    return NULL;
}

layer* make_hidden_layer(layer*** prev, int numNodes, int numPrevLayers, char activation_function)
{
    // Allocate space for the layer
    layer* hiddenLayer = (layer *)calloc(1, sizeof(layer));
    if(hiddenLayer == NULL) return NULL;

    // Set number of previous layers that feed into this layer and number of next layers that this layer feeds into
    // These make it easier to implement models with more complex structures than traditional NNs which would set these at 1
    // Also helps with backward passes
    hiddenLayer->numPrevLayers = numPrevLayers;
    hiddenLayer->numPrevNodes = 0;

    for(int i = 0; i < numPrevLayers; i++) hiddenLayer->numPrevNodes += (*prev[i])->numNodes;

    hiddenLayer->biases = (float *)calloc(numNodes, sizeof(float)); // Bias for each neuron
    if(hiddenLayer->biases == NULL) goto error;

    hiddenLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(hiddenLayer->outputs == NULL) goto error;
    
    hiddenLayer->preActivations = (float *)calloc(numNodes, sizeof(float));
    if(hiddenLayer->preActivations == NULL) goto error;

    hiddenLayer->backErrors = (float *)calloc((numNodes), sizeof(float));
    if(hiddenLayer->backErrors == NULL) goto error;

    hiddenLayer->activationDerivatives = (float *)calloc(numNodes, sizeof(float));
    if(hiddenLayer->activationDerivatives == NULL) goto error;

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING CAREFULLY
    hiddenLayer->prevLayers = (layer ***)malloc(numPrevLayers * sizeof(layer**));
    if(hiddenLayer->prevLayers == NULL) goto error;

    memcpy(hiddenLayer->prevLayers, prev, numPrevLayers*sizeof(layer**));

    hiddenLayer->weights = (float **)calloc(numNodes, sizeof(float*));
    if(hiddenLayer->weights == NULL) goto error;

    // All weights are contained in a single continuous allocation arena to encourage spatial locality of memory for more optimal access procedures
    hiddenLayer->weights[0] = (float *)calloc(hiddenLayer->numPrevNodes * numNodes, sizeof(float));
    if(hiddenLayer->weights[0] == NULL) goto error;

    for(int i = 1; i < numNodes; i++) hiddenLayer->weights[i] = hiddenLayer->weights[0] + (i * hiddenLayer->numPrevNodes); // Each column is a connection to each neuron in the previous layer

    hiddenLayer->numNodes = numNodes;
    hiddenLayer->layerID = -1;
    hiddenLayer->activationFunction = activation_function;
    hiddenLayer->layerType = 'h';

    return hiddenLayer;

error:
    hakai_layer(&hiddenLayer);

    return NULL;
}

layer* make_output_layer(layer*** prev, int numNodes, int numPrevLayers, char activation_function)
{
    layer* outLayer = make_hidden_layer(prev, numNodes, numPrevLayers, activation_function);
    if(outLayer == NULL) return NULL;    
    
    outLayer->layerType = 'o';

    return outLayer;
}

layer* make_window_layer(layer*** prev, int numNodes, int numPrevNodes, int numPrevLayers, char activationFunction)
{
    layer* windowLayer = (layer*)calloc(1, sizeof(layer));
    if(windowLayer == NULL) return NULL;

    windowLayer->numPrevLayers = numPrevLayers;
    windowLayer->numPrevNodes = numPrevNodes;

    windowLayer->weights = NULL;
    
    // These can be used for previous timestep bias sensitivity gradient vector storage
    windowLayer->biases = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->biases == NULL) goto error;

    windowLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->outputs == NULL) goto error;

    // These will be used for previous timestep hidden layer sensitvity gradient vector storage
    windowLayer->preActivations = (float *)calloc(numPrevNodes, sizeof(float));
    if(windowLayer->preActivations == NULL) goto error;

    windowLayer->backErrors = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->backErrors == NULL) goto error;;

    windowLayer->activationDerivatives = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->activationDerivatives == NULL) goto error;

    windowLayer->activationFunction = activationFunction;
    windowLayer->numNodes = numNodes;
    windowLayer->layerID = -1;
    windowLayer->layerType = 'w';

    windowLayer->prevLayers = (layer ***)calloc(numPrevLayers, sizeof(layer**)); // Possibly one prev layer, none if last in window
    if(windowLayer->prevLayers == NULL) goto error;

    memcpy(windowLayer->prevLayers, prev, numPrevLayers * sizeof(layer**));

    return windowLayer;

error:
    hakai_layer(&windowLayer);
    return NULL;
}

/* 
layer* make_convolutional_layers(layer*** prevLayers, int numPrevLayers, int numFilters, int numDims, int convDims, int hasPadding)
{
    // Allocate space for the layer
    layer* convLayer = (layer *)malloc(sizeof(layer));
    if(convLayer == NULL) return NULL;

    convLayer->numPrevLayers = numDims;
    convLayer->numPrevNodes = (*prevLayers[0])->numNodes * numDims;

    convLayer->biases = NULL;
    
    convLayer->backErrors = (float *)calloc((numNodes), sizeof(float));
    if(convLayer->backErrors == NULL) goto error;

    convLayer->outputs = (float *)malloc(numNodes * sizeof(float));
    if(convLayer->outputs == NULL) goto error;
    
    convLayer->preActivations = (float *)malloc(numNodes* sizeof(float));
    if(convLayer->preActivations == NULL) goto error;

    convLayer->activationDerivatives = (float *)calloc(numNodes, sizeof(float));
    if(convLayer->activationDerivatives == NULL) goto error;

    convLayer->weights = (float **)calloc(numNodes, sizeof(float*)); // Initially zeroed out so it can be freed easier later if error occurs when constructing the matrix dynamically
    if(convLayer->weights == NULL) goto error;

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING CAREFULLY
    convLayer->prevLayers = (layer ***)malloc(sizeof(layer**) * numPrevLayers);
    if(convLayer->prevLayers == NULL) goto error;

    // Set the previous layers as the previous layers
    for(int i = 0; i < numPrevLayers; i++)
    {
        convLayer->prevLayers[i] = prev[i]; // Could just use memcpy but this approach induces better cache hits
        convLayer->numPrevNodes += (*prev[i])->numNodes;
    }

    for(int i = 0; i < numNodes; i++)
    {
        convLayer->weights[i] = (float *)malloc(sizeof(float) * (convLayer->numPrevNodes)); // Each column is a connection to each neuron in the previous layer
        if(convLayer->weights[i] == NULL) goto error;
    }
    
    convLayer->numNodes = numNodes;
    convLayer->activationFunction = activation_function;
    convLayer->layerID = -1;
    convLayer->layerType = 'h';

    return convLayer;

error:
    hakai_layer(&convLayer);

    return NULL;
}
*/

layer* make_attention_layer(layer** myPrevLayer, int windowSize, char activation)
{
    layer** prevLayer;
    int numPrevs = 1;

    prevLayer = myPrevLayer;

    if((*prevLayer)->layerType != 'h') return NULL;

    layer* attnLayer = calloc(1, sizeof(layer));
    if(attnLayer == NULL) return NULL;

    attnLayer->outputs = calloc((*myPrevLayer)->numNodes, sizeof(float));
    if(attnLayer->outputs == NULL) goto error;

    // Some slight overhead in the model building phase because I know some smartass will get themself a seg fault trying to make
    // an attention layer with discrepancy between window size and actual number of window layers
    while(numPrevs < windowSize && (*(*prevLayer)->prevLayers[(*prevLayer)->numPrevLayers - 1])->layerType == 'w')
    {
        prevLayer = (*prevLayer)->prevLayers[(*prevLayer)->numPrevLayers - 1];
        numPrevs++;
    }
    
    attnLayer->numPrevLayers = numPrevs;

    attnLayer->preActivations = calloc(numPrevs, sizeof(float));
    if(attnLayer->preActivations == NULL) goto error;

    attnLayer->backErrors = calloc((*myPrevLayer)->numNodes, sizeof(float));
    if(attnLayer->backErrors == NULL) goto error;
    
    attnLayer->prevLayers = malloc(numPrevs * sizeof(layer**));
    if(attnLayer->prevLayers == NULL) goto error;

    prevLayer = myPrevLayer;
    attnLayer->prevLayers[0] = prevLayer;
    numPrevs = 1;
    while(numPrevs < windowSize && (*(*prevLayer)->prevLayers[(*prevLayer)->numPrevLayers - 1])->layerType == 'w')
    {
        prevLayer = (*prevLayer)->prevLayers[(*prevLayer)->numPrevLayers - 1];
        attnLayer->prevLayers[numPrevs] = prevLayer;
        numPrevs++;
    }
    
    attnLayer->numPrevNodes = (*myPrevLayer)->numNodes;
    attnLayer->weights = NULL;
    attnLayer->biases = NULL;
    attnLayer->activationDerivatives = NULL; // Also might be useful later
    attnLayer->numNodes = (*myPrevLayer)->numNodes;
    attnLayer->layerID = -1;
    attnLayer->activationFunction = (activation == 'x') ? 'x' : 'f';
    attnLayer->layerType = 'a';


    return attnLayer;

error:
    hakai_layer(&attnLayer);
    return NULL;
}
