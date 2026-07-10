#pragma once

#include "layer_destruct.h"
#include <string.h>
#include <time.h>
#include <math.h>

static inline void glorot_uniform_init(layer* myLayer) 
{
    float limit = sqrt(6.0 / (myLayer->numPrevNodes + myLayer->numNodes));
    for (int i = 0; i < myLayer->numNodes; ++i) for(int j = 0; j < myLayer->numPrevNodes; j++) myLayer->weights[i][j] = ((float)rand() / RAND_MAX) * 2.0 * limit - limit;
}

// Solely to load input values into the model in a form where layer operations can be generalized into
layer* make_input_layer(int numNodes)
{
    srand(time(NULL));
    // Allocate space for the input layer
    layer *inLayer = (layer*)malloc(sizeof(layer));
    if(inLayer == NULL) return NULL;

    inLayer->numPrevLayers = 0;
    inLayer->numPrevNodes = 0;
    inLayer->layerType = 'i';
    inLayer->prevLayers = NULL; // No previous layers for an input layer
    inLayer->weights = NULL;    // Input layer just accepts inputs, doesn't need actual weights, just something to facilitate forwarding values
    inLayer->biases = NULL;
    inLayer->backErrors = NULL;  // Input layer doesn't need backErrors
    inLayer->preActivations = NULL;
    
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
    layer *hiddenLayer = (layer *)malloc(sizeof(layer));
    if(hiddenLayer == NULL) return NULL;

    // Set number of previous layers that feed into this layer and number of next layers that this layer feeds into
    // These make it easier to implement models with more complex structures than traditional NNs which would set these at 1
    // Also helps with backward passes
    hiddenLayer->numPrevLayers = numPrevLayers;
    hiddenLayer->numPrevNodes = 0;

    for(int i = 0; i < numPrevLayers; i++) hiddenLayer->numPrevNodes += (*prev[i])->numNodes;

    hiddenLayer->weights = (float **)calloc(numNodes, sizeof(float*));
    if(hiddenLayer->weights == NULL) goto error1;

    hiddenLayer->biases = (float *)calloc(numNodes, sizeof(float)); // Bias for each neuron
    if(hiddenLayer->biases == NULL) goto error2;
    
    hiddenLayer->backErrors = (float *)calloc((numNodes), sizeof(float));
    if(hiddenLayer->backErrors == NULL) goto error3;

    hiddenLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(hiddenLayer->outputs == NULL) goto error4;
    
    hiddenLayer->preActivations = (float *)calloc(numNodes, sizeof(float));
    if(hiddenLayer->preActivations == NULL) goto error5;

     // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING CAREFULLY
    hiddenLayer->prevLayers = (layer ***)malloc(sizeof(layer**) * numPrevLayers);
    if(hiddenLayer->prevLayers == NULL) goto error6;

    // Set the previous layers as the previous layers
    for(int i = 0; i < numPrevLayers; i++) hiddenLayer->prevLayers[i] = prev[i];

    for(int i = 0; i < numNodes; i++)
    {
        hiddenLayer->weights[i] = (float *)malloc(sizeof(float) * (hiddenLayer->numPrevNodes)); // Each column is a connection to each neuron in the previous layer pus a bias
        if(hiddenLayer->weights[i] == NULL) goto error7;
    }
    
    hiddenLayer->numNodes = numNodes;
    hiddenLayer->activationFunction = activation_function;
    hiddenLayer->layerID = -1;
    hiddenLayer->layerType = 'h';

    glorot_uniform_init(hiddenLayer);

    return hiddenLayer;

error7:
    free(hiddenLayer->prevLayers);
error6:
    free(hiddenLayer->preActivations);
error5:
    free(hiddenLayer->outputs);
error4:
    free(hiddenLayer->backErrors);
error3:
    free(hiddenLayer->biases);
error2:
    hakai_matrix(&(hiddenLayer->weights), numNodes);
error1:
    free(hiddenLayer);
    hiddenLayer = NULL;

    return NULL;
}

layer* make_output_layer(layer*** prev, int numNodes, int numPrevLayers, char activation_function)
{
    layer *outLayer = (layer *)malloc(sizeof(layer));
    if(outLayer == NULL) return NULL;

    outLayer->numPrevLayers = numPrevLayers;
    outLayer->numPrevNodes = 0;
    for(int i = 0; i < numPrevLayers; i++) outLayer->numPrevNodes += (*prev[i])->numNodes;
    outLayer->weights = (float **)calloc(numNodes, sizeof(float*)); // Each row is a neuron
    if(outLayer->weights == NULL) goto error1;

    outLayer->biases = (float *)calloc(numNodes, sizeof(float)); // Bias for each neuron
    if(outLayer->biases == NULL) goto error2;

    outLayer->backErrors = (float *)calloc(numNodes, sizeof(float));
    if(outLayer->backErrors == NULL) goto error3;

    outLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(outLayer->outputs == NULL) goto error4;
    
    outLayer->preActivations = (float *)calloc(numNodes, sizeof(float));
    if(outLayer->preActivations == NULL) goto error5;

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING
    outLayer->prevLayers = (layer ***)malloc(sizeof(layer**) * numPrevLayers);
    if(outLayer->prevLayers == NULL) goto error6;

    for(int i = 0; i < numPrevLayers; i++) outLayer->prevLayers[i] = prev[i];

    for(int i = 0; i < numNodes; i++)
    {
        outLayer->weights[i] = (float *)malloc(sizeof(float) * (outLayer->numPrevNodes + 1));
        if(outLayer->weights[i] == NULL) goto error7;
    }

    outLayer->numNodes = numNodes;
    outLayer->activationFunction = activation_function;
    outLayer->layerID = -1;
    outLayer->layerType = 'o';

    glorot_uniform_init(outLayer);

    return outLayer;

error7:
    free(outLayer->prevLayers);
error6:
    free(outLayer->preActivations);
error5:
    free(outLayer->outputs);
error4:
    free(outLayer->backErrors);
error3:
    free(outLayer->biases);
error2:
    hakai_matrix(&(outLayer->weights), numNodes);
error1:
    free(outLayer);

    return NULL;
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

    windowLayer->prevLayers = (layer ***)calloc(numPrevLayers, sizeof(layer**)); // Possibly one prev layer, none if last in window
    if(windowLayer->prevLayers == NULL) goto error1;

    windowLayer->backErrors = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->backErrors == NULL) goto error2;

    windowLayer->preActivations = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->preActivations == NULL) goto error3;
    
    windowLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->outputs == NULL) goto error4;

    windowLayer->numNodes = numNodes;
    windowLayer->activationFunction = activationFunction;
    windowLayer->layerID = -1;

    for(int i = 0; i < numPrevLayers; i++) windowLayer->prevLayers[i] = prev[i];

    return windowLayer;

error4:
    free(windowLayer->preActivations);
error3:
    free(windowLayer->backErrors);
error2:
    free(windowLayer->prevLayers);
error1:
    free(windowLayer);
    windowLayer = NULL;

    return NULL;
}

// Not Yet Implemented 
layer** make_convolutional_layers(layer*** prevLayers, int numPrevLayers, int numFilters, int numDims, int* dims, int hasPadding);

// Not yet implemented
layer* make_attention_layer(layer** windowLayers, int windowSize, char activationFunction);
