#pragma once

#include "layer_destruct.h"
#include <string.h>
#include <time.h>
#include <math.h>

void glorot_uniform_init(layer* myLayer) 
{
    float limit = sqrt(6.0 / (myLayer->numPrevNodes + myLayer->numNodes));
    for (int i = 0; i < myLayer->numNodes; ++i) {
        for(int j = 0; j < myLayer->numPrevNodes; j++) myLayer->weights[i][j] = ((float)rand() / RAND_MAX) * 2.0 * limit - limit;
    }
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
    inLayer->switchVar = '0';

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

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING CAREFULLY
    hiddenLayer->prevLayers = (layer ***)malloc(sizeof(layer**) * numPrevLayers);
    if(hiddenLayer->prevLayers == NULL) goto error1;

    // Set the previous layers as the previous layers
    for(int i = 0; i < numPrevLayers; i++) hiddenLayer->prevLayers[i] = prev[i];

    // Make this layer a next layer for all previous layers
    for(int i = 0; i < numPrevLayers; i++) hiddenLayer->numPrevNodes += (*hiddenLayer->prevLayers[i])->numNodes;

    hiddenLayer->weights = (float **)malloc(numNodes * sizeof(float*));
    if(hiddenLayer->weights == NULL) goto error2;

    hiddenLayer->biases = (float *)calloc(numNodes, sizeof(float)); // Bias for each neuron
    if(hiddenLayer->biases == NULL) goto error3;

    for(int i = 0; i < numNodes; i++)
    {
        hiddenLayer->weights[i] = (float *)malloc(sizeof(float) * (hiddenLayer->numPrevNodes)); // Each column is a connection to each neuron in the previous layer pus a bias
        if(hiddenLayer->weights[i] == NULL) goto error4;
    }
    
    hiddenLayer->backErrors = (float *)calloc((numNodes), sizeof(float));
    if(hiddenLayer->backErrors == NULL) goto error4;

    hiddenLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(hiddenLayer->outputs == NULL) goto error5;
    
    hiddenLayer->preActivations = (float *)calloc(numNodes, sizeof(float));
    if(hiddenLayer->preActivations == NULL) goto error6;
    
    hiddenLayer->numNodes = numNodes;
    hiddenLayer->activationFunction = activation_function;
    hiddenLayer->layerID = -1;
    hiddenLayer->switchVar = '0';
    hiddenLayer->layerType = 'h';

    glorot_uniform_init(hiddenLayer);

    return hiddenLayer;


error6:
    free(hiddenLayer->outputs);
    hiddenLayer->outputs = NULL;
error5:
    free(hiddenLayer->backErrors);
    hiddenLayer->backErrors = NULL;
error4:
    hakai_matrix(&(hiddenLayer->weights), numNodes);
error3:
    free(hiddenLayer->biases);
    hiddenLayer->biases = NULL;
error2:
    free(hiddenLayer->prevLayers);
    hiddenLayer->prevLayers = NULL;
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

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING
    outLayer->prevLayers = (layer ***)malloc(sizeof(layer**) * numPrevLayers);
    if(outLayer->prevLayers == NULL) goto error1;

    for(int i = 0; i < numPrevLayers; i++) outLayer->prevLayers[i] = prev[i];

    for(int i = 0; i < numPrevLayers; i++) outLayer->numPrevNodes += (*outLayer->prevLayers[i])->numNodes;

    outLayer->weights = (float **)malloc(numNodes * sizeof(float*)); // Each row is a neuron
    if(outLayer->weights == NULL) goto error2;

    outLayer->biases = (float *)calloc(numNodes, sizeof(float)); // Bias for each neuron
    if(outLayer->biases == NULL) goto error3;

    for(int i = 0; i < numNodes; i++)
    {
        outLayer->weights[i] = (float *)malloc(sizeof(float) * (outLayer->numPrevNodes + 1));
        if(outLayer->weights[i] == NULL) goto error4;
    }

    outLayer->backErrors = (float *)calloc(numNodes, sizeof(float));
    if(outLayer->backErrors == NULL) goto error4;

    outLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(outLayer->outputs == NULL) goto error5;
    
    outLayer->preActivations = (float *)calloc(numNodes, sizeof(float));
    if(outLayer->preActivations == NULL) goto error6;


    outLayer->numNodes = numNodes;
    outLayer->activationFunction = activation_function;
    outLayer->layerID = -1;
    outLayer->switchVar = '0';
    outLayer->layerType = 'o';

    glorot_uniform_init(outLayer);

    return outLayer;

error6:
    free(outLayer->outputs);
    outLayer->outputs = NULL;
error5:
    free(outLayer->backErrors);
    outLayer->backErrors = NULL;
error4:
    hakai_matrix(&(outLayer->weights), numNodes);
error3:
    free(outLayer->biases);
    outLayer->biases = NULL;
error2:
    free(outLayer->prevLayers);
    outLayer->prevLayers = NULL;
error1:
    free(outLayer);
    outLayer = NULL;

    return NULL;
}

// Currently not in use, can be a space saving way to do RNN inference
layer* make_referential_layer(layer*** prev, int numNodes, int numPrevLayers, char activation_function, layer** thisLayerAddress)
{
    // Allocate space for the layer
    layer *referentialLayer = (layer *)malloc(sizeof(layer));
    if(referentialLayer == NULL) return NULL;

    // Set number of previous layers that feed into this layer and number of next layers that this layer feeds into
    // These make it easier to implement models with more complex structures than traditional NNs which would set these at 1
    // Also helps with backward passes
    referentialLayer->numPrevLayers = numPrevLayers + 1;
    referentialLayer->numPrevNodes = 0;

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING CAREFULLY
    referentialLayer->prevLayers = (layer ***)malloc(sizeof(layer**) * (numPrevLayers + 1));
    if(referentialLayer->prevLayers == NULL) goto error1;

    // Set the previous layers as the previous layers
    for(int i = 0; i < numPrevLayers; i++) referentialLayer->prevLayers[i] = prev[i];

    referentialLayer->prevLayers[numPrevLayers] = thisLayerAddress;

    // Make this layer a next layer for all previous layers
    for(int i = 0; i < numPrevLayers; i++) referentialLayer->numPrevNodes += (*referentialLayer->prevLayers[i])->numNodes;

    referentialLayer->numPrevNodes += numNodes;

    referentialLayer->weights = (float **)malloc(numNodes * sizeof(float*));
    if(referentialLayer->weights == NULL) goto error2;

    referentialLayer->biases = (float *)calloc(numNodes, sizeof(float)); // Bias for each neuron
    if(referentialLayer->biases == NULL) goto error3;

    for(int i = 0; i < numNodes; i++)
    {
        referentialLayer->weights[i] = (float *)malloc(sizeof(float) * (referentialLayer->numPrevNodes)); // Each column is a connection to each neuron in the previous layer pus a bias
        if(referentialLayer->weights[i] == NULL) goto error4;
    }
    
    referentialLayer->backErrors = (float *)calloc((numNodes), sizeof(float));
    if(referentialLayer->backErrors == NULL) goto error4;

    referentialLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(referentialLayer->outputs == NULL) goto error5;
    
    referentialLayer->preActivations = (float *)calloc(numNodes, sizeof(float));
    if(referentialLayer->preActivations == NULL) goto error6;
    
    referentialLayer->numNodes = numNodes;
    referentialLayer->activationFunction = activation_function;
    referentialLayer->layerID = -1;
    referentialLayer->switchVar = '0';
    referentialLayer->layerType = 'r'; 

    glorot_uniform_init(referentialLayer);

    return referentialLayer;


error6:
    free(referentialLayer->outputs);
    referentialLayer->outputs = NULL;
error5:
    free(referentialLayer->backErrors);
    referentialLayer->backErrors = NULL;
error4:
    hakai_matrix(&(referentialLayer->weights), numNodes);
error3:
    free(referentialLayer->biases);
    referentialLayer->biases = NULL;
error2:
    free(referentialLayer->prevLayers);
    referentialLayer->prevLayers = NULL;
error1:
    free(referentialLayer);
    referentialLayer = NULL;

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
    windowLayer->weights = NULL; // Just holds previous values, doesn't need actual weights, just something to facilitate forwarding values
    windowLayer->biases = NULL;

    windowLayer->prevLayers = (layer ***)calloc(numPrevLayers, sizeof(layer**)); // Possibly one prev layer, none if last in window
    if(windowLayer->prevLayers == NULL) goto error1;

    windowLayer->backErrors = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->backErrors == NULL) goto error1;

    windowLayer->preActivations = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->preActivations == NULL) goto error1;

    
    windowLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(windowLayer->outputs == NULL) goto error1;

    windowLayer->numNodes = numNodes;
    windowLayer->activationFunction = activationFunction;
    windowLayer->layerID = -1;
    windowLayer->switchVar = '0';

    for(int i = 0; i < numPrevLayers; i++) windowLayer->prevLayers[i] = prev[i];

    return windowLayer;

error1:
    free(windowLayer);
    windowLayer = NULL;

    return NULL;
}

// Not Yet Implemented 
layer* make_convolutional_layer(layer*** prevLayers, int numPrevLayers, int numFilters, int numDims, int* dims, int hasPadding);

// Not yet implemented
layer* make_attention_layer(layer** windowLayers, int windowSize, char activationFunction);
