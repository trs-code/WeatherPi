#pragma once

#include <time.h>
#include "layer.h"

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

error2:
    free(inLayer->outputs);
    inLayer->outputs = NULL;
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

    hiddenLayer->biases = (float *)malloc(numNodes * sizeof(float)); // Bias for each neuron
    if(hiddenLayer->biases == NULL) goto error3;

    for(int i = 0; i < numNodes; i++)
    {
        hiddenLayer->weights[i] = (float *)malloc(sizeof(float) * (hiddenLayer->numPrevNodes)); // Each column is a connection to each neuron in the previous layer pus a bias
        if(hiddenLayer->weights[i] == NULL) goto error4;
        
        for(int j = 0; j < hiddenLayer->numPrevNodes; j++) hiddenLayer->weights[i][j] = ((rand() % 100000) + 50000)/100000; 
        hiddenLayer->biases[i] = ((rand() % 100000) + 50000)/100000; // Initialize biases
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

    outLayer->biases = (float *)malloc(numNodes * sizeof(float)); // Bias for each neuron
    if(outLayer->biases == NULL) goto error3;

    for(int i = 0; i < numNodes; i++)
    {
        outLayer->weights[i] = (float *)malloc(sizeof(float) * (outLayer->numPrevNodes + 1));
        if(outLayer->weights[i] == NULL) goto error4;
        
        for(int j = 0; j < outLayer->numPrevNodes; j++) outLayer->weights[i][j] = ((rand() % 100000) + 50000)/100000; // Initialize weight connections
        outLayer->biases[i] = ((rand() % 100000) + 50000)/100000;
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

layer* make_referential_layer(layer*** prev, int numNodes, int numPrevLayers, char activation_function, layer** thisLayer)
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

    referentialLayer->prevLayers[numPrevLayers] = thisLayer;

    // Make this layer a next layer for all previous layers
    for(int i = 0; i < numPrevLayers; i++) referentialLayer->numPrevNodes += (*referentialLayer->prevLayers[i])->numNodes;

    referentialLayer->numPrevNodes += numNodes;

    referentialLayer->weights = (float **)malloc(numNodes * sizeof(float*));
    if(referentialLayer->weights == NULL) goto error2;

    referentialLayer->biases = (float *)malloc(numNodes * sizeof(float)); // Bias for each neuron
    if(referentialLayer->biases == NULL) goto error3;

    for(int i = 0; i < numNodes; i++)
    {
        referentialLayer->weights[i] = (float *)malloc(sizeof(float) * (referentialLayer->numPrevNodes)); // Each column is a connection to each neuron in the previous layer pus a bias
        if(referentialLayer->weights[i] == NULL) goto error4;
        
        for(int j = 0; j < referentialLayer->numPrevNodes; j++) referentialLayer->weights[i][j] = ((rand() % 100000) + 50000)/100000; 
        referentialLayer->biases[i] = ((rand() % 100000) + 50000)/100000; // Initialize biases
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

// layer* make_normalization_layer();
// layer* make_convolutional_layer();
