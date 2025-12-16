#ifndef NN_LAYER_OPS
#define NN_LAYER_OPS

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
    inLayer->numNextLayers = 0; // Not true but this variable isn't currently utilized for anything so no need to waste our time on it
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

layer* make_dense_layer(layer*** prev, int numNodes, int numPrevLayers, int numNextLayers, char activation_function)
{
    // int j = 0;

    // Allocate space for the layer
    layer *denseLayer = (layer *)malloc(sizeof(layer));
    if(denseLayer == NULL) return NULL;

    // Set number of previous layers that feed into this layer and number of next layers that this layer feeds into
    // These make it easier to implement models with more complex structures than traditional NNs which would set these at 1
    // Also helps with backward passes
    denseLayer->numPrevLayers = numPrevLayers;
    denseLayer->numNextLayers = numNextLayers;
    denseLayer->numPrevNodes = 0;

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING CAREFULLY
    denseLayer->prevLayers = (layer ***)malloc(sizeof(layer**) * numPrevLayers);
    if(denseLayer->prevLayers == NULL) goto error1;

    // Set the previous layers as the previous layers
    for(int i = 0; i < numPrevLayers; i++) denseLayer->prevLayers[i] = prev[i];

    // Make this layer a next layer for all previous layers
    for(int i = 0; i < numPrevLayers; i++) denseLayer->numPrevNodes += (*denseLayer->prevLayers[i])->numNodes;

    denseLayer->weights = (float **)malloc(numNodes * sizeof(float*));
    if(denseLayer->weights == NULL) goto error2;

    denseLayer->biases = (float *)malloc(numNodes * sizeof(float)); // Bias for each neuron
    if(denseLayer->biases == NULL) goto error3;

    for(int i = 0; i < numNodes; i++)
    {
        denseLayer->weights[i] = (float *)malloc(sizeof(float) * (denseLayer->numPrevNodes)); // Each column is a connection to each neuron in the previous layer pus a bias
        if(denseLayer->weights[i] == NULL) goto error4;
        
        for(int j = 0; j < denseLayer->numPrevNodes; j++) denseLayer->weights[i][j] = ((rand() % 100000) + 50000)/100000; 
        denseLayer->biases[i] = ((rand() % 100000) + 50000)/100000; // Initialize biases
    }
    
    denseLayer->backErrors = (float *)calloc((numNodes), sizeof(float));
    if(denseLayer->backErrors == NULL) goto error4;

    denseLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(denseLayer->outputs == NULL) goto error5;
    
    denseLayer->preActivations = (float *)calloc(numNodes, sizeof(float));
    if(denseLayer->preActivations == NULL) goto error6;
    
    denseLayer->numNodes = numNodes;
    denseLayer->activationFunction = activation_function;
    denseLayer->layerID = -1;
    denseLayer->switchVar = '0';

    return denseLayer;


error6:
    free(denseLayer->outputs);
    denseLayer->outputs = NULL;
error5:
    free(denseLayer->backErrors);
    denseLayer->backErrors = NULL;
error4:
    hakai_matrix(&(denseLayer->weights), numNodes);
error3:
    free(denseLayer->biases);
    denseLayer->biases = NULL;
error2:
    free(denseLayer->prevLayers);
    denseLayer->prevLayers = NULL;
error1:
    free(denseLayer);
    denseLayer = NULL;

    return NULL;
}

layer* make_output_layer(layer*** prev, int numNodes, int numPrevLayers, char activation_function)
{
    // int j = 0;

    layer *outLayer = (layer *)malloc(sizeof(layer));
    if(outLayer == NULL) return NULL;

    outLayer->numPrevLayers = numPrevLayers;
    outLayer->numNextLayers = 0;
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

// layer* make_normalization_layer();
// layer* make_convolutional_layer();

#endif