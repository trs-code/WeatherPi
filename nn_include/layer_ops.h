#ifndef NN_LAYER_OPS
#define NN_LAYER_OPS

#include "layer.h"

void hakai_matrix(float** mat)
{
    int i = 0;
    while(mat[i] != NULL)
    {
        free(mat[i]);
        mat[i] = NULL;
        i += 1;
    }

    free(mat);
    mat = NULL;
}

// Solely to load input values into the model in a form where layer operations can be generalized into
struct layer* make_input_layer(int numNodes, int numNextLayers, int layerID)
{
    // Allocate space for the input layer
    struct layer *inLayer = (struct layer*)malloc(sizeof(struct layer));
    if(inLayer == NULL) return NULL;

    inLayer->numPrevLayers = 0;
    inLayer->numPrevNodes = 0;
    inLayer->numNextLayers = numNextLayers;
    // No previous layers for an input layer
    inLayer->prevLayers = NULL;
    // Input layer just accepts inputs, doesn't need actual weights, just something to facilitate forwarding values
    inLayer->weights = NULL;
    inLayer->gradients = NULL; // Input layer doesn't need gradients

    // Allocate space for the following layers so a forward pass is easier to implement and also navigating the layers
    inLayer->nextLayers = (struct layer**)calloc(numNextLayers, sizeof(struct layer*));
    if(inLayer->nextLayers == NULL) goto error1;

    inLayer->activations = NULL;
    
    inLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(inLayer->outputs == NULL) goto error2;

    inLayer->numNodes = numNodes;
    inLayer->activation = 'i';
    inLayer->layerID = layerID;

    return inLayer;

error2:
    free(inLayer->nextLayers);
    inLayer->nextLayers = NULL;
error1:
    free(inLayer);
    inLayer = NULL;

    return NULL;
}

struct layer* make_dense_layer(struct layer** prev, int numNodes, int numPrevLayers, int numNextLayers, int layerID, char norm)
{
    int j = 0;

    // Allocate space for the layer
    struct layer *denseLayer = (struct layer *)malloc(sizeof(struct layer));
    if(denseLayer == NULL) return NULL;

    // Set number of previous layers that feed into this layer and number of next layers that this layer feeds into
    // These make it easier to implement models with more complex structures than traditional NNs which would set these at 1
    // Also helps with backward passes
    denseLayer->numPrevLayers = numPrevLayers;
    denseLayer->numNextLayers = numNextLayers;
    denseLayer->numPrevNodes = 0;

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING CAREFULLY
    denseLayer->prevLayers = (struct layer **)malloc(sizeof(struct layer*) * numPrevLayers);
    if(denseLayer->prevLayers == NULL) goto error1;

    // Set the previous layers as the previous layers
    memcpy(denseLayer->prevLayers, prev, sizeof(struct layer*) * numPrevLayers);

    // Make this layer a next layer for all previous layers
    for(int i = 0; i < numPrevLayers; i++)
    {
        while(prev[i]->nextLayers[j] != NULL)
        {
            j += 1;
        }

        denseLayer->numPrevNodes += prev[i]->numNodes;
        prev[i]->nextLayers[j] = denseLayer;
        j = 0;
    }

    // Allocate space for the next layers using provided parameter
    denseLayer->nextLayers = (struct layer **)calloc(numNextLayers, sizeof(struct layer*));
    if(denseLayer->nextLayers == NULL) goto error2;

    denseLayer->weights = (float **)malloc(numNodes * sizeof(float*));
    if(denseLayer->weights == NULL) goto error3;

    for(int i = 0; i < numNodes; i++)
    {
        denseLayer->weights[i] = (float *)malloc(sizeof(float) * (denseLayer->numPrevNodes + 1)); // Each column is a connection to each neuron in the previous layer pus a bias
        if(denseLayer->weights[i] == NULL) goto error4;
        
        for(int j = 0; j < denseLayer->numPrevNodes; j++) denseLayer->weights[i][j] = 1.0f; 
        denseLayer->weights[i][denseLayer->numPrevNodes] = 0.0f; // Initialize biases
    }
        
    denseLayer->gradients = (float **)malloc(numNodes * sizeof(float)); // Each row is a neuron in this layer
    if(denseLayer->gradients == NULL) goto error4;
    
    for(int i = 0; i < numNodes; i++)
    {
        denseLayer->gradients[i] = (float *)calloc((denseLayer->numPrevNodes + 1), sizeof(float)); // Each column is a connection to each neuron in the previous layers plus a bias
        if(denseLayer->weights[i] == NULL) goto error5;
    }

    denseLayer->activations = (float *)calloc(numNodes, sizeof(float));
    if(denseLayer->activations == NULL) goto error5;
    
    denseLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(denseLayer->outputs == NULL) goto error6;
    
    denseLayer->numNodes = numNodes;
    denseLayer->activation = 'r';
    denseLayer->layerID = layerID;

    return denseLayer;

error6:
    free(denseLayer->activations);
    denseLayer->activations = NULL;
error5:
    hakai_matrix(denseLayer->gradients);
error4:
    hakai_matrix(denseLayer->weights);
error3:
    free(denseLayer->nextLayers);
    denseLayer->nextLayers = NULL;
error2:
    free(denseLayer->prevLayers);
    denseLayer->prevLayers = NULL;
error1:
    free(denseLayer);
    denseLayer = NULL;

    return NULL;
}

struct layer* make_output_layer(struct layer** prev, int numNodes, int numPrevLayers, int layerID)
{
    int j = 0;

    struct layer *outLayer = (struct layer *)malloc(sizeof(struct layer));
    if(outLayer == NULL) return NULL;

    outLayer->numPrevLayers = numPrevLayers;
    outLayer->numNextLayers = 0;
    outLayer->nextLayers = NULL; // No next layers for an output layer
    outLayer->numPrevNodes = 0;

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING
    outLayer->prevLayers = (struct layer **)malloc(sizeof(struct layer*) * numPrevLayers);
    if(outLayer->prevLayers == NULL) goto error1;

    memcpy(outLayer->prevLayers, prev, sizeof(struct layer*) * numPrevLayers);

    for(int i = 0; i < numPrevLayers; i++)
    {
        while(prev[i]->nextLayers[j] != NULL)
        {
            j += 1;
        }

        outLayer->numPrevNodes += prev[i]->numNodes;
        prev[i]->nextLayers[j] = outLayer;
        j = 0;
    }

    outLayer->weights = (float **)malloc((numNodes) * sizeof(float*)); // Each row is a neuron
    if(outLayer->weights == NULL) goto error2;

    for(int i = 0; i < numNodes; i++)
    {
        outLayer->weights[i] = (float *)malloc(sizeof(float) * (outLayer->numPrevNodes + 1));
        if(outLayer->weights[i] == NULL) goto error3;
        
        for(int j = 0; j < outLayer->numPrevNodes; j++) outLayer->weights[i][j] = 1.0f; // Initialize weight connections
        outLayer->weights[i][outLayer->numPrevNodes] = 0.0f;
    }
    
    outLayer->gradients = (float **)calloc(numNodes, sizeof(float*)); // Neuron weights plus a bias weight
    if(outLayer->gradients == NULL) goto error3;

    for(int i = 0; i < numNodes; i++)
    {
        outLayer->gradients[i] = (float *)calloc((outLayer->numPrevNodes + 1), sizeof(float));
        if(outLayer->gradients[i] == NULL) goto error4;
    }

    outLayer->activations = (float *)calloc(numNodes, sizeof(float));
    if(outLayer->activations == NULL) goto error4;
    
    outLayer->outputs = (float *)calloc(numNodes, sizeof(float));
    if(outLayer->outputs == NULL) goto error5;

    outLayer->numNodes = numNodes;
    outLayer->activation = 't';
    outLayer->layerID = layerID;

    return outLayer;

error5:
    free(outLayer->activations);
    outLayer->activations = NULL;
error4:
    hakai_matrix(outLayer->gradients);
error3:
    hakai_matrix(outLayer->weights);
error2:
    free(outLayer->prevLayers);
    outLayer->prevLayers = NULL;
error1:
    free(outLayer);
    outLayer = NULL;

    return NULL;
}

/*
struct layer* make_normalization_layer(struct layer* prev, int numNextLayers = 1)
{
    int layerWeightsSize = sizeof(prev->currLayerWeights);
    int prevSize = sizeof(prev);
    struct layer *normLayer = (struct layer*)malloc(sizeof(struct layer));
    if(normLayer == NULL)
    {
        return NULL;
    }

    normLayer->numNodes = prev->numNodes;
    normLayer->numPrevLayers = 1;
    normLayer->numNextLayers = numNextLayers;
    
    normLayer->prevLayers = (layer **)malloc(prevSize);
    if(normLayer->prevLayers == NULL)
    {
        goto error1;
    }

    normLayer->nextLayers = (layer **)malloc(sizeof(struct layer*)*numNextLayers);
    if(normLayer->prevLayers == NULL)
    {
        goto error2;
    }
    
    memcpy(normLayer->prevLayers, prev, prevSize);
    normLayer->currLayerWeights = (float *)malloc(layerWeightsSize);
    if(normLayer->currLayerWeights == NULL)
    {
        goto error3;
    }

    memcpy(normLayer->currLayerWeights, prev->currLayerWeights, layerWeightsSize);

    minMaxNorm(normLayer->currLayerWeights, normLayer->numNodes);
    
    normLayer->activation = 'r';

    return normLayer;

error3:
    free(normLayer->nextLayers);
    normLayer->nextLayers = NULL;
error2:
    free(normLayer->prevLayers);
    normLayer->prevLayers = NULL;
error1:
    free(normLayer);
    normLayer = NULL;

    return NULL;
}
*/

#endif