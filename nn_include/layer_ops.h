#ifndef NN_LAYER_OPS
#define NN_LAYER_OPS

#include "layer.h"

void hakai_weight_matrix(float** weightMat)
{
    int i = 0;
    while(weightMat[i] != NULL)
    {
        free(weightMat[i]);
        weightMat[i] = NULL;
        i += 1;
    }

    free(weightMat);
    weightMat = NULL;
}

struct layer* make_input_layer(int numNodes, int numNextLayers, int layer_id)
{
    // Allocate space for the input layer
    struct layer *inLayer = (struct layer*)malloc(sizeof(struct layer));
    if(inLayer == NULL)
    {
        return NULL;
    }

    inLayer->numPrevLayers = 0;
    inLayer->numNextLayers = numNextLayers;
    // Enter this function with the outArray of the model and let it do its thing
    // No previous layers for an input layer
    inLayer->prevLayers = NULL;

    // Allocate space for the following layers so a forward pass is easier to implement and also navigating the layers
    inLayer->nextLayers = (struct layer**)calloc(numNextLayers, sizeof(struct layer*));
    if(inLayer->nextLayers == NULL)
    {
        goto error1;
    }

    // Input layer just accepts inputs, doesn't need weights
    inLayer->currLayerWeights = (float**)calloc(numNodes, sizeof(float*));
    if(inLayer->currLayerWeights == NULL)
    {
        goto error2;
    }

    for(int i = 0; i < numNodes; i++)
    {
        inLayer->currLayerWeights[i] = (float *)malloc(sizeof(float));
        if(inLayer->currLayerWeights[i] == NULL) goto error3;
        inLayer->currLayerWeights[i][0] = 1.0f;
    }

    inLayer->currLayerGradients = (float**)calloc(numNodes, sizeof(float*));
    if(inLayer->currLayerGradients == NULL)
    {
        goto error3;
    }

    inLayer->numNodes = numNodes;
    inLayer->activation = 'i';
    inLayer->layer_id = layer_id;
    inLayer->numPrevNodes = {};

    return inLayer;

error3:
    hakai_weight_matrix(inLayer->currLayerWeights);
error2:
    free(inLayer->nextLayers);
    inLayer->nextLayers = NULL;
error1:
    free(inLayer);
    inLayer = NULL;

    return NULL;
}

struct layer* make_dense_layer(struct layer** prev, int numNodes, int numPrevLayers, int numNextLayers, int layer_id, char norm)
{
    int j = 0;
    int sumPrevs = 0;

    // Allocate space for the layer
    struct layer *denseLayer = (struct layer *)malloc(sizeof(struct layer));
    if(denseLayer == NULL)
    {
        return NULL;
    }

    // Set number of previous layers that feed into this layer and number of next layers that this layer feeds into
    // These make it easier to implement models with more complex structures than traditional NNs which would set these at 1
    // Also helps with backward passes
    denseLayer->numPrevLayers = numPrevLayers;
    denseLayer->numNextLayers = numNextLayers;
    if(denseLayer->numPrevNodes == NULL) goto error1;

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING CAREFULLY
    denseLayer->prevLayers = (struct layer **)malloc(sizeof(struct layer*) * numPrevLayers);
    if(denseLayer->prevLayers == NULL) goto error2;

    // Set the previous layers as the previous layers
    memcpy(denseLayer->prevLayers, prev, sizeof(struct layer*) * numPrevLayers);

    // Make this layer a next layer for all previous layers
    for(int i = 0; i < numPrevLayers; i++)
    {
        while(prev[i]->nextLayers[j] != NULL)
        {
            j += 1;
        }

        sumPrevs += prev[i]->numNodes;
        prev[i]->nextLayers[j] = denseLayer;
        j = 0;
    }

    // Allocate space for the next layers using provided parameter
    denseLayer->nextLayers = (struct layer **)calloc(numNextLayers, sizeof(struct layer*));
    if(denseLayer->nextLayers == NULL) goto error2;

    denseLayer->currLayerWeights = (float **)malloc((numNodes + 1) * sizeof(float*)); // Neuron weights plus a bias weight
    if(denseLayer->currLayerWeights == NULL) goto error3;

    for(int i = 0; i < numNodes; i++)
    {
        denseLayer->currLayerWeights[i] = (float *)malloc(sizeof(float) * sumPrevs);
        if(denseLayer->currLayerWeights[i] == NULL) goto error4;
        
        for(int j = 0; j < sumPrevs; j++) denseLayer->currLayerWeights[i][j] = {1.0f}; // Each column is a connection to each neuron in the previous layer pus a bias
        denseLayer->currLayerWeights[i][sumPrevs] = 0.0f; // Initialize biases
    }
    
    for(int i = 0; i < sumPrevs; i++) denseLayer->currLayerWeights[numNodes][i] = 1.0;
    
    denseLayer->currLayerGradients = (float **)calloc((numNodes + 1), sizeof(float)); // Each row is a neuron in this layer
    if(denseLayer->currLayerGradients == NULL) goto error4;
    
    for(int i = 0; i < numNodes; i++)
    {
        denseLayer->currLayerGradients[i] = (float *)calloc(sumPrevs + 1, sizeof(float)); // Each column is a connection to each neuron in the previous layers plus a bias
        if(denseLayer->currLayerWeights[i] == NULL) goto error5;
        
        for(int j = 0; j < sumPrevs; j++) denseLayer->currLayerGradients[i][j] = 1.0f;
    }

    for(int i = 0; i < numNodes; i++)
    {
        for(int j = 0; j < sumPrevs; j++)
        {
            denseLayer->currLayerWeights[i][j] = 1; // Fix later to randomly initialize
        }
    }

    denseLayer->numNodes = numNodes;
    denseLayer->activation = 'r';
    denseLayer->layer_id = layer_id;
    denseLayer->numPrevNodes = sumPrevs;

    return denseLayer;

error5:
    hakai_weight_matrix(denseLayer->currLayerGradients);
error4:
    hakai_weight_matrix(denseLayer->currLayerWeights);
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

struct layer* make_output_layer(struct layer** prev, int numNodes, int numPrevLayers, int layer_id)
{
    int j = 0;
    int sumPrevs = 0;
    struct layer *outLayer = (struct layer *)malloc(sizeof(struct layer));
    if(outLayer == NULL) return NULL;

    outLayer->numPrevLayers = numPrevLayers;
    outLayer->numNextLayers = 0;
    outLayer->nextLayers = NULL; // No next layers for an output layer

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
        prev[i]->nextLayers[j] = outLayer;
        j = 0;
    }

    outLayer->currLayerWeights = (float **)malloc((numNodes) * sizeof(float*)); // Each row is a neuron
    if(outLayer->currLayerWeights == NULL) goto error2;

    for(int i = 0; i < numNodes; i++)
    {
        outLayer->currLayerWeights[i] = (float *)malloc(sizeof(float) * (sumPrevs)); // Each column is a weight from the neuron row to an input or a 
        if(outLayer->currLayerWeights[i] == NULL) goto error3;
        
        for(int j = 0; j < sumPrevs; j++) outLayer->currLayerWeights[i][j] = 1.0f; // Initialize weight connections
    }
    
    outLayer->currLayerGradients = (float **)calloc(numNodes, sizeof(float*)); // Neuron weights plus a bias weight
    if(outLayer->currLayerGradients == NULL) goto error3;

    for(int i = 0; i < numNodes; i++)
    {
        for(int j = 0; j < sumPrevs; j++)
        {
            outLayer->currLayerWeights[i][j] = 1; // Fix later to randomly initialize
        }
    }

    outLayer->numNodes = numNodes;
    outLayer->activation = 't';
    outLayer->layer_id = layer_id;
    outLayer->numPrevNodes = sumPrevs;

    return outLayer;

error3:
    hakai_weight_matrix(outLayer->currLayerWeights);
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