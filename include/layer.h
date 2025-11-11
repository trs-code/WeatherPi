#ifndef NN_LAYER
#define NN_LAYER
/* 
#include <linux/slab.h>
#include <linux/kernel.h>
*/
#include <stdlib.h>
#include <string.h>

//          VERY FUCKING IMPORTANT
// MAKE SURE AN EXIT PROCESS FREES THE MEMORY FOR
//
//                      EVERY
//                      SINGLE
//                      LAYER
//
// STARTING FROM THE OUTPUT AND ITERATE THROUGH THE LINKED LIST FREEING THE MEMORY FOR EACH LAYER WEIGHTS AND LAYER
// WHEN OPERATIONS ARE CONCLUDED
// OR I WILL PERSONALLY HUNT YOU DOWN AND STICK A NEURAL NETWORK
// IN A PLACE WHERE THE LIGHT DON'T SHINE

// 40 Bytes
// 8 extra bytes for each previous layer, 8 extra bytes for each next layer
// 4 extra bytes for each layer weight
struct layer
{
    layer **prevLayers;
    layer **nextLayers;
    float *currLayerWeights;
    float *currLayerGradients;
    int numNodes;
    int numPrevLayers;
    int numNextLayers;
    int layer_id;    
    char activation;
    bool norm;
};

float relu(float x)
{
    // Leaky ReLU
    return (x > 0 ? x : 0.01*x);
}

float tanh(float x)
{
    float x2 = x * x;
    return 0.5 + (x * (27.0 + x2) / (54.0 + 18.0 * x2));
}

float relu_derivative(float x)
{
    // Leaky ReLU derivative
    return (x > 0) ? 1 : 0.01;
}

float tanh_derivative(float x)
{
    float x2 = x * x;
    return (((x * x2) + (9 * x2) + (27 * x) + 27)/((18 * x2) + 54));
}

float findMax(float* arr, int size)
{
    float currMax = arr[0];

    for(int i = 0; i < size; i++)
    {
        currMax = arr[i] > currMax ? arr[i] : currMax;
    }

    return currMax;
}

float findMin(float* arr, int size)
{
    float currMin = arr[0];

    for(int i = 0; i < size; i++)
    {
        currMin = arr[i] < currMin ? arr[i] : currMin;
    }

    return currMin;
}

void minMaxNorm(float* arr, int size)
{
    float min = findMin(arr, size);
    float diff = size > 1 ? findMax(arr, size) - min : 1;

    for(int i = 0; i < size; i++)
    {
        arr[i] = (arr[i] - min) / diff;
    }
}

layer* make_input_layer(int numNodes, int numNextLayers, int layer_id, bool norm)
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
    inLayer->nextLayers = (struct layer**)calloc(numNextLayers, sizeof(layer*));
    if(inLayer->nextLayers == NULL)
    {
        goto error1;
    }

    // Input layer just accepts inputs, doesn't need weights
    inLayer->currLayerWeights = (float*)calloc(numNodes, sizeof(float));
    if(inLayer->currLayerWeights == NULL)
    {
        goto error2;
    }

    inLayer->currLayerGradients = (float*)calloc(numNodes, sizeof(float));
    if(inLayer->currLayerGradients == NULL)
    {
        goto error3;
    }

    inLayer->numNodes = numNodes;
    inLayer->activation = 'i';
    inLayer->layer_id = layer_id;
    inLayer->norm = norm;

    return inLayer;

error3:
    free(inLayer->currLayerWeights);
    inLayer->currLayerGradients = NULL;
error2:
    free(inLayer->nextLayers);
    inLayer->nextLayers = NULL;
error1:
    free(inLayer);
    inLayer = NULL;

    return NULL;
}

layer* make_dense_layer(layer** prev, int numNodes, int numPrevLayers, int numNextLayers, int layer_id, bool norm)
{
    int j = 0;

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

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING CAREFULLY
    denseLayer->prevLayers = (struct layer **)malloc(sizeof(layer*) * numPrevLayers);
    if(denseLayer->prevLayers == NULL)
    {
        goto error1;
    }

    // Set the previous layers as the previous layers
    memcpy(denseLayer->prevLayers, prev, sizeof(layer*) * numPrevLayers);

    // Make this layer a next layer for all previous layers
    for(int i = 0; i < numPrevLayers; i++)
    {
        while(prev[i]->nextLayers[j] != NULL)
        {
            j += 1;
        }

        prev[i]->nextLayers[j] = denseLayer;
        j = 0;
    }

    // Allocate space for the next layers using provided parameter
    denseLayer->nextLayers = (struct layer **)calloc(numNextLayers, sizeof(layer*));
    if(denseLayer->nextLayers == NULL)
    {
        goto error2;
    }

    denseLayer->currLayerWeights = (float *)malloc((numNodes + 1) * sizeof(float)); // Neuron weights plus a bias weight
    if(denseLayer->currLayerWeights == NULL)
    {
        goto error3;
    }

    denseLayer->currLayerGradients = (float *)calloc((numNodes + 1) * sizeof(float)); // Neuron weights plus a bias weight
    if(denseLayer->currLayerGradients == NULL)
    {
        goto error4;
    }

    for(int i = 0; i < (numNodes + 1); i++)
    {
        denseLayer->currLayerWeights[i] = 0; // Fix later to randomly initialize
    }

    denseLayer->numNodes = numNodes;
    denseLayer->activation = 'r';
    denseLayer->layer_id = layer_id;
    denseLayer->norm = norm;

    return denseLayer;

error4:
    free(denseLayer->currLayerWeights);
    denseLayer->currLayerWeights = NULL;
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

layer* make_output_layer(layer** prev, int numNodes, int numPrevLayers, int layer_id)
{
    int j = 0;
    struct layer *outLayer = (struct layer *)malloc(sizeof(struct layer));
    if(outLayer == NULL)
    {
        goto error1;
    }

    outLayer->numPrevLayers = numPrevLayers;
    outLayer->numNextLayers = 0;
    outLayer->nextLayers = NULL; // No next layers for an output layer

    // Allocate space for the previous layers using provided parameter - DESIGN YOUR MODEL BEFORE IMPLEMENTING
    outLayer->prevLayers = (struct layer **)malloc(sizeof(layer *) * numPrevLayers);
    if(outLayer->prevLayers == NULL)
    {
        goto error1;
    }

    memcpy(outLayer->prevLayers, prev, sizeof(layer*) * numPrevLayers);

    for(int i = 0; i < numPrevLayers; i++)
    {
        while(prev[i]->nextLayers[j] != NULL)
        {
            j += 1;
        }
        prev[i]->nextLayers[j] = outLayer;
        j = 0;
    }

    outLayer->currLayerWeights = (float *)malloc((numNodes + 1) * sizeof(float)); // Neuron weights plus a bias weight
    if(outLayer->currLayerWeights == NULL)
    {
        goto error2;
    }

    outLayer->currLayerGradients = (float *)calloc((numNodes + 1) * sizeof(float)); // Neuron weights plus a bias weight
    if(outLayer->currLayerGradients == NULL)
    {
        goto error3;
    }

    for(int i = 0; i < (numNodes + 1); i++)
    {
        outLayer->currLayerWeights[i] = 0; // Fix later to randomly initialize
    }

    outLayer->numNodes = numNodes;
    outLayer->activation = 't';
    outLayer->layer_id = layer_id;
    outLayer->norm = false;

    return outLayer;

error3:
    free(outLayer->currLayerWeights);
    outLayer->currLayerWeights = NULL;
error2:
    free(outLayer->prevLayers);
    outLayer->prevLayers = NULL;
error1:
    free(outLayer);
    outLayer = NULL;

    return NULL;
}

/*
layer* make_normalization_layer(layer* prev, int numNextLayers = 1)
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

    normLayer->nextLayers = (layer **)malloc(sizeof(layer*)*numNextLayers);
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

// Layer loading
void load_layer(layer* layer, float weights[])
{
    //Size of weights array must match number of nodes in inLayer
    for(int i = 0; i < (layer->numNodes + 1); i++)
    {
        memcpy(layer->currLayerWeights, weights, (layer->numNodes + 1));
    }
}

// Works with a single layer to get an output
float layer_forward(layer* layer, float x)
{
    float forwardVal = 0;

    for(int i = 0; i < layer->numNodes; i++)
    {
        forwardVal += layer->currLayerWeights[i] * x;
    }

    forwardVal += layer->currLayerWeights[layer->numNodes];

    switch(layer->activation)
    {
        case 'r':
            return relu(forwardVal);
        case 't':
            return tanh(forwardVal);
        case 'i':
            return forwardVal;
        default:
            return forwardVal;
    }    
}

float* layer_backward(layer* layer, float* gradients)
{

}

// Destroy an individual layer after operations are concluded
void hakai_layer(layer* lay)
{
    free(lay->currLayerWeights);
    free(lay->nextLayers);
    free(lay->prevLayers);
    free(lay);
}

#endif