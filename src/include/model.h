#pragma once

#include "layer_construct.h"

// 56 Bytes for an empty model husk, likely padded by compiler with three extra bytes to be 56 bytes total
// Extra # of bytes: 8(i + l) + 8n-> i input layers, l total number of layers, and n nodes in output layer
typedef struct
{
    int numLayers;          // Number of total layers in the NN
    int numInLayers;        // Number of input layers in the NN
    layer*** layerList;     // List of all layers in the model - for operations to encourage spatial locality, array of pointers to layer allocation pointers
    layer*** inLayers;      // References to the input layers of the model - entry point for model operations, array of pointers to layer allocation pointers
    layer** outLayer;       // References the output layer of the model - entry point for model operations - pointer to layer allocation pointer
    float* lossDerivatives; // Derivatives of the losses of the output layer (outputs vs. targets)
    float* targets;         // target values for the current training iteration
    float learningRate;     // Learning rate for the NN
    char loss_fn;
    char padding[3];
} model;

