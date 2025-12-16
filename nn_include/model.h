#ifndef NN_MODEL
#define NN_MODEL

#include "layer_ops.h"

// 40 Bytes for an empty model husk - update this after training testing is complete
typedef struct
{   
    layer ***inLayers;      // References to the input layers of the model - entry point for model operations, array of pointers to layer allocation pointers
    layer **outLayer;       // References the output layers of the model - entry point for model operations - pointer to layer allocation pointer
    float *targets;         // target values for the current training iteration
    float learning_rate;    // Learning rate for the NN
    int numLayers;          // Number of total layers in the NN
    int numInLayers;        // Number of input layers in the NN
    char loss_fn;
} model;

#endif