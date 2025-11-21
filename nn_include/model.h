#ifndef NN_MODEL
#define NN_MODEL

#include "layer_ops.h"

// 48 Bytes for an empty model husk - update this after training testing is complete
struct model
{   
    struct layer **inLayers;    // References to the input layers of the model - entry point for model operations
    struct layer **layer_refs;  // References for each hung layer while they are getting deconstructed
    struct layer *outLayer;     // References the output layers of the model - entry point for model operations
    float *targets;             // target values for the current training iteration
    float learning_rate;        // Learning rate for the NN
    int numLayers;              // Number of total layers in the NN
    int numInLayers;            // Number of input layers in the NN
    int numOutputs;             // Number of outputs in the model
};

#endif