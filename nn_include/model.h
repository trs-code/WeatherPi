#ifndef NN_MODEL
#define NN_MODEL
#include "layer_ops.h"


// 52 Bytes for an empty model husk - update this after training testing is complete
// 16 extra bytes for input layer for the model - numInLayers
// 16 extra bytes for each layer in model - numLayers
struct model
{   
    struct layer **inLayers; // References to the input layers of the model - entry point for model operations
    struct layer **layer_refs; // References for each hung layer while they are getting deconstructed
    struct layer *outLayer; // References the output layers of the model - entry point for model operations
    int *layer_ids; // Checks for the presence of a layer in the model - layer_id[id # of the layer] = 0 if absent 1 if constructed
    float **neuron_inputs; // Reference for output values of models - helps do DP forward pass on DAG graph
    float *targets; // target values for the current training iteration
    float *model_outs; // Outputs of output layer of the model
    float learning_rate; // Learning rate for the NN
    int numLayers; // Number of total layers in the NN
    int numInLayers; // Number of input layers in the NN
    int numOutputs;
};

#endif