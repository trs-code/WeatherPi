#pragma once


//                                         VERY IMPORTANT
//                           MAKE SURE AN EXIT PROCESS FREES THE MEMORY FOR
//
//                                              EVERY
//                                              SINGLE
//                                              LAYER
//
// STARTING FROM THE OUTPUT AND ITERATE THROUGH THE LINKED TREE FREEING THE MEMORY FOR EACH LAYER WEIGHTS AND LAYER
//                                      WHEN OPERATIONS ARE CONCLUDED
// I tried my best to build the functionality for this into the library itself, but the possibility of human error is NEVER zero


// 66 Bytes to allocate for the structure at base - likely padded by compiler with 6 extra bytes to be 72 bytes total
// Extra # bytes for each layer: 8m + 4np + 16n  -> m previous layers, n current nodes, p previous nodes

typedef struct layer layer;
struct layer
{
    layer*** prevLayers; // Very necessary to operate model - array of pointers(memory addresses) to layer allocation pointers
    float** weights; // n nodes * p previous nodes - weight matrix
    float* biases; // n biases - 1 for each node
    float* backErrors; // Only necessary for backpropagation, not necessary for an inference model - n values
    float* outputs; // Activation value passed through activation function, output of the node that is passed forward - n values
    float* preActivations; // Sum of all previous nodes according to each previous node weight - n values
    int numPrevNodes; // Helps set up the model and also operate it
    int numNodes; // Helps set up the model and also operate it
    int numPrevLayers; // Very necessary for all roads spring forth from rome approach - helps operate the model
    int layerID; // A unique number from [0, (# of layers in the Model) - 1] - can be phased out by having layer assignment done directly but honestly it makes debugging easier
    char layerType;
    char activationFunction;
};

