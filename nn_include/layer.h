#ifndef NN_LAYER
#define NN_LAYER

#include <stdlib.h>
#include <string.h>
#include <time.h>

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

// 72 Bytes to allocate for the structure at base
// Extra # bytes for each layer: 40 + 8m + 8n + 4np + 16n  -> m previous layers, n current nodes, p previous nodes
struct layer
{
    struct layer **prevLayers; // Very necessary to operate model
    // struct layer **nextLayers; // Might not be needed for actual operation of the model
    float **weights; // n nodes * p previous nodes - weight matrix
    float *biases;
    float *backErrors; // Only necessary for backpropagation, not necessary for an inference model - n values
    float *outputs; // Activation value passed through activation function, output of the node that is passed forward - n values
    float *preActivations; // Sum of all previous nodes according to each previous node weight - n values
    int numPrevNodes; // Helps set up the model and also operate it
    int numNodes; // Helps set up the model and also operate it
    int numPrevLayers; // Very necessary for all roads spring forth from rome approach - helps operate the model
    int numNextLayers; // Only needed for backpropagation - not needed for an inference model
    int layerID; // A unique number from [0, (# of layers in the Model) - 1] - maybe made redundant through switchVar?  
    char activationFunction;
    char switchVar;
};

#endif