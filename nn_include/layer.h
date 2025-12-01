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

// 72 Bytes to allocate for the structure at base - can add 1 more int or float or 2 more chars without adding to size of struct
struct layer
{
    struct layer **prevLayers;
    struct layer **nextLayers;
    float **weights;
    float *backErrors;
    float *activations;
    float *outputs;
    int numPrevNodes;
    int numNodes;
    int numPrevLayers;
    int numNextLayers;
    int layerID; // A unique number from [0, (# of layers in the Model) - 1] - maybe made redundant through switchVar?  
    char activationFunction;
    char switchVar;
};


#endif