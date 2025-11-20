#ifndef NN_LAYER
#define NN_LAYER

#include <stdlib.h>
#include <string.h>

#define LAYER_SIZE 56

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

// 56 Bytes
struct layer
{
    struct layer **prevLayers;
    struct layer **nextLayers;
    float **currLayerWeights;
    float **currLayerGradients;
    int numPrevNodes;
    int numNodes;
    int numPrevLayers;
    int numNextLayers;
    int layer_id;    
    char activation;
};


#endif