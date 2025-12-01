#ifndef NN_MODEL_OPS
#define NN_MODEL_OPS

#include "model.h"

struct model* construct_model(int numLayers, int numInLayers, float learning_rate, int numOutputs)
{
    struct model *myModel = (struct model*)malloc(sizeof(struct model));
    if(myModel == NULL) return NULL;

    myModel->inLayers = (struct layer **)calloc(numInLayers, sizeof(struct layer*));
    if(myModel->inLayers == NULL) goto error1;

    myModel->layer_refs = (struct layer **)calloc(numLayers, sizeof(struct layer*));
    if(myModel->layer_refs == NULL) goto error2;

    myModel->targets = (float *)calloc(numOutputs, sizeof(float));
    if(myModel->targets == NULL) goto error3;

    myModel->numLayers = numLayers;
    myModel->learning_rate = learning_rate;
    myModel->numInLayers = numInLayers;
    myModel->numOutputs = numOutputs;

    return myModel;

error3:
    free(myModel->layer_refs);
    myModel->layer_refs = NULL;
error2:
    free(myModel->inLayers);
    myModel->inLayers = NULL;
error1:
    free(myModel);
    myModel = NULL;

    return NULL;
}

// Destroy an individual layer after operations are concluded
void hakai_layer(struct layer* layer, struct model* myModel)
{
    if(myModel->layer_refs[layer->layerID] != NULL) return;

    free(layer->backErrors);
    layer->backErrors = NULL;

    free(layer->outputs);
    layer->outputs = NULL;

    free(layer->activations);
    layer->activations = NULL;

    hakai_matrix(layer->weights);

    free(layer->nextLayers);
    layer->nextLayers = NULL;
    
    myModel->layer_refs[layer->layerID] = layer;

}

// Enter this function with the outArray of the model and let it do its thing
// One big advantage of the doubly linked list structure is being able to exploit the convergence of the model on the output layer
void clear_model(struct layer** layerArr, struct model* myModel, int layerNums)
{
    if(layerArr == NULL) return;

    for(int i = 0; i < layerNums; i++)
    {
        if(myModel->layer_refs[layerArr[i]->layerID] != NULL) continue;
        
        clear_model(layerArr[i]->prevLayers, myModel, layerArr[i]->numPrevLayers);
        hakai_layer(layerArr[i], myModel);
        
        layerArr[i] = NULL;
    }
    
    free(layerArr);
    layerArr = NULL;
}

void hakai_model(struct model* myModel)
{
    struct layer **outArr = (struct layer**)malloc(sizeof(struct layer*));
    if(outArr == NULL) return;
    
    outArr[0] = myModel->outLayer;
    clear_model(outArr, myModel, 1);

    free(myModel->inLayers);
    myModel->inLayers = NULL;

    for(int i = 0; i < myModel->numLayers; i++)
    {
        free(myModel->layer_refs[i]);
        myModel->layer_refs[i] = NULL;
    }

    free(myModel->layer_refs);
    myModel->layer_refs = NULL;

    free(myModel->targets);
    myModel->targets = NULL;

    free(myModel);
    myModel = NULL;
}

int save_model(struct model* saveMod);

struct model* load_model(char* filename[]);




#endif