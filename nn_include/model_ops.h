#ifndef NN_MODEL_OPS
#define NN_MODEL_OPS

#include "model.h"
#include "layer_ops.h"

struct model* construct_model(int numLayers, int numInLayers, float learning_rate, int numOutputs)
{
    struct model *myModel = (struct model*)malloc(sizeof(struct model));
    if(myModel == NULL)
    {
        return NULL;
    }

    myModel->layer_ids = (int *)calloc(numLayers, sizeof(int));
    if(myModel->layer_ids == NULL)
    {
        goto error1;
    }

    myModel->layer_outs = (float *)calloc(numLayers, sizeof(float));
    if(myModel->layer_ids == NULL)
    {
        goto error2;
    }

    myModel->inLayers = (struct layer **)calloc(numInLayers, sizeof(struct layer*));
    if(myModel->inLayers == NULL)
    {
        goto error3;
    }

    myModel->layer_refs = (struct layer **)calloc(numLayers, sizeof(struct layer*));
    if(myModel->layer_refs == NULL)
    {
        goto error4;
    }

    myModel->targets = (float *)calloc(numOutputs, sizeof(float));
    if(myModel->targets == NULL)
    {
        goto error5;
    }

    myModel->model_outs = (float *)calloc(numOutputs, sizeof(float));
    if(myModel->targets == NULL)
    {
        goto error6;
    }


    myModel->numLayers = numLayers;
    myModel->learning_rate = learning_rate;
    myModel->numInLayers = numInLayers;
    myModel->numOutputs = numOutputs;

    return myModel;

error6:
    free(myModel->targets);
    myModel->targets = NULL;
error5:
    free(myModel->layer_refs);
    myModel->layer_refs = NULL;
error4:
    free(myModel->inLayers);
    myModel->inLayers = NULL;
error3:
    free(myModel->layer_outs);
    myModel->layer_outs = NULL;
error2:
    free(myModel->layer_ids);
    myModel->layer_outs = NULL;
error1:
    free(myModel);
    myModel = NULL;

    return NULL;
}


// For clearing the outputs once no longer needed, and to also prime for next forward pass
void hakai_layer_outs(struct model* myModel)
{
    for(int i = 0; i < (myModel->numLayers); i++)
    {
        myModel->layer_outs[i] = 0;
    }
}

// Destroy an individual layer after operations are concluded
void hakai_layer(struct layer* layer, struct model* myModel)
{
    int i = 0;
    
    if(myModel->layer_ids[layer->layer_id] == 0)
    {
        return;
    }

    free(layer->currLayerGradients);
    layer->currLayerGradients = NULL;

    hakai_weight_matrix(layer->currLayerWeights);

    free(layer->nextLayers);
    layer->nextLayers = NULL;
    
    //free(lay->prevLayers);
    layer->prevLayers = NULL;

    myModel->layer_ids[layer->layer_id] = 0;

    while(myModel->layer_refs[i] != NULL)
    {
        myModel->layer_refs[i] = layer;
        i += 1;
    }

}

// Enter this function with the outArray of the model and let it do its thing
// One big advantage of the doubly linked list structure is being able to exploit the convergence of the model on the output layer
void clear_model(struct layer** layerArr, struct model* myModel, int layerNums)
{
    if(layerArr == NULL)
    {
        return;
    }
    for(int i = 0; i < layerNums; i++)
    {
        if(myModel->layer_ids[layerArr[i]->layer_id] == 0)
        {
            continue;
        }
        clear_model(layerArr[i]->prevLayers, myModel, layerArr[i]->numPrevLayers);
        hakai_layer(layerArr[i], myModel);
        myModel->layer_ids[layerArr[i]->layer_id] = 0;
        layerArr[i] = NULL;
    }
    free(layerArr);
    layerArr = NULL;
}

void hakai_model(struct model* myModel)
{
    struct layer **outArr = (struct layer**)malloc(sizeof(struct layer*));
    if(outArr == NULL)
    {
        return;
    }

    outArr[0] = myModel->outLayer;
    clear_model(outArr, myModel, myModel->numLayers);
    
    free(myModel->layer_outs);
    myModel->layer_outs = NULL;

    free(myModel->layer_ids);
    myModel->layer_ids = NULL;

    for(int i = myModel->numLayers - 1; i >= 0; i--)
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


void load_target_values(float);

int save_model(struct model* saveMod);

struct model* load_model(char* filename[]);




#endif