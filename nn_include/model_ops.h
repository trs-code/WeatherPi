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
    for(int i = 0; i < numLayers; i++) myModel->layer_refs[i] = NULL;

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

// Destroy an individual layer after operations are concluded but a model hasn't been built yet
void hakai_layer_mfree(struct layer* layer)
{    
    free(layer->outputs);
    layer->outputs = NULL;

    free(layer->biases);
    layer->biases = NULL;

    if(layer->activationFunction == 'i') goto free;
    
    free(layer->backErrors);
    layer->backErrors = NULL;

    free(layer->prevLayers);
    layer->prevLayers = NULL;

    free(layer->preActivations);
    layer->preActivations = NULL;

    hakai_matrix(layer->weights, layer->numNodes);
    
    // if(layer->numNextLayers == 0) return;

    // free(layer->nextLayers);
    // layer->nextLayers = NULL;

free:
    free(layer);
    layer = NULL;
}

void hakai_model_mfree(struct model* myModel)
{
    free(myModel->inLayers);
    myModel->inLayers = NULL;

    free(myModel->layer_refs);
    myModel->layer_refs = NULL;

    free(myModel->targets);
    myModel->targets = NULL;

    free(myModel);
    myModel = NULL;
}

// Destroy an individual layer after operations are concluded
void hakai_layer(struct layer* layer, struct model* myModel)
{
    if(layer == NULL) return;
    if(myModel->layer_refs[layer->layerID] != NULL) return;

    free(layer->backErrors);
    layer->backErrors = NULL;

    free(layer->preActivations);
    layer->preActivations = NULL;

    free(layer->outputs);
    layer->outputs = NULL;

    free(layer->biases);
    layer->biases = NULL;

    if(layer->activationFunction != 'i') hakai_matrix(layer->weights, layer->numNodes);

    // free(layer->nextLayers);
    // layer->nextLayers = NULL;
    
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
        
        
        //myModel->layer_refs[layerArr[i]->layerID] = layerArr[i];
        
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

//  Gets an output from the target layer
void forward_out(struct layer* layer)
{
    if(layer->switchVar == '1') return;

    layer->switchVar = '1';

    if(layer->numPrevLayers != 0)
    {
        int numPrevsTraversed = 0;
        
        for(int i = 0; i < layer->numPrevLayers; i++) forward_out(layer->prevLayers[i]);

        for(int i = 0; i < layer->numNodes; i++) 
        {
            for(int j = 0; j < layer->numPrevLayers; j++)
            {
                for(int k = 0; k < layer->prevLayers[j]->numNodes; k++) layer->preActivations[i] += layer->prevLayers[j]->outputs[k] * layer->weights[i][numPrevsTraversed + k];
                numPrevsTraversed += layer->prevLayers[j]->numNodes;
            }
            layer->preActivations[i] += layer->biases[i];
            numPrevsTraversed = 0;
        } 
    }

    switch(layer->activationFunction)
    {
        case 'r':
            for(int i = 0; i < layer->numNodes; i++) layer->outputs[i] = leaky_relu(layer->preActivations[i]);
            break;
        case 't':
            for(int i = 0; i < layer->numNodes; i++) layer->outputs[i] = tanh(layer->preActivations[i]);
            break;
        default:
            break;
    }
}

// Run on each output layer and then apply grads before clearing the layer backerrors - All roads spring forth from Rome algorithm
// We pass the backerrors to each previous layer to calculate grads later
// Backerrors can be accumulated from multiple successor layers to calculate grads due to matrix distributivity
void sgd_backprop(struct layer* layer, struct model* myModel)
{ // start at output layer and calculate backerrors for each previous layer
    if(layer->switchVar == '2') return;

    layer->switchVar = '2';

    if(layer->numNextLayers == 0)
    {
        // backErrorsForOutputLayer = mseLossDerivative · activationFunctionDerivative(preActivations) - for output layer
        for(int i = 0; i < layer->numNodes; i++) layer->backErrors[i] = -1 * mse_loss_derivative_func(myModel->targets[i], layer->outputs[i]) * tanh_derivative(layer->preActivations[i]);
    }
    
    // backErrorsForPreviousLayers += (thisLayersBackErrors)(thisLayersWeightMatrixWithRespectToCurrentPreviousLayer) · activationFunctionDerivative(previousLayers)
    int prevsTraversed = 0;
    for(int i = 0; i < layer->numPrevLayers; i++)
    {
        if(layer->prevLayers[i]->numPrevLayers == 0) continue;
        for(int j = 0; j < layer->prevLayers[i]->numNodes; j++)
        {
            for(int k = 0; k < layer->numNodes; k++) layer->prevLayers[i]->backErrors[j] += layer->backErrors[k] * layer->weights[k][prevsTraversed + j] * leaky_relu_derivative(layer->prevLayers[i]->preActivations[j]);                
        }
        prevsTraversed += layer->prevLayers[i]->numNodes;
    }

    for(int i = 0; i < layer->numPrevLayers; i++) if(layer->prevLayers[i]->numPrevLayers != 0) sgd_backprop(layer->prevLayers[i], myModel);
    // calculate backErrors for previous layers' previous layers according to already established layers' backErrors - All roads spring forth from Rome
}

// Another all roads spring forth from Rome approach - go to the convergence point of the model and use it as the root of the undirected, cyclic graph that is this model
void calculate_and_apply_grads(struct layer* layer, float learningRate)
{
    if(layer->switchVar == '3') return;

    layer->switchVar = '3';

    if(layer->numPrevLayers == 0) return;

    for(int i = 0; i < layer->numPrevLayers; i++) calculate_and_apply_grads(layer->prevLayers[i], learningRate);

    for(int i = 0; i < layer->numNodes; i++) layer->biases[i] -= learningRate * layer->backErrors[i];

    int prevsTraversed = 0;

    // calculate gradients from backerrors and activations for each layer and apply them to the weights_mm256_mul_ps(vector_a, scalar_vector);
    for(int i = 0; i < layer->numPrevLayers; i ++)
    {
        for(int j = 0; j < layer->prevLayers[i]->numNodes; j++)
        {
            for(int k = 0; k < layer->numNodes; k++) layer->weights[k][j + prevsTraversed] -= learningRate * layer->prevLayers[i]->outputs[j] * layer->backErrors[k];
        }
        prevsTraversed += layer->prevLayers[i]->numNodes;
    }
}

// For clearing the backErrors once no longer needed, and to also prime for next forward and backward pass
// Use by passing the output layer of the model into the function 
void zero_everything(struct layer* layer)
{
    if(layer->switchVar == '0') return;

    layer->switchVar = '0';

    if(layer->numPrevLayers != 0)
    {
        for(int i = 0; i < layer->numPrevLayers; i++) zero_everything(layer->prevLayers[i]);
    }
    else return;

    memset(layer->backErrors, 0.0f, layer->numNodes * sizeof(float));
    memset(layer->preActivations, 0.0f, layer->numNodes * sizeof(float));
    memset(layer->outputs, 0.0f, layer->numNodes * sizeof(float));
}

int save_model(struct model* saveMod, FILE* modelFile);

struct model* load_model(char* filename[]);




#endif