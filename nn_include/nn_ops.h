#ifndef NN_OPS
#define NN_OPS
#include <immintrin.h>
#include "model_ops.h"

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
                for(int k = 0; k < layer->prevLayers[j]->numNodes; k++) layer->outputs[i] += layer->prevLayers[j]->activations[k] * layer->weights[i][numPrevsTraversed + k];
                numPrevsTraversed += layer->prevLayers[j]->numNodes;
            }
            layer->outputs[i] += layer->biases[i];
            numPrevsTraversed = 0;
        } 
    }

    switch(layer->activationFunction)
    {
        case 'r':
            for(int i = 0; i < layer->numNodes; i++) layer->activations[i] = leaky_relu(layer->outputs[i]);
            break;
        case 't':
            for(int i = 0; i < layer->numNodes; i++) layer->activations[i] = tanh(layer->outputs[i]);
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
        // backErrorsForOutputLayer = mseLossDerivative · activationFunctionDerivative(outputs) - for output layer
        for(int i = 0; i < layer->numNodes; i++) layer->backErrors[i] = -1 * mse_loss_derivative_func(myModel->targets[i], layer->activations[i]) * tanh_derivative(layer->outputs[i]);
    }
    
    // backErrorsForPreviousLayers += (thisLayersBackErrors)(thisLayersWeightMatrixWithRespectToCurrentPreviousLayer) · activationFunctionDerivative(previousLayers)
    int prevsTraversed = 0;
    for(int i = 0; i < layer->numPrevLayers; i++)
    {
        if(layer->prevLayers[i]->numPrevLayers == 0) continue;
        for(int j = 0; j < layer->prevLayers[i]->numNodes; j++)
        {
            for(int k = 0; k < layer->numNodes; k++) layer->prevLayers[i]->backErrors[j] += layer->backErrors[k] * layer->weights[k][prevsTraversed + j] * leaky_relu_derivative(layer->prevLayers[i]->outputs[j]);                
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
            for(int k = 0; k < layer->numNodes; k++) layer->weights[k][j + prevsTraversed] -= learningRate * layer->prevLayers[i]->activations[j] * layer->backErrors[k];
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
    memset(layer->outputs, 0.0f, layer->numNodes * sizeof(float));
    memset(layer->activations, 0.0f, layer->numNodes * sizeof(float));
}


#endif