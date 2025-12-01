#ifndef NN_OPS
#define NN_OPS
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
        float lossGrad = 0;
        int prevsTraversed = 0;

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
            prevsTraversed += layer->prevLayers[i]->numNodes;                
        }
    }

    for(int i = 0; i < layer->numPrevLayers; i++) if(layer->prevLayers[i]->numPrevLayers != 0) sgd_backprop(layer->prevLayers[i], myModel);
    // calculate backErrors for previous layers' previous layers according to already established layers' backErrors - All roads spring forth from Rome
}

void calculate_and_apply_grads(struct layer* layer, struct model* myModel)
{
    if(layer->switchVar == '3') return;

    layer->switchVar = '3';

    for(int i = 0; i < layer->numPrevLayers; i++) calculate_and_apply_grads(layer->prevLayers[i], myModel);

    if(layer->numPrevLayers == 0) return;

    // calculate gradients from backerrors and activations for each layer and apply them to the weights
}

void zero_everything(struct layer* layer)
{
    if(layer->switchVar == '0') return;

    layer->switchVar = '0';

    for(int i = 0; i < layer->numPrevLayers; i++) zero_everything(layer->prevLayers[i]);
    
    for(int i = 0; i < layer->numNodes; i++)
    {
        for(int j = 0; j < layer->numPrevNodes; j++) 
        {
            layer->weights[i][j] = 0;
            layer->backErrors[i] = 0;
        }

        layer->activations[i] = 0;
        layer->outputs[i] = 0;
    }
}


#endif