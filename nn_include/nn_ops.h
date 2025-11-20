#ifndef NN_OPS
#define NN_OPS
#include "model_ops.h"
#include "nn_math.h"

// Layer loading
void load_layer(struct layer* layer, float values[])
{
    // Size of weights array must match number of nodes in inLayer
    memcpy(layer->currLayerWeights, values, sizeof(float)*(layer->numNodes));
}

// For clearing the gradients once no longer needed, and to also prime for next forward pass
// Use by passing the output layer of the model into the function 
void hakai_layer_grads(struct layer* layer)
{
    if(absolute(layer->currLayerGradients[0]) - 0 > 0.0000001)
    {
        return;
    }

    if(layer->numNextLayers != 0)
    {
        for(int i = 0; i < layer->numNextLayers; i++)
        {
            hakai_layer_grads(layer->nextLayers[i]);
        }
    }

    memset(layer->currLayerGradients, 0, layer->numNodes * sizeof(float));
}


//  Gets an output from the target layer
float forward_out(struct layer* layer, struct model* myModel)
{
    float x = 0;

    if(absolute(myModel->layer_outs[layer->layer_id]) - 0 > 0.00000001)
    {
        return myModel->layer_outs[layer->layer_id];
    }

    if(layer->numPrevLayers != 0)
    {
        for(int i = 0; i < layer->numPrevLayers; i++)
        {
            x += forward_out(layer->prevLayers[i], myModel);
        }
        
    }
    else
    {
        x = 1.0;
    }

    if(layer->numNextLayers != 0)
    {
        float layerOut = 0;
        for(int i = 0; i < layer->numNodes; i++)
        {
            layerOut += layer->currLayerWeights[i][0] * x;
        }
        
        myModel->layer_outs[layer->layer_id] = layerOut;

        layerOut += layer->currLayerWeights[layer->numNodes][0];

        switch(layer->activation)
        {
            case 'r':
                layerOut = relu(layerOut);
                break;
            case 't':
                layerOut = tanh(layerOut);
                break;
            case 'i':
                break;
            default:
                return layerOut;
        }

        return layerOut;
    }
    else
    {
        for(int i = 0; i < layer->numNodes; i++)
        {
            myModel->model_outs[i] = layer->currLayerWeights[i][0] * x;
        }
        return 0.0;
    }
}

// Run on each input layer before clearing the layer gradients
void sgd_backprop(struct layer* layer, struct model* myModel)
{
    if(absolute(layer->currLayerGradients[0]) - 0 > 0.0000001)
    {
        return;
    }

    if(layer->numNextLayers != 0)
    {
        for(int i = 0; i < layer->numNextLayers; i++)
        {
            struct layer* next = layer->nextLayers[i];

            sgd_backprop(layer->nextLayers[i], myModel);

            for(int j = 0; j < layer->numNodes; j++)
            {
                for(int k = 0; k < layer->nextLayers[i]->numNodes; k++)
                {
                    layer->currLayerGradients[j] += layer->nextLayers[i]->currLayerGradients[k] * layer->nextLayers[i]->currLayerWeights[i * layer->nextLayers[i]->numNodes + k][0];
                }
            }
        }

        for(int i = 0; i < layer->numNodes; i++)
        {
            switch(layer->activation)
            {
                case 'r':
                    layer->currLayerGradients[i] *= leaky_relu_derivative(layer->currLayerGradients[i]);
                    break;
                case 't':
                    layer->currLayerGradients[i] *= tanh_derivative(layer->currLayerGradients[i]);
                    break;
                case 'i':
                    layer->currLayerGradients[i] *= 0;
                    break;
                default:
                    break;
            }
        }
    }
    else
    {
        for(int i = 0; i < layer->numNodes; i++)
        {
            float nodeLossDerivative = mse_loss_derivative(myModel->targets[i], myModel->model_outs[i]);
        }
    }
    
    for(int i = 0; i < layer->numPrevLayers; i++) for(int j = 0; j < layer->prevLayers[i]->numNodes; j++) layer->currLayerWeights[i][j] -= myModel->learning_rate * layer->currLayerGradients[i];
}



#endif