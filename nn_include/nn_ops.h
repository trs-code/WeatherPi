#ifndef NN_OPS
#define NN_OPS
#include "model_ops.h"
#include "nn_math.h"

// For clearing the gradients once no longer needed, and to also prime for next forward pass
// Use by passing the output layer of the model into the function 
void hakai_layer_grads(struct layer* layer)
{
    if(absolute(layer->gradients[0][0]) - 0 > 0.0000001)
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

    memset(layer->gradients, 0, layer->numNodes * sizeof(float));
}


//  Gets an output from the target layer
float forward_out(struct layer* layer, struct model* myModel)
{
    float x = 0;

    if(absolute(myModel->model_outs[layer->layer_id]) - 0 > 0.00000001)
    {
        return myModel->model_outs[layer->layer_id];
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
            layerOut += layer->weights[i][0] * x;
        }
        
        myModel->model_outs[layer->layer_id] = layerOut;

        layerOut += layer->weights[layer->numNodes][0];

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
            myModel->model_outs[i] = layer->weights[i][0] * x;
        }
        return 0.0;
    }
}

// Run on each input layer before clearing the layer gradients
void sgd_backprop(struct layer* layer, struct model* myModel)
{
    if(absolute(layer->gradients[0][0]) - 0 > 0.0000001)
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
                for(int k = 0; k < layer->prevLayers[i]->numNodes; k++)
                {
                    layer->gradients[j] += 0;
                }
            }
        }

        for(int i = 0; i < layer->numNodes; i++)
        {
            switch(layer->activation)
            {
                case 'r':
                    layer->gradients[i][0] *= leaky_relu_derivative(layer->gradients[i][0]);
                    break;
                case 't':
                    layer->gradients[i][0] *= tanh_derivative(layer->gradients[i][0]);
                    break;
                case 'i':
                    layer->gradients[i][0] *= 0;
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
    
    for(int i = 0; i < layer->numPrevLayers; i++) for(int j = 0; j < layer->prevLayers[i]->numNodes; j++) layer->weights[i][j] -= myModel->learning_rate * layer->gradients[i][i];
}



#endif