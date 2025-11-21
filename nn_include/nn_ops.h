#ifndef NN_OPS
#define NN_OPS
#include "model_ops.h"
#include "nn_math.h"

// For clearing the gradients once no longer needed, and to also prime for next backward pass
// Use by passing the output layer of the model into the function 
void hakai_layer_grads(struct layer* layer)
{
    if(absolute(layer->gradients[0][0]) - 0.0f > 0.0000001) return;

    if(layer->numNextLayers != 0)
    {
        for(int i = 0; i < layer->numNextLayers; i++) hakai_layer_grads(layer->nextLayers[i]);
    }

    for(int i = 0; i < layer->numNodes; i++) memset(layer->gradients[i], 0, layer->numPrevNodes * sizeof(float));
}


//  Gets an output from the target layer
float* forward_out(struct layer* layer, struct model* myModel)
{
    float x = 0;

    if(absolute(myModel->outLayer->outputs[0]) - 0 > 0.00000001)
    {
        return myModel->outLayer;
    }

    if(layer->numPrevLayers != 0)
    {
        for(int i = 0; i < layer->numPrevLayers; i++)
        {
            x += forward_out(layer->prevLayers[i], myModel)[0];
        }
        
    }
    else
    {
        x = 1.0;
    }

    if(layer->numNextLayers != 0)
    {
        float layerOut[] = {0};
        for(int i = 0; i < layer->numNodes; i++)
        {
            layerOut[0] += layer->weights[i][0] * x;
        }
        

        layerOut[0] += layer->weights[layer->numNodes][0];

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
            myModel->outLayer->outputs[i] = layer->weights[i][0] * x;
        }
        return {0.0};
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
            float nodeLossDerivative = mse_loss_derivative(myModel->targets[i], myModel->outLayer->outputs[i]);
        }
    }
    
    for(int i = 0; i < layer->numPrevLayers; i++) for(int j = 0; j < layer->prevLayers[i]->numNodes; j++) layer->weights[i][j] -= myModel->learning_rate * layer->gradients[i][i];
}



#endif