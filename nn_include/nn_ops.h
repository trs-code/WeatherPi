#ifndef NN_OPS
#define NN_OPS
#include "model_ops.h"

//  Gets an output from the target layer
void forward_out(struct layer* layer)
{
    if(absolute(layer->activations[0]) - 0 > 0.00000001)
    {
        return;
    }

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

// Run on each input layer before clearing the layer gradients
// void sgd_backprop(struct layer* layer, struct model* myModel)
// {
//     if(absolute(layer->gradients[0][0]) - 0 > 0.0000001)
//     {
//         return;
//     }

//     if(layer->numNextLayers != 0)
//     {
//         for(int i = 0; i < layer->numNextLayers; i++)
//         {
//             struct layer* next = layer->nextLayers[i];

//             sgd_backprop(layer->nextLayers[i], myModel);

//             for(int j = 0; j < layer->numNodes; j++)
//             {
//                 for(int k = 0; k < layer->prevLayers[i]->numNodes; k++)
//                 {
//                     layer->gradients[j] += 0;
//                 }
//             }
//         }

//         for(int i = 0; i < layer->numNodes; i++)
//         {
//             switch(layer->activationFunction)
//             {
//                 case 'r':
//                     layer->gradients[i][0] *= leaky_relu_derivative(layer->gradients[i][0]);
//                     break;
//                 case 't':
//                     layer->gradients[i][0] *= tanh_derivative(layer->gradients[i][0]);
//                     break;
//                 case 'i':
//                     layer->gradients[i][0] *= 0;
//                     break;
//                 default:
//                     break;
//             }
//         }
//     }
//     else
//     {
//         for(int i = 0; i < layer->numNodes; i++)
//         {
//             float nodeLossDerivative = mse_loss_derivative(myModel->targets[i], myModel->outLayer->outputs[i]);
//         }
//     }
    
//     for(int i = 0; i < layer->numPrevLayers; i++) for(int j = 0; j < layer->prevLayers[i]->numNodes; j++) layer->weights[i][j] -= myModel->learning_rate * layer->gradients[i][i];
// }



#endif