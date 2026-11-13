#pragma once

#include "model.h"

//Loss functions and their derivatives
float mse_loss(model* myModel)
{
    float sum = 0;
    for(int i = 0; i < (*myModel->outLayer)->numNodes; i++) sum += ((myModel->targets[i]) - (*myModel->outLayer)->outputs[i]) * ((myModel->targets[i]) - (*myModel->outLayer)->outputs[i]);
    return (sum / (*myModel->outLayer)->numNodes);
}

float mse_loss_derivative(float target, float yHat, int n)
{
    return 2 * (yHat - target) / n;
}

float mae_loss(model* myModel)
{
    float sum = 0;
    for(int i = 0; i < (*myModel->outLayer)->numNodes; i++) sum += absolute((*myModel->outLayer)->outputs[i] - (myModel->targets[i]));
    return (sum / (*myModel->outLayer)->numNodes);
}

float mae_loss_derivative(float target, float yHat, int n)
{
    return sign(yHat - target) / n;
}

float mbe_loss(model* myModel)
{
    float sum = 0;
    for(int i = 0; i < (*myModel->outLayer)->numNodes; i++) sum += ((*myModel->outLayer)->outputs[i] - (myModel->targets[i]));
    return (sum / (*myModel->outLayer)->numNodes);
}

float mbe_loss_derivative(int n)
{
    return 1 / (float)n;
}

float huber_loss(model* myModel)
{
    float sum = 0.0;
    float error = 0.0;
    for(int i = 0; i < (*myModel->outLayer)->numNodes; i++)
    {
        error = absolute((*myModel->outLayer)->outputs[i] - (myModel->targets[i]));
        if(error > 0.01) sum += 0.01 * (error - 0.005);
        else sum +=  0.5 * error * error;
    }

    return (sum / (*myModel->outLayer)->numNodes);
}

float huber_loss_derivative(float target, float yHat, int n)
{
    float error = yHat - target;
    if(absolute(error) > 0.01) return 0.01 * sign(error);
    else return error / (float)n;
}

float binary_cross_entropy_loss(model* myModel, float* classWeights)
{
    if((*myModel->outLayer)->numNodes > 1) exit(EXIT_FAILURE);
    float sum = 0.0f;
    float eps = 1e-7f;
    float y = myModel->targets[0];
    float yHat = (*myModel->outLayer)->outputs[0];
    if(yHat < eps) yHat = eps;
    if(yHat > 1.0f - eps) yHat = 1.0f - eps;
    
    sum -= (((classWeights) ? classWeights[(y == 0) ? 0 : 1] : 1) * y * logf(yHat)) + (((classWeights) ? classWeights[(y == 0) ? 1 : 0] : 1) * (1.0f-y) * logf(1.0f - yHat));
    
    return (sum);
}

// Erroneous and will never actually be implemented because model construction ensures that BCE Loss is paired with sigmoid output activation to produce cleaner backerror calculation case
// float binary_cross_entropy_loss_derivative(float target, float yHat, float classWeight)
// {
//     float eps = 1e-8f;
//     return (-1 * ( target / (yHat + eps))) + ((1 - target) / (1 - yHat + eps));
// }

float fast_binary_cross_entropy_loss(model* myModel, float* classWeights)
{
    if((*myModel->outLayer)->numNodes > 1) exit(EXIT_FAILURE);
    float sum = 0.0f;
    float eps = 1e-8f;
    float y = myModel->targets[0];
    float yHat = (*myModel->outLayer)->outputs[0];
    if(yHat < eps) yHat = eps;
    if(yHat > 1.0f - eps) yHat = 1.0f - eps;
    
    sum -= (((classWeights) ? classWeights[(y == 0) ? 0 : 1] : 1) * y * fast_ln(yHat)) + (((classWeights) ? classWeights[(y == 0) ? 1 : 0] : 1.0f) * (1.0f-y) * fast_ln(1.0f - yHat));
    
    return (sum);
}

float categorical_cross_entropy_loss(model* myModel, float* classWeights)
{
    float sum = 0;
    
    for(int i = 0; i < (*myModel->outLayer)->numNodes; i++) sum -= ((classWeights) ? classWeights[i] : 1) * (myModel->targets[i]) * logf((*myModel->outLayer)->outputs[i] + 1e-7);
    
    return sum;
}

// Erroneous and will never actually be implemented because model construction ensures that CCE Loss is paired with softmax output activation to produce cleaner backerror calculation case
// float categorical_cross_entropy_loss_derivative(float target, float yHat, float classWeight)
// {
//     return (yHat - target);
// }

float fast_categorical_cross_entropy_loss(model* myModel, float* classWeights)
{
    float sum = 0;
    
    for(int i = 0; i < (*myModel->outLayer)->numNodes; i++) sum -= ((classWeights) ? classWeights[i] : 1) * (myModel->targets[i]) * fast_ln((*myModel->outLayer)->outputs[i] + 1e-7);
    return sum;
}


float loss_function(model* myModel, float* classWeights)
{
    switch(myModel->loss_fn)
    {
        case 'q':
            return mse_loss(myModel);
        case 'a':
            return mae_loss(myModel);
        case 'b':
            return mbe_loss(myModel);
        case 'h':
            return huber_loss(myModel);
        case 'n':
            return binary_cross_entropy_loss(myModel, classWeights);
        case 'r':
            return fast_binary_cross_entropy_loss(myModel, classWeights);
        case 'c':
            return categorical_cross_entropy_loss(myModel, classWeights);
        case 'x':
            return fast_categorical_cross_entropy_loss(myModel, classWeights);
        default:
            return 1;
    }
}

float loss_derivative(float target, float yHat, model* myModel, float classWeight)
{
    float numNodes = (*myModel->outLayer)->numNodes;

    switch(myModel->loss_fn)
    {
        case 'q':
            return mse_loss_derivative(target, yHat, numNodes);
        case 'a':
            return mae_loss_derivative(target, yHat, numNodes);
        case 'b':
            return mbe_loss_derivative(numNodes);
        case 'h':
            return huber_loss_derivative(target, yHat, numNodes);
        default:
            return 1;
    }
}
