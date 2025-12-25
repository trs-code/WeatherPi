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
    return -2 * (yHat - target) / n;
}

float mae_loss(model* myModel)
{
    float sum = 0;
    for(int i = 0; i < (*myModel->outLayer)->numNodes; i++) sum += abs((*myModel->outLayer)->outputs[i] - (myModel->targets[i]));
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
        error = abs((*myModel->outLayer)->outputs[i] - (myModel->targets[i]));
        if(error > 0.01) sum += 0.01 * (error - 0.005);
        else sum +=  0.5 * error * error;
    }

    return (sum / (*myModel->outLayer)->numNodes);
}

float huber_loss_derivative(float target, float yHat, int n)
{
    float error = yHat - target;
    if(abs(error) > 0.01) return 0.01 * sign(error);
    else return error / (float)n;
}

float binary_cross_entropy_loss(model* myModel)
{
    if((*myModel->outLayer)->numNodes > 1) exit(EXIT_FAILURE);
    float sum = 0;
    
    sum -= ((myModel->targets[0]) * log((*myModel->outLayer)->outputs[0])) + ((1-(myModel->targets[0])) * log((1 - (*myModel->outLayer)->outputs[0])));
    
    return (sum / (*myModel->outLayer)->numNodes);
}

float binary_cross_entropy_loss_derivative(float target, float yHat)
{
    return (yHat - target);
}

float fast_binary_cross_entropy_loss(model* myModel)
{
    if((*myModel->outLayer)->numNodes > 1) exit(EXIT_FAILURE);
    float sum = 0;
    
    sum -= ((myModel->targets[0]) * fast_ln((*myModel->outLayer)->outputs[0])) + ((1-(myModel->targets[0])) * fast_ln((1 - (*myModel->outLayer)->outputs[0])));
    
    return ( sum / (*myModel->outLayer)->numNodes);
}

float fast_binary_cross_entropy_loss_derivative(float target, float yHat)
{
    return (yHat - target);
}

float categorical_cross_entropy_loss(model* myModel)
{
    float sum = 0;
    
    for(int i = 0; i < (*myModel->outLayer)->numNodes; i++) sum -= ((*myModel->outLayer)->outputs[i] * log(myModel->targets[i]));
    
    return (sum / (*myModel->outLayer)->numNodes);
}

float categorical_cross_entropy_loss_derivative(float target, float yHat, int n)
{
    return (yHat - target)/(float)n;
}

float fast_categorical_cross_entropy_loss(model* myModel)
{
    float sum = 0;
    
    for(int i = 0; i < (*myModel->outLayer)->numNodes; i++) sum -= ((*myModel->outLayer)->outputs[i] * fast_ln(myModel->targets[i]));
    return (sum / (*myModel->outLayer)->numNodes);
}

float fast_categorical_cross_entropy_loss_derivative(float target, float yHat, int n)
{
    return (yHat - target) / (float)n;
}

float loss_function(model* myModel)
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
            return binary_cross_entropy_loss(myModel);
        case 'r':
            return fast_binary_cross_entropy_loss(myModel);
        case 'c':
            return categorical_cross_entropy_loss(myModel);
        case 'x':
            return fast_categorical_cross_entropy_loss(myModel);
        default:
            return 1;
    }
}

float loss_derivative(float target, float yHat, model* myModel)
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
        case 'n':
            return binary_cross_entropy_loss_derivative(target, yHat);
        case 'r':
            return fast_binary_cross_entropy_loss_derivative(target, yHat);
        case 'c':
            return categorical_cross_entropy_loss_derivative(target, yHat, numNodes);
        case 'x':
            return fast_categorical_cross_entropy_loss_derivative(target, yHat, numNodes);
        default:
            return 1;
    }
}




