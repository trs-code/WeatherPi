#pragma once

#include "nn_math.h"
#include "layer.h"

// Activation functions and their derivatives 
static inline float relu(float x)
{
    return x > 0 ? x : 0.0;
}

static inline float relu_derivative(float x)
{
    return (x > 0) ? 1 : 0.0;
}

static inline float leaky_relu(float x)
{
    return (x > 0 ? x : 0.01*x);
}

static inline float leaky_relu_derivative(float x)
{
    return (x > 0) ? 1 : (0.01);
}

float fast_tanh(float x) // tanh fast approximation
{
    return clip(2* (x / (1.3f + 1.6f * absolute(x))), 1.0f, 0.0f);
}

float fast_tanh_derivative(float x) // d(tanhx)/dx fast approximation
{
    float x2 = (x) / (0.8 + (1.1 * absolute(x)));
    float res = 1 - (2*x2*x2);
    return (res > 0.0001) ? res : 0.0001;
}

float tanh_derivative(float x)
{
    float x2 = tanh(x) * tanh(x); 
    return (1-x2);
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-1*x));
}

float sigmoid_derivative(float x) {
    float yHat = sigmoid(x);
    return yHat * (1.0 - yHat);
}

float fast_sigmoid(float x) {
    return 0.5 + 0.5 * (x / (1.3 + absolute(x)));
}

float fast_sigmoid_derivative(float x) {
    float yHat = fast_sigmoid(x);
    return yHat * (1.0 - yHat);
}

void softmax(layer** myLayer)
{
    int n = (*myLayer)->numNodes;
    float maxVal = findMax((*myLayer)->outputs, n);
    float sum = 0.0f;
    
    for (int i = 0; i < n; i++) 
    {
        (*myLayer)->outputs[i] = exp((*myLayer)->outputs[i] - maxVal);
        sum += (*myLayer)->outputs[i];
    }

    for (int i = 0; i < n; i++)
    {
        (*myLayer)->outputs[i] /= sum;
    }
}

float softmax_derivative(float x, layer* myLayer, int currNode)
{
    return myLayer->outputs[currNode] - x;
}

void fast_softmax(layer** myLayer)
{
    int n = (*myLayer)->numNodes;
    float maxVal = findMax((*myLayer)->outputs, n);
    float sum = 0.0f;
    
    for (int i = 0; i < n; i++) 
    {
        (*myLayer)->outputs[i] = fast_exp((*myLayer)->outputs[i] - maxVal);
        sum += (*myLayer)->outputs[i];
    }

    for (int i = 0; i < n; i++)
    {
        (*myLayer)->outputs[i] /= sum;
    }
}

float fast_softmax_derivative(float x, layer* myLayer, int currNode)
{
    return myLayer->outputs[currNode] - x;
}


float activation_function(float x, char activationFunction)
{
    switch(activationFunction)
    {
        case 'l':
            return x;
        case 'i':
            return x;
        case 'u':
            return leaky_relu(x);
        case 'r':
            return relu(x);
        case 't':
            return tanh(x);
        case 'h':
            return fast_tanh(x);
        case 's':
            return sigmoid(x);
        case 'g':
            return fast_sigmoid(x);
        default:
            return 0.0;
    }
}

float activation_derivative(float x, char activationFunction, layer* myLayer, int currNode)
{
    switch(activationFunction)
    {
        case 'l':
            return 1;
        case 'i':
            return 1;
        case 'u':
            return leaky_relu_derivative(x);
        case 'r':
            return relu_derivative(x);
        case 't':
            return tanh_derivative(x);
        case 'h':
            return fast_tanh_derivative(x);
        case 's':
            return sigmoid_derivative(x);
        case 'g':
            return fast_sigmoid_derivative(x);
        case 'x':
            return softmax_derivative(x, myLayer, currNode);
        case 'f':
            return fast_softmax_derivative(x, myLayer, currNode);
        default:
            return 0.0;
    }
}