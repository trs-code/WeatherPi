#ifndef NN_MATH
#define NN_MATH
#include <math.h>
#include "model.h"

float findMax(float* arr, int size)
{
    float currMax = arr[0];

    for(int i = 0; i < size; i++)
    {
        currMax = arr[i] > currMax ? arr[i] : currMax;
    }

    return currMax;
}

float findMin(float* arr, int size)
{
    float currMin = arr[0];

    for(int i = 0; i < size; i++)
    {
        currMin = arr[i] < currMin ? arr[i] : currMin;
    }

    return currMin;
}

void minMaxNorm(float* arr, int size)
{
    float min = findMin(arr, size);
    float diff = (size > 1) ? findMax(arr, size) - min : 1;

    for(int i = 0; i < size; i++)
    {
        arr[i] = (arr[i] - min) / diff;
    }
}

float clip(float x, float upper, float lower)
{
    if(x > upper) return upper;
    if(x < lower) return lower;
    return x;
}

float sign(float x)
{
    if(x > 0) return 1;
    if(x < 0) return -1;
    return 0;
}

static inline float absolute(float x)
{
    return (x > 0) ? x : -1.0*x;
}

void add_array(float* dest, float* arr1, float* arr2, __ssize_t size)
{
    for(int i = 0; i < size; i++)
    {
        dest[i] = arr1[i] + arr2[i];
    }
}

void subtract_array(float* dest, float* arr1, float* arr2, __ssize_t size)
{
    for(int i = 0; i < size; i++)
    {
        dest[i] = arr1[i] - arr2[i];
    }
}

void dot_product(float* dest, float* arr1, float* arr2, __ssize_t size)
{
    for(int i = 0; i < size; i++)
    {
        dest[i] = arr1[i] * arr2[i];
    }
}

void dot_product_value(float* dest, float* arr1, float value, __ssize_t size)
{
    for(int i = 0; i < size; i++)
    {
        dest[i] = arr1[i] * value;
    }
}

void dot_product_matrix(float** dest, float** arr1, float** arr2, __ssize_t rows, __ssize_t cols)
{
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++) dest[i][j] = arr1[i][j] * arr2[i][j];
    }
}

void dot_product_value_matrix(float** dest, float** arr1, float value, __ssize_t rows, __ssize_t cols)
{
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++) dest[i][j] = arr1[i][j] * value;
    }
}

static inline int max(int a, int b)
{
    return (a > b) ? a : b;
}

static inline int min(int a, int b)
{
    return (a < b) ? a : b;
}

float fast_ln(float x) // Courtesy of LingDong Huang(gist.github.com/LingDong-)
{
    unsigned int bx = * (unsigned int *) (&x);
    
    unsigned int ex = bx >> 23;
    
    signed int t = (signed int)ex-(signed int)127;
    
    unsigned int s = (t < 0) ? (-t) : t;
    
    bx = 1065353216 | (bx & 8388607);
    x = * (float *) (&bx);
    return -1.49278+(2.11263+(-0.729104+0.10969*x)*x)*x+0.6931471806*t;
}


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
    return 0.5 + 0.5 * (x / (1.3 + abs(x)));
}

float fast_sigmoid_derivative(float x) {
    float yHat = fast_sigmoid(x);
    return yHat * (1.0 - yHat);
}

float activation_function(float x, char activationFunction)
{
    switch(activationFunction)
    {
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
        case 'l':
            return x;
        case 'i':
            return x;
        default:
            return 1;
    }
}

float activation_derivative(float x, char activationFunction)
{
    switch(activationFunction)
    {
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
        case 'l': // Don't use unless you know what you're doing - Gradients WILL Explode and lead to NaN overflow values
            return 1;
        case 'i':
            return 1;
        default:
            return 1;
    }
}


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
    float sum = 0;
    
    for(int i = 0; i < (*myModel->outLayer)->numNodes; i++) sum -= ((myModel->targets[i]) * log((*myModel->outLayer)->outputs[i])) + ((1-(myModel->targets[i])) * log((1 - (*myModel->outLayer)->outputs[i])));
    
    return (sum / (*myModel->outLayer)->numNodes);
}

float binary_cross_entropy_loss_derivative(float target, float yHat, int n)
{
    return (yHat - target)/(float)n;
}

float fast_binary_cross_entropy_loss(model* myModel)
{
    float sum = 0;
    
    for(int i = 0; i < (*myModel->outLayer)->numNodes; i++) sum -= ((myModel->targets[i]) * fast_ln((*myModel->outLayer)->outputs[i])) + ((1-(myModel->targets[i])) * fast_ln((1 - (*myModel->outLayer)->outputs[i])));
    
    return ( sum / (*myModel->outLayer)->numNodes);
}

float fast_binary_cross_entropy_loss_derivative(float target, float yHat, int n)
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
            return binary_cross_entropy_loss_derivative(target, yHat, numNodes);
        case 'r':
            return fast_binary_cross_entropy_loss_derivative(target, yHat, numNodes);
        default:
            return 1;
    }
}

// float accuracy(model* myModel)
// {
//     float sum = 0;
//     for(int i = 0; i < (*myModel->outLayer)->numNodes; i++) sum += ((*myModel->outLayer)->outputs[i] - (myModel->targets[i])) / (myModel->targets[i]);
//     return absolute(sum / (*myModel->outLayer)->numNodes);
// }

float accuracy(model* myModel)
{
    float sum = 0.0f;
    int n = (*myModel->outLayer)->numNodes;

    for (int i = 0; i < n; i++)
    {
        float pred   = (*myModel->outLayer)->outputs[i];
        float target = myModel->targets[i];

        // target guaranteed != 0
        sum += absolute(pred - target) / absolute(target);
    }

    return 1.0f - (sum / n);
}

void shuffle(float*** arr1, float*** arr2, int n) 
{
    float* temp;
    for (int i = n - 1; i > 0; i--) 
    {        
        int j = rand() % (i + 1);
        
        temp = (*arr1)[i];
        (*arr1)[i] = (*arr1)[j];
        (*arr1)[j] = temp;

        temp = (*arr2)[i];
        (*arr2)[i] = (*arr2)[j];
        (*arr2)[j] = temp;
    }
}

#endif