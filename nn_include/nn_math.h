#ifndef NN_MATH
#define NN_MATH
#include <math.h>

float clip(float x, float upper, float lower)
{
    if(x > upper) return upper;
    if(x < lower) return lower;
    return x;
}

static inline float absolute(float x)
{
    return (x > 0) ? x : -1.0*x;
}

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

float cross_entropy_loss(float target, float y)
{
    return -1; // finish later
}

static inline float mse_loss(float target, float y)
{
    return 0.5 * (target - y) * (target - y);
}

static inline float mse_loss_derivative(float target, float y)
{
    return (target - y);
}


void add_array(float* dest, float* arr1, float* arr2, __ssize_t size)
{
    for(int i = 0; i < size; i++)
    {
        dest[i] = arr1[i] + arr2[i];
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

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-1*x));
}

float sigmoid_derivative(float x) {
    float y = sigmoid(x);
    return y * (1.0 - y);
}

float fast_sigmoid(float x) {
    return 0.5 + 0.5 * (x / (1.3 + abs(x)));
}

float fast_sigmoid_derivative(float x) {
    float y = fast_sigmoid(x);
    return y * (1.0 - y);
}

float loss_derivative(float target, float y, char lossFunction)
{
    switch(lossFunction)
    {
        case 'q':
            return mse_loss_derivative(target, y);
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
        case 'i':
            return 1;
        default:
            return 1;
    }
}

float loss_function(float target, float y, char lossFunction)
{
    switch(lossFunction)
    {
        case 'q':
            return mse_loss(target, y);
        default:
            return 1;
    }
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
        case 'i':
            return x;
        default:
            return 1;
    }
}

#endif