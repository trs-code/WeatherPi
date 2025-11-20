#ifndef NN_MATH
#define NN_MATH

float clip(x)
{
    if(x > 1.0f) return 1;
    if(x < 0.0f) return 0;
    return x;
}

inline float absolute(float x)
{
    return (x > 0) ? x : -1.0*x;
}

inline float relu(float x)
{
    return x > 0 ? x : 0.0;
}

inline float relu_derivative(float x)
{
    return (x > 0) ? 1 : 0.0;
}

inline float leaky_relu(float x)
{
    return (x > 0 ? x : 0.01*x);
}

inline float leaky_relu_derivative(float x)
{
    return (x > 0) ? 1 : (0.01);
}

float tanh(float x)
{
    return clip(0.5f + (0.5f* x / (0.06f + 2.0f * absolute(x))));
}

float tanh_derivative(float x)
{
    float x2 = (x) / (1 + (0.95 * absolute(x)));
    float res = 0.5 - (x2*x2);
    return (res > 0) ? res : 0;
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

inline float mse_loss(float target, float y)
{
    return 0.5 * (y - target) * (y - target);
}

inline float mse_loss_derivative(float target, float y)
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

void dot_product(float* dest, float* arr1, float value, __ssize_t size)
{
    for(int i = 0; i < size; i++)
    {
        dest[i] = arr1[i] * value;
    }
}

#endif