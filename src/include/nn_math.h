#pragma once

#include <math.h>
#include <stdlib.h>
#include <string.h>

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

float fastExp(float num) // Taken from Johan Rade (github.com/jrade/)
{
    float a = (1 << 23) / 0.69314718f;
    float b = (1 << 23) * (127 - 0.043677448f);
    float x = a * num + b;

    float c = (1 << 23);
    float d = (1 << 23) * 255;
    if (x < c || x > d)
        x = (x < c) ? 0.0f : d;

    __uint32_t n = (__uint32_t)(x);
    memcpy(&x, &n, 4);
    return x;
}