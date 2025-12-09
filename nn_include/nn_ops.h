#ifndef NN_OPS
#define NN_OPS
#include <immintrin.h>
#include "model_ops.h"

// IMPORTANT
// Engineer data according to dimensions of input layers so first n inputs correspond to n nodes of first input layer
// Next m inputs correspond to m nodes of second input layer
// So on and so forth

void train_model_sgd(model* myModel, int epochs, int numSamples, float** inputs, float *targets);

void batch_train_sgd(model* myModel, int epochs, int numSamples, int batchSize, float** inputs, float *targets); //Automatically normalizes the batch into a single sample

void train_model_adam(model* myModel, int epochs, int numSamples, float** inputs, float *targets, float initialFirstMomentum, float initialSecondMomentum);

void batch_train_adam(model* myModel, int epochs, int numSamples, int batchSize, float** inputs, float *targets, float initialFirstMomentum, float initialSecondMomentum);

float* model_inference(model* myModel, float** inputs);

#endif