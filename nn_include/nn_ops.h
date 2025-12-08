#ifndef NN_OPS
#define NN_OPS
#include <immintrin.h>
#include "model_ops.h"

void train_model_sgd(struct model* myModel, int epochs, int numSamples, float** inputs, float *targets);

void batch_train_sgd(struct model* myModel, int epochs, int numSamples, int batchSize, float** inputs, float *targets); //Automatically normalizes the batch into a single sample

void train_model_adam(struct model* myModel, int epochs, int numSamples, float** inputs, float *targets, float initialFirstMomentum, float initialSecondMomentum);

void batch_train_adam(struct model* myModel, int epochs, int numSamples, int batchSize, float** inputs, float *targets, float initialFirstMomentum, float initialSecondMomentum);

float* model_inference(struct model* myModel, float** inputs);

#endif