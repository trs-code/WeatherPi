#pragma once

//#include <immintrin.h>
#include "model_ops.h"

// IMPORTANT
// Engineer data according to dimensions of input layers so first n inputs correspond to n nodes of first input layer
// Next m inputs correspond to m nodes of second input layer
// So on and so forth

void train_model_sgd(model* myModel, int epochs, int numSamples, float** inputs, float **targets, float valSplit)
{
    int inputsTraversed = 0;
    float trainingLoss = 0.0;
    float validationAcc = 0.0;
    int trainSamples = (int)(valSplit * numSamples);
    int valSamples = numSamples - trainSamples;
    struct timespec start, end;
    double timeElapsed;

    
    for(int e = 1; e < (epochs + 1); e++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start);
        trainingLoss = 0.0;
        validationAcc = 0.0;

        shuffle(&inputs, &targets, numSamples);
        
        for(int i = 0; i < trainSamples; i++)
        {
            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                memcpy((*myModel->inLayers[j])->outputs, &(inputs[i][inputsTraversed]), sizeof(float) * (*myModel->inLayers[j])->numNodes);
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }

            memcpy(myModel->targets, targets[i], sizeof(float) * (*myModel->outLayer)->numNodes);
            
            forward_out(myModel->outLayer);
            sgd_backprop(myModel->outLayer, &myModel);
            calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);

            trainingLoss += loss_function(myModel);
            zero_everything(myModel->outLayer);
        }
        
        for(int i = trainSamples; i < numSamples; i++)
        {
            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                for(int k = 0; k < (*myModel->inLayers[j])->numNodes; k++) (*myModel->inLayers[j])->outputs[k] = inputs[i][k + inputsTraversed];
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }

            memcpy(myModel->targets, targets[i], sizeof(float) * (*myModel->outLayer)->numNodes);
            
            forward_out(myModel->outLayer);

            validationAcc += loss_function(myModel);
            zero_everything(myModel->outLayer);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        
        timeElapsed = (end.tv_nsec - start.tv_nsec) / 1000000.0;
        timeElapsed =  (timeElapsed >= 0) ? timeElapsed : 1000.0 + timeElapsed;

        validationAcc = (validationAcc / valSamples);
        trainingLoss /= trainSamples;
        printf("Epoch %d - Training Loss: %f, Validation Loss: %f - %.1lfms\n", e, trainingLoss, validationAcc, timeElapsed);
    }
}

void train_context_model_sgd(model* myModel, layer** windowLayers, int epochs, int numSamples, float** inputs, float **targets, float valSplit, int windowSize)
{
    int inputsTraversed = 0;
    float trainingLoss = 0.0;
    float validationAcc = 0.0;
    int trainSamples = (int)(valSplit * numSamples);
    int valSamples = numSamples - trainSamples;
    struct timespec start, end;
    double timeElapsed;

    
    for(int e = 1; e < (epochs + 1); e++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start);
        trainingLoss = 0.0;
        validationAcc = 0.0;

        // shuffle(inputs, targets, numSamples); - need this to be time series data so no shuffling
        
        for(int i = 0; i < trainSamples; i++)
        {
            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                load_context_window(windowLayers, inputs[i], windowSize);
                memcpy((*myModel->inLayers[j])->outputs, &(inputs[i][inputsTraversed]), sizeof(float) * (*myModel->inLayers[j])->numNodes);
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }

            memcpy(myModel->targets, targets[i], sizeof(float) * (*myModel->outLayer)->numNodes);
            
            forward_out(myModel->outLayer);
            sgd_backprop(myModel->outLayer, &myModel);
            calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);

            trainingLoss += loss_function(myModel);
            zero_everything(myModel->outLayer);
        }
        
        for(int i = trainSamples; i < numSamples; i++)
        {
            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                load_context_window(windowLayers, inputs[i], windowSize);
                memcpy((*myModel->inLayers[j])->outputs, &(inputs[i][inputsTraversed]), sizeof(float) * (*myModel->inLayers[j])->numNodes);
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }

            memcpy(myModel->targets, targets[i], sizeof(float) * (*myModel->outLayer)->numNodes);
            
            forward_out(myModel->outLayer);

            validationAcc += loss_function(myModel);
            zero_everything(myModel->outLayer);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        
        timeElapsed = (end.tv_nsec - start.tv_nsec) / 1000000.0;
        timeElapsed =  (timeElapsed >= 0) ? timeElapsed : 1000.0 + timeElapsed;

        validationAcc = (validationAcc / valSamples);
        trainingLoss /= trainSamples;
        printf("Epoch %d - Training Loss: %f, Validation Loss: %f - %.1lfms\n", e, trainingLoss, validationAcc, timeElapsed);
    }
}

void mini_batch_train_sgd(model* myModel, int epochs, int numSamples, int batchSize, float** inputs, float**targets, _Bool normBatch) //Automatically normalizes the batch into a single sample
{
    int inputsTraversed = 0;
    int numFullBatches = 0;
    int numLeftOver = 0;
    float averageLoss = 0.0;
    
    numFullBatches = numSamples / batchSize;
    numLeftOver = numSamples % batchSize;
    for(int e = 1; e < (epochs + 1); e++)
    {
        averageLoss = 0.0;
        for(int i = 0; i < numFullBatches; i++)
        {

            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                for(int k = 0; k < (*myModel->inLayers[j])->numNodes; k++) (*myModel->inLayers[j])->outputs[k] += inputs[i][k + inputsTraversed];
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }

            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                for(int k = 0; k < (*myModel->inLayers[j])->numNodes; k++) (*myModel->inLayers[j])->outputs[k] /= batchSize;
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }

            memcpy(myModel->targets, targets[i], sizeof(float) * (*myModel->outLayer)->numNodes);
            
            forward_out(myModel->outLayer);
            sgd_backprop(myModel->outLayer, &myModel);
            calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);

            averageLoss += loss_function(myModel);
            zero_everything(myModel->outLayer);
        }

        averageLoss /= (numFullBatches + (numLeftOver / batchSize));
        printf("Average training loss for epoch %d: %f\n", e, averageLoss);  

    }
}

void train_model_adam(model* myModel, int epochs, int numSamples, float** inputs, float *targets, float initialFirstMomentum, float initialSecondMomentum);

void batch_train_adam(model* myModel, int epochs, int numSamples, int batchSize, float** inputs, float *targets, float initialFirstMomentum, float initialSecondMomentum);

float* model_inference(model* myModel, float* inputs)
{
    int inputsTraversed = 0;
    for(int i = 0; i < myModel->numInLayers; i++)
    {
        for(int j = 0; j < (*myModel->inLayers[i])->numNodes; j++) (*myModel->inLayers[i])->outputs[j] = inputs[j + inputsTraversed];
        inputsTraversed += (*myModel->inLayers[i])->numNodes;
    }

    forward_out(myModel->outLayer);

    return (*myModel->outLayer)->outputs;
}