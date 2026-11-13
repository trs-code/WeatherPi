#pragma once

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

#include "model_ops.h"
#include <sys/time.h>

// IMPORTANT
// Engineer data according to dimensions of input layers so first n inputs correspond to n nodes of first input layer
// Next m inputs correspond to m nodes of second input layer
// So on and so forth

void train_model_sgd(model* myModel, int epochs, int numSamples, float** inputs, float **targets, float valSplit, float dropoutVal)
{
    float* classWeights = NULL;
    float trainingLoss = 0.0;
    float testingLoss = 0.0;
    int binFlag = 0;
    int inputsTraversed = 0;
    int trainSamples = clip((int)(valSplit * numSamples), numSamples, 0);
    int testSamples = numSamples - trainSamples;
    int numTrainLosses = (trainSamples != 0) ? trainSamples : 1;
    int numTestLosses = (testSamples != 0) ? testSamples : 1;
    struct timespec start, end, tStart, tEnd;
    double timeElapsed;
    
    if(myModel->loss_fn == 'c' || myModel->loss_fn == 'x') class_weights((*myModel->outLayer)->numNodes, numSamples, targets, &classWeights, 0x00);
    if(myModel->loss_fn == 'n' || myModel->loss_fn == 'r') class_weights(2, numSamples, targets, &classWeights, 0x01);

    clock_gettime(CLOCK_MONOTONIC, &tStart);
    
    for(int e = 1; e < (epochs + 1); e++)
    {
        trainingLoss = 0.0;
        testingLoss = 0.0;

        shuffle(&inputs, &targets, numSamples);

        clock_gettime(CLOCK_MONOTONIC, &start);
        
        for(int i = 0; i < trainSamples; i++)
        {
            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                memcpy((*myModel->inLayers[j])->outputs, &(inputs[i][inputsTraversed]), sizeof(float) * (*myModel->inLayers[j])->numNodes);
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }

            memcpy(myModel->targets, targets[i], sizeof(float) * (*myModel->outLayer)->numNodes);
            
            forward_out(myModel, dropoutVal);
            sgd_backprop(myModel, classWeights);
            calculate_and_apply_grads(myModel);

            trainingLoss += loss_function(myModel, classWeights);
        }
        
        for(int i = trainSamples; i < numSamples; i++)
        {
            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                memcpy((*myModel->inLayers[j])->outputs, &inputs[i][inputsTraversed], (*myModel->inLayers[j])->numNodes * sizeof(float));
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }

            memcpy(myModel->targets, targets[i], sizeof(float) * (*myModel->outLayer)->numNodes);
            
            forward_out(myModel, 0.0);

            testingLoss += loss_function(myModel, NULL);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        timeElapsed = ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_nsec - start.tv_nsec) / 1000000.0);

        testingLoss /=  numTestLosses;
        trainingLoss /= numTrainLosses;
        printf("Epoch %d - Training Loss: %f, Testing Loss: %f - %.1lfms\n", e, trainingLoss, testingLoss, timeElapsed);
    }

    clock_gettime(CLOCK_MONOTONIC, &tEnd);
    
    timeElapsed = ((tEnd.tv_sec - tStart.tv_sec) * 1000.0) + ((tEnd.tv_nsec - tStart.tv_nsec) / 1000000.0);

    printf("\nTotal training time: %lfms\n", timeElapsed);
}

void train_rnn_sgd(model* myModel, int epochs, int numSamples, float** inputs, float **targets, float valSplit, int seqLength, float dropoutVal)
{
    float* classWeights = NULL;
    int inputsTraversed = 0;
    // int currTargs = 0;
    float trainingLoss = 0.0;
    float testingLoss = 0.0;
    int trainSamples = clip((int)(valSplit * numSamples), numSamples, 0);
    int testSamples = numSamples - trainSamples;
    int numTrainLosses = (trainSamples != 0) ? trainSamples : 1;
    int numTestLosses = (testSamples != 0) ? testSamples : 1;;
    struct timespec start, end, tStart, tEnd;
    double timeElapsed;

    if(myModel->loss_fn == 'c' || myModel->loss_fn == 'x') class_weights((*myModel->outLayer)->numNodes, numSamples, targets, &classWeights, 0x00);
    if(myModel->loss_fn == 'n' || myModel->loss_fn == 'r') class_weights(2, numSamples, targets, &classWeights, 0x01);

    clock_gettime(CLOCK_MONOTONIC, &tStart);
    
    for(int e = 1; e < (epochs + 1); e++)
    {
        trainingLoss = 0.0;
        testingLoss = 0.0;

        clock_gettime(CLOCK_MONOTONIC, &start);
        
        for(int i = 0; i < trainSamples; i += seqLength)
        {                
            for(int j = i; j < i + seqLength && j < trainSamples; j++)
            {
                inputsTraversed = 0;
                for(int k = 0; k < myModel->numInLayers; k++)
                {
                    memcpy((*myModel->inLayers[k])->outputs, &(inputs[j][inputsTraversed]), sizeof(float) * (*myModel->inLayers[k])->numNodes);
                    inputsTraversed += (*myModel->inLayers[k])->numNodes;
                }

                forward_out(myModel, dropoutVal);
                sgd_backprop(myModel, classWeights);
                shift_rnn(myModel, 't');
                calculate_and_apply_grads(myModel);
                
                memcpy(myModel->targets, targets[j], sizeof(float) * (*myModel->outLayer)->numNodes);
                trainingLoss += loss_function(myModel, classWeights);
            }

            for(int l = 0; l < myModel->numLayers; l++) 
            {
                layer* currLayer = (*myModel->layerList[l]);
                
                if(currLayer->layerType == 'h')
                {
                    layer* prevLayer = (*currLayer->prevLayers[currLayer->numPrevLayers - 1]);
                    
                    if(prevLayer->layerType == 'w')
                    {
                        memset(prevLayer->preActivations, 0.0, (*myModel->layerList[l])->numPrevNodes);
                        memset(prevLayer->biases, 0.0, (*myModel->layerList[l])->numNodes);
                    }
                }
            }
        }
        
        for(int i = trainSamples; i < numSamples; i++)
        {
            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                memcpy((*myModel->inLayers[j])->outputs, &(inputs[i][inputsTraversed]), sizeof(float) * (*myModel->inLayers[j])->numNodes);
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }
            
            forward_out(myModel, dropoutVal);
            memcpy(myModel->targets, targets[i], sizeof(float) * (*myModel->outLayer)->numNodes);
            testingLoss += loss_function(myModel, NULL);
            
            shift_rnn(myModel, 'i');
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        
        timeElapsed = ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_nsec - start.tv_nsec) / 1000000.0);

        testingLoss /=  numTestLosses;
        trainingLoss /= numTrainLosses;
        printf("Epoch %d - Training Loss: %f, Testing Loss: %f - %.1lfms\n", e, trainingLoss, testingLoss, timeElapsed);
    }

    clock_gettime(CLOCK_MONOTONIC, &tEnd);
    
    timeElapsed = ((tEnd.tv_sec - tStart.tv_sec) * 1000.0) + ((tEnd.tv_nsec - tStart.tv_nsec) / 1000000.0);

    printf("\nTotal training time: %lfms\n", timeElapsed);
}

void model_inference(model* myModel, float* inputs, float** outputs) //(model*, float*, &float[])
{
    int inputsTraversed = 0;
    for(int i = 0; i < myModel->numInLayers; i++)
    {
        for(int j = 0; j < (*myModel->inLayers[i])->numNodes; j++) (*myModel->inLayers[i])->outputs[j] = inputs[j + inputsTraversed];
        inputsTraversed += (*myModel->inLayers[i])->numNodes;
    }

    forward_out(myModel, 0.0);

    memcpy(*outputs, (*myModel->outLayer)->outputs, sizeof(float) * (*myModel->outLayer)->numNodes);
}

void rnn_model_inference(model* myModel, float inputs[], float outputs[]) //(model*, float*, &float[])
{
    int inputsTraversed = 0;
    for(int i = 0; i < myModel->numInLayers; i++)
    {
        for(int j = 0; j < (*myModel->inLayers[i])->numNodes; j++) (*myModel->inLayers[i])->outputs[j] = inputs[j + inputsTraversed];
        inputsTraversed += (*myModel->inLayers[i])->numNodes;
    }

    forward_out(myModel, 0.0);
    shift_rnn(myModel, 'i');

    memcpy(outputs, (*myModel->outLayer)->outputs, sizeof(float) * (*myModel->outLayer)->numNodes);
}

void train_model_adam(model* myModel, int epochs, int numSamples, float** inputs, float *targets, float initialFirstMomentum, float initialSecondMomentum); //Still needs to be implemented


#if defined(__AVX__) || defined(__AVX2__)
void train_model_sgd_fast(model* myModel, int epochs, int numSamples, float** inputs, float **targets, float valSplit, float dropoutVal)
{
    float* classWeights;
    int inputsTraversed = 0;
    float trainingLoss = 0.0;
    float testingLoss = 0.0;
    int trainSamples = clip((int)(valSplit * numSamples), numSamples, 0);
    int testSamples = numSamples - trainSamples;
    int numTrainLosses = (trainSamples != 0) ? trainSamples : 1;
    int numTestLosses = (testSamples != 0) ? testSamples : 1;
    struct timespec start, end, tStart, tEnd;
    double timeElapsed;

    if(myModel->loss_fn == 'c' || myModel->loss_fn == 'x') class_weights((*myModel->outLayer)->numNodes, numSamples, targets, &classWeights, 0x00);
    if(myModel->loss_fn == 'n' || myModel->loss_fn == 'r') class_weights(2, numSamples, targets, &classWeights, 0x01);

    clock_gettime(CLOCK_MONOTONIC, &tStart);
    
    for(int e = 1; e < (epochs + 1); e++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start);

        trainingLoss = 0.0;
        testingLoss = 0.0;

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
            
            _mm256_forward_out(myModel, dropoutVal);
            sgd_backprop(myModel, classWeights);
            calculate_and_apply_grads(myModel);

            trainingLoss += loss_function(myModel, NULL);
        }
        
        for(int i = trainSamples; i < numSamples; i++)
        {
            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                memcpy((*myModel->inLayers[j])->outputs, &(inputs[i][inputsTraversed]), sizeof(float) * (*myModel->inLayers[j])->numNodes);
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }

            memcpy(myModel->targets, targets[i], sizeof(float) * (*myModel->outLayer)->numNodes);
            
            _mm256_forward_out(myModel, 0.0);

            testingLoss += loss_function(myModel, NULL);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        
        timeElapsed = ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_nsec - start.tv_nsec) / 1000000.0);
        timeElapsed =  (timeElapsed >= 0) ? timeElapsed : 1000.0 + timeElapsed;

        testingLoss /=  numTestLosses;
        trainingLoss /= numTrainLosses;
        printf("Epoch %d - Training Loss: %f, Testing Loss: %f - %.1lfms\n", e, trainingLoss, testingLoss, timeElapsed);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &tEnd);
    
    timeElapsed = ((tEnd.tv_sec - tStart.tv_sec) * 1000.0) + ((tEnd.tv_nsec - tStart.tv_nsec) / 1000000.0);

    printf("\nTotal training time: %lfms\n", timeElapsed);

}

void train_rnn_sgd_fast(model* myModel, int epochs, int numSamples, float** inputs, float **targets, float valSplit, int seqLength, float dropoutVal)
{
    float* classWeights = NULL;
    int inputsTraversed = 0;
    int currTargs = 0;
    float trainingLoss = 0.0;
    float testingLoss = 0.0;
    int trainSamples = clip((int)(valSplit * numSamples), numSamples, 0);
    int testSamples = numSamples - trainSamples;
    int numTrainLosses = (trainSamples != 0) ? trainSamples : 1;
    int numTestLosses = (testSamples != 0) ? testSamples : 1;
    struct timespec start, end, tStart, tEnd;
    double timeElapsed;

    if(myModel->loss_fn == 'c' || myModel->loss_fn == 'x') class_weights((*myModel->outLayer)->numNodes, numSamples, targets, &classWeights, 0x00);
    if(myModel->loss_fn == 'n' || myModel->loss_fn == 'r') class_weights(2, numSamples, targets, &classWeights, 0x01);

    clock_gettime(CLOCK_MONOTONIC, &tStart);
    
    for(int e = 1; e < (epochs + 1); e++)
    {
        trainingLoss = 0.0;
        testingLoss = 0.0;

        clock_gettime(CLOCK_MONOTONIC, &start);
        
        for(int i = 0; i < trainSamples; i += seqLength)
        {                
            for(int j = i; j < i + seqLength && j < trainSamples; j++)
            {
                inputsTraversed = 0;
                for(int k = 0; k < myModel->numInLayers; k++)
                {
                    memcpy((*myModel->inLayers[k])->outputs, &(inputs[j][inputsTraversed]), sizeof(float) * (*myModel->inLayers[k])->numNodes);
                    inputsTraversed += (*myModel->inLayers[k])->numNodes;
                }

                _mm256_forward_out(myModel, dropoutVal);
                sgd_backprop(myModel, classWeights);
                calculate_and_apply_grads(myModel);
                shift_rnn(myModel, 't');
                

                memcpy(myModel->targets, targets[currTargs], sizeof(float) * (*myModel->outLayer)->numNodes);
                trainingLoss += loss_function(myModel, classWeights);
            }

            
            for(int l = 0; l < myModel->numLayers; l++) 
            {
                layer* currLayer = (*myModel->layerList[l]);
                
                if(currLayer->layerType == 'h')
                {
                    layer* prevLayer = (*currLayer->prevLayers[currLayer->numPrevLayers - 1]);
                    
                    if(prevLayer->layerType == 'w')
                    {
                        memset(prevLayer->preActivations, 0.0, (*myModel->layerList[l])->numPrevNodes);
                        memset(prevLayer->biases, 0.0, (*myModel->layerList[l])->numNodes);
                    }
                }
            }
            
        }
        
        for(int i = trainSamples; i < numSamples; i++)
        {
            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                memcpy((*myModel->inLayers[j])->outputs, &(inputs[i][inputsTraversed]), sizeof(float) * (*myModel->inLayers[j])->numNodes);
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }

            printf("Got to here %d\n", i);
            
            _mm256_forward_out(myModel, 0.0);
            memcpy(myModel->targets, targets[i], sizeof(float) * (*myModel->outLayer)->numNodes);
            testingLoss += loss_function(myModel, NULL);
            
            shift_rnn(myModel, 'i');
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        
        timeElapsed = ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_nsec - start.tv_nsec) / 1000000.0);
        timeElapsed =  (timeElapsed >= 0) ? timeElapsed : 1000.0 + timeElapsed;

        testingLoss /=  numTestLosses;
        trainingLoss /= numTrainLosses;
        printf("Epoch %d - Training Loss: %f, Testing Loss: %f - %.1lfms\n", e, trainingLoss, testingLoss, timeElapsed);
    }

    clock_gettime(CLOCK_MONOTONIC, &tEnd);
    
    timeElapsed = ((tEnd.tv_sec - tStart.tv_sec) * 1000.0) + ((tEnd.tv_nsec - tStart.tv_nsec) / 1000000.0);

    printf("\nTotal training time: %lfms\n", timeElapsed);
}

void model_inference_fast(model* myModel, float* inputs, float** outputs) //(model*, float*, &float[])
{
    int inputsTraversed = 0;
    for(int i = 0; i < myModel->numInLayers; i++)
    {
        for(int j = 0; j < (*myModel->inLayers[i])->numNodes; j++) (*myModel->inLayers[i])->outputs[j] = inputs[j + inputsTraversed];
        inputsTraversed += (*myModel->inLayers[i])->numNodes;
    }

    _mm256_forward_out(myModel, 0.0);

    memcpy(*outputs, (*myModel->outLayer)->outputs, sizeof(float) * (*myModel->outLayer)->numNodes);
}

void rnn_model_inference_fast(model* myModel, float* inputs, float** outputs) //(model*, float*, &float[])
{
    int inputsTraversed = 0;
    for(int i = 0; i < myModel->numInLayers; i++)
    {
        for(int j = 0; j < (*myModel->inLayers[i])->numNodes; j++) (*myModel->inLayers[i])->outputs[j] = inputs[j + inputsTraversed];
        inputsTraversed += (*myModel->inLayers[i])->numNodes;
    }

    _mm256_forward_out(myModel, 0.0);
    shift_rnn(myModel, 'i');

    memcpy(*outputs, (*myModel->outLayer)->outputs, sizeof(float) * (*myModel->outLayer)->numNodes);
}

#endif

#if defined(__ARM_NEON)
void model_inference_fast(model* myModel, float* inputs, float** outputs) //(model*, float*, &float[])
{
    int inputsTraversed = 0;
    for(int i = 0; i < myModel->numInLayers; i++)
    {
        for(int j = 0; j < (*myModel->inLayers[i])->numNodes; j++) (*myModel->inLayers[i])->outputs[j] = inputs[j + inputsTraversed];
        inputsTraversed += (*myModel->inLayers[i])->numNodes;
    }

    vforward_out(myModel, 0.0);

    memcpy(*outputs, (*myModel->outLayer)->outputs, sizeof(float) * (*myModel->outLayer)->numNodes);
}

void rnn_model_inference_fast(model* myModel, float* inputs, float** outputs) //(model*, float*, &float[])
{
    int inputsTraversed = 0;
    for(int i = 0; i < myModel->numInLayers; i++)
    {
        for(int j = 0; j < (*myModel->inLayers[i])->numNodes; j++) (*myModel->inLayers[i])->outputs[j] = inputs[j + inputsTraversed];
        inputsTraversed += (*myModel->inLayers[i])->numNodes;
    }

    vforward_out(myModel, 0.0);
    shift_rnn(myModel, 'i');

    memcpy(*outputs, (*myModel->outLayer)->outputs, sizeof(float) * (*myModel->outLayer)->numNodes);
}

#endif