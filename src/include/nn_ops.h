#pragma once

#include "model_ops.h"
#include <sys/time.h>

// IMPORTANT
// Engineer data according to dimensions of input layers so first n inputs correspond to n nodes of first input layer
// Next m inputs correspond to m nodes of second input layer
// So on and so forth

void train_model_sgd(model* myModel, int epochs, int numSamples, float** inputs, float **targets, float valSplit)
{
    int inputsTraversed = 0;
    float trainingLoss = 0.0;
    float testingLoss = 0.0;
    int trainSamples = clip((int)(valSplit * numSamples), numSamples, 0);
    int testSamples = numSamples - trainSamples;
    int numTrainLosses = 0;
    int numTestLosses = 0;
    double timeElapsed;
    long tStart, tEnd, eStart, eEnd;
    struct timeval timecheck;

    gettimeofday(&timecheck, NULL);
    tStart = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
    
    for(int e = 1; e < (epochs + 1); e++)
    {
        gettimeofday(&timecheck, NULL);
        eStart = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

        trainingLoss = 0.0;
        testingLoss = 0.0;
        numTrainLosses = (trainSamples != 0) ? 0 : 1;
        numTestLosses = (testSamples != 0) ? 0 : 1;

        //shuffle(&inputs, &targets, numSamples);
        
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
            numTrainLosses++;
            zero_everything(myModel->outLayer);
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
            
            forward_out(myModel->outLayer);

            testingLoss += loss_function(myModel);
            numTestLosses++;
            zero_everything(myModel->outLayer);
        }
        
        gettimeofday(&timecheck, NULL);
        eEnd = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
        
        trainingLoss /= numTrainLosses;
        testingLoss /=  numTestLosses;
        
        printf("Epoch %d - Training Loss: %f, Testing Loss: %f - %.1ldms\n", e, trainingLoss, testingLoss, eEnd - eStart);
    }

    gettimeofday(&timecheck, NULL);
    tEnd = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    printf("\nTotal training time: %ldms\n", (tEnd-tStart));
}

void train_rnn_sgd(model* myModel, int epochs, int numSamples, float** inputs, float **targets, float valSplit, int seqLength)
{
    int inputsTraversed = 0;
    int currTargs = 0;
    float trainingLoss = 0.0;
    float testingLoss = 0.0;
    int trainSamples = clip((int)(valSplit * numSamples), numSamples, 0);
    int testSamples = numSamples - trainSamples;
    int numTrainLosses = 0;
    int numTestLosses = 0;
    double timeElapsed;
    long tStart, tEnd, eStart, eEnd;
    struct timeval timecheck;

    gettimeofday(&timecheck, NULL);
    tStart = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    for(int e = 1; e < (epochs + 1); e++)
    {
        trainingLoss = 0.0;
        testingLoss = 0.0;
        numTrainLosses = (trainSamples != 0) ? 0 : 1;
        numTestLosses = (testSamples != 0) ? 0 : 1;

        gettimeofday(&timecheck, NULL);
        eStart = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
        
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

                forward_out(myModel->outLayer);
                
                shift_model(myModel->outLayer, 't');

                memcpy(myModel->targets, targets[currTargs], sizeof(float) * (*myModel->outLayer)->numNodes);
                trainingLoss += loss_function(myModel);
                numTrainLosses++;
            }

            sgd_backprop(myModel->outLayer, &myModel);
            calculate_and_apply_grads_through_time(myModel->outLayer, myModel->learning_rate);
            zero_everything(myModel->outLayer);
        }
        
        for(int i = trainSamples; i < numSamples; i++)
        {
            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                memcpy((*myModel->inLayers[j])->outputs, &(inputs[i][inputsTraversed]), sizeof(float) * (*myModel->inLayers[j])->numNodes);
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }
            
            forward_out(myModel->outLayer);
            memcpy(myModel->targets, targets[i], sizeof(float) * (*myModel->outLayer)->numNodes);
            testingLoss += loss_function(myModel);
            
            shift_model(myModel->outLayer, 'i');
            numTestLosses++;
            zero_base_model(myModel->outLayer);
        }

        gettimeofday(&timecheck, NULL);
        eEnd = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

        trainingLoss /= numTrainLosses;
        testingLoss /= numTestLosses;
        printf("Epoch %d - Training Loss: %f, Testing Loss: %f - %.2ldms\n", e, trainingLoss, testingLoss, eEnd - eStart);
    }
    
    gettimeofday(&timecheck, NULL);
    tEnd = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    printf("\nTotal training time: %ldms\n", (tEnd-tStart));
}

void train_model_adam(model* myModel, int epochs, int numSamples, float** inputs, float *targets, float initialFirstMomentum, float initialSecondMomentum);

void batch_train_adam(model* myModel, int epochs, int numSamples, int batchSize, float** inputs, float *targets, float initialFirstMomentum, float initialSecondMomentum);

void model_inference(model* myModel, float* inputs, float** outputs) //(model*, float*, &float[])
{
    int inputsTraversed = 0;
    for(int i = 0; i < myModel->numInLayers; i++)
    {
        for(int j = 0; j < (*myModel->inLayers[i])->numNodes; j++) (*myModel->inLayers[i])->outputs[j] = inputs[j + inputsTraversed];
        inputsTraversed += (*myModel->inLayers[i])->numNodes;
    }

    forward_out(myModel->outLayer);

    memcpy(*outputs, (*myModel->outLayer)->outputs, sizeof(float) * (*myModel->outLayer)->numNodes);
    zero_everything(myModel->outLayer);
}

void rnn_model_inference(model* myModel, float* inputs, float** outputs) //(model*, float*, &float[])
{
    int inputsTraversed = 0;
    for(int i = 0; i < myModel->numInLayers; i++)
    {
        for(int j = 0; j < (*myModel->inLayers[i])->numNodes; j++) (*myModel->inLayers[i])->outputs[j] = inputs[j + inputsTraversed];
        inputsTraversed += (*myModel->inLayers[i])->numNodes;
    }

    forward_out(myModel->outLayer);
    shift_model(myModel->outLayer, 'i');

    memcpy(*outputs, (*myModel->outLayer)->outputs, sizeof(float) * (*myModel->outLayer)->numNodes);
    zero_base_model(myModel->outLayer);
}

#if defined(__AVX__) || defined(__AVX2__)
void train_model_sgd_fast(model* myModel, int epochs, int numSamples, float** inputs, float **targets, float valSplit)
{
    int inputsTraversed = 0;
    float trainingLoss = 0.0;
    float testingLoss = 0.0;
    int trainSamples = (int)(valSplit * numSamples);
    int testSamples = numSamples - trainSamples;
    struct timespec start, end;
    double timeElapsed;
    long tStart, tEnd;
    struct timeval timecheck;

    gettimeofday(&timecheck, NULL);
    tStart = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    testSamples = (testSamples != 0) ? testSamples : -1;
    trainSamples = (trainSamples != 0) ? trainSamples : -1;
    
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
            
            _mm256_forward_out(myModel->outLayer);
            sgd_backprop(myModel->outLayer, &myModel);
            _mm256_calculate_and_apply_grads(myModel->outLayer, myModel->learning_rate);

            trainingLoss += loss_function(myModel);
            zero_everything(myModel->outLayer);
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
            
            _mm256_forward_out(myModel->outLayer);

            testingLoss += loss_function(myModel);
            zero_everything(myModel->outLayer);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        
        timeElapsed = (end.tv_nsec - start.tv_nsec) / 1000000.0;
        timeElapsed =  (timeElapsed >= 0) ? timeElapsed : 1000.0 + timeElapsed;

        testingLoss /=  testSamples;
        trainingLoss /= trainSamples;
        printf("Epoch %d - Training Loss: %f, Testing Loss: %f - %.1lfms\n", e, trainingLoss, testingLoss, timeElapsed);
    }
    
    gettimeofday(&timecheck, NULL);
    tEnd = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    printf("\nTotal training time: %ldms\n", (tEnd-tStart));

}

void train_rnn_sgd_fast(model* myModel, int epochs, int numSamples, float** inputs, float **targets, float valSplit, int seqLength)
{
    int inputsTraversed = 0;
    int currTargs = 0;
    float trainingLoss = 0.0;
    float testingLoss = 0.0;
    int trainSamples = clip((int)(valSplit * numSamples), numSamples, 0);
    int testSamples = numSamples - trainSamples;
    int numTrainLosses = 0;
    int numTestLosses = 0;
    double timeElapsed;
    long tStart, tEnd, eStart, eEnd;
    struct timeval timecheck;

    gettimeofday(&timecheck, NULL);
    tStart = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    for(int e = 1; e < (epochs + 1); e++)
    {
        trainingLoss = 0.0;
        testingLoss = 0.0;
        numTrainLosses = (trainSamples != 0) ? 0 : 1;
        numTestLosses = (testSamples != 0) ? 0 : 1;

        gettimeofday(&timecheck, NULL);
        eStart = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
        
        for(int i = 0; i < trainSamples; i += seqLength)
        {                
            inputsTraversed = 0;
            for(int j = i; j < i + seqLength && j < trainSamples; j++)
            {
                for(int k = 0; k < myModel->numInLayers; k++)
                {
                    memcpy((*myModel->inLayers[k])->outputs, &(inputs[j][inputsTraversed]), sizeof(float) * (*myModel->inLayers[k])->numNodes);
                    inputsTraversed += (*myModel->inLayers[k])->numNodes;
                }

                _mm256_forward_out(myModel->outLayer);
                
                shift_model(myModel->outLayer, 't');

                memcpy(myModel->targets, targets[currTargs], sizeof(float) * (*myModel->outLayer)->numNodes);
                trainingLoss += loss_function(myModel);
                numTrainLosses++;
            }

            sgd_backprop(myModel->outLayer, &myModel);
            _mm256_calculate_and_apply_grads_through_time(myModel->outLayer, myModel->learning_rate);
            zero_everything(myModel->outLayer);
        }
        
        for(int i = trainSamples; i < numSamples; i++)
        {
            inputsTraversed = 0;
            for(int j = 0; j < myModel->numInLayers; j++)
            {
                memcpy((*myModel->inLayers[j])->outputs, &(inputs[i][inputsTraversed]), sizeof(float) * (*myModel->inLayers[j])->numNodes);
                inputsTraversed += (*myModel->inLayers[j])->numNodes;
            }
            
            _mm256_forward_out(myModel->outLayer);
            memcpy(myModel->targets, targets[i], sizeof(float) * (*myModel->outLayer)->numNodes);
            testingLoss += loss_function(myModel);
            
            shift_model(myModel->outLayer, 'i');
            numTestLosses++;
            zero_base_model(myModel->outLayer);
        }

        gettimeofday(&timecheck, NULL);
        eEnd = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

        trainingLoss /= numTrainLosses;
        testingLoss /= numTestLosses;
        printf("Epoch %d - Training Loss: %f, Testing Loss: %f - %.2ldms\n", e, trainingLoss, testingLoss, eEnd - eStart);
    }
    
    gettimeofday(&timecheck, NULL);
    tEnd = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    printf("\nTotal training time: %ldms\n", (tEnd-tStart));
}

void model_inference_fast(model* myModel, float* inputs, float** outputs) //(model*, float*, &float[])
{
    int inputsTraversed = 0;
    for(int i = 0; i < myModel->numInLayers; i++)
    {
        for(int j = 0; j < (*myModel->inLayers[i])->numNodes; j++) (*myModel->inLayers[i])->outputs[j] = inputs[j + inputsTraversed];
        inputsTraversed += (*myModel->inLayers[i])->numNodes;
    }

    _mm256_forward_out(myModel->outLayer);

    memcpy(*outputs, (*myModel->outLayer)->outputs, sizeof(float) * (*myModel->outLayer)->numNodes);
    zero_everything(myModel->outLayer);
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

    vforward_out(myModel->outLayer);

    memcpy(*outputs, (*myModel->outLayer)->outputs, sizeof(float) * (*myModel->outLayer)->numNodes);
    zero_everything(myModel->outLayer);
}

void rnn_model_inference(model* myModel, float* inputs, float** outputs) //(model*, float*, &float[])
{
    int inputsTraversed = 0;
    for(int i = 0; i < myModel->numInLayers; i++)
    {
        for(int j = 0; j < (*myModel->inLayers[i])->numNodes; j++) (*myModel->inLayers[i])->outputs[j] = inputs[j + inputsTraversed];
        inputsTraversed += (*myModel->inLayers[i])->numNodes;
    }

    vforward_out(myModel->outLayer);
    shift_model(myModel->outLayer, 'i');

    memcpy(*outputs, (*myModel->outLayer)->outputs, sizeof(float) * (*myModel->outLayer)->numNodes);
    zero_base_model(myModel->outLayer);
}

#endif