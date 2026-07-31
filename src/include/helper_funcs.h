#pragma once

#include "layer_destruct.h"
#include "nn_math.h"
#include "model.h"

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif



void flush_buffer(char* buffer, int size)
{
    for(int i = 0; i < size; i++) buffer[i] = '\0';
}

void int2bin(int x, int numBits, char* bitBuff)
{
    int myX = x;

    for(int i = 0; i < numBits; i++)
    {
        bitBuff[i] = (myX & 1) ? '1' : '0';
        myX >>= 1; 
    }
}

int bin2int(const char* bin, int size)
{
    int retVal = 0;
    int currCount = 1;

    for(int i = 0; i < size; i++)
    {
        if(bin[i] == '1')
        {
            retVal += currCount;
        }
        currCount <<= 1;
    }

    return retVal;
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

// If you have trouble accessing a CSV or model save file make sure the program is being run in the same directory as the file that it is referencing
// Or provide the full path to the file in the filename 
int read_csv(const char* fileName, int numSamples, int numInputs, int numOutputs, float*** inArrs, float*** outArrs)
{
    FILE *datFile = fopen(fileName, "r");
    if(datFile == NULL) goto error1;

    char buffer[128] = {'\0'};   
    char fltBuffer[24] = {'\0'};
    int fltTraversed = 0;
    int offset = 0;

    if(fgets(buffer, 128, datFile) == NULL) goto error2; // Sacrificial getline for the header info line
    flush_buffer(buffer, 80);
    
    *inArrs = (float**)malloc(numSamples * sizeof(float *));
    *outArrs = (float**)malloc(numSamples * sizeof(float *));

    float** inArrsRef = *inArrs;
    float** outArrsRef = *outArrs; 

    for(int i = 0; i < numSamples; i++)
    {
        if(fgets(buffer, 128, datFile) == NULL) goto error2;

        offset = 0;
        inArrsRef[i] = (float*)malloc(numInputs * sizeof(float));
        outArrsRef[i] = (float*)malloc(numOutputs * sizeof(float));
        
        for(int j = 0; j < numInputs; j++)
        {
            while(buffer[offset] != ',')
            {
                if(offset > 127) goto error1;
                
                fltBuffer[fltTraversed] = buffer[offset];
                offset += 1;
                fltTraversed += 1;
            }

            offset += 1;
            inArrsRef[i][j] = atof(fltBuffer);
            flush_buffer(fltBuffer, 24);
            fltTraversed = 0;
        }

        for(int j = 0; j < numOutputs; j++)
        {
            while(buffer[offset] != ',' && buffer[offset] != '\n')
            {
                fltBuffer[fltTraversed] = buffer[offset];
                offset += 1;
                fltTraversed += 1;
            }
            
            offset += 1;
            outArrsRef[i][j] = atof(fltBuffer);
            flush_buffer(fltBuffer, 24);
            fltTraversed = 0;
        }

        flush_buffer(buffer, 128);
    }

    fclose(datFile);
    datFile = NULL;
    return 0;

error2:
    hakai_matrix(inArrs, numSamples);
    hakai_matrix(outArrs, numSamples);
    fclose(datFile);
    datFile = NULL;
error1:
    return -1;
}

#if defined(__AVX__) || defined(__AVX2__)

static inline float _mm256_sum_manual(__m256 v)
{
    float tmp[8];
    _mm256_storeu_ps(tmp, v);

    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

int vectorized_forward_out_calc(layer* myLayer)
{
    float prevNodeOuts[myLayer->numPrevNodes];
    int maskHelp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    layer* prevLayer;
    __m256 preActs = _mm256_setzero_ps();
    __m256 prevOuts = _mm256_setzero_ps();
    __m256 mulWeights = _mm256_setzero_ps();
    int numPrevsTraversed = 0;
    int leftoverBatch = myLayer->numPrevNodes % 8;
    int batchesOfEight = (myLayer->numPrevNodes - leftoverBatch) / 8;
    
    for(int i = 0; i < leftoverBatch; i++) maskHelp[i] = -1;

    __m256i _load_mask = _mm256_loadu_si256((const __m256i*)maskHelp);

    for(int i = 0; i < myLayer->numPrevLayers; i++)
    {
        prevLayer = *myLayer->prevLayers[i];

        memcpy(&prevNodeOuts[numPrevsTraversed], prevLayer->outputs, prevLayer->numNodes * sizeof(float));
        numPrevsTraversed += prevLayer->numNodes;
    }

    for(int i = 0; i < myLayer->numNodes; i++)
    {
        for(int j = 0; j < batchesOfEight; j++)
        {
            prevOuts = _mm256_loadu_ps(&prevNodeOuts[8 * j]);
            mulWeights = _mm256_loadu_ps(&myLayer->weights[i][(8 * j)]);

            preActs = _mm256_fmadd_ps(prevOuts, mulWeights, preActs);
        }

        if(leftoverBatch > 0)
        {
            prevOuts = _mm256_maskload_ps(&prevNodeOuts[8 * batchesOfEight], _load_mask);
            mulWeights = _mm256_maskload_ps(&myLayer->weights[i][(8 * batchesOfEight)], _load_mask);

            preActs = _mm256_fmadd_ps(prevOuts, mulWeights, preActs);
        }
        
        myLayer->preActivations[i] = _mm256_sum_manual(preActs) + myLayer->biases[i];
        preActs = _mm256_setzero_ps();
    }

    return 0;
}

// void vectorized_sgd_backprop_output_calc(layer* myLayer, model* myModel)
// { // start at output layer and calculate backerrors for each previous layer
    
//     float lossDer[myLayer->numNodes];
//     float actDer[myLayer->numNodes];
//     float f[8];
//     int maskHelp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
//     layer* prevLayer;
//     __m256 lossDerivatives = _mm256_setzero_ps();
//     __m256 activationDerivatives = _mm256_setzero_ps();
//     __m256 backErrors = _mm256_setzero_ps();
    
//     int leftoverBatch = myLayer->numNodes % 8;
//     int batchesOfEight = (myLayer->numNodes - leftoverBatch) / 8;
//     // backErrorsForOutputLayer = lossDerivative · activationFunctionDerivative(preActivations) - for output layer

//     for(int i = 0; i < leftoverBatch; i++) maskHelp[i] = -1;

//     __m256i _load_mask = _mm256_loadu_si256((const __m256i*)maskHelp);

//     for(int i = 0; i < myLayer->numNodes; i++)
//     {
//         lossDer[i] = -1 * loss_derivative(myModel->targets[i], myLayer->outputs[i], myModel);
//         actDer[i] = activation_derivative(myLayer->preActivations[i], myLayer, i);
//     }

//     for(int j = 0; j < batchesOfEight; j++)
//     {
//         lossDerivatives = _mm256_loadu_ps(&lossDer[8 * j]);
//         activationDerivatives = _mm256_loadu_ps(&actDer[8 * j]);

//         backErrors = _mm256_mul_ps(lossDerivatives, activationDerivatives);
//         _mm256_storeu_ps(f, backErrors);

//         memcpy(&myLayer->backErrors[8 * j], f, sizeof(float) * 8);
//     }

//     if(leftoverBatch > 0)
//     {
//         lossDerivatives = _mm256_maskload_ps(&lossDer[8 * batchesOfEight], _load_mask);
//         activationDerivatives = _mm256_maskload_ps(&actDer[8 * batchesOfEight], _load_mask);

//         backErrors = _mm256_mul_ps(lossDerivatives, activationDerivatives);
//         _mm256_storeu_ps(f, backErrors);

//         memcpy(&myLayer->backErrors[8 * batchesOfEight], f, sizeof(float) * leftoverBatch);
//     }   
// }

// // void vectorized_sgd_backprop_calc(layer* myLayer, model* myModel, int prevLayerNum, int prevsTraversed)
// // { // start at output layer and calculate backerrors for each previous layer
    
// //     int maskHelp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
// //     float prevLayerPreActivations[myLayer->numNodes];
// //     float f[8];
// //     layer* prevLayer;
// //     __m256 myBackErrors = _mm256_setzero_ps();
// //     __m256 prevBackErrors = _mm256_setzero_ps();
// //     __m256 prevPreActivations = _mm256_setzero_ps();
// //     __m256 weights =_mm256_setzero_ps();
    
// //     int leftoverBatchPrevs = (*myLayer->prevLayers[prevLayerNum])->numNodes % 8;
// //     int batchesOfEightPrevs = ((*myLayer->prevLayers[prevLayerNum])->numNodes - leftoverBatchPrevs) / 8;
// //     int leftoverBatch = myLayer->numNodes % 8;
// //     int batchesOfEight = (myLayer->numNodes - leftoverBatch) / 8;

// //     for(int j = 0; j < batchesOfEight; j++)
// //     {
// //         biases = _mm256_loadu_ps(&myLayer->biases[8 * j]);
// //         backErrors = _mm256_loadu_ps(&myLayer->backErrors[8 * j]);

// //         biases = _mm256_fnmadd_ps(backErrors, learningRates, biases);
// //         _mm256_storeu_ps(f, biases);

// //         memcpy(&myLayer->biases[8 * j], f, sizeof(float) * 8);
// //     }

// //     if(leftoverBatch > 0)
// //     {
// //         biases = _mm256_maskload_ps(&myLayer->biases[8 * batchesOfEight], _load_mask);
// //         backErrors = _mm256_maskload_ps(&myLayer->backErrors[8 * batchesOfEight], _load_mask);

// //         biases = _mm256_fnmadd_ps(backErrors, learningRates, biases);
// //         _mm256_storeu_ps(f, biases);

// //         memcpy(&myLayer->biases[8 * batchesOfEight], f, sizeof(float) * leftoverBatch);
// //     }  

// //     // for(int i = 0; i < leftoverBatch; i++) maskHelp[i] = -1;

// //     // // backErrorsForPreviousLayers[j] = SUM_OVER_I((thisLayersBackErrors[i])(thisLayersWeightMatrix[i][j]) · activationFunctionDerivative(previousLayersPreActivation[j])) - where j is considered to be a traversal of all previous 'J' layers' 'K' values as one vector
// //     // // e.g. J = 3 prev layers with K = 5 nodes each are considered as one prev layer with J = 15 nodes in this formulation
// //     // for(int i = 0; i < myLayer->numPrevLayers; i++)
// //     //     for(int j = 0; j < prevLayer->numNodes; j++) 
// //     //         for(int k = 0; k < myLayer->numNodes; k++) 
// //     //             prevLayer->backErrors[j] += myLayer->backErrors[i] * myLayer->weights[k][prevsTraversed + j] * activation_derivative(prevLayer->preActivations[j], prevLayer->activationFunction, myLayer, i);
// //     //     prevsTraversed += prevLayer->numNodes;
// //     // }
// // }

// int vectorized_calculate_and_apply_grads(layer* myLayer, float learningRate)
// {
//     int maskHelp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
//     float prevNodeOuts[myLayer->numPrevNodes];
//     float f[8];
//     layer* prevLayer;
//     __m256 learningRates = _mm256_set1_ps(learningRate);
//     __m256 biases = _mm256_setzero_ps();
//     __m256 backErrors = _mm256_setzero_ps();
//     __m256 prevOuts = _mm256_setzero_ps();
//     __m256 weights =_mm256_setzero_ps();
//     int numPrevsTraversed = 0;
//     int leftoverBatchPrevs = myLayer->numPrevNodes % 8;
//     int batchesOfEightPrevs = (myLayer->numPrevNodes - leftoverBatchPrevs) / 8;
//     int leftoverBatch = myLayer->numNodes % 8;
//     int batchesOfEight = (myLayer->numNodes - leftoverBatch) / 8;
    
    
//     for(int i = 0; i < leftoverBatch; i++) maskHelp[i] = -1;

//     __m256i _load_mask = _mm256_loadu_si256((const __m256i*)maskHelp);

//     for(int j = 0; j < batchesOfEight; j++)
//     {
//         biases = _mm256_loadu_ps(&myLayer->biases[8 * j]);
//         backErrors = _mm256_loadu_ps(&myLayer->backErrors[8 * j]);

//         biases = _mm256_fnmadd_ps(backErrors, learningRates, biases);
//         _mm256_storeu_ps(f, biases);

//         memcpy(&myLayer->biases[8 * j], f, sizeof(float) * 8);
//     }

//     if(leftoverBatch > 0)
//     {
//         biases = _mm256_maskload_ps(&myLayer->biases[8 * batchesOfEight], _load_mask);
//         backErrors = _mm256_maskload_ps(&myLayer->backErrors[8 * batchesOfEight], _load_mask);

//         biases = _mm256_fnmadd_ps(backErrors, learningRates, biases);
//         _mm256_storeu_ps(f, biases);

//         memcpy(&myLayer->biases[8 * batchesOfEight], f, sizeof(float) * leftoverBatch);
//     }        

//     memset(maskHelp, 0, sizeof(maskHelp));
//     for(int i = 0; i < leftoverBatchPrevs; i++) maskHelp[i] = -1;

//     _load_mask = _mm256_loadu_si256((const __m256i*)maskHelp);

//     for(int i = 0; i < myLayer->numPrevLayers; i++)
//     {
//         prevLayer = *myLayer->prevLayers[i];
        
//         memcpy(&prevNodeOuts[numPrevsTraversed], prevLayer->outputs, prevLayer->numNodes * sizeof(float));
//         numPrevsTraversed += prevLayer->numNodes;
//     }

//     for(int i = 0; i < myLayer->numNodes; i++)
//     {
//         backErrors = _mm256_set1_ps(myLayer->backErrors[i]);
//         backErrors = _mm256_mul_ps(backErrors, learningRates);
//         numPrevsTraversed = 0;

//         for(int j = 0; j < batchesOfEightPrevs; j++)
//         {
//             prevOuts = _mm256_loadu_ps(&prevNodeOuts[8 * j]);
//             weights = _mm256_loadu_ps(&myLayer->weights[i][(8 * j)]);

//             weights = _mm256_fnmadd_ps(prevOuts, backErrors, weights);
//             _mm256_storeu_ps(f, weights);

//             memcpy(&myLayer->weights[i][8 * j], f, sizeof(float) * 8);
//         }

//         if(leftoverBatchPrevs > 0)
//         {
//             prevOuts = _mm256_maskload_ps(&prevNodeOuts[8 * batchesOfEightPrevs], _load_mask);
//             weights = _mm256_maskload_ps(&myLayer->weights[i][(8 * batchesOfEightPrevs)], _load_mask);

//             weights = _mm256_fnmadd_ps(prevOuts, backErrors, weights);
//             _mm256_storeu_ps(f, weights);

//             memcpy(&myLayer->weights[i][8 * batchesOfEightPrevs], f, sizeof(float) * leftoverBatchPrevs);
//         }
//     }

//     return 0;
// }


#endif

#if defined(__ARM_NEON)

int vectorized_forward_out_calc(layer* myLayer)
{
    layer* prevLayer;
    float32x4_t preActs = vdupq_n_f32(0.0f);
    float32x4_t prevOuts = vdupq_n_f32(0.0f);
    float32x4_t mulWeights = vdupq_n_f32(0.0f);
    float tailSum = 0.0;
    int numPrevsTraversed = 0;
    int leftoverBatch = myLayer->numPrevNodes % 4;
    int batchesOfFour = (myLayer->numPrevNodes - leftoverBatch) / 4;

    float* prevNodeOuts = (float*)calloc(myLayer->numPrevNodes, sizeof(float));
    if(prevNodeOuts == NULL) return -1;
    for(int i = 0; i < myLayer->numPrevLayers; i++)
    {
        prevLayer = *myLayer->prevLayers[i];
        
        memcpy(&prevNodeOuts[numPrevsTraversed], prevLayer->outputs, prevLayer->numNodes * sizeof(float));
        numPrevsTraversed += prevLayer->numNodes;
    }

    for(int i = 0; i < myLayer->numNodes; i++)
    {
        for(int j = 0; j < batchesOfFour; j++)
        {
            prevOuts = vld1q_f32(&prevNodeOuts[4 * j]);
            mulWeights = vld1q_f32(&myLayer->weights[i][(4 * j)]);

            preActs = vfmaq_f32(preActs, prevOuts, mulWeights);
        }

        for(int i = 0; i < leftoverBatch; i++) tailSum += prevNodeOuts[(4 * batchesOfFour) + i] * myLayer->weights[i][(4 * batchesOfFour) + i];
        
        myLayer->preActivations[i] = vaddvq_f32(preActs) + tailSum + myLayer->biases[i];
        preActs = vdupq_n_f32(0.0f);
    }

    free(prevNodeOuts);
    prevNodeOuts = NULL;

    return 0;
}

#endif