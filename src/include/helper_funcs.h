#pragma once

#include "nn_math.h"
#include "model.h"
#include <immintrin.h>

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

void flush_buffer(char* buffer, int size)
{
    for(int i = 0; i < size; i++) buffer[i] = '\0';
}

int read_csv(const char* fileName, int numSamples, int numInputs, int numOutputs, float*** inArrs, float*** outArrs)
{
    FILE *datFile = fopen(fileName, "r");
    if(datFile == NULL) goto error1;

    char buffer[128];
    flush_buffer(buffer, 128);
    
    char fltBuffer[24];
    flush_buffer(fltBuffer, 24);

    int fltTraversed = 0;
    int offset = 0;

    if(fgets(buffer, 128, datFile) == NULL) goto error2; // Sacrificial getline for the header info line
    flush_buffer(buffer, 80);
    
    *inArrs = (float**)malloc(numSamples * sizeof(float *));
    *outArrs = (float**)malloc(numSamples * sizeof(float *));

    for(int i = 0; i < numSamples; i++)
    {
        if(fgets(buffer, 128, datFile) == NULL) goto error2;

        offset = 0;
        (*inArrs)[i] = (float*)malloc(numInputs * sizeof(float));
        (*outArrs)[i] = (float*)malloc(numOutputs * sizeof(float));
        
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
            (*inArrs)[i][j] = atof(fltBuffer);
            flush_buffer(fltBuffer, 24);
            fltTraversed = 0;
        }

        for(int j = 0; j < numOutputs; j++)
        {
            while(buffer[offset] != ',' && buffer[offset] != ';')
            {
                fltBuffer[fltTraversed] = buffer[offset];
                offset += 1;
                fltTraversed += 1;
            }
            
            offset += 1;
            (*outArrs)[i][j] = atof(fltBuffer);
            flush_buffer(fltBuffer, 24);
            fltTraversed = 0;
        }

        flush_buffer(buffer, 128);
    }

    fclose(datFile);
    datFile = NULL;
    return 0;

error2:
    fclose(datFile);
    datFile = NULL;
error1:
    return -1;
}

static inline float _mm256_sum_manual(__m256 v)
{
    float tmp[8];
    _mm256_storeu_ps(tmp, v);

    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

// for(int i = 0; i < (*myLayer)->numNodes; i++) 
// {
//     for(int j = 0; j < (*myLayer)->numPrevLayers; j++)
//     {
//         for(int k = 0; k < (*(*myLayer)->prevLayers[j])->numNodes; k++) (*myLayer)->preActivations[i] += (*(*myLayer)->prevLayers[j])->outputs[k] * (*myLayer)->weights[i][numPrevsTraversed + k];
//         numPrevsTraversed += (*(*myLayer)->prevLayers[j])->numNodes;
//     }
//     (*myLayer)->preActivations[i] += (*myLayer)->biases[i];
//     numPrevsTraversed = 0;
// }

int vectorized_forward_out_calc(layer** myLayer)
{
    int maskHelp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    __m256 preActs = _mm256_setzero_ps();
    __m256 prevOuts = _mm256_setzero_ps();
    __m256 mulWeights = _mm256_setzero_ps();
    int numPrevsTraversed = 0;
    int leftoverBatch = (*myLayer)->numPrevNodes % 8;
    int batchesOfEight = ((*myLayer)->numPrevNodes - leftoverBatch) / 8;
    
    for(int i = 0; i < leftoverBatch; i++)
    {
        maskHelp[i] = -1;
    }

    __m256i _load_mask = _mm256_loadu_si256((const __m256i*)maskHelp);

    float* prevNodeOuts = (float*)calloc((*myLayer)->numPrevNodes, sizeof(float));
    if(prevNodeOuts == NULL) return -1;
    for(int i = 0; i < (*myLayer)->numPrevLayers; i++)
    {
        memcpy(&prevNodeOuts[numPrevsTraversed], (*(*myLayer)->prevLayers[i])->outputs, (*(*myLayer)->prevLayers[i])->numNodes * sizeof(float));
        numPrevsTraversed += (*(*myLayer)->prevLayers[i])->numNodes;
    }

    for(int i = 0; i < (*myLayer)->numNodes; i++)
    {
        for(int j = 0; j < batchesOfEight; j++)
        {
            prevOuts = _mm256_loadu_ps(&prevNodeOuts[8 * j]);
            mulWeights = _mm256_loadu_ps(&(*myLayer)->weights[i][(8 * j)]);

            preActs = _mm256_fmadd_ps(prevOuts, mulWeights, preActs);
        }

        if(leftoverBatch > 0)
        {
            prevOuts = _mm256_maskload_ps(&prevNodeOuts[8 * batchesOfEight], _load_mask);
            mulWeights = _mm256_maskload_ps(&(*myLayer)->weights[i][(8 * batchesOfEight)], _load_mask);

            preActs = _mm256_fmadd_ps(prevOuts, mulWeights, preActs);
        }
        
        (*myLayer)->preActivations[i] = _mm256_sum_manual(preActs);
        preActs = _mm256_setzero_ps();
    }

    free(prevNodeOuts);
    prevNodeOuts = NULL;

    return 0;
}

int vectorized_calculate_and_apply_grads(layer** myLayer, float learningRate)
{
    int maskHelp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float f[8];
    __m256 learningRates = _mm256_set1_ps(learningRate);
    __m256 biases = _mm256_setzero_ps();
    __m256 backErrors = _mm256_setzero_ps();
    __m256 prevOuts = _mm256_setzero_ps();
    __m256 weights =_mm256_setzero_ps();
    int numPrevsTraversed = 0;
    int leftoverBatchPrevs = (*myLayer)->numPrevNodes % 8;
    int batchesOfEightPrevs = ((*myLayer)->numPrevNodes - leftoverBatchPrevs) / 8;
    int leftoverBatch = (*myLayer)->numNodes % 8;
    int batchesOfEight = ((*myLayer)->numNodes - leftoverBatch) / 8;
    
    
    for(int i = 0; i < leftoverBatch; i++)
    {
        maskHelp[i] = -1;
    }

    __m256i _load_mask = _mm256_loadu_si256((const __m256i*)maskHelp);


    for(int i = 0; i < (*myLayer)->numNodes; i++)
    {
        for(int j = 0; j < batchesOfEight; j++)
        {
            biases = _mm256_loadu_ps(&(*myLayer)->biases[8 * j]);
            backErrors = _mm256_loadu_ps(&(*myLayer)->backErrors[8 * j]);

            biases = _mm256_fmsub_ps(backErrors, learningRates, biases);
            _mm256_storeu_ps(f, biases);

            memcpy(&(*myLayer)->biases[8 * j], f, sizeof(float) * 8);
        }

        if(leftoverBatch > 0)
        {
            biases = _mm256_maskload_ps(&(*myLayer)->biases[8 * batchesOfEight], _load_mask);
            backErrors = _mm256_maskload_ps(&(*myLayer)->backErrors[8 * batchesOfEight], _load_mask);

            biases = _mm256_fmsub_ps(backErrors, learningRates, biases);
            _mm256_storeu_ps(f, biases);

            memcpy(&(*myLayer)->biases[8 * batchesOfEight], f, sizeof(float) * leftoverBatch);
        }        
    }

    memset(maskHelp, 0, sizeof(maskHelp));
    for(int i = 0; i < leftoverBatchPrevs; i++)
    {
        maskHelp[i] = -1;
    }

    _load_mask = _mm256_loadu_si256((const __m256i*)maskHelp);

    float* prevNodeOuts = (float*)calloc((*myLayer)->numPrevNodes, sizeof(float));
    if(prevNodeOuts== NULL) return -1;
    for(int i = 0; i < (*myLayer)->numPrevLayers; i++)
    {
        memcpy(&prevNodeOuts[numPrevsTraversed], (*(*myLayer)->prevLayers[i])->outputs, (*(*myLayer)->prevLayers[i])->numNodes * sizeof(float));
        numPrevsTraversed += (*(*myLayer)->prevLayers[i])->numNodes;
    }

    for(int i = 0; i < (*myLayer)->numNodes; i++)
    {
        backErrors = _mm256_set1_ps((*myLayer)->backErrors[i]);
        backErrors = _mm256_mul_ps(backErrors, learningRates);
        numPrevsTraversed = 0;

        for(int j = 0; j < batchesOfEightPrevs; j++)
        {
            prevOuts = _mm256_loadu_ps(&prevNodeOuts[8 * j]);
            weights = _mm256_loadu_ps(&(*myLayer)->weights[i][(8 * j)]);

            weights = _mm256_fmsub_ps(prevOuts, backErrors, weights);
            _mm256_storeu_ps(f, weights);

            memcpy(&(*myLayer)->weights[i][8 * j], f, sizeof(float) * 8);
        }

        if(leftoverBatchPrevs > 0)
        {
            prevOuts = _mm256_maskload_ps(&prevNodeOuts[8 * batchesOfEightPrevs], _load_mask);
            weights = _mm256_maskload_ps(&(*myLayer)->weights[i][(8 * batchesOfEightPrevs)], _load_mask);

            weights = _mm256_fmsub_ps(prevOuts, backErrors, weights);
            _mm256_storeu_ps(f, weights);

            memcpy(&(*myLayer)->weights[i][8 * batchesOfEightPrevs], f, sizeof(float) * leftoverBatchPrevs);
        }
        
    }

    free(prevNodeOuts);
    prevNodeOuts = NULL;

    return 0;
}