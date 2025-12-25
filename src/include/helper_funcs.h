#pragma once

#include "nn_math.h"
#include "model.h"

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

    fgets(buffer, 128, datFile); // Sacrificial getline for the header info line
    flush_buffer(buffer, 80);
    
    *inArrs = (float**)malloc(numSamples * sizeof(float *));
    *outArrs = (float**)malloc(numSamples * sizeof(float *));

    for(int i = 0; i < numSamples; i++)
    {
        fgets(buffer, 128, datFile);

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

error1:
    return -1;
}
