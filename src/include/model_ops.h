#pragma once

#include "layer_construct.h"
#include "model_construct.h"
#include "activation.h"
#include "loss.h"
#include "helper_funcs.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef union 
{
    float flt;
    unsigned char chars[sizeof(float)];
} fltChars;

typedef union 
{
    uint32_t num;
    unsigned char chars[4];
} intChars;

// Gets outputs from the layers, is essentially also a inference function
void forward_out(model* myModel, float dropoutVal)
{
    layer* currLayer;
    layer* prevLayer;
    // layer* otherPrev;
    float* currCol;
    float* activationDerivatives;
    float preAct;
    int currNumNodes;
    int prevNumNodes;
    int numPrevsTraversed = 0;

    if(myModel == NULL) return;

    for(int l = 0; l < myModel->numLayers; l++)
    {        
        currLayer = *myModel->layerList[l];

        if(currLayer->layerType == 'i' || currLayer->layerType == 't' || currLayer->layerType == 'w') continue;

        currNumNodes = currLayer->numNodes;

        // if(currLayer->layerType == 'a')
        // {
        //     memset(currLayer->preActivations, 0.0, currLayer->numPrevLayers * sizeof(float));
        //     memset(currLayer->outputs, 0.0, currNumNodes * sizeof(float));
        //     for(int i = 0; i < currLayer->numPrevLayers; i++)
        //     {
        //         prevLayer = *currLayer->prevLayers[i];
        //         for(int j = 0; j <= i; j++)
        //         {
        //             otherPrev = *currLayer->prevLayers[j];
        //             for(int k = 0; k < currNumNodes; k++) currLayer->preActivations[i] += prevLayer->outputs[k] * otherPrev->outputs[k];
        //         }

        //         currLayer->preActivations[i] /= sqrt(currNumNodes);
        //     }

        //     softmax(currLayer->preActivations, currLayer->numPrevLayers);

        //     for(int i = 0; i < currLayer->numPrevLayers; i++)
        //     {
        //         prevLayer = *currLayer->prevLayers[i];
        //         for(int j = 0; j < currNumNodes; j++) currLayer->outputs[j] += currLayer->preActivations[i] * prevLayer->outputs[j];
        //     }

        //     continue;
        // }
        
        // preActivation[i] = SUM_OVER_J(prevNodeOutputs[j] * weights[i][j]) 
        for(int i = 0; i < currNumNodes; i++, numPrevsTraversed = 0) 
        {
            preAct = 0.0;
            currCol = currLayer->weights[i];
            for(int j = 0; j < currLayer->numPrevLayers; j++)
            {
                prevLayer = *currLayer->prevLayers[j];
                prevNumNodes = prevLayer->numNodes;
                
                for(int k = 0; k < prevNumNodes; k++) preAct += prevLayer->outputs[k] * currCol[numPrevsTraversed + k];
                numPrevsTraversed += prevNumNodes;
            }
            currLayer->preActivations[i] =  preAct + currLayer->biases[i];
        }
        
        numPrevsTraversed = 0;

        // In case of softmax activation, we do a function on the layer instead of point-wise on the outputs
        if(currLayer->activationFunction == 'x')
        {
            memcpy(currLayer->outputs, currLayer->preActivations, sizeof(float) * currNumNodes);
            softmax(currLayer->outputs, currNumNodes);
        }
        else if(currLayer->activationFunction == 'f')
        {
            memcpy(currLayer->outputs, currLayer->preActivations, sizeof(float) * currNumNodes);
            fast_softmax(currLayer->outputs, currNumNodes);
        }
        else
        {
            // outputs[i] = activation_function(preActivations[i])
            activationDerivatives = currLayer->activationDerivatives;
            for(int i = 0; i < currNumNodes; i++)
            {
                currLayer->outputs[i] = activation_function(currLayer->preActivations[i], currLayer->activationFunction);
                activationDerivatives[i] = activation_derivative(currLayer->preActivations[i], currLayer, i);
            }
        }

        if(currLayer->layerType == 'o' || dropoutVal <= 0.0f) continue;;
        float scalingFactor = 1/(1-dropoutVal);
        for(int i = 0; i < currNumNodes; i++) currLayer->outputs[i] *= ((float)((rand() % 999))/1000.0 < dropoutVal) ? 0 : scalingFactor;
    }
}

// We pass the backerrors to each previous layer to calculate grads later
// Backerrors can be accumulated from multiple successor layers to calculate grads due to matrix distributivity
void sgd_backprop(model* myModel, float* classWeights)
{
    layer* currLayer;
    layer* prevLayer;
    float* lossDerivatives;
    float* activationDerivatives;
    float prevBackerror = 0.0;
    // float accum1 = 0.0;
    // float accum2 = 0.0;
    // float invDim;
    int prevsTraversed = 0;
    int currNumNodes;
    int prevNumNodes;
    
    if(myModel == NULL) return;

    lossDerivatives = myModel->lossDerivatives;

    currLayer = *myModel->outLayer;
    
    // If the loss functions are CCE, fast CCE, BCE, or fast BCE, the output activation functions are guaranteed to be softmax, fast softmax, sigmoid, or fast sigmoid respectively by model construction initiative
    switch(myModel->loss_fn)
    {
        case 'c':
        case 'x':
            for(int i = 0; i < currLayer->numNodes; i++) currLayer->backErrors[i] = classWeights[i] * (currLayer->outputs[i] - myModel->targets[i]);
            break;
        case 'n':
        case 'r':
            for(int i = 0; i < currLayer->numNodes; i++) currLayer->backErrors[i] = classWeights[(myModel->targets[i] == 0) ? 0 : 1] * (currLayer->outputs[i] - myModel->targets[i]);
            break;
        default:
            activationDerivatives = currLayer->activationDerivatives;

            // backErrorsForOutputLayer = lossDerivative · activationFunctionDerivative(preActivations)
            for(int i = 0; i < currLayer->numNodes; i++) 
            {
                myModel->lossDerivatives[i] = loss_derivative(myModel->targets[i], currLayer->outputs[i], myModel, 0); // Will break for binary cross entropy derivative without sigmoid activation
                currLayer->backErrors[i] = lossDerivatives[i] * activationDerivatives[i];
            }
    }

    for(int l = myModel->numLayers - 2; l > -1; l--)
    {        
        currLayer = *myModel->layerList[l];
        if(currLayer->layerType == 'i' || currLayer->layerType == 't' ) continue;
        activationDerivatives = currLayer->activationDerivatives;
        for(int i = 0; i < currLayer->numNodes; i++) activationDerivatives[i] = activation_derivative(currLayer->preActivations[i], currLayer, i);
    }

    // Start at output layer and calculate backerrors for each previous layer
    for(int l = myModel->numLayers - 1; l > -1; l--, prevsTraversed = 0)
    {
        currLayer = *myModel->layerList[l];

        if(currLayer->numPrevLayers == 0 || currLayer->layerType =='w') continue;

        currNumNodes = currLayer->numNodes;

        // backErrorsForPreviousLayers[j] = SUM_OVER_I((thisLayersBackErrors[i])(thisLayersWeightMatrix[i][j]) · activationFunctionDerivative(previousLayersPreActivation[j])) - where j is considered to be a traversal of all previous 'J' layers' 'K' values as one vector
        // e.g. J = 3 prev layers with K = 5 nodes each are considered as one prev layer with J = 15 nodes in this formulation
        for(int i = 0; i < currLayer->numPrevLayers; i++)
        {
            prevLayer = *currLayer->prevLayers[i];
            
            if(!prevLayer->numPrevLayers) continue;
            if(prevLayer->layerType == 'w')
            {
                prevsTraversed += prevNumNodes; // Will be redundant when rnn is fixed
                continue;
            }

            prevNumNodes = prevLayer->numNodes;

            if (currLayer->layerType == 'a')
            {
                /*invDim = 1/sqrt(currNumNodes);

                for(int j = 0; j < prevNumNodes; j++, prevBackerror = 0.0f, accum1 = 0.0, accum2 = 0.0)
                {
                    prevBackerror += currLayer->preActivations[i] * currLayer->backErrors[j];

                    for(int r = 0; r <= i; r++) accum1 += (*currLayer->prevLayers[r])->outputs[j];
                    accum1 *= (currLayer->preActivations[i] * currLayer->preActivations[i] * invDim);

                    for(int p = 0; p < currNumNodes; p++) accum2 += currLayer->backErrors[p] * (prevLayer->outputs[p] - currLayer->outputs[p]);

                    prevBackerror += accum1 * accum2;
                    accum1 = 0.0f;
                    accum2 = 0.0f;

                    for(int n = i + 1; n < currLayer->numPrevLayers; n++, accum2 = 0.0f)
                    {
                        for(int p = 0; p < currNumNodes; p++) accum2 += currLayer->backErrors[p] * ((*currLayer->prevLayers[n])->outputs[p] - currLayer->outputs[p]);
                        accum1 += currLayer->preActivations[n] * currLayer->preActivations[n] * (*currLayer->prevLayers[n])->outputs[j] * accum2;
                    }

                    prevBackerror += invDim * accum1;

                    prevBackerror *= activation_derivative(prevLayer->preActivations[j], prevLayer, j);

                    prevLayer->backErrors[j] += prevBackerror;
                }
                
                */continue;
            }

            activationDerivatives = prevLayer->activationDerivatives;
            
            for(int j = 0; j < prevNumNodes; j++, prevBackerror = 0.0f)
            {
                for(int k = 0; k < currNumNodes; k++) prevBackerror += currLayer->backErrors[k] * currLayer->weights[k][prevsTraversed + j] * activationDerivatives[j];
                prevLayer->backErrors[j] += prevBackerror;
            }
            prevsTraversed += prevNumNodes;
        }
    }
}

void calculate_and_apply_grads(model* myModel)
{
    layer* currLayer;
    layer* prevLayer;
    layer* windowLayer;
    float* backErrors;
    float* prevOutputs;
    float learningRate = myModel->learningRate;
    float backError;
    float currSum = 0.0;
    int currNumNodes;
    int prevNumNodes;
    int prevsTraversed = 0;

    if(myModel == NULL) return;

    for(int l = 0; l < myModel->numLayers; l++)
    {
        currLayer = *myModel->layerList[l];
        if(currLayer->layerType == 'a') memset(prevLayer->backErrors, 0.0, prevNumNodes * sizeof(float));
        if(!currLayer->numPrevLayers || currLayer->layerType == 'w' || currLayer->layerType == 'a') continue;

        currNumNodes = currLayer->numNodes;
        backErrors = currLayer->backErrors;

        prevsTraversed = 0;

        // If we are dealing with a context window, update sensitivity gradient vectors and then multiply them by the backerrors to obtain 
        if((*currLayer->prevLayers[currLayer->numPrevLayers - 1])->layerType == 'w')
        {
            continue;
            prevsTraversed = 0;
            windowLayer = (*currLayer->prevLayers[currLayer->numPrevLayers - 1]);

            for(int j = 0; j < currLayer->numPrevLayers; j++)
            {
                prevLayer = *currLayer->prevLayers[j];
                prevNumNodes = prevLayer->numNodes;
                for(int k = 0; k < prevLayer->numNodes; k++, currSum = 0.0)
                {
                    for(int i = 0; i < currNumNodes; i++) currSum += currLayer->weights[i][k + prevsTraversed] * windowLayer->activationDerivatives[i];
                    currSum *= windowLayer->preActivations[k + prevsTraversed];
                    windowLayer->preActivations[k + prevsTraversed] = currSum + prevLayer->outputs[k];
                }

                prevsTraversed += prevNumNodes;
            }

            prevsTraversed = 0;

            for(int i = 0; i < currNumNodes; i++, prevsTraversed = 0)
            {
                backError = backErrors[i];

                // Calculate gradient sensitivites for current timestep base
                windowLayer->biases[i] = 1 + windowLayer->biases[i] * windowLayer->activationDerivatives[i];
                
                currLayer->biases[i] -= learningRate * backError * windowLayer->biases[i];
                
                for(int j = 0; j < currLayer->numPrevLayers; j++)
                {
                    prevLayer = *currLayer->prevLayers[j];
                    prevNumNodes = prevLayer->numNodes;
                    
                    for(int k = 0; k < prevNumNodes; k++) 
                    {
                        currLayer->weights[i][k + prevsTraversed] -= learningRate * backError * windowLayer->preActivations[k + prevsTraversed];
                    }
                    prevsTraversed += prevNumNodes;
                }
            }
            memset(currLayer->backErrors, 0, currNumNodes * sizeof(float));
            continue;
        }


        // newBias[i] = oldBias[i] - (learningRate * backErrors[i])
        for(int i = 0; i < currNumNodes; i++) currLayer->biases[i] -= learningRate * currLayer->backErrors[i];

        // newWeights[i][j] = oldWeights[i][j] - (learningRate * (prevNodeOuts[j] * backError[i]))
        for(int i = 0; i < currNumNodes; i++, prevsTraversed = 0)
        {
            backError = backErrors[i];

            for(int j = 0; j < currLayer->numPrevLayers; j++)
            {
                prevLayer = *currLayer->prevLayers[j];
                prevNumNodes = prevLayer->numNodes;
                prevOutputs = prevLayer->outputs;
                
                for(int k = 0; k < prevNumNodes; k++) currLayer->weights[i][k + prevsTraversed] -= learningRate * prevOutputs[k] * backError;
                prevsTraversed += prevNumNodes;
            }
        }

        memset(currLayer->backErrors, 0, currNumNodes * sizeof(float));
    }
}

// Encode and save a model to file
// Care is needed when using, this implementation assumes that a float is always 32 bits and endianness is little if not defined;
int save_model(model* saveModel, char* modelFileName)
{
    FILE* modFile = NULL;
    unsigned char* line = NULL;
    fltChars fVal;
    intChars iVal;
    int offset = 0;
    uint32_t lineLength = 13;

    modFile = fopen(modelFileName, "wb");
    if(modFile == NULL) goto error1;

    #if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
    if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) fputc('L', modFile); 
    else if(__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) fputc('B', modFile); 
    #else
    fputc('L', modFile);  // Making a big assumption here
    #endif

    lineLength += 4 * saveModel->numInLayers;

    line = (unsigned char *)calloc(lineLength, sizeof(unsigned char));
    if(line == NULL) goto error2;

    iVal.num = saveModel->numLayers;
    memcpy(&line[offset], iVal.chars, 4);
    offset += 4;

    iVal.num = saveModel->numInLayers;    
    memcpy(&line[offset], iVal.chars, 4);
    offset += 4;


    for(int i = 0; i < saveModel->numInLayers; i++)
    {
        iVal.num = (*saveModel->inLayers[i])->layerID;    
        memcpy(&line[offset], iVal.chars, 4);
        offset += 4;
    }


    fVal.flt = saveModel->learningRate;
    for(int l = 0; l < 4; l++) line[offset++] = fVal.chars[l];

    line[offset++] = saveModel->loss_fn;

    iVal.num = lineLength;
        
    fwrite(iVal.chars, sizeof(unsigned char), 4, modFile);
    fwrite(line, sizeof(unsigned char), lineLength, modFile);

    free(line);
    line = NULL;

    for(int i = 0; i < saveModel->numLayers; i++)
    {
        lineLength = 9;
        if((*saveModel->layerList[i])->layerType == 'h' || (*saveModel->layerList[i])->layerType == 'o' || (*saveModel->layerList[i])->layerType == 'a')
        {
            lineLength += 4 * (*saveModel->layerList[i])->numPrevLayers;
            lineLength += 4 * ((*saveModel->layerList[i])->numNodes * ((*saveModel->layerList[i])->numPrevNodes + 1));
            lineLength += 9;
        }
        else if((*saveModel->layerList[i])->layerType == 'w')
        {
            lineLength += 4 * (*saveModel->layerList[i])->numPrevLayers;
            lineLength += 9;
        }
        else if((*saveModel->layerList[i])->layerType == 'a')
        {
            lineLength += 9;
        }
        offset = 0;

        line = (unsigned char *)calloc(lineLength, sizeof(unsigned char));
        if(line == NULL) goto error2;

        line[offset++] = (*saveModel->layerList[i])->layerType;
        
        iVal.num = (*saveModel->layerList[i])->layerID;
        for(int j = 0; j < 4; j++) line[offset++] = iVal.chars[j];

        iVal.num = (*saveModel->layerList[i])->numNodes;
        for(int j = 0; j < 4; j++) line[offset++] = iVal.chars[j];

        if((*saveModel->layerList[i])->layerType == 'i' || (*saveModel->layerList[i])->layerType == 't')
        {            
            iVal.num = lineLength;
        
            fwrite(iVal.chars, 4, 1, modFile);
            fwrite(line, lineLength, 1, modFile);
            
            free(line);
            line = (unsigned char*)NULL;
            continue;
        }

        iVal.num = (*saveModel->layerList[i])->numPrevNodes;
        for(int j = 0; j < 4; j++) line[offset++] = iVal.chars[j];
        
        iVal.num = (*saveModel->layerList[i])->numPrevLayers;
        for(int j = 0; j < 4; j++) line[offset++] = iVal.chars[j];
        
        line[offset++] = (*saveModel->layerList[i])->activationFunction;

        if((*saveModel->layerList[i])->layerType == 'a')
        {
            iVal.num = (*(*saveModel->layerList[i])->prevLayers[0])->layerID;
            for(int k = 0; k < 4; k++) line[offset++] = iVal.chars[k];

            iVal.num = lineLength;
        
            fwrite(iVal.chars, 4, 1, modFile);
            fwrite(line, lineLength, 1, modFile);
            
            free(line);
            line = (unsigned char*)NULL;
            
            continue;
        }

        for(int j = 0; j < (*saveModel->layerList[i])->numPrevLayers; j++)
        {
            iVal.num = (*(*saveModel->layerList[i])->prevLayers[j])->layerID;
            for(int k = 0; k < 4; k++) line[offset++] = iVal.chars[k];
        }

        if((*saveModel->layerList[i])->layerType == 'w')
        {            
            iVal.num = lineLength;
        
            fwrite(iVal.chars, 4, 1, modFile);
            
            fwrite(line, lineLength, 1, modFile);
            
            free(line);
            line = (unsigned char*)NULL;
            continue;
        }

        for(int j = 0; j < (*saveModel->layerList[i])->numNodes; j++)
        {
            for(int k = 0; k < (*saveModel->layerList[i])->numPrevNodes; k++)
            {
                fVal.flt = (*saveModel->layerList[i])->weights[j][k];
                for(int l = 0; l < 4; l++) line[offset++] = fVal.chars[l];
            }
        }

        for(int j = 0; j < (*saveModel->layerList[i])->numNodes; j++)
        {
            fVal.flt = (*saveModel->layerList[i])->biases[j];
            for(int l = 0; l < 4; l++) line[offset++] = fVal.chars[l];
        }

        iVal.num = lineLength;
        
        fwrite(iVal.chars, 4, 1, modFile);
        fwrite(line, lineLength, 1, modFile);

        free(line);
        line = NULL;
    }

    fclose(modFile);
    modFile = NULL;
    return 0;

error2:
    fclose(modFile);
    modFile = NULL;
error1:
    return -1;
}

// Decode and load a saved model from a file generated by the save model function - should be in model_construct but its cleaner and easier to keep it near save_model
model* load_model(const char* modelFileName, layer*** modelLayers)
{
    layer*** layerArr = NULL;
    model* myModel = NULL;
    int* inLayerIDs = NULL;
    unsigned char* line = NULL;
    fltChars fVal;
    intChars iVal;
    float learningRate = 1.0f;
    int outLayerID = 0;
    uint32_t lineLength = 0;
    int offset = 0;
    int numLayers = 0;
    int numInLayers = 0;
    int numPrevLayers = 0;
    int numPrevNodes = 0;
    int layerID = 0;
    int numNodes = 0;
    char activationFunction = '\0';
    char loss_fn = '\0';
    char layerType = '\0';
    char endianness = '\0';
    char myEndianness;

    #if defined(__BYTE_ORDER__)
    if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) myEndianness = 'L'; 
    else if(__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) myEndianness = 'B';
    #else
    myEndianness = 'L'; // Making a big assumption here
    #endif

    FILE *modFile = fopen(modelFileName, "rb");
    if(modFile == NULL) goto error1;

    endianness = fgetc(modFile);

    if(fread(iVal.chars, sizeof(unsigned char), 4, modFile) != 4) goto error2;
    if(myEndianness != endianness) reverse_uchars(iVal.chars, 4);
    lineLength = iVal.num;

    line = (unsigned char *)calloc(lineLength, sizeof(unsigned char));
    if(line == NULL) goto error2;

    if(fread(line, sizeof(unsigned char), lineLength, modFile) != lineLength) goto error3;

    memcpy(iVal.chars, line, 4 * sizeof(unsigned char));
    if(myEndianness != endianness) reverse_uchars(iVal.chars, 4);
    numLayers = iVal.num;
    
    memcpy(iVal.chars, &line[4], 4 * sizeof(unsigned char));
    if(myEndianness != endianness) reverse_uchars(iVal.chars, 4);
    numInLayers = iVal.num;

    offset += 8;

    inLayerIDs = (int *)calloc(numInLayers, sizeof(int));
    if(inLayerIDs == NULL) goto error3;

    for(int i = 0; i < numInLayers; i++) 
    {
        memcpy(iVal.chars, &line[offset], 4 * sizeof(unsigned char));
        if(myEndianness != endianness) reverse_uchars(iVal.chars, 4);
        inLayerIDs[i] = iVal.num;
        offset += 4;
    }

    memcpy(fVal.chars, &line[offset], 4 * sizeof(unsigned char));
    if(myEndianness != endianness) reverse_uchars(fVal.chars, 4);
    learningRate = fVal.flt;
    offset += 4;

    loss_fn = line[offset];

    free(line);
    line = NULL;

    *modelLayers = (layer**)malloc(numLayers * sizeof(layer*));
    if(*modelLayers == NULL) goto error4;

    for(int i = 0; i < numLayers; i++)
    {
        if(fread(iVal.chars, sizeof(unsigned char), 4, modFile) != 4) goto error5;
        if(myEndianness != endianness) reverse_uchars(iVal.chars, 4);
        lineLength = iVal.num;

        line = (unsigned char *)calloc(lineLength, sizeof(unsigned char));
        if(line == NULL) goto error5;

        if(fread(line, sizeof(unsigned char), lineLength, modFile) != lineLength) goto error6;

        offset = 0;
        numNodes = 0;
        numPrevNodes = 0;

        layerType = line[0];
        offset++;

        memcpy(iVal.chars, &line[offset], 4 * sizeof(unsigned char));
        if(myEndianness != endianness) reverse_uchars(iVal.chars, 4);
        layerID = iVal.num;
        offset += 4;
        
        memcpy(iVal.chars, &line[offset], 4 * sizeof(unsigned char));
        if(myEndianness != endianness) reverse_uchars(iVal.chars, 4);
        numNodes = iVal.num;
        offset += 4;

        if(layerType == 'i' || layerType == 't')
        {
            (*modelLayers)[layerID] = make_input_layer(numNodes);
            if((*modelLayers)[layerID] == NULL) goto error7;

            (*modelLayers)[layerID]->layerType = layerType;

            (*modelLayers)[layerID]->layerID = layerID;
            
            free(line);
            line = NULL;
            
            continue;
        }

        memcpy(iVal.chars, &line[offset], 4 * sizeof(unsigned char));
        if(myEndianness != endianness) reverse_uchars(iVal.chars, 4);
        numPrevNodes = iVal.num;
        offset += 4;

        memcpy(iVal.chars, &line[offset], 4 * sizeof(unsigned char));
        if(myEndianness != endianness) reverse_uchars(iVal.chars, 4);
        numPrevLayers = iVal.num;
        offset += 4;

        activationFunction = line[offset];
        offset+=1;

        if(layerType == 'a')
        {
            memcpy(iVal.chars, &line[offset], 4 * sizeof(unsigned char));
            if(myEndianness != endianness) reverse_uchars(iVal.chars, 4);
            offset += 4;

            (*modelLayers)[layerID] = make_attention_layer(&((*modelLayers)[iVal.num]), numPrevLayers, activationFunction); // Needs to be fixed
            if((*modelLayers)[layerID] == NULL) goto error7;
            continue;
        }

        layerArr = (layer***)malloc(numPrevLayers * sizeof(layer**));
        if(layerArr == NULL) goto error6;

        for(int j = 0; j < numPrevLayers; j++)
        {
            memcpy(iVal.chars, &line[offset], 4 * sizeof(unsigned char));
            if(myEndianness != endianness) reverse_uchars(iVal.chars, 4);
            offset += 4;

            layerArr[j] = &((*modelLayers)[iVal.num]);
        }

        switch (layerType) 
        {
            case 'o':
                (*modelLayers)[layerID] = make_output_layer(layerArr, numNodes, numPrevLayers, activationFunction);
                if((*modelLayers)[layerID] == NULL) goto error8;
                outLayerID = layerID;
                break;
            case 'w':
                (*modelLayers)[layerID] = NULL;//make_window_layer(layerArr, numNodes, numPrevLayers, activationFunction);
                if((*modelLayers)[layerID] == NULL) goto error8;
                (*modelLayers)[layerID]->layerID = layerID;
                free(line);
                line = (unsigned char *)NULL;
                free(layerArr);
                layerArr = (layer***)NULL;
                continue;
            default:
                (*modelLayers)[layerID] = make_hidden_layer(layerArr, numNodes, numPrevLayers, activationFunction);
                if((*modelLayers)[layerID] == NULL) goto error8;
        }

        (*modelLayers)[layerID]->layerID = layerID;

        for(int j = 0; j < numNodes; j++)
        {
            for(int k = 0; k < numPrevNodes; k++)
            {
                memcpy(fVal.chars, &line[offset], 4 * sizeof(unsigned char));
                if(myEndianness != endianness) reverse_uchars(fVal.chars, 4);
                offset += 4;
                
                (*modelLayers)[layerID]->weights[j][k] = fVal.flt;
            }
        }

        for(int j = 0; j < numNodes; j++)
        {
            memcpy(fVal.chars, &line[offset], 4 * sizeof(unsigned char));
            if(myEndianness != endianness) reverse_uchars(fVal.chars, 4);
            offset += 4;
            
            (*modelLayers)[layerID]->biases[j] = fVal.flt;
        }

        free(line);
        line = (unsigned char *)NULL;
        free(layerArr);
        layerArr = (layer***)NULL;
    }

    for(int i = numLayers - 1; i > -1; i--)
    {
        layer* currLayer = (*modelLayers)[i];
        if(currLayer->numPrevLayers == 0) continue;
        layer* lastPrevLayer = *(currLayer->prevLayers[currLayer->numPrevLayers - 1]);
        
        if(lastPrevLayer->layerType == 'w')
        {
            lastPrevLayer->weights = currLayer->weights;
            lastPrevLayer->biases = currLayer->biases;
        }
    }
    

    fclose(modFile);
    modFile = NULL;

    myModel = (model*)malloc(sizeof(model));
    if(myModel == NULL) goto error8;
    
    myModel->inLayers = (layer ***)malloc(numInLayers * sizeof(layer**));
    if(myModel->inLayers == NULL) goto error9;

    myModel->targets = (float *)malloc((*modelLayers)[outLayerID]->numNodes * sizeof(float));
    if(myModel->targets == NULL) goto error10;

    myModel->layerList = (layer ***)malloc(numLayers * sizeof(layer**));
    if(myModel->layerList == NULL) goto error11;

    myModel->lossDerivatives = (float *)malloc((*modelLayers)[outLayerID]->numNodes * sizeof(float));
    if(myModel->lossDerivatives == NULL) goto error12;

    for(int i = 0; i < numInLayers; i++) myModel->inLayers[i] = &(*modelLayers)[inLayerIDs[i]];
    for(int i = 0; i < numLayers; i++) myModel->layerList[i] = &(*modelLayers)[i];

    myModel->outLayer = &(*modelLayers)[outLayerID];
    myModel->numLayers = numLayers;
    myModel->learningRate = learningRate;
    myModel->numInLayers = numInLayers;
    myModel->loss_fn = loss_fn;

    free(inLayerIDs);
    inLayerIDs = NULL;
    free(layerArr);
    layerArr = NULL;

    return myModel;

error12:
    free(myModel->layerList);
    myModel->layerList = NULL;
error11:
    free(myModel->targets);
    myModel->targets = NULL;
error10:
    free(myModel->inLayers);
    myModel->inLayers = NULL;
error9:
    free(myModel);
    myModel = NULL;
error8:
    free(layerArr);
    layerArr = NULL;
error7:
    for(int i = 0; i < numLayers; i++) hakai_layer(&(*modelLayers)[i]);
error6:
    free(line);
    line = NULL;
error5:
    free(*modelLayers);
    *modelLayers = NULL;
error4:
    free(inLayerIDs);
    inLayerIDs = NULL;
error3:
    free(line); // Won't break from error6 because line was set to NULL there
    line = NULL;
error2:
    fclose(modFile);
error1:
    return NULL;
}

void shift_rnn(model* myModel, char opType)
{
    layer* currLayer;
    // layer* prevIns;
    layer* prevWindow;
    // int numInputs = 0;
    int numHiddenNodes = 0;
    // int prevsTraversed = 0;

    for(int l = 0; l < myModel->numLayers; l++)
    {
        currLayer = *myModel->layerList[l];

        if(currLayer->numPrevLayers == 0) continue; // Second if statement after this one will fail without this since input layers don't have prevLayers
        
        if(currLayer->layerType == 'w' && opType == 't')
        {
            prevWindow = (*currLayer->prevLayers[0]);
            numHiddenNodes = currLayer->numNodes;

            memcpy(prevWindow->outputs, currLayer->outputs, sizeof(float) * numHiddenNodes);    // Outputs needed for attention in future
            memcpy(prevWindow->activationDerivatives, currLayer->activationDerivatives, sizeof(float) * numHiddenNodes);  // Preactivations needed for calculating dels at most recent timestep for attention in future
            continue;
        }
        
        if(currLayer->layerType == 'h' && (*currLayer->prevLayers[currLayer->numPrevLayers - 1])->layerType == 'w')
        {
            numHiddenNodes = currLayer->numNodes;
            // numInputs = currLayer->numPrevNodes;
            prevWindow = (*currLayer->prevLayers[currLayer->numPrevLayers - 1]);

            memcpy(prevWindow->outputs, currLayer->outputs, sizeof(float) * numHiddenNodes);
            //memcpy(prevWindow->preActivations, currLayer->preActivations, sizeof(float) * numHiddenNodes);  // Preactivations needed for calculating dels at most recent timestep for attention in future
            // memcpy(prevWindow->backErrors, currLayer->backErrors, sizeof(float) * numHiddenNodes);  // Can be used for previous layer gradient sensitivity storage
            // memcpy(prevWindow->biases, currLayer->biases, sizeof(float) * numHiddenNodes);  // Can be used for input layers gradient sensitivity storage
            memcpy(prevWindow->activationDerivatives, currLayer->activationDerivatives, sizeof(float) * numHiddenNodes);    // Activation derivatives will be needed for gradient sensitivity calculations

            // prevsTraversed = 0;
        }
    }
}

#if defined(__AVX__) || defined(__AVX2__)
//  Gets an output from the target layer, is essentially also a inference function
// Vectorized version of forward out, only really makes a difference on industrial grade models so it will be shelved for now
int _mm256_forward_out(model* myModel, float dropoutVal)
{
    layer* currLayer;
    layer* prevLayer;
    layer* otherPrev;
    float* activationDerivatives;
    float scalingFactor;
    int currNumNodes;

    for(int l = 0; l < myModel->numLayers; l++)
    {
        currLayer = (*myModel->layerList[l]);
        if(currLayer->layerType == 'i'  || currLayer->layerType == 't' || currLayer->layerType == 'w') continue;

        currNumNodes = currLayer->numNodes;

        // if(currLayer->layerType == 'a')
        // {
        //     for(int i = 0; i < currLayer->numPrevLayers; i++)
        //     {
        //         prevLayer = *currLayer->prevLayers[i];
        //         for(int j = 0; j <= i; j++)
        //         {
        //             otherPrev = *currLayer->prevLayers[j];
        //             for(int k = 0; k < currNumNodes; k++) currLayer->preActivations[i] += prevLayer->outputs[k] * otherPrev->outputs[k];
        //         }

        //         currLayer->preActivations[i] /= sqrt(currNumNodes);
        //     }

        //     fast_softmax(currLayer->preActivations, currLayer->numPrevLayers);

        //     for(int i = 0; i < currLayer->numPrevLayers; i++)
        //     {
        //         prevLayer = *currLayer->prevLayers[i];
        //         for(int j = 0; j < currNumNodes; j++) currLayer->outputs[j] += currLayer->preActivations[i] * prevLayer->outputs[j];
        //     }

        //     continue;
        // }
        
        if(vectorized_forward_out_calc(currLayer) != 0) return -1;

        // In case of softmax activation, we do a function on the layer instead of point-wise on the outputs
        if(currLayer->activationFunction == 'x')
        {
            memcpy(currLayer->outputs, currLayer->preActivations, sizeof(float) * currNumNodes);
            softmax(currLayer->outputs, currNumNodes);
        }
        else if(currLayer->activationFunction == 'f')
        {
            memcpy(currLayer->outputs, currLayer->preActivations, sizeof(float) * currNumNodes);
            fast_softmax(currLayer->outputs, currNumNodes);
        }
        else
        {
            // outputs[i] = activation_function(preActivations[i])
            activationDerivatives = currLayer->activationDerivatives;
            for(int i = 0; i < currNumNodes; i++)
            {
                currLayer->outputs[i] = activation_function(currLayer->preActivations[i], currLayer->activationFunction);
                activationDerivatives[i] = activation_derivative(currLayer->preActivations[i], currLayer, i);
            }
        }

        if(currLayer->layerType == 'o' || dropoutVal <= 0.0f) continue;;
        scalingFactor = 1/(1-dropoutVal);
        for(int i = 0; i < currNumNodes; i++) currLayer->outputs[i] *= ((float)((rand() % 999))/1000.0 < dropoutVal) ? 0 : scalingFactor;
    }

    return 0;
}

// void _mm256_sgd_backprop(model* myModel)
// { // start at output layer and calculate backerrors for each previous layer
//     layer* currLayer;
//     layer* prevLayer;
//     int prevsTraversed = 0;
// 
//     for(int l = myModel->numLayers - 1; l > -1; l--)
//     {
//         currLayer = *myModel->layerList[l];
//
//         if(currLayer->layerType == 'i'  || currLayer->layerType == 't' ) continue;
//        
//         // backErrorsForOutputLayer = lossDerivative · activationFunctionDerivative(preActivations) - for output layer
//         if(currLayer->layerType == 'o') vectorized_sgd_backprop_output_calc(currLayer, myModel);
//        
//         // backErrorsForPreviousLayers[j] = SUM_OVER_I((thisLayersBackErrors[i])(thisLayersWeightMatrix[i][j]) · activationFunctionDerivative(previousLayersPreActivation[j])) - where j is considered to be a traversal of all previous 'J' layers' 'K' values as one vector
//         // e.g. J = 3 prev layers with K = 5 nodes each are considered as one prev layer with J = 15 nodes in this formulation
//         prevsTraversed = 0;
//         for(int i = 0; i < currLayer->numPrevLayers; i++)
//         {
//             prevLayer = *currLayer->prevLayers[i];
//            
//             if(prevLayer->layerType == 'i') continue;
//            
//             for(int j = 0; j < prevLayer->numNodes; j++) for(int k = 0; k < currLayer->numNodes; k++) prevLayer->backErrors[j] += currLayer->backErrors[k] * currLayer->weights[k][prevsTraversed + j] * activation_derivative(prevLayer->preActivations[j], currLayer, i);
//             prevsTraversed += prevLayer->numNodes;
//         }
//     }
// }

// int _mm256_calculate_and_apply_grads(model* myModel)
// {
//     layer* currLayer;
//
//     for(int l = 0; l < myModel->numLayers; l++)
//     {
//         currLayer = *myModel->layerList[l];
//
//         if(currLayer->layerType == 'i' || currLayer->layerType == 't' ) continue;
//         if(vectorized_calculate_and_apply_grads(currLayer, myModel->learningRate) != 0) return -1;
//     }
//     return 0;
// }

#endif

#if defined(__ARM_NEON)

int vforward_out(model* myModel, float dropoutVal)
{
    layer* currLayer;
    float* activationDerivatives;
    float scalingFactor

    for(int l = 0; l < myModel->numLayers; l++)
    {
        currLayer = *myModel->layerList[l];
        if(currLayer->layerType == 'i' || currLayer->layerType == 't' || currLayer->layerType == 'w') continue;

        memset(currLayer->backErrors, 0.0f, currLayer->numNodes * sizeof(float));
        memset(currLayer->preActivations, 0.0f, currLayer->numNodes * sizeof(float));
        memset(currLayer->outputs, 0.0f, currLayer->numNodes * sizeof(float));

        if(vectorized_forward_out_calc(currLayer) != 0) return -1;

        // In case of softmax activation, we do a function on the layer instead of point-wise on the outputs
        if(currLayer->activationFunction == 'x')
        {
            memcpy(currLayer->outputs, currLayer->preActivations, sizeof(float) * currLayer->numNodes);
            softmax(currLayer);
        }
        else if(currLayer->activationFunction == 'f')
        {
            memcpy(currLayer->outputs, currLayer->preActivations, sizeof(float) * currLayer->numNodes);
            fast_softmax(currLayer->outputs, currLayer->numNodes);
        }
        else
        {
            // outputs[i] = activation_function(preActivations[i])
            activationDerivatives = currLayer->activationDerivatives;
            for(int i = 0; i < currNumNodes; i++)
            {
                currLayer->outputs[i] = activation_function(currLayer->preActivations[i], currLayer->activationFunction);
                activationDerivatives[i] = activation_derivative(currLayer->preActivations[i], currLayer, i);
            }
        }

        if(currLayer->layerType == 'o' || dropoutVal <= 0.0f) continue;;
        scalingFactor = 1/(1-dropoutVal);
        for(int i = 0; i < currNumNodes; i++) currLayer->outputs[i] *= ((float)((rand() % 999))/1000.0 < dropoutVal) ? 0 : scalingFactor;
    }
}

#endif