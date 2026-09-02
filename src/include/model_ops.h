#pragma once

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
    float* currCol;
    float preAct;
    int currNumNodes;
    int prevNumNodes;
    int numPrevsTraversed = 0;

    for(int l = 0; l < myModel->numLayers; l++)
    {        
        currLayer = *myModel->layerList[l];

        if(currLayer->layerType == 'i' || currLayer->layerType == 'w') continue;

        currNumNodes = currLayer->numNodes;
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
            softmax(currLayer);
        }
        else if(currLayer->activationFunction == 'f')
        {
            memcpy(currLayer->outputs, currLayer->preActivations, sizeof(float) * currNumNodes);
            fast_softmax(currLayer);
        }
        else
        {
            // outputs[i] = activation_function(preActivations[i])
            for(int i = 0; i < currNumNodes; i++) currLayer->outputs[i] = activation_function(currLayer->preActivations[i], currLayer->activationFunction);
        }

        if(currLayer->layerType == 'o' || dropoutVal <= 0.0f) continue;;
        float scalingFactor = 1/(1-dropoutVal);
        for(int i = 0; i < currNumNodes; i++) currLayer->outputs[i] *= ((float)((rand() % 999))/1000.0 < dropoutVal) ? 0 : scalingFactor;
    }
}

// We pass the backerrors to each previous layer to calculate grads later
// Backerrors can be accumulated from multiple successor layers to calculate grads due to matrix distributivity
void sgd_backprop(model* myModel)
{ // start at output layer and calculate backerrors for each previous layer
    layer* currLayer;
    layer* prevLayer;
    float* lossDerivatives = myModel->lossDerivatives;
    float* activationDerivatives;
    float prevBackerror = 0.0;
    int prevsTraversed = 0;
    int currNumNodes;
    int prevNumNodes;
    
    currLayer = *myModel->outLayer;
    
    for(int i = 0; i < currLayer->numNodes; i++) lossDerivatives[i] = loss_derivative(myModel->targets[i], currLayer->outputs[i], myModel);

    for(int l = myModel->numLayers - 1; l > -1; l--)
    {        
        currLayer = *myModel->layerList[l];
        if(currLayer->layerType == 'i') continue;
        activationDerivatives = currLayer->activationDerivatives;
        for(int i = 0; i < currLayer->numNodes; i++) activationDerivatives[i] = activation_derivative(currLayer->preActivations[i], currLayer, i);
    }

    currLayer = *myModel->outLayer;
    activationDerivatives = currLayer->activationDerivatives;

    // backErrorsForOutputLayer = lossDerivative · activationFunctionDerivative(preActivations)
    for(int i = 0; i < currLayer->numNodes; i++) currLayer->backErrors[i] = -1 * lossDerivatives[i] * activationDerivatives[i];

    for(int l = myModel->numLayers - 2; l > -1; l--)
    {
        currLayer = *myModel->layerList[l];

        if(currLayer->layerType == 'i') continue;

        currNumNodes = currLayer->numNodes;
        
        // backErrorsForPreviousLayers[j] = SUM_OVER_I((thisLayersBackErrors[i])(thisLayersWeightMatrix[i][j]) · activationFunctionDerivative(previousLayersPreActivation[j])) - where j is considered to be a traversal of all previous 'J' layers' 'K' values as one vector
        // e.g. J = 3 prev layers with K = 5 nodes each are considered as one prev layer with J = 15 nodes in this formulation
        prevsTraversed = 0;
        for(int i = 0; i < currLayer->numPrevLayers; i++)
        {
            prevLayer = *currLayer->prevLayers[i];
            
            if(prevLayer->layerType == 'i') continue;

            prevNumNodes = prevLayer->numNodes;
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
    float* backErrors;
    float* prevOutputs;
    float learningRate = myModel->learningRate;
    float backError;
    int currNumNodes;
    int prevNumNodes;
    int prevsTraversed = 0;

    for(int l = myModel->numLayers - 1; l > -1; l--)
    {
        currLayer = *myModel->layerList[l];

        if(currLayer->layerType == 'i') continue;

        currNumNodes = currLayer->numNodes;
        backErrors = currLayer->backErrors;

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

    line = (char *)calloc(lineLength, sizeof(unsigned char));
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
        if((*saveModel->layerList[i])->layerType == 'h' || (*saveModel->layerList[i])->layerType == 'o')
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
        offset = 0;

        line = (unsigned char *)calloc(lineLength, sizeof(unsigned char));
        if(line == NULL) goto error2;

        line[offset++] = (*saveModel->layerList[i])->layerType;
        
        iVal.num = (*saveModel->layerList[i])->layerID;
        for(int j = 0; j < 4; j++) line[offset++] = iVal.chars[j];

        iVal.num = (*saveModel->layerList[i])->numNodes;
        for(int j = 0; j < 4; j++) line[offset++] = iVal.chars[j];

        if((*saveModel->layerList[i])->layerType == 'i')
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

        for(int j = 0; j < (*saveModel->layerList[i])->numPrevLayers; j++)
        {
            // if((*layerList[i])->layerID == (*(*layerList[i])->prevLayers[j])->layerID) continue;
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

// Decode and load a saved model from a file generated by the save model function
model* load_model(const char* modelFileName, layer*** modelLayers)
{
    layer*** layerArr = (layer***)NULL;
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

    #if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
    if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) myEndianness = 'L'; 
    else if(__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) myEndianness = 'B';
    #else
    myEndianness = 'L'; // Making a big assumption here
    #endif

    FILE *modFile = fopen(modelFileName, "rb");
    if(modFile == NULL) goto error1;

    endianness = fgetc(modFile);

    if(fread(iVal.chars, sizeof(unsigned char), 4, modFile) != 4) goto error2;
    if(myEndianness != endianness) reverse_chars(iVal.chars, 4);
    lineLength = iVal.num;

    line = (unsigned char *)calloc(lineLength, sizeof(unsigned char));
    if(line == NULL) goto error2;

    if(fread(line, sizeof(unsigned char), lineLength, modFile) != lineLength) goto error3;

    memcpy(iVal.chars, line, 4 * sizeof(unsigned char));
    if(myEndianness != endianness) reverse_chars(iVal.chars, 4);
    numLayers = iVal.num;
    
    memcpy(iVal.chars, &line[4], 4 * sizeof(unsigned char));
    if(myEndianness != endianness) reverse_chars(iVal.chars, 4);
    numInLayers = iVal.num;

    offset += 8;

    inLayerIDs = (int *)calloc(numInLayers, sizeof(int));
    if(inLayerIDs == NULL) goto error3;

    for(int i = 0; i < numInLayers; i++) 
    {
        memcpy(iVal.chars, &line[offset], 4 * sizeof(unsigned char));
        if(myEndianness != endianness) reverse_chars(iVal.chars, 4);
        inLayerIDs[i] = iVal.num;
        offset += 4;
    }

    memcpy(fVal.chars, &line[offset], 4 * sizeof(unsigned char));
    if(myEndianness != endianness) reverse_chars(fVal.chars, 4);
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
        if(myEndianness != endianness) reverse_chars(iVal.chars, 4);
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
        if(myEndianness != endianness) reverse_chars(iVal.chars, 4);
        layerID = iVal.num;
        offset += 4;
        
        memcpy(iVal.chars, &line[offset], 4 * sizeof(unsigned char));
        if(myEndianness != endianness) reverse_chars(iVal.chars, 4);
        numNodes = iVal.num;
        offset += 4;

        if(layerType == 'i')
        {
            (*modelLayers)[layerID] = make_input_layer(numNodes);
            if((*modelLayers)[layerID] == NULL) goto error7;

            (*modelLayers)[layerID]->layerID = layerID;
            
            free(line);
            line = NULL;
            
            continue;
        }

        memcpy(iVal.chars, &line[offset], 4 * sizeof(unsigned char));
        if(myEndianness != endianness) reverse_chars(iVal.chars, 4);
        numPrevNodes = iVal.num;
        offset += 4;

        memcpy(iVal.chars, &line[offset], 4 * sizeof(unsigned char));
        if(myEndianness != endianness) reverse_chars(iVal.chars, 4);
        numPrevLayers = iVal.num;
        offset += 4;

        activationFunction = line[offset];
        offset+=1;

        layerArr = (layer***)malloc(numPrevLayers * sizeof(layer**));
        if(layerArr == NULL) goto error6;

        for(int j = 0; j < numPrevLayers; j++)
        {
            memcpy(iVal.chars, &line[offset], 4 * sizeof(unsigned char));
            if(myEndianness != endianness) reverse_chars(iVal.chars, 4);
            offset += 4;

            layerArr[j] = &((*modelLayers)[iVal.num]);
        }

        if(layerType == 'h')
        {
            (*modelLayers)[layerID] = make_hidden_layer(layerArr, numNodes, numPrevLayers, activationFunction);
            if((*modelLayers)[layerID] == NULL) goto error8;
        }
        else if(layerType == 'w')
        {
            (*modelLayers)[layerID] = make_window_layer(layerArr, numNodes, numPrevLayers, activationFunction, numPrevNodes);
            if((*modelLayers)[layerID] == NULL) goto error8;

            (*modelLayers)[layerID]->layerID = layerID;

            free(line);
            line = (unsigned char *)NULL;
            free(layerArr);
            layerArr = (layer***)NULL;

            continue;
        }
        else
        {
            (*modelLayers)[layerID] = make_output_layer(layerArr, numNodes, numPrevLayers, activationFunction);
            if((*modelLayers)[layerID] == NULL) goto error8;

            outLayerID = layerID;
        }

        (*modelLayers)[layerID]->layerID = layerID;

        for(int j = 0; j < numNodes; j++)
        {
            for(int k = 0; k < numPrevNodes; k++)
            {
                memcpy(fVal.chars, &line[offset], 4 * sizeof(unsigned char));
                if(myEndianness != endianness) reverse_chars(fVal.chars, 4);
                offset += 4;
                
                (*modelLayers)[layerID]->weights[j][k] = fVal.flt;
            }
        }

        for(int j = 0; j < numNodes; j++)
        {
            memcpy(fVal.chars, &line[offset], 4 * sizeof(unsigned char));
            if(myEndianness != endianness) reverse_chars(fVal.chars, 4);
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

void shift_model(model* myModel, char opType)
{
    layer* currLayer;
    layer* prevIns;
    layer* prevWindow;
    int numInputs = 0;
    int numHiddenNodes = 0;
    int prevsTraversed = 0;

    for(int l = 0; l < myModel->numLayers; l++)
    {
        currLayer = *myModel->layerList[l];

        if(currLayer->layerType == 'i') continue; // Second if statement after this one will fail without this since input layers don't have prevLayers
        
        if(currLayer->layerType == 'w' && opType == 't' && currLayer->numPrevLayers == 2)
        {
            prevIns = (*currLayer->prevLayers[0]);
            prevWindow = (*currLayer->prevLayers[1]);
            numInputs = prevIns->numNodes;
            numHiddenNodes = currLayer->numNodes;

            memcpy(prevWindow->outputs, currLayer->outputs, sizeof(float) * numHiddenNodes);
            memcpy(prevWindow->preActivations, currLayer->preActivations, sizeof(float) * numHiddenNodes);
            memcpy((*prevWindow->prevLayers[0])->outputs, prevIns->outputs, sizeof(float) * numInputs);
            continue;
        }
        
        if(currLayer->layerType == 'h' && (*currLayer->prevLayers[currLayer->numPrevLayers - 1])->layerType == 'w')// technically would work for every window layer, separate for readability and inference functionality all in single shift function
        {
            numHiddenNodes = currLayer->numNodes;
            prevWindow = (*currLayer->prevLayers[currLayer->numPrevLayers - 1]);

            memcpy(prevWindow->outputs, currLayer->outputs, sizeof(float) * numHiddenNodes);
            memcpy(prevWindow->preActivations, currLayer->preActivations, sizeof(float) * numHiddenNodes);

            for(int i = 0; i < currLayer->numPrevLayers - 1; i++)
            {
                prevIns = *currLayer->prevLayers[i];
                
                memcpy(&(*prevWindow->prevLayers[0])->outputs[prevsTraversed], prevIns->outputs, sizeof(float) * prevIns->numNodes);
                prevsTraversed += prevIns->numNodes;
            }

            prevsTraversed = 0;
        }
    }
}

#if defined(__AVX__) || defined(__AVX2__)
//  Gets an output from the target layer, is essentially also a inference function
// Vectorized version of forward out, only really makes a difference on industrial grade models so it will be shelved for now
int _mm256_forward_out(model* myModel, float dropoutVal)
{
    layer* currLayer;
    int currNumNodes;

    for(int l = 0; l < myModel->numLayers; l++)
    {
        currLayer = (*myModel->layerList[l]);
        if(currLayer->layerType == 'i' || currLayer->layerType == 'w') continue;

        currNumNodes = currLayer->numNodes;
        
        if(vectorized_forward_out_calc(currLayer) != 0) return -1;

        // In case of softmax activation, we do a function on the layer instead of point-wise on the outputs
        if(currLayer->activationFunction == 'x')
        {
            memcpy(currLayer->outputs, currLayer->preActivations, sizeof(float) * currNumNodes);
            softmax(currLayer);
        }
        else if(currLayer->activationFunction == 'f')
        {
            memcpy(currLayer->outputs, currLayer->preActivations, sizeof(float) * currNumNodes);
            fast_softmax(currLayer);
        }
        else
        {            
            for(int i = 0; i < currNumNodes; i++) currLayer->outputs[i] = activation_function(currLayer->preActivations[i], currLayer->activationFunction);
        }

        if(currLayer->layerType == 'o' || dropoutVal <= 0.0) continue;
        float scalingFactor = 1/(1-dropoutVal);
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
//         if(currLayer->layerType == 'i') continue;
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
//         if(currLayer->layerType == 'i') continue;
//         if(vectorized_calculate_and_apply_grads(currLayer, myModel->learningRate) != 0) return -1;
//     }
//     return 0;
// }

#endif

#if defined(__ARM_NEON)

void vforward_out(model* myModel, float dropoutVal)
{
    layer* currLayer;

    for(int l = 0; l < myModel->numLayers; l++)
    {
        currLayer = *myModel->layerList[l];
        if(currLayer->layerType == 'i' || currLayer->layerType == 'w') continue;

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
            fast_softmax(currLayer);
        }
        else
        {            
            for(int i = 0; i < currLayer->numNodes; i++) currLayer->outputs[i] = activation_function(currLayer->preActivations[i], currLayer->activationFunction);
        }

        if(currLayer->layerType == 'o' || dropoutVal <= 0.0) continue;
        float scalingFactor = 1/(1-dropoutVal);
        for(int i = 0; i < currLayer->numNodes; i++) currLayer->outputs[i] *= ((float)((rand() % 999))/1000.0 < dropoutVal) ? 0 : scalingFactor;
    }

    return 0;
}

#endif