#pragma once

#include "model_construct.h"
#include "activation.h"
#include "loss.h"
#include "helper_funcs.h"
#include <stdio.h>
#include <pthread.h>

//  Gets an output from the target layer, is essentially also a inference function
void forward_out(model* myModel, float dropoutVal)
{
    layer* currLayer;
    layer* prevLayer;
    int numPrevsTraversed = 0;

    for(int l = 0; l < myModel->numLayers; l++)
    {        
        currLayer = *myModel->layerList[l];

        if(currLayer->layerType == 'w' || currLayer->layerType == 'i') continue; // If layer has been traversed or if layer is a window layer for a context window or if layer is an input layer
        
        memset(currLayer->backErrors, 0.0f, currLayer->numNodes * sizeof(float));
        memset(currLayer->preActivations, 0.0f, currLayer->numNodes * sizeof(float));

        // preActivation[i] = SUM_OVER_J(prevNodeOutputs[j] * weights[i][j]) 
        for(int i = 0; i < currLayer->numNodes; i++, numPrevsTraversed = 0) 
        {
            for(int j = 0; j < currLayer->numPrevLayers; j++)
            {
                prevLayer = *currLayer->prevLayers[j];
                
                for(int k = 0; k < prevLayer->numNodes; k++) currLayer->preActivations[i] += prevLayer->outputs[k] * currLayer->weights[i][numPrevsTraversed + k];
                numPrevsTraversed += prevLayer->numNodes;
            }
            currLayer->preActivations[i] += currLayer->biases[i];
        }
        
        numPrevsTraversed = 0;

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
            // outputs[i] = activation_function(preActivations[i])
            for(int i = 0; i < currLayer->numNodes; i++) currLayer->outputs[i] = activation_function(currLayer->preActivations[i], currLayer->activationFunction);
        }

        if(currLayer->layerType == 'o' || dropoutVal <= 0.0f) continue;;
        float scalingFactor = 1/(1-dropoutVal);
        for(int i = 0; i < currLayer->numNodes; i++) currLayer->outputs[i] *= ((float)((rand() % 999))/1000.0 < dropoutVal) ? 0 : scalingFactor;
    }
}

// Run on each output layer and then apply grads before clearing the layer backerrors - All roads spring forth from Rome algorithm
// We pass the backerrors to each previous layer to calculate grads later
// Backerrors can be accumulated from multiple successor layers to calculate grads due to matrix distributivity
void sgd_backprop(model* myModel)
{ // start at output layer and calculate backerrors for each previous layer
    layer* currLayer;
    layer* prevLayer;
    int prevsTraversed = 0;
    
    for(int l = myModel->numLayers - 1; l > -1; l--)
    {
        currLayer = *myModel->layerList[l];

        if(currLayer->layerType == 'i' || currLayer->layerType == 'w') continue;
        
        // backErrorsForOutputLayer = lossDerivative · activationFunctionDerivative(preActivations) - for output layer
        if(currLayer->layerType == 'o') for(int i = 0; i < currLayer->numNodes; i++) currLayer->backErrors[i] = -1 * loss_derivative(myModel->targets[i], currLayer->outputs[i], myModel) * activation_derivative(currLayer->preActivations[i], currLayer, i);
        
        // backErrorsForPreviousLayers[j] = SUM_OVER_I((thisLayersBackErrors[i])(thisLayersWeightMatrix[i][j]) · activationFunctionDerivative(previousLayersPreActivation[j])) - where j is considered to be a traversal of all previous 'J' layers' 'K' values as one vector
        // e.g. J = 3 prev layers with K = 5 nodes each are considered as one prev layer with J = 15 nodes in this formulation
        prevsTraversed = 0;
        for(int i = 0; i < currLayer->numPrevLayers; i++)
        {
            prevLayer = *currLayer->prevLayers[i];
            
            if(prevLayer->layerType == 'i' || prevLayer->layerType == 'w') continue;
            
            for(int j = 0; j < prevLayer->numNodes; j++) for(int k = 0; k < currLayer->numNodes; k++) prevLayer->backErrors[j] += currLayer->backErrors[k] * currLayer->weights[k][prevsTraversed + j] * activation_derivative(prevLayer->preActivations[j], currLayer, i);
            prevsTraversed += prevLayer->numNodes;
        }
    }
}

// Another all roads spring forth from Rome approach - go to the convergence point of the model(output layer) and use it as the root this model graph
void calculate_and_apply_grads(model* myModel)
{
    layer* currLayer;
    layer* prevLayer;
    int prevsTraversed = 0;

    for(int l = myModel->numLayers - 1; l > -1; l--)
    {
        currLayer = *myModel->layerList[l];

        if(currLayer->layerType == 'i' || currLayer->layerType == 'w') continue;

        // newBias[i] = oldBias[i] - (learningRate * backErrors[i])
        for(int i = 0; i < currLayer->numNodes; i++) currLayer->biases[i] -= myModel->learningRate * currLayer->backErrors[i];

        prevsTraversed = 0;

        // newWeights[i][j] = oldWeights[i][j] - (learningRate * (prevNodeOuts[j] * backError[i]))
        for(int i = 0; i < currLayer->numNodes; i++)
        {
            for(int j = 0; j < currLayer->numPrevLayers; j++)
            {
                prevLayer = *currLayer->prevLayers[j];
                
                for(int k = 0; k < prevLayer->numNodes; k++) currLayer->weights[i][k + prevsTraversed] -= myModel->learningRate * prevLayer->outputs[k] * currLayer->backErrors[i];
                prevsTraversed += prevLayer->numNodes;
            }
            prevsTraversed = 0;
        }
    }
}

// For clearing the backErrors once no longer needed, and to also prime for next forward and backward pass
// Use by passing the output layer of the model into the function 
void zero_everything(model* myModel)
{
    layer* currLayer;
    for(int l = 0; l < myModel->numLayers; l++)
    {
        currLayer = *myModel->layerList[l];

        if(currLayer->layerType == 'i') continue;

        memset(currLayer->backErrors, 0.0f, currLayer->numNodes * sizeof(float));
        memset(currLayer->preActivations, 0.0f, currLayer->numNodes * sizeof(float));
        memset(currLayer->outputs, 0.0f, currLayer->numNodes * sizeof(float));
    }
}

// Fix required to properly handle referential layers
// Encode and save a model to file
int save_model(model* saveModel, char* modelFileName)
{
    FILE *modFile = NULL;
    char *line = NULL;
    int offset = 0;
    int lineLength = 50;
    char bitBuff[33] = "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
    char fltBuff[20] = "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

    modFile = fopen(modelFileName, "w");
    if(modFile == NULL) goto error1;

    lineLength += 16 * saveModel->numInLayers;

    line = (char *)calloc(lineLength, sizeof(char));
    if(line == NULL) goto error2;
    
    int2bin(saveModel->numLayers, 16, bitBuff);
    for(int i = 0; i < 16; i++) line[offset + i] = bitBuff[i]; // big endian representation
    offset += 16;

    int2bin(saveModel->numInLayers, 16, bitBuff);
    for(int i = 0; i < 16; i++) line[offset + i] = bitBuff[i]; // big endian representation
    offset += 16;

    for(int i = 0; i < saveModel->numInLayers; i++)
    {
        int2bin((*saveModel->inLayers[i])->layerID, 16, bitBuff);
        for(int j = 0; j < 16; j++) line[offset + j] = bitBuff[j]; // big endian representation
        offset += 16;
    }

    snprintf(fltBuff, 19UL, "%.16f", saveModel->learningRate);
    for(int l = 0; l < 16; l++) line[offset + l] = fltBuff[l];
    offset += 16;

    line[offset] = saveModel->loss_fn;

    line[lineLength - 1] = '\0';

    int2bin(lineLength, 24, bitBuff);
        
    for(int i = 0; i < 24; i++) fputc(bitBuff[i], modFile);
    fputc('\n', modFile);
    fputs(line, modFile);
    fputs("\n", modFile);

    free(line);
    line = NULL;

    for(int i = 0; i < saveModel->numLayers; i++)
    {
        lineLength = 29;
        if((*saveModel->layerList[i])->layerType == 'h' || (*saveModel->layerList[i])->layerType == 'o')
        {
            lineLength += 16 * (*saveModel->layerList[i])->numPrevLayers;
            lineLength += 16 * ((*saveModel->layerList[i])->numNodes * ((*saveModel->layerList[i])->numPrevNodes + 1));
            lineLength += 65;
        }
        else if((*saveModel->layerList[i])->layerType == 'w')
        {
            lineLength += 16 * (*saveModel->layerList[i])->numPrevLayers;
            lineLength += 49;
        }
        offset = 0;

        line = (char *)calloc(lineLength, sizeof(char));
        if(line == NULL) goto error2;

        line[0] = (*saveModel->layerList[i])->layerType;
        offset += 1;

        int2bin((*saveModel->layerList[i])->layerID, 11, bitBuff);
        for(int j = 0; j < 11; j++) line[offset + j] = bitBuff[j]; // big endian representation
        offset += 11;

        int2bin((*saveModel->layerList[i])->numNodes, 16, bitBuff);
        for(int j = 0; j < 16; j++) line[offset+ j] = bitBuff[j];
        offset += 16;

        if((*saveModel->layerList[i])->layerType == 'i')
        {
            line[offset] = '\0';
            
            int2bin(lineLength, 24, bitBuff);
            
            for(int j = 0; j < 24; j++) fputc(bitBuff[j], modFile);
            fputc('\n', modFile);
            
            fputs(line, modFile);
            fputs("\n", modFile);
            
            free(line);
            line = (char*)NULL;
            continue;
        }

        int2bin((*saveModel->layerList[i])->numPrevNodes, 32, bitBuff);
        for(int j = 0; j < 32; j++) line[offset + j] = bitBuff[j];
        offset += 32;
        
        int2bin((*saveModel->layerList[i])->numPrevLayers, 16, bitBuff);
        for(int j = 0; j < 16; j++) line[offset + j] = bitBuff[j];
        offset += 16;
        
        line[offset] = (*saveModel->layerList[i])->activationFunction;
        offset += 1;

        for(int j = 0; j < (*saveModel->layerList[i])->numPrevLayers; j++)
        {
            //if((*layerList[i])->layerID == (*(*layerList[i])->prevLayers[j])->layerID) continue;
            int2bin((*(*saveModel->layerList[i])->prevLayers[j])->layerID, 16, bitBuff);
            for(int k = 0; k < 16; k++) line[offset + k] = bitBuff[k];
            offset += 16;
        }

        if((*saveModel->layerList[i])->layerType == 'w')
        {
            line[offset] = '\0';
            
            int2bin(lineLength, 24, bitBuff);
            
            for(int j = 0; j < 24; j++) fputc(bitBuff[j], modFile);
            fputc('\n', modFile);
            
            fputs(line, modFile);
            fputs("\n", modFile);
            
            free(line);
            line = (char*)NULL;
            continue;
        }

        for(int j = 0; j < (*saveModel->layerList[i])->numNodes; j++)
        {
            for(int k = 0; k < (*saveModel->layerList[i])->numPrevNodes; k++)
            {
                if((*saveModel->layerList[i])->weights[j][k] < 0) snprintf(fltBuff, 18UL, "%.15f", (*saveModel->layerList[i])->weights[j][k]);
                else snprintf(fltBuff, 19UL, "%.16f", (*saveModel->layerList[i])->weights[j][k]);
                
                for(int l = 0; l < 16; l++) line[offset + l] = fltBuff[l];
                offset += 16;
            }
        }

        for(int j = 0; j < (*saveModel->layerList[i])->numNodes; j++)
        {
            if((*saveModel->layerList[i])->biases[j] < 0) snprintf(fltBuff, 18UL, "%.15f", (*saveModel->layerList[i])->biases[j]);
            else snprintf(fltBuff, 19UL, "%.16f", (*saveModel->layerList[i])->biases[j]);
            
            for(int l = 0; l < 16; l++) line[offset + l] = fltBuff[l];
            offset += 16;
        }

        line[offset] = '\0';

        int2bin(lineLength, 24, bitBuff);
        
        for(int j = 0; j < 24; j++) fputc(bitBuff[j], modFile);
        fputc('\n', modFile);
        fputs(line, modFile);
        fputs("\n", modFile);

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
    int *inLayerIDs = NULL;
    char* line = NULL;
    float learningRate = 1.0f;
    int outLayerID = 0;
    int lineLength = 0;
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
    char lineLengthBuff[26] = "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"; // Line length of next line string will always be 24 characters
    char fltBuff[17] = "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

    FILE *modFile = fopen(modelFileName, "r");
    if(modFile == NULL) goto error1;

    if(fgets(lineLengthBuff, 26, modFile) == NULL) goto error5;

    lineLength = bin2int(lineLengthBuff, 24) + 1;

    line = (char *)calloc(lineLength, sizeof(char));
    if(line == NULL) goto error2;

    if(fgets(line, lineLength, modFile) == NULL) goto error6;
    numLayers = bin2int(line, 16);
    numInLayers = bin2int(&line[16], 16);
    offset += 32;

    inLayerIDs = (int *)calloc(numInLayers, sizeof(int));
    if(inLayerIDs == NULL) goto error3;

    for(int i = 0; i < numInLayers; i++) 
    {
        inLayerIDs[i] = bin2int(&line[offset], 16);
        offset += 16;
    }

    for(int i = 0; i < 16; i++) fltBuff[i] = line[offset + i];
    offset += 16;

    learningRate = atof(fltBuff);

    loss_fn = line[offset];

    free(line);
    line = NULL;

    *modelLayers = (layer**)malloc(numLayers * sizeof(layer*));
    if(*modelLayers == NULL) goto error4;

    for(int i = 0; i < numLayers; i++)
    {
        if(fgets(lineLengthBuff, 26, modFile) == NULL) goto error5;

        lineLength = bin2int(lineLengthBuff, 24) + 2;

        line = (char *)calloc(lineLength, sizeof(char));
        if(line == NULL) goto error5;

        if(fgets(line, lineLength, modFile) == NULL) goto error6;

        offset = 0;
        numNodes = 0;
        numPrevNodes = 0;

        layerType = line[0];
        offset += 1;

        layerID = bin2int(&line[offset], 11);
        offset += 11;
        
        numNodes = bin2int(&line[offset], 16);
        offset += 16;

        if(layerType == 'i')
        {
            (*modelLayers)[layerID] = make_input_layer(numNodes);
            if((*modelLayers)[layerID] == NULL) goto error7;

            (*modelLayers)[layerID]->layerID = layerID;
            
            free(line);
            line = NULL;
            
            continue;
        }

        numPrevNodes = bin2int(&line[offset], 32);
        offset += 32;

        numPrevLayers = bin2int(&line[offset], 16);
        offset += 16;

        activationFunction = line[offset];
        offset += 1;

        layerArr = (layer***)malloc(numPrevLayers * sizeof(layer**));
        if(layerArr == NULL) goto error6;

        for(int j = 0; j < numPrevLayers; j++)
        {
            layerArr[j] = &((*modelLayers)[bin2int(&line[offset], 16)]);
            offset += 16;
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

            flush_buffer(lineLengthBuff, 26);
            free(line);
            line = (char *)NULL;
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
                for(int l = 0; l < 16; l++) fltBuff[l] = line[offset+l];
                offset += 16;
                (*modelLayers)[layerID]->weights[j][k] = atof(fltBuff);
                flush_buffer(fltBuff, 16);
            }
        }

        for(int j = 0; j < numNodes; j++)
        {
            for(int k = 0; k < 16; k++) fltBuff[k] = line[offset+k];
            offset += 16;
            (*modelLayers)[layerID]->biases[j] = atof(fltBuff);
            flush_buffer(fltBuff, 16);
        }

        flush_buffer(lineLengthBuff, 26);
        free(line);
        line = (char *)NULL;
        free(layerArr);
        layerArr = (layer***)NULL;
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
    free(modelLayers);
    modelLayers = NULL;
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
    layer* prevLayer;
    int numInputs = 0;
    int numHiddenNodes = 0;
    int prevsTraversed = 0;

    for(int l = 0; l < myModel->numLayers; l++)
    {
        currLayer = *myModel->layerList[l];

        if(currLayer->numPrevLayers == 0) continue;
        if(currLayer->layerType == 'w' && opType == 't' && currLayer->numPrevLayers == 2)
        {
            numInputs = (*currLayer->prevLayers[0])->numNodes;
            numHiddenNodes = currLayer->numNodes;

            memcpy((*currLayer->prevLayers[1])->outputs, currLayer->outputs, sizeof(float) * numHiddenNodes);
            memcpy((*currLayer->prevLayers[1])->preActivations, currLayer->preActivations, sizeof(float) * numHiddenNodes);
            memcpy((*(*currLayer->prevLayers[1])->prevLayers[0])->outputs, (*currLayer->prevLayers[0])->outputs, sizeof(float) * numInputs);
        }
        
        if((*currLayer->prevLayers[currLayer->numPrevLayers - 1])->layerType == 'w' && currLayer->layerType == 'h')
        {
            numHiddenNodes = currLayer->numNodes;
            int windowIdx = currLayer->numPrevLayers - 1;

            memcpy((*currLayer->prevLayers[windowIdx])->outputs, currLayer->outputs, sizeof(float) * numHiddenNodes);
            memcpy((*currLayer->prevLayers[windowIdx])->preActivations, currLayer->preActivations, sizeof(float) * numHiddenNodes);

            for(int i = 0; i < currLayer->numPrevLayers - 1; i++)
            {
                prevLayer = *currLayer->prevLayers[i];
                
                memcpy(&(*(*currLayer->prevLayers[windowIdx])->prevLayers[0])->outputs[prevsTraversed], prevLayer->outputs, sizeof(float) * prevLayer->numNodes);
                prevsTraversed += prevLayer->numNodes;
            }

            prevsTraversed = 0;
        }
    }
}

// Applies Back Propagation Through Time procedure for all context windows
void calculate_and_apply_grads_through_time(model* myModel)
{
    layer* currLayer;
    layer* prevLayer;
    int prevsTraversed = 0;
    
    for(int l = myModel->numLayers - 1; l > -1; l--)
    {
        currLayer = *myModel->layerList[l];

        if(currLayer->layerType == 'i' || currLayer->layerType == 'w') continue;

        // newBiases[i] = oldBiases[i] - (learningRate * backErrors[i])
        for(int i = 0; i < currLayer->numNodes; i++) currLayer->biases[i] -= myModel->learningRate * currLayer->backErrors[i];
        if((*currLayer->prevLayers[currLayer->numPrevLayers - 1])->layerType == 'w')
        {
            prevLayer = (*currLayer->prevLayers[currLayer->numPrevLayers - 1]);

            while(prevLayer->numPrevLayers == 2)
            {
                for(int i = 0; i < prevLayer->numNodes; i++) currLayer->biases[i] -= myModel->learningRate * prevLayer->backErrors[i];
                prevLayer = *prevLayer->prevLayers[1];
            }
            
            for(int i = 0; i < prevLayer->numNodes; i++) currLayer->biases[i] -= myModel->learningRate * prevLayer->backErrors[i]; // For the last layer in the window with only one prevLayer
        }


        // newWeights[i][j] = oldWeights[i][j] - (learningRate * (prevNodeOuts[j] * backError[i]))
        for(int i = 0; i < currLayer->numNodes; i++, prevsTraversed = 0)
        {
            for(int j = 0; j < currLayer->numPrevLayers; j++)
            {
                prevLayer = *currLayer->prevLayers[j];
                for(int k = 0; k < prevLayer->numNodes; k++) currLayer->weights[i][k + prevsTraversed] -= myModel->learningRate * prevLayer->outputs[k] * currLayer->backErrors[i];          
                prevsTraversed += prevLayer->numNodes;
            }

            if((*currLayer->prevLayers[currLayer->numPrevLayers - 1])->layerType == 'w')
            {
                prevLayer = (*currLayer->prevLayers[currLayer->numPrevLayers - 1]);
                prevLayer = *prevLayer->prevLayers[1];

                while(prevLayer->numPrevLayers == 2)
                {
                    for(int j = 0; j < (*prevLayer->prevLayers[0])->numNodes; j++) currLayer->weights[i][j] -= myModel->learningRate * (*prevLayer->prevLayers[0])->outputs[j] * prevLayer->backErrors[i];
                    prevLayer = *prevLayer->prevLayers[1];
                }
                for(int j = 0; j < (*prevLayer->prevLayers[0])->numNodes; j++) currLayer->weights[i][j] -= myModel->learningRate * (*prevLayer->prevLayers[0])->outputs[j] * prevLayer->backErrors[i];
            }

        }
    }
}


#if defined(__AVX__) || defined(__AVX2__)
//  Gets an output from the target layer, is essentially also a inference function
// Vectorized version of forward out, only really makes a difference on industrial grade models so it will be shelved for now
int _mm256_forward_out(model* myModel, float dropoutVal)
{
    layer* currLayer;

    for(int l = 0; l < myModel->numLayers; l++)
    {
        currLayer = (*myModel->layerList[l]);
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

// void _mm256_sgd_backprop(model* myModel)
// { // start at output layer and calculate backerrors for each previous layer
//     layer* currLayer;
//     layer* prevLayer;
//     int prevsTraversed = 0;
    
//     for(int l = myModel->numLayers - 1; l > -1; l--)
//     {
//         currLayer = *myModel->layerList[l];

//         if(currLayer->layerType == 'i' || currLayer->layerType == 'w') continue;
        
//         // backErrorsForOutputLayer = lossDerivative · activationFunctionDerivative(preActivations) - for output layer
//         if(currLayer->layerType == 'o') vectorized_sgd_backprop_output_calc(currLayer, myModel);
        
//         // backErrorsForPreviousLayers[j] = SUM_OVER_I((thisLayersBackErrors[i])(thisLayersWeightMatrix[i][j]) · activationFunctionDerivative(previousLayersPreActivation[j])) - where j is considered to be a traversal of all previous 'J' layers' 'K' values as one vector
//         // e.g. J = 3 prev layers with K = 5 nodes each are considered as one prev layer with J = 15 nodes in this formulation
//         prevsTraversed = 0;
//         for(int i = 0; i < currLayer->numPrevLayers; i++)
//         {
//             prevLayer = *currLayer->prevLayers[i];
            
//             if(prevLayer->layerType == 'i' || prevLayer->layerType == 'w') continue;
            
//             for(int j = 0; j < prevLayer->numNodes; j++) for(int k = 0; k < currLayer->numNodes; k++) prevLayer->backErrors[j] += currLayer->backErrors[k] * currLayer->weights[k][prevsTraversed + j] * activation_derivative(prevLayer->preActivations[j], currLayer, i);
//             prevsTraversed += prevLayer->numNodes;
//         }
//     }
// }

// int _mm256_calculate_and_apply_grads(model* myModel)
// {
//     layer* currLayer;

//     for(int l = 0; l < myModel->numLayers; l++)
//     {
//         currLayer = *myModel->layerList[l];

//         if(currLayer->layerType == 'i') continue;
//         if(vectorized_calculate_and_apply_grads(currLayer, myModel->learningRate) != 0) return -1;
//     }
//     return 0;
// }

// int _mm256_calculate_and_apply_grads_through_time(model* myModel)
// {
//     layer* currLayer;

//     for(int l = 0; l < myModel->numLayers; l++)
//     {
//         currLayer = *myModel->layerList[l];

//         if(currLayer->layerType == 'i' || currLayer->layerType == 'w') continue;
//         if(vectorized_calculate_and_apply_grads_through_time(currLayer, myModel->learningRate) != 0) return -1;
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