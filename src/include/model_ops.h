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
    int numPrevsTraversed = 0;

    for(int l = 0; l < myModel->numLayers; l++)
    {        
        if((*myModel->layerList[l])->layerType == 'w' || (*myModel->layerList[l])->layerType == 'i') continue; // If layer has been traversed or if layer is a window layer for a context window or if layer is an input layer
        
        memset((*myModel->layerList[l])->backErrors, 0.0f, (*myModel->layerList[l])->numNodes * sizeof(float));
        memset((*myModel->layerList[l])->preActivations, 0.0f, (*myModel->layerList[l])->numNodes * sizeof(float));

        // preActivation[i] = SUM_OVER_J(prevNodeOutputs[j] * weights[i][j]) 
        for(int i = 0; i < (*myModel->layerList[l])->numNodes; i++, numPrevsTraversed = 0) 
        {
            for(int j = 0; j < (*myModel->layerList[l])->numPrevLayers; j++)
            {
                for(int k = 0; k < (*(*myModel->layerList[l])->prevLayers[j])->numNodes; k++) (*myModel->layerList[l])->preActivations[i] += (*(*myModel->layerList[l])->prevLayers[j])->outputs[k] * (*myModel->layerList[l])->weights[i][numPrevsTraversed + k];
                numPrevsTraversed += (*(*myModel->layerList[l])->prevLayers[j])->numNodes;
            }
            (*myModel->layerList[l])->preActivations[i] += (*myModel->layerList[l])->biases[i];
        }
        
        numPrevsTraversed = 0;

        // In case of softmax activation, we do a function on the layer instead of point-wise on the outputs
        if((*myModel->layerList[l])->activationFunction == 'x')
        {
            memcpy((*myModel->layerList[l])->outputs, (*myModel->layerList[l])->preActivations, sizeof(float) * (*myModel->layerList[l])->numNodes);
            softmax(myModel->layerList[l]);
        }
        else if((*myModel->layerList[l])->activationFunction == 'f')
        {
            memcpy((*myModel->layerList[l])->outputs, (*myModel->layerList[l])->preActivations, sizeof(float) * (*myModel->layerList[l])->numNodes);
            fast_softmax(myModel->layerList[l]);
        }
        else
        {
            // outputs[i] = activation_function(preActivations[i])
            for(int i = 0; i < (*myModel->layerList[l])->numNodes; i++) (*myModel->layerList[l])->outputs[i] = activation_function((*myModel->layerList[l])->preActivations[i], (*myModel->layerList[l])->activationFunction);
        }

        if((*myModel->layerList[l])->layerType == 'o' || dropoutVal <= 0.0f) continue;;
        float scalingFactor = 1/(1-dropoutVal);
        for(int i = 0; i < (*myModel->layerList[l])->numNodes; i++) (*myModel->layerList[l])->outputs[i] *= ((float)((rand() % 999))/1000.0 < dropoutVal) ? 0 : scalingFactor;
    }
}

// Run on each output layer and then apply grads before clearing the layer backerrors - All roads spring forth from Rome algorithm
// We pass the backerrors to each previous layer to calculate grads later
// Backerrors can be accumulated from multiple successor layers to calculate grads due to matrix distributivity
void sgd_backprop(model* myModel)
{ // start at output layer and calculate backerrors for each previous layer
    int prevsTraversed = 0;
    
    for(int l = myModel->numLayers - 1; l > -1; l--)
    {
        if((*myModel->layerList[l])->numPrevLayers == 0 || (*myModel->layerList[l])->layerType == 'w') continue;
        
        // backErrorsForOutputLayer = lossDerivative · activationFunctionDerivative(preActivations) - for output layer
        if((*myModel->layerList[l])->layerType == 'o') for(int i = 0; i < (*myModel->layerList[l])->numNodes; i++) (*myModel->layerList[l])->backErrors[i] = -1 * loss_derivative(myModel->targets[i], (*myModel->layerList[l])->outputs[i], myModel) * activation_derivative((*myModel->layerList[l])->preActivations[i], (*myModel->layerList[l])->activationFunction, *myModel->layerList[l], i);
        
        // backErrorsForPreviousLayers[j] = SUM_OVER_I((thisLayersBackErrors[i])(thisLayersWeightMatrix[i][j]) · activationFunctionDerivative(previousLayersPreActivation[j])) - where j is considered to be a traversal of all previous 'J' layers' 'K' values as one vector
        // e.g. J = 3 prev layers with K = 5 nodes each are considered as one prev layer with J = 15 nodes in this formulation
        prevsTraversed = 0;
        for(int i = 0; i < (*myModel->layerList[l])->numPrevLayers; i++)
        {
            if((*(*myModel->layerList[l])->prevLayers[i])->layerType == 'i' || (*(*myModel->layerList[l])->prevLayers[i])->layerType == 'w') continue;
            for(int j = 0; j < (*(*myModel->layerList[l])->prevLayers[i])->numNodes; j++) for(int k = 0; k < (*myModel->layerList[l])->numNodes; k++) (*(*myModel->layerList[l])->prevLayers[i])->backErrors[j] += (*myModel->layerList[l])->backErrors[k] * (*myModel->layerList[l])->weights[k][prevsTraversed + j] * activation_derivative((*(*myModel->layerList[l])->prevLayers[i])->preActivations[j], (*(*myModel->layerList[l])->prevLayers[i])->activationFunction, *myModel->layerList[l], i);
            prevsTraversed += (*(*myModel->layerList[l])->prevLayers[i])->numNodes;
        }
    }
}

// Another all roads spring forth from Rome approach - go to the convergence point of the model(output layer) and use it as the root this model graph
void calculate_and_apply_grads(model* myModel)
{
    int prevsTraversed = 0;

    for(int l = myModel->numLayers - 1; l > -1; l--)
    {
        if((*myModel->layerList[l])->layerType == 'i' || (*myModel->layerList[l])->layerType == 'w') continue;

        // newBias[i] = oldBias[i] - (learningRate * backErrors[i])
        for(int i = 0; i < (*myModel->layerList[l])->numNodes; i++) (*myModel->layerList[l])->biases[i] -= myModel->learning_rate * (*myModel->layerList[l])->backErrors[i];

        prevsTraversed = 0;

        // newWeights[i][j] = oldWeights[i][j] - (learningRate * (prevNodeOuts[j] * backError[i]))
        for(int i = 0; i < (*myModel->layerList[l])->numNodes; i++)
        {
            for(int j = 0; j < (*myModel->layerList[l])->numPrevLayers; j++)
            {
                for(int k = 0; k < (*(*myModel->layerList[l])->prevLayers[j])->numNodes; k++) (*myModel->layerList[l])->weights[i][k + prevsTraversed] -= myModel->learning_rate * (*(*myModel->layerList[l])->prevLayers[j])->outputs[k] * (*myModel->layerList[l])->backErrors[i];
                
                prevsTraversed += (*(*myModel->layerList[l])->prevLayers[j])->numNodes;
            }
            prevsTraversed = 0;
        }
    }
}

// For clearing the backErrors once no longer needed, and to also prime for next forward and backward pass
// Use by passing the output layer of the model into the function 
void zero_everything(model* myModel)
{
    for(int l = 0; l < myModel->numLayers; l++)
    {
        if((*myModel->layerList[l])->layerType == 'i') continue;

        memset((*myModel->layerList[l])->backErrors, 0.0f, (*myModel->layerList[l])->numNodes * sizeof(float));
        memset((*myModel->layerList[l])->preActivations, 0.0f, (*myModel->layerList[l])->numNodes * sizeof(float));
        memset((*myModel->layerList[l])->outputs, 0.0f, (*myModel->layerList[l])->numNodes * sizeof(float));
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

    snprintf(fltBuff, 19UL, "%.16f", saveModel->learning_rate);
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
    myModel->learning_rate = learningRate;
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
    for(int i = 0; i < numLayers; i++)
    {
        hakai_layer(&(*modelLayers)[i]);
    }
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
    int numInputs = 0;
    int numHiddenNodes = 0;
    int prevsTraversed = 0;
    for(int l = 0; l < myModel->numLayers; l++)
    {
        if((*myModel->layerList[l])->numPrevLayers == 0) continue;
        if((*myModel->layerList[l])->layerType == 'w' && opType == 't' && (*myModel->layerList[l])->numPrevLayers == 2)
        {
            numInputs = (*(*myModel->layerList[l])->prevLayers[0])->numNodes;
            numHiddenNodes = (*myModel->layerList[l])->numNodes;

            memcpy((*(*myModel->layerList[l])->prevLayers[1])->outputs, (*myModel->layerList[l])->outputs, sizeof(float) * numHiddenNodes);
            memcpy((*(*myModel->layerList[l])->prevLayers[1])->preActivations, (*myModel->layerList[l])->preActivations, sizeof(float) * numHiddenNodes);
            memcpy((*(*(*myModel->layerList[l])->prevLayers[1])->prevLayers[0])->outputs, (*(*myModel->layerList[l])->prevLayers[0])->outputs, sizeof(float) * numInputs);
        }
        
        if((*(*myModel->layerList[l])->prevLayers[(*myModel->layerList[l])->numPrevLayers - 1])->layerType == 'w' && (*myModel->layerList[l])->layerType == 'h')
        {
            numHiddenNodes = (*myModel->layerList[l])->numNodes;
            int windowIdx = (*myModel->layerList[l])->numPrevLayers - 1;

            memcpy((*(*myModel->layerList[l])->prevLayers[windowIdx])->outputs, (*myModel->layerList[l])->outputs, sizeof(float) * numHiddenNodes);
            memcpy((*(*myModel->layerList[l])->prevLayers[windowIdx])->preActivations, (*myModel->layerList[l])->preActivations, sizeof(float) * numHiddenNodes);

            for(int i = 0; i < (*myModel->layerList[l])->numPrevLayers - 1; i++)
            {
                memcpy(&(*(*(*myModel->layerList[l])->prevLayers[windowIdx])->prevLayers[0])->outputs[prevsTraversed], (*(*myModel->layerList[l])->prevLayers[i])->outputs, sizeof(float) * (*(*myModel->layerList[l])->prevLayers[i])->numNodes);
                prevsTraversed += (*(*myModel->layerList[l])->prevLayers[i])->numNodes;
            }

            prevsTraversed = 0;
        }
    }
}

// Applies Back Propagation Through Time procedure for all context windows
void calculate_and_apply_grads_through_time(model* myModel)
{
    int prevsTraversed = 0;
    for(int l = myModel->numLayers - 1; l > -1; l--)
    {
        if((*myModel->layerList[l])->numPrevLayers == 0 || (*myModel->layerList[l])->layerType == 'w') continue;

        // newBiases[i] = oldBiases[i] - (learningRate * backErrors[i])
        for(int i = 0; i < (*myModel->layerList[l])->numNodes; i++) (*myModel->layerList[l])->biases[i] -= myModel->learning_rate * (*myModel->layerList[l])->backErrors[i];
        if((*(*myModel->layerList[l])->prevLayers[(*myModel->layerList[l])->numPrevLayers - 1])->layerType == 'w')
        {
            layer* currLayer = (*(*myModel->layerList[l])->prevLayers[(*myModel->layerList[l])->numPrevLayers - 1]);

            while(currLayer->numPrevLayers == 2)
            {
                for(int i = 0; i < currLayer->numNodes; i++) (*myModel->layerList[l])->biases[i] -= myModel->learning_rate * currLayer->backErrors[i];
                currLayer = *currLayer->prevLayers[1];
            }
            
            for(int i = 0; i < currLayer->numNodes; i++) (*myModel->layerList[l])->biases[i] -= myModel->learning_rate * currLayer->backErrors[i]; // For the last layer in the window with only one prevLayer
        }


        // newWeights[i][j] = oldWeights[i][j] - (learningRate * (prevNodeOuts[j] * backError[i]))
        for(int i = 0; i < (*myModel->layerList[l])->numNodes; i++, prevsTraversed = 0)
        {
            for(int j = 0; j < (*myModel->layerList[l])->numPrevLayers; j++)
            {
                for(int k = 0; k < (*(*myModel->layerList[l])->prevLayers[j])->numNodes; k++) (*myModel->layerList[l])->weights[i][k + prevsTraversed] -= myModel->learning_rate * (*(*myModel->layerList[l])->prevLayers[j])->outputs[k] * (*myModel->layerList[l])->backErrors[i];          
                prevsTraversed += (*(*myModel->layerList[l])->prevLayers[j])->numNodes;
            }

            if((*(*myModel->layerList[l])->prevLayers[(*myModel->layerList[l])->numPrevLayers - 1])->layerType == 'w')
            {
                layer* currLayer = (*(*myModel->layerList[l])->prevLayers[(*myModel->layerList[l])->numPrevLayers - 1]);
                currLayer = *currLayer->prevLayers[1];

                while(currLayer->numPrevLayers == 2)
                {
                    for(int j = 0; j < (*currLayer->prevLayers[0])->numNodes; j++) (*myModel->layerList[l])->weights[i][j] -= myModel->learning_rate * (*currLayer->prevLayers[0])->outputs[j] * currLayer->backErrors[i];
                    currLayer = *currLayer->prevLayers[1];
                }
                for(int j = 0; j < (*currLayer->prevLayers[0])->numNodes; j++) (*myModel->layerList[l])->weights[i][j] -= myModel->learning_rate * (*currLayer->prevLayers[0])->outputs[j] * currLayer->backErrors[i];
            }

        }
    }
}


#if defined(__AVX__) || defined(__AVX2__)
//  Gets an output from the target layer, is essentially also a inference function
// Vectorized version of forward out, only really makes a difference on industrial grade models so it will be shelved for now
int _mm256_forward_out(model* myModel, float dropoutVal)
{
    for(int l = 0; l < myModel->numLayers; l++)
    {
        if((*myModel->layerList[l])->layerType == 'i' || (*myModel->layerList[l])->layerType == 'w') continue;
        
        memset((*myModel->layerList[l])->backErrors, 0.0f, (*myModel->layerList[l])->numNodes * sizeof(float));
        memset((*myModel->layerList[l])->preActivations, 0.0f, (*myModel->layerList[l])->numNodes * sizeof(float));
        memset((*myModel->layerList[l])->outputs, 0.0f, (*myModel->layerList[l])->numNodes * sizeof(float));

        if(vectorized_forward_out_calc(myModel->layerList[l]) != 0) return -1;

        // In case of softmax activation, we do a function on the layer instead of point-wise on the outputs
        if((*myModel->layerList[l])->activationFunction == 'x')
        {
            memcpy((*myModel->layerList[l])->outputs, (*myModel->layerList[l])->preActivations, sizeof(float) * (*myModel->layerList[l])->numNodes);
            softmax(myModel->layerList[l]);
        }
        else if((*myModel->layerList[l])->activationFunction == 'f')
        {
            memcpy((*myModel->layerList[l])->outputs, (*myModel->layerList[l])->preActivations, sizeof(float) * (*myModel->layerList[l])->numNodes);
            fast_softmax(myModel->layerList[l]);
        }
        else
        {            
            for(int i = 0; i < (*myModel->layerList[l])->numNodes; i++) (*myModel->layerList[l])->outputs[i] = activation_function((*myModel->layerList[l])->preActivations[i], (*myModel->layerList[l])->activationFunction);
        }

        if((*myModel->layerList[l])->layerType == 'o' || dropoutVal <= 0.0) continue;
        float scalingFactor = 1/(1-dropoutVal);
        for(int i = 0; i < (*myModel->layerList[l])->numNodes; i++) (*myModel->layerList[l])->outputs[i] *= ((float)((rand() % 999))/1000.0 < dropoutVal) ? 0 : scalingFactor;
    }

    return 0;
}

// void _mm256_sgd_backprop(layer** myLayer, model** myModel)
// { // start at output layer and calculate backerrors for each previous layer
//     if((*myLayer)->switchVar == '2') return;
//
//     (*myLayer)->switchVar = '2';
//
//     vectorized_sgd_backprop_calc(myLayer, myModel);
//
//     for(int i = 0; i < (*myLayer)->numPrevLayers; i++) if((*(*myLayer)->prevLayers[i])->numPrevLayers != 0 && (*(*myLayer)->prevLayers[i])->layerType != 'w') sgd_backprop((*myLayer)->prevLayers[i], myModel);
//     // calculate backErrors for previous layers' previous layers according to already established layers' backErrors - All roads spring forth from Rome
// }

int _mm256_calculate_and_apply_grads(model* myModel)
{
    for(int l = 0; l < myModel->numLayers; l++)
    {
        if((*myModel->layerList[l])->layerType == 'i') continue;
        if(vectorized_calculate_and_apply_grads(myModel->layerList[l], myModel->learning_rate) != 0) return -1;
    }
    return 0;
}

int _mm256_calculate_and_apply_grads_through_time(model* myModel)
{
    for(int l = 0; l < myModel->numLayers; l++)
    {
        if((*myModel->layerList[l])->layerType == 'i' || (*myModel->layerList[l])->layerType == 'w') continue;
        if(vectorized_calculate_and_apply_grads_through_time(myModel->layerList[l], myModel->learning_rate) != 0) return -1;
    }
    return 0;
}

#endif

#if defined(__ARM_NEON)

void vforward_out(layer** myLayer, float dropoutVal)
{
    for(int l = 0; l < myModel->numLayers; l++)
    {
        if((*myModel->layerList[l])->layerType == 'i' && (*myModel->layerList[l])->layerType == 'w') continue;

        memset((*myModel->layerList[l])->backErrors, 0.0f, (*myModel->layerList[l])->numNodes * sizeof(float));
        memset((*myModel->layerList[l])->preActivations, 0.0f, (*myModel->layerList[l])->numNodes * sizeof(float));
        memset((*myModel->layerList[l])->outputs, 0.0f, (*myModel->layerList[l])->numNodes * sizeof(float));

        if(vectorized_forward_out_calc(myModel->layerList[l]) != 0) return -1;

        // In case of softmax activation, we do a function on the layer instead of point-wise on the outputs
        if((*myModel->layerList[l])->activationFunction == 'x')
        {
            memcpy((*myModel->layerList[l])->outputs, (*myModel->layerList[l])->preActivations, sizeof(float) * (*myModel->layerList[l])->numNodes);
            softmax(myModel->layerList[l]);
        }
        else if((*myModel->layerList[l])->activationFunction == 'f')
        {
            memcpy((*myModel->layerList[l])->outputs, (*myModel->layerList[l])->preActivations, sizeof(float) * (*myModel->layerList[l])->numNodes);
            fast_softmax(myModel->layerList[l]);
        }
        else
        {            
            for(int i = 0; i < (*myModel->layerList[l])->numNodes; i++) (*myModel->layerList[l])->outputs[i] = activation_function((*myModel->layerList[l])->preActivations[i], (*myModel->layerList[l])->activationFunction);
        }

        if((*myModel->layerList[l])->layerType == 'o' || dropoutVal <= 0.0) continue;
        float scalingFactor = 1/(1-dropoutVal);
        for(int i = 0; i < (*myModel->layerList[l])->numNodes; i++) (*myModel->layerList[l])->outputs[i] *= ((float)((rand() % 999))/1000.0 < dropoutVal) ? 0 : scalingFactor;
    }

    return 0;
}

#endif