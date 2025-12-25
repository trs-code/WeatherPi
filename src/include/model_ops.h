#pragma once

#include "model_construct.h"
#include "activation.h"
#include "loss.h"
#include "helper_funcs.h"
#include <stdio.h>

//  Gets an output from the target layer, is essentially also a inference function
void forward_out(layer** myLayer)
{
    if((*myLayer)->switchVar == '1') return;

    (*myLayer)->switchVar = '1';

    if((*myLayer)->numPrevLayers != 0)
    {
        int numPrevsTraversed = 0;
        
        for(int i = 0; i < (*myLayer)->numPrevLayers; i++) forward_out((*myLayer)->prevLayers[i]);

        for(int i = 0; i < (*myLayer)->numNodes; i++) 
        {
            for(int j = 0; j < (*myLayer)->numPrevLayers; j++)
            {
                for(int k = 0; k < (*(*myLayer)->prevLayers[j])->numNodes; k++) (*myLayer)->preActivations[i] += (*(*myLayer)->prevLayers[j])->outputs[k] * (*myLayer)->weights[i][numPrevsTraversed + k];
                numPrevsTraversed += (*(*myLayer)->prevLayers[j])->numNodes;
            }
            (*myLayer)->preActivations[i] += (*myLayer)->biases[i];
            numPrevsTraversed = 0;
        }
        
        for(int i = 0; i < (*myLayer)->numNodes; i++) (*myLayer)->outputs[i] = activation_function((*myLayer)->preActivations[i], (*myLayer)->activationFunction, *myLayer, i);
    }
}


// Run on each output layer and then apply grads before clearing the layer backerrors - All roads spring forth from Rome algorithm
// We pass the backerrors to each previous layer to calculate grads later
// Backerrors can be accumulated from multiple successor layers to calculate grads due to matrix distributivity
void sgd_backprop(layer** myLayer, model** myModel)
{ // start at output layer and calculate backerrors for each previous layer
    if((*myLayer)->switchVar == '2') return;

    (*myLayer)->switchVar = '2';
    
    // backErrorsForOutputLayer = lossDerivative · activationFunctionDerivative(preActivations) - for output layer
    if((*myLayer)->layerType == 'o') for(int i = 0; i < (*myLayer)->numNodes; i++) (*myLayer)->backErrors[i] = -1 * loss_derivative((*myModel)->targets[i], (*myLayer)->outputs[i], (*myModel)) * activation_derivative((*myLayer)->preActivations[i], (*myLayer)->activationFunction, *myLayer, i);
    
    // backErrorsForPreviousLayers += (thisLayersBackErrors)(thisLayersWeightMatrixWithRespectToCurrentPreviousLayer) · activationFunctionDerivative(previousLayers)
    int prevsTraversed = 0;
    for(int i = 0; i < (*myLayer)->numPrevLayers; i++)
    {
        if((*(*myLayer)->prevLayers[i])->layerType == 'i' || (*(*myLayer)->prevLayers[i])->layerType == 't' ) continue;
        for(int j = 0; j < (*(*myLayer)->prevLayers[i])->numNodes; j++) for(int k = 0; k < (*myLayer)->numNodes; k++) (*(*myLayer)->prevLayers[i])->backErrors[j] += (*myLayer)->backErrors[k] * (*myLayer)->weights[k][prevsTraversed + j] * activation_derivative((*(*myLayer)->prevLayers[i])->preActivations[j], (*myLayer)->activationFunction, *myLayer, i);
        prevsTraversed += (*(*myLayer)->prevLayers[i])->numNodes;
    }

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) if((*(*myLayer)->prevLayers[i])->numPrevLayers != 0) sgd_backprop((*myLayer)->prevLayers[i], myModel);
    // calculate backErrors for previous layers' previous layers according to already established layers' backErrors - All roads spring forth from Rome
}


// Another all roads spring forth from Rome approach - go to the convergence point of the model and use it as the root of the undirected, cyclic graph that is this model
void calculate_and_apply_grads(layer** myLayer, float learningRate)
{
    if((*myLayer)->switchVar == '3') return;

    (*myLayer)->switchVar = '3';

    if((*myLayer)->numPrevLayers == 0) return;

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) calculate_and_apply_grads(((*myLayer)->prevLayers[i]), learningRate);

    for(int i = 0; i < (*myLayer)->numNodes; i++) (*myLayer)->biases[i] -= learningRate * (*myLayer)->backErrors[i];

    int prevsTraversed = 0;

    // calculate gradients from backerrors and activations for each layer and apply them to the weights_mm256_mul_ps(vector_a, scalar_vector);
    for(int i = 0; i < (*myLayer)->numPrevLayers; i ++)
    {
        for(int j = 0; j < (*(*myLayer)->prevLayers[i])->numNodes; j++)
        {
            for(int k = 0; k < (*myLayer)->numNodes; k++) (*myLayer)->weights[k][j + prevsTraversed] -= learningRate * (*(*myLayer)->prevLayers[i])->outputs[j] * (*myLayer)->backErrors[k];
        }
        prevsTraversed += (*(*myLayer)->prevLayers[i])->numNodes;
    }
}

// For clearing the backErrors once no longer needed, and to also prime for next forward and backward pass
// Use by passing the output layer of the model into the function 
void zero_everything(layer** myLayer)
{
    if((*myLayer)->switchVar == '0') return;

    (*myLayer)->switchVar = '0';

    if((*myLayer)->layerType == 'i' || (*myLayer)->layerType == 't') return;

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) zero_everything((*myLayer)->prevLayers[i]);

    memset((*myLayer)->backErrors, 0.0f, (*myLayer)->numNodes * sizeof(float));
    memset((*myLayer)->preActivations, 0.0f, (*myLayer)->numNodes * sizeof(float));
    memset((*myLayer)->outputs, 0.0f, (*myLayer)->numNodes * sizeof(float));
}

int save_model(model** saveModel, char* modelFileName)
{
    FILE *modFile = NULL;
    char *line = NULL;
    int offset = 0;
    int lineLength = 42;
    char bitBuff[33] = "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
    char fltBuff[20] = "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
    
    layer*** layerList = (layer ***)calloc((*saveModel)->numLayers, sizeof(layer**));
    if(layerList == NULL) goto error1;

    traverse_model_fill_layer_list((*saveModel)->outLayer, layerList);

    modFile = fopen(modelFileName, "w");
    if(modFile == NULL) goto error2;

    lineLength += 16 * (*saveModel)->numInLayers;

    line = (char *)calloc(lineLength, sizeof(char));
    if(line == NULL) goto error3;
    
    int2bin((*saveModel)->numLayers, 16, bitBuff);
    for(int i = 0; i < 16; i++) line[offset + i] = bitBuff[i]; // big endian representation
    offset += 16;

    int2bin((*saveModel)->numInLayers, 16, bitBuff);
    for(int i = 0; i < 16; i++) line[offset + i] = bitBuff[i]; // big endian representation
    offset += 16;

    for(int i = 0; i < (*saveModel)->numInLayers; i++)
    {
        int2bin((*(*saveModel)->inLayers[i])->layerID, 16, bitBuff);
        for(int j = 0; j < 16; j++) line[offset + j] = bitBuff[j]; // big endian representation
        offset += 16;
    }

    snprintf(fltBuff, 10UL, "%.8f", (*saveModel)->learning_rate);
    for(int l = 0; l < 8; l++) line[offset + l] = fltBuff[l];
    offset += 8;

    line[offset] = (*saveModel)->loss_fn;

    line[lineLength - 1] = '\0';

    int2bin(lineLength, 24, bitBuff);
        
    for(int i = 0; i < 24; i++) fputc(bitBuff[i], modFile);
    fputc('\n', modFile);
    fputs(line, modFile);
    fputs("\n", modFile);

    free(line);
    line = NULL;

    for(int i = 0; i < (*saveModel)->numLayers; i++)
    {
        lineLength = 29;
        if((*layerList[i])->layerType != 'i')
        {
            lineLength += 16 * (*layerList[i])->numPrevLayers;
            lineLength += 16 * ((*layerList[i])->numNodes * ((*layerList[i])->numPrevNodes + 1));
            lineLength += 65;
        }
        offset = 0;

        line = (char *)calloc(lineLength, sizeof(char));
        if(line == NULL) goto error3;

        line[0] = (*layerList[i])->layerType;
        offset += 1;

        int2bin((*layerList[i])->layerID, 11, bitBuff);
        for(int j = 0; j < 11; j++) line[offset + j] = bitBuff[j]; // big endian representation
        offset += 11;

        int2bin((*layerList[i])->numNodes, 16, bitBuff);
        for(int j = 0; j < 16; j++) line[offset+ j] = bitBuff[j];
        offset += 16;

        if((*layerList[i])->layerType == 'i' || (*layerList[i])->layerType == 't')
        {
            line[offset] = '\0';
            
            int2bin(lineLength, 24, bitBuff);
            
            for(int j = 0; j < 24; j++) fputc(bitBuff[j], modFile);
            fputc('\n', modFile);
            
            fputs(line, modFile);
            fputs("\n", modFile);
            
            free(line);
            line = NULL;
            continue;
        }

        int2bin((*layerList[i])->numPrevNodes, 32, bitBuff);
        for(int j = 0; j < 32; j++) line[offset + j] = bitBuff[j];
        offset += 32;
        
        if((*layerList[i])->layerType != 'r') int2bin((*layerList[i])->numPrevLayers, 16, bitBuff);
        else int2bin((*layerList[i])->numPrevLayers - 1, 16, bitBuff);
        for(int j = 0; j < 16; j++) line[offset + j] = bitBuff[j];
        offset += 16;
        
        line[offset] = (*layerList[i])->activationFunction;
        offset += 1;

        for(int j = 0; j < (*layerList[i])->numPrevLayers; j++)
        {
            //if((*layerList[i])->layerID == (*(*layerList[i])->prevLayers[j])->layerID) continue;
            int2bin((*(*layerList[i])->prevLayers[j])->layerID, 16, bitBuff);
            for(int k = 0; k < 16; k++) line[offset + k] = bitBuff[k];
            offset += 16;
        }

        for(int j = 0; j < (*layerList[i])->numNodes; j++)
        {
            for(int k = 0; k < (*layerList[i])->numPrevNodes; k++)
            {
                if((*layerList[i])->weights[j][k] < 0) snprintf(fltBuff, 17UL, "%.15f", (*layerList[i])->weights[j][k]);
                else snprintf(fltBuff, 17UL, "%.16f", (*layerList[i])->weights[j][k]);
                
                for(int l = 0; l < 16; l++) line[offset + l] = fltBuff[l];
                offset += 16;
            }
        }

        for(int j = 0; j < (*layerList[i])->numNodes; j++)
        {
            if((*layerList[i])->biases[j] < 0) snprintf(fltBuff, 17UL, "%.15f", (*layerList[i])->biases[j]);
            else snprintf(fltBuff, 17UL, "%.16f", (*layerList[i])->biases[j]);
            
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
    free(layerList);
    layerList = NULL;
    modFile = NULL;
    return 0;

error3:
    fclose(modFile);
    modFile = NULL;
error2:
    free(layerList);
    layerList = NULL;
error1:
    return -1;
}

model* load_model(const char* modelFileName, layer*** modelLayers)
{
    layer*** layerArr = (layer***)NULL;
    model* myModel = NULL;
    int *inLayerIDs = NULL;
    char* line = NULL;
    int outLayerID = 0;
    int lineLength = 0;
    int offset = 0;
    int numLayers = 0;
    int numInLayers = 0;
    int numPrevLayers = 0;
    int numPrevNodes = 0;
    int layerID = 0;
    int numNodes = 0;
    float learningRate = 1.0f;
    char activationFunction = '\0';
    char loss_fn = '\0';
    char layerType = '\0';
    char lineLengthBuff[26] = "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"; // Line length of next line string will always be 24 characters
    char fltBuff[17] = "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

    FILE *modFile = fopen(modelFileName, "r");
    if(modFile == NULL) goto error1;

    fgets(lineLengthBuff, 26, modFile);

    lineLength = bin2int(lineLengthBuff, 24) + 2;

    line = (char *)calloc(lineLength, sizeof(char));
    if(line == NULL) goto error2;

    fgets(line, lineLength, modFile);
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

    for(int i = 0; i < 8; i++) fltBuff[i] = line[offset + i];
    offset += 8;

    learningRate = atof(fltBuff);

    loss_fn = line[offset];

    free(line);
    line = NULL;

    *modelLayers = (layer**)malloc(numLayers * sizeof(layer**));
    if(*modelLayers == NULL) goto error4;

    for(int i = 0; i < numLayers; i++)
    {
        fgets(lineLengthBuff, 26, modFile);

        lineLength = bin2int(lineLengthBuff, 24) + 2;

        line = (char *)calloc(lineLength, sizeof(char));
        if(line == NULL) goto error5;

        fgets(line, lineLength, modFile);

        offset = 0;
        numNodes = 0;
        numPrevNodes = 0;
        learningRate = 1.0f;

        layerType = line[0];
        offset += 1;

        layerID = bin2int(&line[offset], 11);
        offset += 11;
        
        numNodes = bin2int(&line[offset], 16);
        offset += 16;

        if(layerType == 'i')
        {
            (*modelLayers)[layerID] = make_input_layer(numNodes);
            
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
        }
        else if(layerType == 'r')
        {
            (*modelLayers)[layerID] = make_referential_layer(layerArr, numNodes, numPrevLayers, activationFunction, &(*modelLayers)[layerID]);
        }
        else
        {
            (*modelLayers)[layerID] = make_output_layer(layerArr, numNodes, numPrevLayers, activationFunction);
            outLayerID = layerID;
        }

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

    layerArr = (layer***)malloc(numInLayers * sizeof(layer*));

    for(int i = 0; i < numInLayers; i++) layerArr[i] = &(*(modelLayers)[inLayerIDs[i]]);

    myModel = construct_model(layerArr, &(*modelLayers)[outLayerID], numLayers, numInLayers, learningRate, loss_fn);

    free(inLayerIDs);
    inLayerIDs = NULL;
    free(layerArr);
    layerArr = NULL;

    return myModel;

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
    free(line);
    line = NULL;
error2:
    fclose(modFile);
error1:
    return NULL;
}

void extend_context(layer** myLayer, int windowSize, layer*** windowLayers) // Extend context windows 
{
    if((*myLayer)->layerType != 'h' || (*(*myLayer)->prevLayers[0])->layerType != 'i' || windowSize < 1) return;
    
    layer*** tmp0 = NULL;
    float* tmp1 = NULL;

    *windowLayers = (layer**)calloc(2 * (windowSize + 1), sizeof(layer**));
    if(*windowLayers == NULL) return;

    int hiddenNodes = (*myLayer)->numNodes;
    int inNodes = (*(*myLayer)->prevLayers[0])->numNodes;
    char hiddenActivationFunction = (*myLayer)->activationFunction;

    (*windowLayers)[0] = *(*myLayer)->prevLayers[0];
    (*windowLayers)[1] = *(myLayer);

    (*windowLayers)[2 * windowSize] = make_input_layer(inNodes);
    if((*windowLayers)[2 * windowSize] == NULL) goto error1;
    (*windowLayers)[(2 * windowSize) + 1] = make_hidden_layer((layer**[]){&(*windowLayers)[2 * windowSize]}, hiddenNodes, 1, hiddenActivationFunction);
    if((*windowLayers)[(2 * windowSize) + 1] == NULL) goto error1;
    (*windowLayers)[2 * windowSize]->layerType = 't';
    (*windowLayers)[(2 * windowSize) + 1]->layerType = 'r';

    for(int i = 0; i < hiddenNodes; i++) memcpy((*windowLayers)[(2 * windowSize) + 1]->weights[i], (*myLayer)->weights[i], sizeof(float) * (*myLayer)->numPrevNodes);
    memcpy((*windowLayers)[(2 * windowSize) + 1]->biases, (*myLayer)->biases, sizeof(float) * hiddenNodes);

    (*myLayer)->numPrevLayers += 1;
    (*myLayer)->numPrevNodes += hiddenNodes;
    tmp0 = (layer ***)realloc((*myLayer)->prevLayers, sizeof(layer**) * (*myLayer)->numPrevLayers);
    if(tmp0 == NULL) goto error1;
    (*myLayer)->prevLayers = tmp0;
    tmp0 = NULL;

    (*myLayer)->prevLayers[(*myLayer)->numPrevLayers - 1] = &(*windowLayers)[3];

    for(int i = 0; i < hiddenNodes; i++)
    {
        tmp1 = (float*)realloc((*myLayer)->weights[i], sizeof(float*) * (*myLayer)->numPrevNodes);
        if(tmp1 == NULL) goto error1;
        (*myLayer)->weights[i] = tmp1;
        tmp1 = NULL;

        for(int j = (*myLayer)->numPrevNodes - hiddenNodes; j < (*myLayer)->numPrevNodes; j++) (*myLayer)->weights[i][j] = ((rand() % 100000) + 50000)/100000;
    }

    for(int i = windowSize - 1; i > 0; i--)
    {
        (*windowLayers)[2 * i] = make_input_layer(inNodes);
        if((*windowLayers)[(2 * i)] == NULL) goto error1;
        (*windowLayers)[(2 * i) + 1] = make_hidden_layer((layer**[]){&(*windowLayers)[2 * i], &(*windowLayers)[(2 * i) + 3]}, hiddenNodes, 2, hiddenActivationFunction);
        if((*windowLayers)[(2 * i) + 1] == NULL) goto error1;
        
        for(int j = 0; j < hiddenNodes; j++) if(i < windowSize - 1) memcpy((*windowLayers)[2 * i + 1]->weights[j], (*myLayer)->weights[j], sizeof(float) * (*myLayer)->numPrevNodes);
        memcpy((*windowLayers)[2 * i + 1]->biases, (*myLayer)->biases, sizeof(float) * hiddenNodes);
        (*windowLayers)[(2 * i)]->layerType = 't';
        (*windowLayers)[(2 * i) + 1]->layerType = 'r';
    }
    
    return;

error1:
    hakai_context_window(windowLayers, windowSize);
}

void load_context_window(layer** windowLayers, float* inputs, int windowSize)
{
    if(windowSize < 1) return;

    int numInputs = windowLayers[0]->numNodes;
    int numHiddenNodes = windowLayers[1]->numNodes;

    for(int i = 0; i < numHiddenNodes; i++) memcpy(windowLayers[(2 * windowSize) + 1]->weights[i], windowLayers[1]->weights[i], sizeof(float) * (windowLayers[1]->numPrevNodes - numHiddenNodes));
    memcpy(windowLayers[(2 * windowSize) + 1]->biases, windowLayers[1]->biases, sizeof(float) * numHiddenNodes);

    for(int i = windowSize - 1; i > 1; i--)
    {
        memcpy(windowLayers[(2 * i)]->outputs, windowLayers[(2 * i) - 2]->outputs, sizeof(float) * numInputs);
        
        for(int j = 0; j < numHiddenNodes; j++) if(i < windowSize) memcpy(windowLayers[2 * i + 1]->weights[j], windowLayers[1]->weights[j], sizeof(float) * windowLayers[0]->numPrevNodes);
        memcpy(windowLayers[2 * i + 1]->biases, windowLayers[1]->biases, sizeof(float) * numHiddenNodes);
    }
    memcpy(windowLayers[2]->outputs, windowLayers[0]->outputs, sizeof(float) * numInputs);
    memcpy(windowLayers[0]->outputs, inputs, numInputs * sizeof(float));
}

void sgd_backprop_through_time(layer** myLayer, model** myModel, int timeStep)
{ // start at output layer and calculate backerrors for each previous layer
    if((*myLayer)->switchVar == '2') return;

    (*myLayer)->switchVar = '2';
    
    // backErrorsForOutputLayer = lossDerivative · activationFunctionDerivative(preActivations) - for output layer
    if((*myLayer)->layerType == 'o') for(int i = 0; i < (*myLayer)->numNodes; i++) (*myLayer)->backErrors[i] = -1 * loss_derivative((*myModel)->targets[i], (*myLayer)->outputs[i], (*myModel)) * activation_derivative((*myLayer)->preActivations[i], (*myLayer)->activationFunction, *myLayer, i);
    
    // backErrorsForPreviousLayers += (thisLayersBackErrors)(thisLayersWeightMatrixWithRespectToCurrentPreviousLayer) · activationFunctionDerivative(previousLayers)
    int prevsTraversed = 0;
    for(int i = 0; i < (*myLayer)->numPrevLayers; i++)
    {
        if((*(*myLayer)->prevLayers[i])->layerType == 'i') continue;
        for(int j = 0; j < (*(*myLayer)->prevLayers[i])->numNodes; j++) for(int k = 0; k < (*myLayer)->numNodes; k++) (*(*myLayer)->prevLayers[i])->backErrors[j] += (*myLayer)->backErrors[k] * (*myLayer)->weights[k][prevsTraversed + j] * activation_derivative((*(*myLayer)->prevLayers[i])->preActivations[j], (*myLayer)->activationFunction, *myLayer, i);
        prevsTraversed += (*(*myLayer)->prevLayers[i])->numNodes;
    }

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++)
    {
        if((*(*myLayer)->prevLayers[i])->numPrevLayers != 0) sgd_backprop_through_time((*myLayer)->prevLayers[i], myModel, timeStep);
    }
    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) if((*(*myLayer)->prevLayers[i])->numPrevLayers != 0) sgd_backprop((*myLayer)->prevLayers[i], myModel);
    // calculate backErrors for previous layers' previous layers according to already established layers' backErrors - All roads spring forth from Rome
}