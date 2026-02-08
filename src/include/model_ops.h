#pragma once

#include "model_construct.h"
#include "activation.h"
#include "loss.h"
#include "helper_funcs.h"
#include <stdio.h>
#include <pthread.h>

//  Gets an output from the target layer, is essentially also a inference function
void forward_out(layer** myLayer)
{
    if((*myLayer)->switchVar == '1' || (*myLayer)->layerType == 'w') return; // If layer has been traversed or if layer is a window layer for a context window

    (*myLayer)->switchVar = '1';

    if((*myLayer)->numPrevLayers != 0)
    {
        int numPrevsTraversed = 0;
        
        for(int i = 0; i < (*myLayer)->numPrevLayers; i++) forward_out((*myLayer)->prevLayers[i]);

        // preActivation[i] = SUM_OVER_J(prevNodeOutputs[j] * weights[i][j]) 
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

        // In case of softmax activation, we do a function on the layer instead of point-wise on the outputs
        if((*myLayer)->activationFunction == 'x')
        {
            memcpy((*myLayer)->outputs, (*myLayer)->preActivations, sizeof(float) * (*myLayer)->numNodes);
            softmax(myLayer);
            return;
        }
        else if((*myLayer)->activationFunction == 'f')
        {
            memcpy((*myLayer)->outputs, (*myLayer)->preActivations, sizeof(float) * (*myLayer)->numNodes);
            fast_softmax(myLayer);
            return;
        }
        
        // outputs[i] = activation_function(preActivations[i])
        for(int i = 0; i < (*myLayer)->numNodes; i++) (*myLayer)->outputs[i] = activation_function((*myLayer)->preActivations[i], (*myLayer)->activationFunction);
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
    
    // backErrorsForPreviousLayers[j] = SUM_OVER_I((thisLayersBackErrors[i])(thisLayersWeightMatrix[i][j]) · activationFunctionDerivative(previousLayersPreActivation[j])) - where j is considered to be a traversal of all previous 'J' layers' 'K' values as one vector
    // e.g. J = 3 prev layers with K = 5 nodes each are considered as one prev layer with J = 15 nodes in this formulation
    int prevsTraversed = 0;
    for(int i = 0; i < (*myLayer)->numPrevLayers; i++)
    {
        if((*(*myLayer)->prevLayers[i])->layerType == 'i' || (*(*myLayer)->prevLayers[i])->layerType == 't') continue;
        for(int j = 0; j < (*(*myLayer)->prevLayers[i])->numNodes; j++) for(int k = 0; k < (*myLayer)->numNodes; k++) (*(*myLayer)->prevLayers[i])->backErrors[j] += (*myLayer)->backErrors[k] * (*myLayer)->weights[k][prevsTraversed + j] * activation_derivative((*(*myLayer)->prevLayers[i])->preActivations[j], (*(*myLayer)->prevLayers[i])->activationFunction, *myLayer, i);
        prevsTraversed += (*(*myLayer)->prevLayers[i])->numNodes;
    }

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) if((*(*myLayer)->prevLayers[i])->numPrevLayers != 0 && (*(*myLayer)->prevLayers[i])->layerType != 'w') sgd_backprop((*myLayer)->prevLayers[i], myModel);
    // calculate backErrors for previous layers' previous layers according to already established layers' backErrors - All roads spring forth from Rome
}

// Another all roads spring forth from Rome approach - go to the convergence point of the model(output layer) and use it as the root this model graph
void calculate_and_apply_grads(layer** myLayer, float learningRate)
{
    if((*myLayer)->switchVar == '3') return;

    (*myLayer)->switchVar = '3';

    if((*myLayer)->numPrevLayers == 0 || (*myLayer)->layerType == 'w') return;

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) calculate_and_apply_grads(((*myLayer)->prevLayers[i]), learningRate);

    // newBias[i] = oldBias[i] - (learningRate * backErrors[i])
    for(int i = 0; i < (*myLayer)->numNodes; i++) (*myLayer)->biases[i] -= learningRate * (*myLayer)->backErrors[i];

    int prevsTraversed = 0;

    // newWeights[i][j] = oldWeights[i][j] - (learningRate * (prevNodeOuts[j] * backError[i]))
    for(int i = 0; i < (*myLayer)->numNodes; i++)
    {
        for(int j = 0; j < (*myLayer)->numPrevLayers; j++)
        {
            for(int k = 0; k < (*(*myLayer)->prevLayers[j])->numNodes; k++) (*myLayer)->weights[i][k + prevsTraversed] -= learningRate * (*(*myLayer)->prevLayers[j])->outputs[k] * (*myLayer)->backErrors[i];
            
            prevsTraversed += (*(*myLayer)->prevLayers[j])->numNodes;
        }
        prevsTraversed = 0;
    }
}

// For clearing the backErrors once no longer needed, and to also prime for next forward and backward pass
// Use by passing the output layer of the model into the function 
void zero_everything(layer** myLayer)
{
    if((*myLayer)->switchVar == '0') return;

    (*myLayer)->switchVar = '0';

    if((*myLayer)->layerType == 'i') return;

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) zero_everything((*myLayer)->prevLayers[i]);

    memset((*myLayer)->backErrors, 0.0f, (*myLayer)->numNodes * sizeof(float));
    memset((*myLayer)->preActivations, 0.0f, (*myLayer)->numNodes * sizeof(float));
    memset((*myLayer)->outputs, 0.0f, (*myLayer)->numNodes * sizeof(float));
}

// Encode and save a model to file
int save_model(model** saveModel, char* modelFileName)
{
    FILE *modFile = NULL;
    char *line = NULL;
    int offset = 0;
    int lineLength = 50;
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

    snprintf(fltBuff, 19UL, "%.16f", (*saveModel)->learning_rate);
    for(int l = 0; l < 16; l++) line[offset + l] = fltBuff[l];
    offset += 16;

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
        if((*layerList[i])->layerType == 'h' || (*layerList[i])->layerType == 'o')
        {
            lineLength += 16 * (*layerList[i])->numPrevLayers;
            lineLength += 16 * ((*layerList[i])->numNodes * ((*layerList[i])->numPrevNodes + 1));
            lineLength += 65;
        }
        else if((*layerList[i])->layerType == 'w')
        {
            lineLength += 16 * (*layerList[i])->numPrevLayers;
            lineLength += 49;
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

        if((*layerList[i])->layerType == 'i')
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

        if((*layerList[i])->layerType == 'w')
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

        for(int j = 0; j < (*layerList[i])->numNodes; j++)
        {
            for(int k = 0; k < (*layerList[i])->numPrevNodes; k++)
            {
                if((*layerList[i])->weights[j][k] < 0) snprintf(fltBuff, 18UL, "%.15f", (*layerList[i])->weights[j][k]);
                else snprintf(fltBuff, 19UL, "%.16f", (*layerList[i])->weights[j][k]);
                
                for(int l = 0; l < 16; l++) line[offset + l] = fltBuff[l];
                offset += 16;
            }
        }

        for(int j = 0; j < (*layerList[i])->numNodes; j++)
        {
            if((*layerList[i])->biases[j] < 0) snprintf(fltBuff, 18UL, "%.15f", (*layerList[i])->biases[j]);
            else snprintf(fltBuff, 19UL, "%.16f", (*layerList[i])->biases[j]);
            
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

// Decode and load a saved model from a file generated by the save model function
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

    *modelLayers = (layer**)malloc(numLayers * sizeof(layer**));
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
        else if(layerType == 'r')
        {
            (*modelLayers)[layerID] = make_referential_layer(layerArr, numNodes, numPrevLayers, activationFunction, &(*modelLayers)[layerID]);
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

    layerArr = (layer***)calloc(numInLayers, sizeof(layer**));
    if(layerArr == NULL) goto error7;

    for(int i = 0; i < numInLayers; i++) layerArr[i] = &(*modelLayers)[inLayerIDs[i]];

    myModel = construct_model(layerArr, &(*modelLayers)[outLayerID], numLayers, numInLayers, learningRate, loss_fn);
    if(myModel == NULL) goto error8;

    free(inLayerIDs);
    inLayerIDs = NULL;
    free(layerArr);
    layerArr = NULL;

    return myModel;

error8:
    free(layerArr);
    layerArr = NULL;
error7:
    for(int i = 0; i < numLayers; i++)
    {
        hakai_layer_mfree(&(*modelLayers)[i]);
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

// Use to shift the values of a context window to the previous timestep before retraining for the next timestep
void shift_context_train(layer** thisLayer, layer** prevLayer)
{
    if((*prevLayer)->numPrevLayers == 2) shift_context_train(prevLayer, (*prevLayer)->prevLayers[1]);
    
    if((*thisLayer)->layerType == 'w')
    {
        int numInputs = (*(*thisLayer)->prevLayers[0])->numNodes;
        int numHiddenNodes = (*thisLayer)->numNodes;

        memcpy((*prevLayer)->outputs, (*thisLayer)->outputs, sizeof(float) * numHiddenNodes);
        memcpy((*prevLayer)->preActivations, (*thisLayer)->preActivations, sizeof(float) * numHiddenNodes);
        memcpy((*(*prevLayer)->prevLayers[0])->outputs, (*(*thisLayer)->prevLayers[0])->outputs, sizeof(float) * numInputs);
    }
    else
    {
        int numPrevsTraversed = 0;
        int numHiddenNodes = (*thisLayer)->numNodes;

        memcpy((*prevLayer)->outputs, (*thisLayer)->outputs, sizeof(float) * numHiddenNodes);
        memcpy((*prevLayer)->preActivations, (*thisLayer)->preActivations, sizeof(float) * numHiddenNodes);

        for(int i = 0; i < (*thisLayer)->numPrevLayers - 1; i++)
        {
            memcpy(&(*(*prevLayer)->prevLayers[0])->outputs[numPrevsTraversed], (*(*thisLayer)->prevLayers[i])->outputs , sizeof(float) * (*(*thisLayer)->prevLayers[i])->numNodes);
            numPrevsTraversed += (*(*thisLayer)->prevLayers[i])->numNodes;
        }
    }
}

void shift_context_infer(layer** thisLayer, layer** prevLayer)
{
    memcpy((*prevLayer)->outputs, (*thisLayer)->outputs, sizeof(float) * (*thisLayer)->numNodes);
}

void shift_model(layer** myLayer, char opType)
{
    if((*(*myLayer)->prevLayers[(*myLayer)->numPrevLayers - 1])->layerType == 'w')
    {
        if(opType == 't') shift_context_train(myLayer, (*myLayer)->prevLayers[(*myLayer)->numPrevLayers - 1]);
        else shift_context_infer(myLayer, (*myLayer)->prevLayers[(*myLayer)->numPrevLayers - 1]);
    }

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++)
    {
        if((*(*myLayer)->prevLayers[i])->numPrevLayers == 0 || (*(*myLayer)->prevLayers[i])->layerType == 'w') continue;
        shift_model((*myLayer)->prevLayers[i], opType);
    }
}

// Used to clear the base model of an RNN while preserving the values of the previous timesteps
void zero_base_model(layer** myLayer)
{
    if((*myLayer)->switchVar == 'b') return;

    (*myLayer)->switchVar = 'b';

    if((*myLayer)->layerType == 'i' || (*myLayer)->layerType == 'w') return;

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) zero_everything((*myLayer)->prevLayers[i]);

    memset((*myLayer)->backErrors, 0.0f, (*myLayer)->numNodes * sizeof(float));
    memset((*myLayer)->preActivations, 0.0f, (*myLayer)->numNodes * sizeof(float));
    memset((*myLayer)->outputs, 0.0f, (*myLayer)->numNodes * sizeof(float));
}

// Applies Back Propagation Through Time procedure for all context windows
void calculate_and_apply_grads_through_time(layer** myLayer, float learningRate)
{
    if((*myLayer)->switchVar == '3') return;
    
    (*myLayer)->switchVar = '3';

    if((*myLayer)->numPrevLayers == 0 || (*myLayer)->layerType == 'w') return;

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) calculate_and_apply_grads(((*myLayer)->prevLayers[i]), learningRate);

    // newBias[i] = oldBias[i] - (learningRate * backErrors[i])
    for(int i = 0; i < (*myLayer)->numNodes; i++) (*myLayer)->biases[i] -= learningRate * (*myLayer)->backErrors[i];
    if((*(*myLayer)->prevLayers[(*myLayer)->numPrevLayers - 1])->layerType == 'w')
    {
        layer* currLayer = (*(*myLayer)->prevLayers[(*myLayer)->numPrevLayers - 1]);

        while(currLayer->numPrevLayers == 2)
        {
            for(int i = 0; i < currLayer->numNodes; i++) (*myLayer)->biases[i] -= learningRate * currLayer->backErrors[i];
            currLayer = *currLayer->prevLayers[1];
        }
        
        for(int i = 0; i < currLayer->numNodes; i++) (*myLayer)->biases[i] -= learningRate * currLayer->backErrors[i]; // For the last layer in the window with only one prevLayer
    }

    int prevsTraversed = 0;

    // newWeights[i][j] = oldWeights[i][j] - (learningRate * (prevNodeOuts[j] * backError[i]))
    for(int i = 0; i < (*myLayer)->numNodes; i++)
    {
        for(int j = 0; j < (*myLayer)->numPrevLayers; j++)
        {
            for(int k = 0; k < (*(*myLayer)->prevLayers[j])->numNodes; k++) (*myLayer)->weights[i][k + prevsTraversed] -= learningRate * (*(*myLayer)->prevLayers[j])->outputs[k] * (*myLayer)->backErrors[i];          
            prevsTraversed += (*(*myLayer)->prevLayers[j])->numNodes;
        }

        if((*(*myLayer)->prevLayers[(*myLayer)->numPrevLayers - 1])->layerType == 'w')
        {
            layer* currLayer = (*(*myLayer)->prevLayers[(*myLayer)->numPrevLayers - 1]);

            while(currLayer->numPrevLayers == 2)
            {
                for(int j = 0; j < currLayer->numNodes; j++) (*myLayer)->weights[i][j] -= learningRate * (*currLayer->prevLayers[0])->outputs[j] * currLayer->backErrors[i];
                currLayer = *currLayer->prevLayers[1];
            }
            for(int j = 0; j < currLayer->numNodes; j++) (*myLayer)->weights[i][j] -= learningRate * (*currLayer->prevLayers[0])->outputs[j] * currLayer->backErrors[i];
        }
        prevsTraversed = 0;
    }
}


#if defined(__AVX__) || defined(__AVX2__)
//  Gets an output from the target layer, is essentially also a inference function
// Vectorized version of forward out, only really makes a difference on industrial grade models so it will be shelved for now
int _mm256_forward_out(layer** myLayer)
{
    if((*myLayer)->switchVar == '1') return 0;

    (*myLayer)->switchVar = '1';

    if((*myLayer)->numPrevLayers != 0 && (*myLayer)->layerType != 'w')
    {        
        for(int i = 0; i < (*myLayer)->numPrevLayers; i++) if(_mm256_forward_out((*myLayer)->prevLayers[i]) != 0) return -1;

        if(vectorized_forward_out_calc(myLayer) != 0) return -1;

        // In case of softmax activation, we do a function on the layer instead of point-wise on the outputs
        if((*myLayer)->activationFunction == 'x')
        {
            memcpy((*myLayer)->outputs, (*myLayer)->preActivations, sizeof(float) * (*myLayer)->numNodes);
            softmax(myLayer);
            return 0;
        }
        else if((*myLayer)->activationFunction == 'f')
        {
            memcpy((*myLayer)->outputs, (*myLayer)->preActivations, sizeof(float) * (*myLayer)->numNodes);
            fast_softmax(myLayer);
            return 0;
        }
        
        for(int i = 0; i < (*myLayer)->numNodes; i++) (*myLayer)->outputs[i] = activation_function((*myLayer)->preActivations[i], (*myLayer)->activationFunction);
    }

    return 0;
}

//Need to finish
void _mm256_sgd_backprop(layer** myLayer, model** myModel)
{ // start at output layer and calculate backerrors for each previous layer
    if((*myLayer)->switchVar == '2') return;

    (*myLayer)->switchVar = '2';
    
    vectorized_sgd_backprop_calc(myLayer, myModel);

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) if((*(*myLayer)->prevLayers[i])->numPrevLayers != 0 && (*(*myLayer)->prevLayers[i])->layerType != 'w') sgd_backprop((*myLayer)->prevLayers[i], myModel);
    // calculate backErrors for previous layers' previous layers according to already established layers' backErrors - All roads spring forth from Rome
}


int _mm256_calculate_and_apply_grads(layer** myLayer, float learningRate)
{
    if((*myLayer)->switchVar == '3') return 0;

    (*myLayer)->switchVar = '3';

    if((*myLayer)->numPrevLayers == 0) return 0;

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) if(_mm256_calculate_and_apply_grads(((*myLayer)->prevLayers[i]), learningRate) != 0) return -1;

    if(vectorized_calculate_and_apply_grads(myLayer, learningRate) != 0) return -1;

    return 0;
}

int _mm256_calculate_and_apply_grads_through_time(layer** myLayer, float learningRate)
{
    if((*myLayer)->switchVar == '3') return 0;

    (*myLayer)->switchVar = '3';

    if((*myLayer)->numPrevLayers == 0 || (*myLayer)->layerType == 'w') return 0;

    for(int i = 0; i < (*myLayer)->numPrevLayers; i++) if(_mm256_calculate_and_apply_grads_through_time(((*myLayer)->prevLayers[i]), learningRate) != 0) return -1;

    if(vectorized_calculate_and_apply_grads_through_time(myLayer, learningRate) != 0) return -1;

    return 0;
}

#endif

#if defined(__ARM_NEON)

void vforward_out(layer** myLayer)
{
    if((*myLayer)->switchVar == '1') return;

    (*myLayer)->switchVar = '1';

    if((*myLayer)->numPrevLayers != 0)
    {
        int numPrevsTraversed = 0;
        
        for(int i = 0; i < (*myLayer)->numPrevLayers; i++) vforward_out((*myLayer)->prevLayers[i]);

        vectorized_forward_out_calc(myLayer);

        // In case of softmax activation, we do a function on the layer instead of point-wise on the outputs
        if((*myLayer)->activationFunction == 'x')
        {
            memcpy((*myLayer)->outputs, (*myLayer)->preActivations, sizeof(float) * (*myLayer)->numNodes);
            softmax(myLayer);
            return;
        }
        else if((*myLayer)->activationFunction == 'f')
        {
            memcpy((*myLayer)->outputs, (*myLayer)->preActivations, sizeof(float) * (*myLayer)->numNodes);
            fast_softmax(myLayer);
            return;
        }
        
        for(int i = 0; i < (*myLayer)->numNodes; i++) (*myLayer)->outputs[i] = activation_function((*myLayer)->preActivations[i], (*myLayer)->activationFunction);
    }
}

#endif