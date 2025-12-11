#ifndef NN_MODEL_OPS
#define NN_MODEL_OPS

#include <stdio.h>
#include "model.h"
#include "nn_math.h"

int assign_layer_ids(layer* myLayer, int currID)
{
    //Post order traversal so the layers can be readily identified before any of their dependencies
    int myID = currID;
    if(myLayer->layerID != -1) return currID;

    for (int i = 0; i < myLayer->numPrevLayers; i++)
    {
        myID = assign_layer_ids(myLayer->prevLayers[i], myID);
    }

    myLayer->layerID = myID;
    return myID + 1;
    
}

model* construct_model(layer** inLayers, layer* outLayer, int numLayers, int numInLayers, float learning_rate)
{
    model *myModel = (model*)malloc(sizeof(model));
    if(myModel == NULL) return NULL;

    myModel->layer_refs = (layer **)calloc(numLayers, sizeof(layer*));
    if(myModel->layer_refs == NULL) goto error2;
    for(int i = 0; i < numLayers; i++) myModel->layer_refs[i] = NULL;

    myModel->inLayers = inLayers;
    myModel->outLayer = outLayer;
    myModel->numLayers = numLayers;
    myModel->learning_rate = learning_rate;
    myModel->numInLayers = numInLayers;

    myModel->targets = (float *)calloc(myModel->outLayer->numNodes, sizeof(float));
    if(myModel->targets == NULL) goto error3;

    assign_layer_ids(outLayer, 0);

    return myModel;

error3:
    free(myModel->layer_refs);
    myModel->layer_refs = NULL;
error2:
    free(myModel->inLayers);
    myModel->inLayers = NULL;
error1:
    free(myModel);
    myModel = NULL;

    return NULL;
}

// Destroy an individual layer after operations are concluded but a model hasn't been built yet
void hakai_layer_mfree(layer* myLayer)
{    
    free(myLayer->outputs);
    myLayer->outputs = NULL;

    free(myLayer->biases);
    myLayer->biases = NULL;

    if(myLayer->activationFunction == 'i') goto free;
    
    free(myLayer->backErrors);
    myLayer->backErrors = NULL;

    free(myLayer->prevLayers);
    myLayer->prevLayers = NULL;

    free(myLayer->preActivations);
    myLayer->preActivations = NULL;

    hakai_matrix(myLayer->weights, myLayer->numNodes);
    
    // if(myLayer->numNextLayers == 0) return;

    // free(myLayer->nextLayers);
    // myLayer->nextLayers = NULL;

free:
    free(myLayer);
    myLayer = NULL;
}

void hakai_model_mfree(model* myModel)
{
    free(myModel->layer_refs);
    myModel->layer_refs = NULL;

    free(myModel->targets);
    myModel->targets = NULL;

    free(myModel);
    myModel = NULL;
}

// Destroy an individual layer after operations are concluded
void hakai_layer(layer* myLayer, model* myModel)
{
    if(myLayer == NULL) return;
    if(myModel->layer_refs[myLayer->layerID] != NULL) return;

    free(myLayer->backErrors);
    myLayer->backErrors = NULL;

    free(myLayer->preActivations);
    myLayer->preActivations = NULL;

    free(myLayer->outputs);
    myLayer->outputs = NULL;

    free(myLayer->biases);
    myLayer->biases = NULL;

    if(myLayer->activationFunction != 'i') hakai_matrix(myLayer->weights, myLayer->numNodes);

    // free(myLayer->nextLayers);
    // myLayer->nextLayers = NULL;
    
    myModel->layer_refs[myLayer->layerID] = myLayer;
}


// Enter this function with the outArray of the model and let it do its thing
// One big advantage of the doubly linked list structure is being able to exploit the convergence of the model on the output layer
void clear_model(layer** layerArr, model* myModel, int layerNums)
{
    if(layerArr == NULL) return;

    for(int i = 0; i < layerNums; i++)
    {
        if(myModel->layer_refs[layerArr[i]->layerID] != NULL) continue;
        
        
        //myModel->layer_refs[layerArr[i]->layerID] = layerArr[i];
        
        clear_model(layerArr[i]->prevLayers, myModel, layerArr[i]->numPrevLayers);
        hakai_layer(layerArr[i], myModel);
        
        layerArr[i] = NULL;
    }
    
    free(layerArr);
    layerArr = NULL;
}


void hakai_model(model* myModel)
{
    layer **outArr = (layer**)malloc(sizeof(layer*));
    if(outArr == NULL) return;
    
    outArr[0] = myModel->outLayer;
    clear_model(outArr, myModel, 1);

    for(int i = 0; i < myModel->numLayers; i++)
    {
        free(myModel->layer_refs[i]);
        myModel->layer_refs[i] = NULL;
    }

    free(myModel->layer_refs);
    myModel->layer_refs = NULL;

    free(myModel->targets);
    myModel->targets = NULL;

    free(myModel);
    myModel = NULL;
}

//  Gets an output from the target layer
void forward_out(layer* myLayer)
{
    if(myLayer->switchVar == '1') return;

    myLayer->switchVar = '1';

    if(myLayer->numPrevLayers != 0)
    {
        int numPrevsTraversed = 0;
        
        for(int i = 0; i < myLayer->numPrevLayers; i++) forward_out(myLayer->prevLayers[i]);

        for(int i = 0; i < myLayer->numNodes; i++) 
        {
            for(int j = 0; j < myLayer->numPrevLayers; j++)
            {
                for(int k = 0; k < myLayer->prevLayers[j]->numNodes; k++) myLayer->preActivations[i] += myLayer->prevLayers[j]->outputs[k] * myLayer->weights[i][numPrevsTraversed + k];
                numPrevsTraversed += myLayer->prevLayers[j]->numNodes;
            }
            myLayer->preActivations[i] += myLayer->biases[i];
            numPrevsTraversed = 0;
        } 
    }

    switch(myLayer->activationFunction)
    {
        case 'r':
            for(int i = 0; i < myLayer->numNodes; i++) myLayer->outputs[i] = leaky_relu(myLayer->preActivations[i]);
            break;
        case 't':
            for(int i = 0; i < myLayer->numNodes; i++) myLayer->outputs[i] = tanh(myLayer->preActivations[i]);
            break;
        default:
            break;
    }
}

// Run on each output layer and then apply grads before clearing the layer backerrors - All roads spring forth from Rome algorithm
// We pass the backerrors to each previous layer to calculate grads later
// Backerrors can be accumulated from multiple successor layers to calculate grads due to matrix distributivity
void sgd_backprop(layer* myLayer, model* myModel)
{ // start at output layer and calculate backerrors for each previous layer
    if(myLayer->switchVar == '2') return;

    myLayer->switchVar = '2';

    if(myLayer->numNextLayers == 0)
    {
        // backErrorsForOutputLayer = mseLossDerivative · activationFunctionDerivative(preActivations) - for output layer
        for(int i = 0; i < myLayer->numNodes; i++) myLayer->backErrors[i] = -1 * mse_loss_derivative_func(myModel->targets[i], myLayer->outputs[i]) * tanh_derivative(myLayer->preActivations[i]);
    }
    
    // backErrorsForPreviousLayers += (thisLayersBackErrors)(thisLayersWeightMatrixWithRespectToCurrentPreviousLayer) · activationFunctionDerivative(previousLayers)
    int prevsTraversed = 0;
    for(int i = 0; i < myLayer->numPrevLayers; i++)
    {
        if(myLayer->prevLayers[i]->numPrevLayers == 0) continue;
        for(int j = 0; j < myLayer->prevLayers[i]->numNodes; j++)
        {
            for(int k = 0; k < myLayer->numNodes; k++) myLayer->prevLayers[i]->backErrors[j] += myLayer->backErrors[k] * myLayer->weights[k][prevsTraversed + j] * leaky_relu_derivative(myLayer->prevLayers[i]->preActivations[j]);                
        }
        prevsTraversed += myLayer->prevLayers[i]->numNodes;
    }

    for(int i = 0; i < myLayer->numPrevLayers; i++) if(myLayer->prevLayers[i]->numPrevLayers != 0) sgd_backprop(myLayer->prevLayers[i], myModel);
    // calculate backErrors for previous layers' previous layers according to already established layers' backErrors - All roads spring forth from Rome
}

// Another all roads spring forth from Rome approach - go to the convergence point of the model and use it as the root of the undirected, cyclic graph that is this model
void calculate_and_apply_grads(layer* myLayer, float learningRate)
{
    if(myLayer->switchVar == '3') return;

    myLayer->switchVar = '3';

    if(myLayer->numPrevLayers == 0) return;

    for(int i = 0; i < myLayer->numPrevLayers; i++) calculate_and_apply_grads(myLayer->prevLayers[i], learningRate);

    for(int i = 0; i < myLayer->numNodes; i++) myLayer->biases[i] -= learningRate * myLayer->backErrors[i];

    int prevsTraversed = 0;

    // calculate gradients from backerrors and activations for each layer and apply them to the weights_mm256_mul_ps(vector_a, scalar_vector);
    for(int i = 0; i < myLayer->numPrevLayers; i ++)
    {
        for(int j = 0; j < myLayer->prevLayers[i]->numNodes; j++)
        {
            for(int k = 0; k < myLayer->numNodes; k++) myLayer->weights[k][j + prevsTraversed] -= learningRate * myLayer->prevLayers[i]->outputs[j] * myLayer->backErrors[k];
        }
        prevsTraversed += myLayer->prevLayers[i]->numNodes;
    }
}

// For clearing the backErrors once no longer needed, and to also prime for next forward and backward pass
// Use by passing the output layer of the model into the function 
void zero_everything(layer* myLayer)
{
    if(myLayer->switchVar == '0') return;

    myLayer->switchVar = '0';

    if(myLayer->numPrevLayers != 0)
    {
        for(int i = 0; i < myLayer->numPrevLayers; i++) zero_everything(myLayer->prevLayers[i]);
    }
    else return;

    memset(myLayer->backErrors, 0.0f, myLayer->numNodes * sizeof(float));
    memset(myLayer->preActivations, 0.0f, myLayer->numNodes * sizeof(float));
    memset(myLayer->outputs, 0.0f, myLayer->numNodes * sizeof(float));
}

void traverse_model_fill_layer_list(layer* myLayer, layer** layerList)
{
    if(layerList[myLayer->layerID] != NULL) return;
    for(int i = 0; i < myLayer->numPrevLayers; i++) traverse_model_fill_layer_list(myLayer->prevLayers[i], layerList); // if input layer then the for loop won't even initiate
    layerList[myLayer->layerID] = myLayer;
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

int save_model(model* saveModel, const char* modelFileName)
{
    layer** layerList = (layer **)calloc(saveModel->numLayers, sizeof(layer*));
    if(layerList == NULL) goto error1;

    traverse_model_fill_layer_list(saveModel->outLayer, layerList);

    FILE *modFile = fopen(modelFileName, "w");
    if(modFile == NULL) goto error2;

    int offset = 0;
    int lineLength = 41;
    char bitBuff[33] = "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
    char fltBuff[20] = "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

    lineLength += 16 * saveModel->numInLayers;

    char *line = (char *)calloc(lineLength, sizeof(char));
    if(line == NULL) goto error3;
    
    int2bin(saveModel->numLayers, 16, bitBuff);
    for(int i = 0; i < 16; i++) line[offset + i] = bitBuff[i]; // big endian representation
    offset += 16;

    int2bin(saveModel->numInLayers, 16, bitBuff);
    for(int i = 0; i < 16; i++) line[offset + i] = bitBuff[i]; // big endian representation
    offset += 16;

    for(int i = 0; i < saveModel->numInLayers; i++)
    {
        int2bin(saveModel->inLayers[i]->layerID, 16, bitBuff);
        for(int j = 0; j < 16; j++) line[offset + j] = bitBuff[j]; // big endian representation
        offset += 16;
    }

    snprintf(fltBuff, 9UL, "%.8f", saveModel->learning_rate);
    for(int l = 0; l < 8; l++) line[offset + l] = fltBuff[l];
    offset += 8;

    line[lineLength - 1] = '\0';

    fputs(line, modFile);
    fputs("\n", modFile);

    free(line);
    line = NULL;

    for(int i = 0; i < saveModel->numLayers; i++)
    {
        lineLength = 29;
        if(layerList[i]->numPrevLayers != 0)
        {
            lineLength += 16 * layerList[i]->numPrevLayers;
            lineLength += 16 * (layerList[i]->numNodes * (layerList[i]->numPrevNodes + 1));
            lineLength += 65;
        }
        offset = 0;

        line = (char *)calloc(lineLength, sizeof(char));
        if(line == NULL) goto error3;

        if(layerList[i]->numNextLayers == 0) 
        {
            line[0] = '2'; // Only reason for separate else if and else branches is differentiating the outlayer with hidden layers
        }
        else
        {
            line[0] = '1';
        }

        offset += 1;

        int2bin(layerList[i]->layerID, 11, bitBuff);
        for(int j = 0; j < 11; j++) line[offset + j] = bitBuff[j]; // big endian representation
        offset += 11;

        int2bin(layerList[i]->numNodes, 16, bitBuff);
        for(int j = 0; j < 16; j++) line[offset+ j] = bitBuff[j];
        offset += 16;

        if(layerList[i]->numPrevLayers == 0)
        {
            line[0] = '0';

            line[offset] = '\0';
            
            fputs(line, modFile);
            fputs("\n", modFile);
            
            free(line);
            line = NULL;
            continue;
        }

        int2bin(layerList[i]->numNextLayers, 16, bitBuff);
        for(int j = 0; j < 16; j++) line[offset+ j] = bitBuff[j];
        offset += 16;

        int2bin(layerList[i]->numPrevNodes, 32, bitBuff);
        for(int j = 0; j < 32; j++) line[offset+ j] = bitBuff[j];
        offset += 32;
        
        int2bin(layerList[i]->numPrevLayers, 16, bitBuff);
        for(int j = 0; j < 16; j++) line[offset+ j] = bitBuff[j];
        offset += 16;
        
        line[offset] = layerList[i]->activationFunction;
        offset += 1;

        for(int j = 0; j < layerList[i]->numPrevLayers; j++)
        {
            int2bin(layerList[i]->prevLayers[j]->layerID, 16, bitBuff);
            for(int k = 0; k < 16; k++) line[offset + k] = bitBuff[k];
            offset += 16;
        }

        for(int j = 0; j < layerList[i]->numNodes; j++)
        {
            for(int k = 0; k < layerList[i]->numPrevNodes; k++)
            {
                if(layerList[i]->weights[j][k] < 0) snprintf(fltBuff, 17UL, "%.15f", layerList[i]->weights[j][k]);
                else snprintf(fltBuff, 17UL, "%.16f", layerList[i]->weights[j][k]);
                
                for(int l = 0; l < 16; l++) line[offset + l] = fltBuff[l];
                offset += 16;
            }
        }

        for(int j = 0; j < layerList[i]->numNodes; j++)
        {
            if(layerList[i]->biases[j] < 0) snprintf((char*)fltBuff, 17UL, "%.15f", layerList[i]->biases[j]);
            else snprintf((char*)fltBuff, 17UL, "%.15f", layerList[i]->biases[j]);
            
            for(int l = 0; l < 16; l++) line[offset + l] = fltBuff[l];
            offset += 16;
        }

        line[lineLength - 1] = '\0';
        
        fputs(line, modFile);
        fputs("\n", modFile);

        free(line);
        line = NULL;
    }

    fclose(modFile);
    free(layerList);
    free(line);
    layerList = NULL;
    return 0;

error3:
    fclose(modFile);
error2:
    free(layerList);
    layerList = NULL;
error1:
    return -1;
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
        currCount <<= 2;
    }

    return retVal;
}

model* load_model(const char* filename);




#endif