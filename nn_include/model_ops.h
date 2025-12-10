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

model* construct_model(layer** inLayers, layer* outLayer,int numLayers, int numInLayers, float learning_rate)
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

int save_model(model* saveModel, const char* modelFileName)
{
    layer** layerList = (layer **)calloc(saveModel->numLayers, sizeof(layer*));
    if(layerList == NULL) goto error1;

    traverse_model_fill_layer_list(saveModel->outLayer, layerList);

    FILE *modFile = fopen(modelFileName, "w");
    if(modFile == NULL) goto error2;

    for(int i = 0; i < saveModel->numLayers; i++)
    {
        int offset = 6;
        int lineLength = 6;
        if(layerList[i]->numPrevLayers != 0)
        {
            offset += 1;
            lineLength += layerList[i]->numPrevLayers;
            lineLength += 16 * (layerList[i]->numNodes * (layerList[i]->numPrevNodes + 1));
            lineLength += 2;
        }

        char *line = (char *)malloc(lineLength);
        if(line == NULL) goto error3;

        if(layerList[i]->numPrevLayers == 0)
        {
            line[0] = (char)(layerList[i]->layerID + 33); // not likely to be more than 93 so we can just encode with the printable ascii value
            line[1] = '0';
            line[2] = (char)(layerList[i]->numNodes + 33); // would need to be adjusted for larger applications like chatgpt which can have just under 13k nodes in a layer
            line[3] = (char)(layerList[i]->numNextLayers + 33);
            line[4] = layerList[i]->activationFunction;
            line[5] = '\0';
            
            fputs(line, modFile);
            fputs("\n", modFile);
            continue;
        }
        else if(layerList[i]->numNextLayers == 0) 
        {
            line[1] = '2'; // Only reason for separate else if and else branches is differentiating the outlayer with hidden layers
        }
        else
        {
            line[1] = '1';
        }

        char fltBuff[16];

        line[0] = (char)(layerList[i]->layerID + 33); // not likely to be more than 255 so we can just encode with the ascii value
        line[2] = (char)(layerList[i]->numNodes + 33);
        line[3] = (char)(layerList[i]->numNextLayers + 33);
        line[4] = (char)(layerList[i]->numPrevNodes + 33);
        line[5] = (char)(layerList[i]->numPrevLayers + 33);
        line[6] = layerList[i]->activationFunction;

        for(int j = 0; j < layerList[i]->numPrevLayers; j++)
        {
            line[offset] = (char)(layerList[i]->prevLayers[j]->layerID + 33);
            offset += 1;
        }

        for(int j = 0; j < layerList[i]->numNodes; j++)
        {
            for(int k = 0; k < layerList[i]->numPrevNodes; k++)
            {
                snprintf(fltBuff, 16UL, "%.14f", layerList[i]->weights[j][k]);
                for(int l = 0; l < 15; l++) line[offset + l] = fltBuff[l];
                offset += 15;
            }
        }

        for(int j = 0; j < layerList[i]->numNodes; j++)
        {
            snprintf((char*)fltBuff, 16UL, "%.14f", layerList[i]->biases[j]);
            for(int l = 0; l < 16; l++) line[offset + l] = fltBuff[l];
            offset += 16;
        }

        line[lineLength - 1] = '\0';
        
        fputs(line, modFile);
        fputs("\n", modFile);

        free(line);
    }

    fclose(modFile);
    free(layerList);
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

model* load_model(const char* filename);




#endif