#include <stdio.h>
#include "./nn_include/model.h"

int main()
{
    struct model *myModel = construct_model(3, 1, 0.01f);
    if(myModel == NULL)
    {
        printf("Memory allocation failed at model\n");
        goto error1;
    }
    

    struct layer* layer0 = make_input_layer(3, 1, 0);
    if(layer0 == NULL)
    {
        printf("Memory allocation failed at layer0\n");
        goto error2;
    }
    myModel->layer_ids[layer0->layer_id] = 1;
    
    struct layer* layer1_in[] = {layer0};
    struct layer* layer1 = make_dense_layer(layer1_in, 3, 1, 1, 1, 'f');
    if(layer1 == NULL)
    {
        printf("Memory allocation failed at layer1\n");
        goto error3;
    }
    myModel->layer_ids[layer1->layer_id] = 1;

    struct layer* layer2_in[] = {layer1};
    struct layer* layer2 = make_output_layer(layer2_in, 2, 1, 2);
    if(layer1 == NULL)
    {
        printf("Memory allocation failed at layer0\n");
        goto error4;
    }
    myModel->layer_ids[layer2->layer_id] = 1;

    myModel->inLayers[0] = layer0;
    myModel->outLayer = layer2;

    printf("Model creation successful\n");

    // float weights
    // load_layer()

    hakai_model(myModel);
    printf("Test Successful\n");
    return 0;

error4:
    hakai_layer(layer2, myModel);
    free(layer2);
    layer2 = NULL;
error3:
    hakai_layer(layer1, myModel);
    free(layer1);
    layer1 = NULL;
error2:
    hakai_model(myModel);
    free(myModel);
    myModel = NULL;
error1:
    exit(EXIT_FAILURE);
}