#include <stdio.h>
#include "../nn_include/nn.h"

int main()
{
    layer** layerArray = (layer**)NULL; // Necessary so model creator retains ownership of all allocated resources - array of layer struct allocation pointers
    model *myModel = load_model("testModel1.cml", &layerArray);
    if(myModel == NULL)
    {
        printf("Memory allocation failed at model\n");
        goto error1;
    }

    printf("Model creation successful\n\n");

    float values[] = {0.05, 0.10, 0.15};
    memcpy((*myModel->inLayers[0])->outputs, (float[]){0.05, 0.10, 0.15}, 3*sizeof(float));
    memcpy(myModel->targets, (float[]){0.905405}, sizeof(float));

    forward_out(myModel->outLayer);
    
    printf("\nModel output is: %f\nTarget is : %f\n", (*myModel->outLayer)->outputs[0], myModel->targets[0]);

    hakai_model(&myModel);

    free(layerArray);
    layerArray = NULL;
    printf("Test Successful\n");
    return 0;


error1:
    exit(EXIT_FAILURE);
}