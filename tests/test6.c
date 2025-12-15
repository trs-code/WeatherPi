#include <stdio.h>
#include "../nn_include/nn.h"

int main()
{
    char filename[] = "THPDiffs.csv";
    int numIns = 9;
    int numOuts = 1;
    int numSamples = 20;

    float** inArrays = (float**)NULL;
    float** outArrays = (float**)NULL; 

    read_csv(filename, numSamples, numIns, numOuts, &inArrays, &outArrays);

    for(int i = 0; i < numSamples; i++)
    {
        for(int j = 0; j < numIns; j++) printf("%f, ", inArrays[i][j]);

        printf("%f\n", outArrays[i][0]);
    }

    hakai_matrix(inArrays, numSamples);
    hakai_matrix(outArrays, numSamples);
    return 0;

}
