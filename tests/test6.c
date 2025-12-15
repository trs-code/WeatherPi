#include <stdio.h>
#include "../nn_include/nn.h"

int main()
{
    char filename[] = "THPDiffs.csv";
    int numIns = 9;
    int numOuts = 1;
    int numSamples = 21;

    float** inArrays = (float**)NULL;
    float** outArrays = (float**)NULL; 

    int success = read_csv(filename, numSamples, numIns, numOuts, &inArrays, &outArrays);
    printf("%d\n", success);
    return 0;

}
