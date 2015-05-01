/******************************************
*******************************************
***		Ariel Levin						***
***		ariel.lvn89@gmail.com			***
***		http://about.me/ariel.levin		***
*******************************************
******************************************/

#include "global.h"


void outputPathToFile(int path[], int len, FILE *f);

bool loadArrayFromFile(float arr[], int size, char fpath[]);

bool loadMatrixFromFile(float *mat[], int rows, int cols, char fpath[], bool load_transposed);

void clearOutputFiles();

void fileOutputAllMaxStates(STATE *mat[], MAX_STATE *arr, int size);

