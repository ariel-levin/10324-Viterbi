/******************************************
*******************************************
***		Ariel Levin						***
***		ariel.lvn89@gmail.com			***
***		http://about.me/ariel.levin		***
*******************************************
******************************************/

#include "global.h"


float** allocateFloatMatrix(int rows, int cols);

STATE** allocateStateMatrix(int rows, int cols);

void generateMatrix(float *mat[], int rows, int cols);

void printMatrix(float *mat[], int rows, int cols);

void printMatrix(STATE *mat[], int rows, int cols);

void freeMatrix(float *mat[], int rows);

void freeMatrix(STATE *mat[], int rows);

void normalizeMatrix(float *mat[], int rows, int cols);

void logMatrixValues(float *mat[], int rows, int cols);

