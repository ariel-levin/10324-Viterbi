/******************************************
*******************************************
***		Ariel Levin						***
***		ariel.lvn89@gmail.com			***
***		http://about.me/ariel.levin		***
*******************************************
******************************************/

#include "global.h"


/* allocate space and returns doubles matrix with number of rows and cols received */
float** allocateFloatMatrix(int rows, int cols)
{
	float **mat;
	mat = (float**)calloc(rows, sizeof(float*));
	for (int i = 0; i < rows; i++)
	{
		mat[i] = (float*)calloc(cols, sizeof(float));
	}
	return mat;
}

/* allocate space and returns states matrix with number of rows and cols received */
STATE** allocateStateMatrix(int rows, int cols)
{
	STATE **mat;
	mat = (STATE**)malloc(rows * sizeof(STATE*));
	for (int i = 0; i < rows; i++)
	{
		mat[i] = (STATE*)malloc(cols * sizeof(STATE));
	}
	return mat;
}

/* generate random matrix values between 0 and 1, with number of rows and cols received */
void generateMatrix(float *mat[], int rows, int cols)
{
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			mat[i][j] = (float)rand() / RAND_MAX;
		}
	}
}

/* prints matrix with number of rows and cols received */
void printMatrix(float *mat[], int rows, int cols)
{
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			printf("%12.4f", mat[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

/* prints matrix with number of rows and cols received */
void printMatrix(STATE *mat[], int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			printf("%12.4f ", mat[i][j].prob);
		}
		printf("\n");
	}
	printf("\n");
}

/* free matrix from memory with number of rows received */
void freeMatrix(float *mat[], int rows)
{
	int i;
	for (i = 0; i < rows; i++)
	{
		free(mat[i]);
	}
	free(mat);
}

/* free matrix from memory with number of rows received */
void freeMatrix(STATE *mat[], int rows)
{
	int i;
	for (i = 0; i < rows; i++)
	{
		free(mat[i]);
	}
	free(mat);
}

void normalizeMatrix(float *mat[], int rows, int cols)
{
	float sum;

	for (int i = 0; i < rows; i++)
	{
		sum = 0;
		for (int j = 0; j < cols; j++)
		{
			sum += mat[i][j];
		}
#pragma omp parallel for
		for (int j = 0; j < cols; j++)
		{
			mat[i][j] = mat[i][j] / sum;
		}
	}
}

/* commit log function to every value on the matrix with OpenMP */
void logMatrixValues(float *mat[], int rows, int cols)
{
#pragma omp parallel for
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			mat[i][j] = log(mat[i][j]);
		}
	}
}

