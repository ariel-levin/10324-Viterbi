/******************************************
*******************************************
***		Ariel Levin						***
***		ariel.lvn89@gmail.com			***
***		http://about.me/ariel.levin		***
*******************************************
******************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <string>
#include <mpi.h>
#include <omp.h>
#include "iostream"


/***************************************************************************
****************************	 DEFINES		****************************
***************************************************************************/


#ifndef __MAIN_H
#define __MAIN_H


#define NUM_OF_STATES	1000			// number of states, default: 1000
#define NUM_OF_OBSRV	30000			// number of time slices, default: 30000


#define FILE_PATH		"C:\\Users\\Ariel\\workspace\\afeka\\2015a\\10324-parallel\\Project\\Project\\Files\\"


typedef struct STATE
{
	float	prob;
	int		parent;

} STATE;

typedef struct MAX_STATE
{
	int		obsrv;
	int		state_num;
	STATE	state;

} MAX_STATE;


#endif	//__MAIN_H



/***************************************************************************
****************************	 SETTINGS		****************************
***************************************************************************/


// initialize consts
static const bool	TEST_VALUES = false;					// small scale 3x5 testing values
static const bool	WITH_FILES = true;						// working with input and output files
static const bool	GENERATE_ZEROES = false;				// when using random values: generate zeroes in observations
static const int	MAX_ZERO_RANGE = NUM_OF_OBSRV / 2;		// how often to generate zeros (if using random values)
static const bool	NORMALIZE_TRANS = false;				// normalize transition matrix

// how to calc consts
static const bool	WITH_LOGS = true;						// working with logs
static const bool	WITH_CUDA = true;						// working with cuda
static const bool	PRINT_OBSRV = false;					// print observation progress during calculation
															// (may slow the process)



/***************************************************************************
***************************		DECLARATIONS	****************************
***************************************************************************/


void allocateSpace(float ***trans, float ***ab, float **obsrv, float **emission,
	int **max_states_idx, int mpi_proc_num);

void initializeValues(float **trans, float **ab, float *obsrv, float **cuda_emission, float **cuda_a, float **cuda_b);

float emissionFunc(float aj, float bj, float oi);

int getMaxStateIndex(STATE arr[], int range[2]);

void getRange(int range[], int rank, int mpi_proc_num);

void testValues(float *trans[], float *ab[], float obsrv[]);

int getNextIndexToCalc(float obsrv[], int current);

int getMaxIndexFromIndexArray(int max_states_idx[], int mpi_proc_num, STATE arr[]);

void addMaxToMaxArray(MAX_STATE *max_arr[], int *max_states_num, int max_idx, int obsrv, STATE arr[]);

MPI_Datatype createMpiStateType();

int calcEmission(float emission[], float *ab[], float obsrv, float *cuda_emission, float *cuda_a, float *cuda_b);

void viterbi(STATE current[], STATE next[], int state_calc_num, float *trans[], float emission[], int range[2]);

void initStateColumn(STATE *mat[], int i);

int getPathLength(MAX_STATE *arr, int i);

int* getPath(MAX_STATE *arr, int i, STATE *mat[]);

int freeAll(int rank, float *trans[], float *ab[], float obsrv[], float emission[],
	int max_states_idx[], STATE *mat[], MAX_STATE max_states_arr[], STATE current[], STATE next[],
	float *cuda_emission, float *cuda_a, float *cuda_b);

void printPath(int path[], int len);

void printAllMaxStates(STATE *mat[], MAX_STATE *arr, int size);

void printArray(STATE arr[], int size);

void printArray(float arr[], int size);

void printArray(int arr[], int size);

void generateArray(float arr[], int size);

void copyArray(int a[], int b[], int start, int end);

float** allocateFloatMatrix(int rows, int cols);

STATE** allocateStateMatrix(int rows, int cols);

void generateMatrix(float *mat[], int rows, int cols);

void printMatrix(float *mat[], int rows, int cols);

void printMatrix(STATE *mat[], int rows, int cols);

void freeMatrix(float *mat[], int rows);

void freeMatrix(STATE *mat[], int rows);

void normalizeMatrix(float *mat[], int rows, int cols);

void logMatrixValues(float *mat[], int rows, int cols);

cudaError_t initCuda(float **cuda_a, float **cuda_b, float **cuda_emission, float a[], float b[],
	unsigned int num_of_states, bool withLog);

cudaError_t emissionWithCuda(float emission[], float cuda_emission[], float cuda_a[], float cuda_b[],
	float obsrv, unsigned int num_of_states);

void freeCuda(float cuda_emission[], float cuda_a[], float cuda_b[]);

void outputPathToFile(int path[], int len, FILE *f);

bool loadArrayFromFile(float arr[], int size, char fpath[]);

bool loadMatrixFromFile(float *mat[], int rows, int cols, char fpath[], bool load_transposed);

void clearOutputFiles();

void fileOutputAllMaxStates(STATE *mat[], MAX_STATE *arr, int size);

