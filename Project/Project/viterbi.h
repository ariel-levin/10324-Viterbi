/******************************************
*******************************************
***		Ariel Levin						***
***		ariel.lvn89@gmail.com			***
***		http://about.me/ariel.levin		***
*******************************************
******************************************/

#include <mpi.h>
#include "global.h"


void viterbi(STATE current[], STATE next[], int state_calc_num, float *trans[], float emission[], int range[2]);

float emissionFunc(float aj, float bj, float oi);

int calcEmission(float emission[], float *ab[], float obsrv, float *cuda_emission, float *cuda_a, float *cuda_b);

int getMaxStateIndex(STATE arr[], int range[2]);

void getRange(int range[], int rank, int mpi_proc_num);

int getNextIndexToCalc(float obsrv[], int current);

int getMaxIndexFromIndexArray(int max_states_idx[], int mpi_proc_num, STATE arr[]);

void addMaxToMaxArray(MAX_STATE *max_arr[], int *max_states_num, int max_idx, int obsrv, STATE arr[]);

void initStateColumn(STATE *mat[], int i);

int getPathLength(MAX_STATE *arr, int i);

int* getPath(MAX_STATE *arr, int i, STATE *mat[]);

void allocateSpace(float ***trans, float ***ab, float **obsrv, float **emission,
	int **max_states_idx, int mpi_proc_num);

void initializeValues(float **trans, float **ab, float *obsrv, float **cuda_emission, float **cuda_a, float **cuda_b);

MPI_Datatype createMpiStateType();

void testValues(float *trans[], float *ab[], float obsrv[]);

int freeAll(int rank, float *trans[], float *ab[], float obsrv[], float emission[],
	int max_states_idx[], STATE *mat[], MAX_STATE max_states_arr[], STATE current[], STATE next[],
	float *cuda_emission, float *cuda_a, float *cuda_b);

