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
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "global.h"
#include "viterbi.h"
#include "cuda.h"
#include "matrix.h"
#include "array.h"
#include "file.h"



void viterbi(STATE current[], STATE next[], int state_calc_num, float *trans[], float emission[], int range[2])
{

#pragma omp parallel for
	for (int m = 0; m < state_calc_num; m++)
	{
		if (WITH_LOGS)
			next[m].prob = current[0].prob + trans[0][m + range[0]] + emission[0];
		else
			next[m].prob = current[0].prob * trans[0][m + range[0]] * emission[0];

		next[m].parent = 0;

		for (int j = 1; j < NUM_OF_STATES; j++)
		{
			float tmp_calc;

			if (WITH_LOGS)
				tmp_calc = current[j].prob + trans[j][m + range[0]] + emission[j];
			else
				tmp_calc = current[j].prob * trans[j][m + range[0]] * emission[j];

			if (tmp_calc > next[m].prob)
			{
				next[m].prob = tmp_calc;
				next[m].parent = j;
			}
		}

	}
}

float emissionFunc(float aj, float bj, float oi)
{
	return aj * exp(-pow(oi - bj, 2));
}

int calcEmission(float emission[], float *ab[], float obsrv, float *cuda_emission, float *cuda_a, float *cuda_b)
{
	cudaError_t cudaStatus;

	if (WITH_CUDA)
	{
		cudaStatus = emissionWithCuda(emission, cuda_emission, cuda_a, cuda_b, obsrv, NUM_OF_STATES);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "ERROR: Calculate emission failed!");
			return 1;
		}
	}
	else
	{
		for (int i = 0; i < NUM_OF_STATES; i++)
		{
			emission[i] = emissionFunc(ab[0][i], ab[1][i], obsrv);
		}
	}

	return 0;
}

/* returns the state index with the maximum probability from the array */
int getMaxStateIndex(STATE arr[], int range[2])
{
	int max_idx = range[0];

	for (int i = range[0] + 1; i < range[1]; i++)
	{
		if (arr[i].prob > arr[max_idx].prob)
			max_idx = i;
	}

	return max_idx;
}

/* returns the range of a specific slave to work on */
void getRange(int range[], int rank, int mpi_proc_num)
{
	int calc_per_proc = NUM_OF_STATES / (mpi_proc_num - 1);

	range[0] = (rank - 1) * calc_per_proc;

	if (rank == mpi_proc_num - 1)
		range[1] = NUM_OF_STATES;
	else
		range[1] = range[0] + calc_per_proc;
}

/* returns the next index i in observation array that obsrv[i] is not zero */
int getNextIndexToCalc(float obsrv[], int current)
{
	int i = current + 1;
	while ((obsrv[i] == 0) && (i < NUM_OF_OBSRV - 1))
		i++;

	if (i >= NUM_OF_OBSRV - 1)
		return -1;
	else
		return i;
}

/* returns the index of the state with the biggest prob, from the array of indices */
int getMaxIndexFromIndexArray(int max_states_idx[], int mpi_proc_num, STATE arr[])
{
	// finding the maximum from the maximums the slaves found
	int tmp_idx;
	int max_idx = max_states_idx[1];
	for (int i = 2; i < mpi_proc_num; i++)
	{
		tmp_idx = max_states_idx[i];
		if (arr[tmp_idx].prob > arr[max_idx].prob)
			max_idx = tmp_idx;
	}
	return max_idx;
}

void addMaxToMaxArray(MAX_STATE *max_arr[], int *max_states_num, int max_idx, int obsrv, STATE arr[])
{
	(*max_states_num)++;
	*max_arr = (MAX_STATE*)realloc(*max_arr, (*max_states_num + 1) * sizeof(MAX_STATE));
	(*max_arr)[*max_states_num - 1].obsrv = obsrv;
	(*max_arr)[*max_states_num - 1].state_num = max_idx;
	(*max_arr)[*max_states_num - 1].state = arr[max_idx];
}

void initStateColumn(STATE *mat[], int i)
{
#pragma omp parallel for
	for (int j = 0; j < NUM_OF_STATES; j++)
	{
		if (WITH_LOGS)
			mat[i][j].prob = 0;		// log(1) = 0
		else
			mat[i][j].prob = 1;

		mat[i][j].parent = -1;		// no parent
	}
}

int getPathLength(MAX_STATE *arr, int i)
{
	int start, end = arr[i].obsrv;

	if (i == 0)
		start = 0;
	else
		start = arr[i - 1].obsrv + 1;

	return end - start + 1;
}

int* getPath(MAX_STATE *arr, int i, STATE *mat[])
{
	STATE max_state = arr[i].state;
	int o = arr[i].obsrv;

	int len = getPathLength(arr, i);

	int *path = (int*)calloc(len, sizeof(int));
	path[len - 1] = arr[i].state_num;
	for (int j = 2; j <= len; j++)
	{
		path[len - j] = max_state.parent;
		max_state = mat[o + 1 - j][max_state.parent];
	}

	return path;
}

void allocateSpace(float ***trans, float ***ab, float **obsrv, float **emission,
	int **max_states_idx, int mpi_proc_num)
{
	*trans = allocateFloatMatrix(NUM_OF_STATES, NUM_OF_STATES);
	*ab = allocateFloatMatrix(2, NUM_OF_STATES);
	*obsrv = (float*)calloc(NUM_OF_OBSRV, sizeof(float));
	*emission = (float*)calloc(NUM_OF_STATES, sizeof(float));
	*max_states_idx = (int*)malloc(mpi_proc_num * sizeof(int));
}

void initializeValues(float **trans, float **ab, float *obsrv, float **cuda_emission, float **cuda_a, float **cuda_b)
{
	if (TEST_VALUES)
	{
		testValues(trans, ab, obsrv);
	}
	else if (WITH_FILES)
	{
		bool success1 = loadMatrixFromFile(trans, NUM_OF_STATES, NUM_OF_STATES, FILE_PATH"Transition.txt", false);
		bool success2 = loadMatrixFromFile(ab, 2, NUM_OF_STATES, FILE_PATH"AB.txt", true);
		bool success3 = loadArrayFromFile(obsrv, NUM_OF_OBSRV, FILE_PATH"Observation.txt");
		if (!success1 || !success2 || !success3)
		{
			printf("\nERROR: couldn't load files properly..\n"); fflush(stdout);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	else
	{
		generateMatrix(trans, NUM_OF_STATES, NUM_OF_STATES);
		generateMatrix(ab, 2, NUM_OF_STATES);
		generateArray(obsrv, NUM_OF_OBSRV);
	}

	if (NORMALIZE_TRANS)
		normalizeMatrix(trans, NUM_OF_STATES, NUM_OF_STATES);

	if (WITH_CUDA)
		initCuda(cuda_a, cuda_b, cuda_emission, ab[0], ab[1], NUM_OF_STATES);
}

MPI_Datatype createMpiStateType()
{
	MPI_Datatype	STATE_MPI_TYPE;
	MPI_Datatype	type[2] = { MPI_FLOAT, MPI_INT };
	MPI_Aint		offset[2];
	int				blocklen[2] = { 1, 1 };

	offset[0] = offsetof(STATE, prob);
	offset[1] = offsetof(STATE, parent);
	MPI_Type_create_struct(2, blocklen, offset, type, &STATE_MPI_TYPE);
	MPI_Type_commit(&STATE_MPI_TYPE);

	return STATE_MPI_TYPE;
}

void testValues(float *trans[], float *ab[], float obsrv[])
{
	trans[0][0] = 0.8f;	trans[0][1] = 0.5f;	trans[0][2] = 0.3f;
	trans[1][0] = 0.4f;	trans[1][1] = 0.5f;	trans[1][2] = 0.9f;
	trans[2][0] = 0.3f;	trans[2][1] = 0.8f;	trans[2][2] = 0.9f;

	ab[0][0] = 2;	ab[0][1] = 50;	ab[0][2] = 5;
	ab[1][0] = 9;	ab[1][1] = 5;	ab[1][2] = 8;

	obsrv[0] = 5;	obsrv[1] = 5;	obsrv[2] = 10;	obsrv[3] = 4;	obsrv[4] = 5;
	//obsrv[0] = 5;	obsrv[1] = 5;	obsrv[2] = 10;	obsrv[3] = 0;	obsrv[4] = 0;
}

int freeAll(int rank, float *trans[], float *ab[], float obsrv[], float emission[],
	int max_states_idx[], STATE *mat[], MAX_STATE max_states_arr[], STATE current[], STATE next[], 
	float *cuda_emission, float *cuda_a, float *cuda_b)
{
	cudaError_t cudaStatus;

	try { freeMatrix(trans, NUM_OF_STATES); }
	catch (...) {}
	try { freeMatrix(ab, 2); }
	catch (...) {}
	try { free(obsrv); }
	catch (...) {}
	try { free(emission); }
	catch (...) {}
	try { free(max_states_idx); }
	catch (...) {}

	if (rank == 0)
	{
		try { freeMatrix(mat, NUM_OF_OBSRV); }
		catch (...) {}
		try { free(max_states_arr); }
		catch (...) {}

		if (WITH_CUDA)
		{
			try {
				freeCuda(cuda_emission, cuda_a, cuda_b);

				cudaStatus = cudaDeviceReset();
				if (cudaStatus != cudaSuccess)
				{
					fprintf(stderr, "ERROR: rank %d >> cudaDeviceReset failed!", rank);
					return 1;
				}
			}
			catch (...) {}
		}

	}
	else
	{
		try { free(current); }
		catch (...) {}
		try { free(next); }
		catch (...) {}
	}

	return 0;
}

