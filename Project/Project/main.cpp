/******************************************
*******************************************
***		Ariel Levin						***
***		ariel.lvn89@gmail.com			***
***		http://about.me/ariel.levin		***
*******************************************
******************************************/

#define _CRT_SECURE_NO_DEPRECATE

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
#include "main.h"

using namespace std;

#define NUM_OF_STATES	1000			// number of states, default: 1000
#define NUM_OF_OBSRV	30000			// number of time slices, default: 30000

#define FILE_PATH	"C:\\Users\\Ariel\\workspace\\afeka\\2015a\\10324-parallel\\Project\\Project\\Files\\"


// initialize consts
static const bool	TEST_VALUES = false;
static const bool	WITH_FILES = true;
static const bool	GENERATE_ZEROES = false;
static const int	MAX_ZERO_RANGE = NUM_OF_OBSRV / 2;		// how often to generate zeros (if using random values)
static const bool	NORMALIZE_TRANS = false;

// how to calc consts
static const bool	WITH_LOGS = true;
static const bool	WITH_CUDA = true;
static const bool	PRINT_OBSRV = false;


float	*cuda_emission, *cuda_a, *cuda_b;		// pointers to memory in GPU


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


cudaError_t initCuda(float **cuda_a, float **cuda_b, float **cuda_emission, float a[], float b[], unsigned int num_of_states);
cudaError_t emissionWithCuda(float emission[], float cuda_emission[], float cuda_a[], float cuda_b[], float obsrv, unsigned int num_of_states);
void freeCuda(float cuda_emission[], float cuda_a[], float cuda_b[]);



float emissionFunc(float aj, float bj, float oi)
{
	return aj * exp(-pow(oi - bj, 2));
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

/* prints the state's given path */
void printPath(int path[], int len)
{
	printf("Path: ");

	// if it's a small path print it all
	if (NUM_OF_STATES <= 10 && NUM_OF_OBSRV <= 10)
	{
		for (int i = 0; i < len; i++)
		{
			printf("%d", path[i]);
			if (i < len - 1)
				printf(" -> ");
		}

	}	// else- print a part of it
	else
		printf("%d -> %d -> ... -> %d -> %d", path[0], path[1], path[len - 2], path[len - 1]);

	printf("\n");
}

void outputPathToFile(int path[], int len, FILE *f)
{
	fprintf(f, "Path: ");

	for (int i = 0; i < len; i++)
	{
		fprintf(f, "%d", path[i]);
		if (i < len - 1)
			fprintf(f, " -> ");
	}

	fprintf(f, "\n");
}

/* prints STATES array */
void printArray(STATE arr[], int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%.3f", arr[i].prob);
		if (i < size - 1)
			printf(", ");
	}
	printf("\n");
}

/* prints doubles array */
void printArray(float arr[], int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		printf("%8.4f", arr[i]);
	}
	printf("\n\n");
}

/* prints int array */
void printArray(int arr[], int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		printf("%4d", arr[i]);
	}
	printf("\n\n");
}

/* generate random array values between 0 and 1 */
void generateArray(float arr[], int size)
{
	int next_zero = rand() % MAX_ZERO_RANGE;

	for (int i = 0; i < size; i++)
	{
		if (GENERATE_ZEROES && i == next_zero)
		{
			arr[i] = 0;
			next_zero = (i + 1) + (rand() % MAX_ZERO_RANGE);
		}
		else
			arr[i] = (float)rand() / RAND_MAX;
	}
}

bool loadArrayFromFile(float arr[], int size, char fpath[])
{
	FILE* f = fopen(fpath, "r+");
	if (f == NULL)
	{
		printf("\nFailed opening the file..\n");
		return false;
	}

	for (int i = 0; i < size; i++)
	{
		fscanf(f, "%f", &arr[i]);
	}

	fclose(f);
	return true;
}

/* copy values from int array a to array b, from index 'start' to 'end' */
void copyArray(int a[], int b[], int start, int end)
{
	int i;
	for (i = start; i <= end; i++)
	{
		b[i] = a[i];
	}
}

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

bool loadMatrixFromFile(float *mat[], int rows, int cols, char fpath[])
{
	FILE* f = fopen(fpath, "r+");
	if (f == NULL)
	{
		printf("\nFailed opening the file..\n");
		return false;
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			fscanf(f, "%f", &mat[i][j]);
		}
	}

	fclose(f);
	return true;
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

int calcEmission(float emission[], float *ab[], float obsrv)
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

int freeAll(int rank, float *trans[], float *ab[], float obsrv[], float emission[],
	int max_states_idx[], STATE *mat[], MAX_STATE max_states_arr[], STATE current[], STATE next[])
{
	cudaError_t cudaStatus;

	try { freeMatrix(trans, NUM_OF_STATES); } catch (...) {}
	try { freeMatrix(ab, 2); } catch (...) {}
	try { free(obsrv); } catch (...) {}
	try { free(emission); } catch (...) {}
	try { free(max_states_idx); } catch (...) {}

	if (rank == 0)
	{
		try { freeMatrix(mat, NUM_OF_OBSRV); } catch (...) {}
		try { free(max_states_arr); } catch (...) {}

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
		try { free(current); } catch (...) {}
		try { free(next); } catch (...) {}
	}

	return 0;
}

void printAllMaxStates(STATE *mat[], MAX_STATE *arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		STATE max_state = arr[i].state;
		printf("\nMax State #%d - Observation %d:\n", i + 1, arr[i].obsrv);
		if (WITH_LOGS)
		{
			if (exp(max_state.prob) != 0)
				printf("State %d >> Final Prob = %e\n", arr[i].state_num, exp(max_state.prob));
			else
				printf("State %d >> Final Prob (log) = %e\n", arr[i].state_num, max_state.prob);
		}
		else
			printf("State %d >> Final Prob = %e\n", arr[i].state_num, max_state.prob);

		int *path = getPath(arr, i, mat);
		printPath(path, getPathLength(arr, i));
		free(path);
	}
}

void clearOutputFiles()
{
	system("IF EXIST "FILE_PATH"Max_States_Result.txt del /q /f "FILE_PATH"Max_States_Result.txt");
}

void fileOutputAllMaxStates(STATE *mat[], MAX_STATE *arr, int size)
{
	clearOutputFiles();

	FILE* f = fopen(FILE_PATH"Max_States_Result.txt", "w+");
	if (f == NULL)
	{
		printf("\nFailed opening the file..\n");
		return;
	}

	for (int i = 0; i < size; i++)
	{
		STATE max_state = arr[i].state;
		fprintf(f, "\nMax State #%d - Observation %d:\n", i + 1, arr[i].obsrv);
		if (WITH_LOGS)
		{
			if (exp(max_state.prob) != 0)
				fprintf(f, "State %d >> Final Prob = %e\n", arr[i].state_num, exp(max_state.prob));
			else
				fprintf(f, "State %d >> Final Prob (log) = %e\n", arr[i].state_num, max_state.prob);
		}
		else
			fprintf(f, "State %d >> Final Prob = %e\n", arr[i].state_num, max_state.prob);

		int *path = getPath(arr, i, mat);
		outputPathToFile(path, getPathLength(arr, i), f);
		free(path);
	}

	fclose(f);
}



/***************************************************************************
****************************		MAIN		****************************
***************************************************************************/


int main(int argc, char* argv[])
{
	int				range[2], state_calc_num, action_flag;
	int				rank, mpi_proc_num, max_idx = 0, *max_states_idx;
	float			**trans, **ab, *obsrv, *emission;
	STATE			**mat, *current, *next;
	MAX_STATE		*max_states_arr;
	MPI_Status		status;
	MPI_Datatype	STATE_MPI_TYPE;


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_proc_num);

	if (mpi_proc_num < 2)
	{
		printf("\nERROR: at least 2 Processes are necessary\n"); fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (mpi_proc_num > NUM_OF_STATES + 1)
	{
		printf("\nERROR: currently running program with %d States.\n", NUM_OF_STATES);
		printf("Please don't run with more than %d Processes\n", NUM_OF_STATES + 1);
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	STATE_MPI_TYPE = createMpiStateType();

	/* initialize random seed: */
	srand((unsigned int)time(NULL));

	// allocate space
	trans = allocateFloatMatrix(NUM_OF_STATES, NUM_OF_STATES);
	ab = allocateFloatMatrix(2, NUM_OF_STATES);
	obsrv = (float*)calloc(NUM_OF_OBSRV, sizeof(float));
	emission = (float*)calloc(NUM_OF_STATES, sizeof(float));
	max_states_idx = (int*)malloc(mpi_proc_num * sizeof(int));


	///////////////////////////////////////////////////////////


	// initialize values
	if (rank == 0)
	{
		if (TEST_VALUES)
		{
			testValues(trans, ab, obsrv);
		}
		else if (WITH_FILES)
		{
			bool success1 = loadMatrixFromFile(trans, NUM_OF_STATES, NUM_OF_STATES, FILE_PATH"Transition.txt");
			bool success2 = loadMatrixFromFile(ab, 2, NUM_OF_STATES, FILE_PATH"AB.txt");
			bool success3 = loadArrayFromFile(obsrv, NUM_OF_OBSRV, FILE_PATH"Observation.txt");
			if (!success1 || !success2 || !success3)
			{
				cout << "\nERROR: couldn't load files properly..\n";
				fflush(stdout);
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

		if (NUM_OF_STATES <= 10 && NUM_OF_OBSRV <= 10)
		{
			printf("rank %d >> Transition Matrix:\n", rank);
			printMatrix(trans, NUM_OF_STATES, NUM_OF_STATES);
			printf("\n");
			printf("rank %d >> a,b Matrix:\n", rank);
			printMatrix(ab, 2, NUM_OF_STATES);
			printf("\n");
			printf("rank %d >> Observations Array:\n", rank);
			printArray(obsrv, NUM_OF_OBSRV);
		}

		if (WITH_CUDA)
			initCuda(&cuda_a, &cuda_b, &cuda_emission, ab[0], ab[1], NUM_OF_STATES);
	}


	///////////////////////////////////////////////////////////

	printf("\n");

	if (rank == 0)		///////////////////////		master
	{
		mat = allocateStateMatrix(NUM_OF_OBSRV, NUM_OF_STATES);
		max_states_arr = (MAX_STATE*)malloc(sizeof(MAX_STATE));

		int max_states_num = 0;
		bool zero_flag = true;

		double omp_start_time = omp_get_wtime();
		double mpi_start_time = MPI_Wtime();								///////////// START TIME

		if (WITH_LOGS)
			logMatrixValues(trans, NUM_OF_STATES, NUM_OF_STATES);

		// distribute the transition matrix to all the slaves
		for (int i = 0; i < NUM_OF_STATES; i++)
			MPI_Bcast(trans[i], NUM_OF_STATES, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// calc the first emission column
		int next_index = -1;
		next_index = getNextIndexToCalc(obsrv, next_index);
		if (next_index != -1)
			calcEmission(emission, ab, obsrv[next_index]);

		for (int i = 0; i < NUM_OF_OBSRV; i++)		// loop on time slices (observations)
		{
			if (zero_flag)
			{
				initStateColumn(mat, i);
				zero_flag = false;
			}

			// sending the current observation's states array
			MPI_Bcast(mat[i], NUM_OF_STATES, STATE_MPI_TYPE, 0, MPI_COMM_WORLD);

			if ((obsrv[i] != 0) && (i < NUM_OF_OBSRV - 1))		// normal observation
			{
				action_flag = 1;		// normal observation - calc next

				MPI_Bcast(&action_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

				MPI_Bcast(emission, NUM_OF_STATES, MPI_FLOAT, 0, MPI_COMM_WORLD);

				// calc the next emission column
				next_index = getNextIndexToCalc(obsrv, next_index);
				if (next_index != -1)
					calcEmission(emission, ab, obsrv[next_index]);

				for (int j = 1; j < mpi_proc_num; j++)
				{
					getRange(range, j, mpi_proc_num);
					state_calc_num = range[1] - range[0];
					MPI_Recv(&mat[i + 1][range[0]], state_calc_num, STATE_MPI_TYPE, j, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				}

			}
			else		// finding current max state
			{
				if (i == NUM_OF_OBSRV - 1)
					action_flag = 3;		// signal that's the last obsrvation
				else
					action_flag = 2;		// signal that obsrvation is zero

				MPI_Bcast(&action_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

				MPI_Gather(&max_idx, 1, MPI_INT, max_states_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);

				// finding the maximum from the maximums the slaves found
				max_idx = getMaxIndexFromIndexArray(max_states_idx, mpi_proc_num, mat[i]);

				addMaxToMaxArray(&max_states_arr, &max_states_num, max_idx, i, mat[i]);

				zero_flag = true;
			}


			if (NUM_OF_STATES <= 10 && NUM_OF_OBSRV <= 10)
			{
				printf("\nMatrix Observation %d:\n", i);
				printMatrix(mat, i + 1, NUM_OF_STATES);
			}

			if (PRINT_OBSRV)
			{
				cout << i << "\n";
				fflush(stdout);
			}

		}

		double mpi_end_time = MPI_Wtime();								///////////// END TIME
		double omp_end_time = omp_get_wtime();

		if (WITH_FILES)
			fileOutputAllMaxStates(mat, max_states_arr, max_states_num);
		else
			printAllMaxStates(mat, max_states_arr, max_states_num);

		printf("\n\nMPI measured time: %lf\n", mpi_end_time - mpi_start_time);
		printf("\nOpenMP measured time: %lf\n\n", omp_end_time - omp_start_time);


	}
	else	///////////////////////		slaves		///////////////////////
	{

		// receive the transition matrix from master
		for (int i = 0; i < NUM_OF_STATES; i++)
			MPI_Bcast(trans[i], NUM_OF_STATES, MPI_FLOAT, 0, MPI_COMM_WORLD);

		getRange(range, rank, mpi_proc_num);
		state_calc_num = range[1] - range[0];

		current = (STATE*)malloc(NUM_OF_STATES * sizeof(STATE));
		next = (STATE*)malloc(state_calc_num * sizeof(STATE));

		bool more_calc = true;

		while (more_calc)
		{
			MPI_Bcast(current, NUM_OF_STATES, STATE_MPI_TYPE, 0, MPI_COMM_WORLD);

			MPI_Bcast(&action_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

			if (action_flag == 1)		// normal observation
			{
				MPI_Bcast(emission, NUM_OF_STATES, MPI_FLOAT, 0, MPI_COMM_WORLD);

				viterbi(current, next, state_calc_num, trans, emission, range);

				MPI_Send(next, state_calc_num, STATE_MPI_TYPE, 0, 0, MPI_COMM_WORLD);
			}
			else
			{
				max_idx = getMaxStateIndex(current, range);

				MPI_Gather(&max_idx, 1, MPI_INT, max_states_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);

				if (action_flag == 3)
					more_calc = false;
			}

		}

	}


	///////////////////////////////////////////////////////////


	freeAll(rank, trans, ab, obsrv, emission, max_states_idx, mat, max_states_arr, current, next);

	printf("\nrank %d >> DONE\n\n", rank);
	MPI_Finalize();

	return 0;

}

