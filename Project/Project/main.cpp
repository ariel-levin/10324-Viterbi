#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "iostream"

using namespace std;


#define NUM_OF_STATES 3			// number of states, default: 1000
#define NUM_OF_OBSRV 5			// number of time slices, default: 30000


static const bool	GENERATE_ZEROES = true;
static const int	MAX_ZERO_RANGE = NUM_OF_OBSRV / 2;
static const bool	TEST_VALUES = true;
static const bool	WITH_LOGS = false;
static const bool	WITH_CUDA = true;
static const bool	PRINT_OBSRV_STATUS = false;


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



float emissionCalc(float aj, float bj, float oi)
{
	return aj * exp(-pow(oi - bj, 2));
}

/* returns the state with the maximum probability from the array */
STATE getMaxState(STATE arr[], int size)
{
	STATE max = arr[0];

	for (int i = 1; i < size; i++)
	{
		if (arr[i].prob > max.prob)
			max = arr[i];
	}

	return max;
}

/* returns the state index with the maximum probability from the array */
int getMaxStateIndex(STATE arr[], int size)
{
	int max = 0;

	for (int i = 1; i < size; i++)
	{
		if (arr[i].prob > arr[max].prob)
			max = i;
	}

	return max;
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
			next_zero = (i+1) + (rand() % MAX_ZERO_RANGE);
		}
		else
			arr[i] = (float)rand() / RAND_MAX;
	}
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
float** allocateDoubleMatrix(int rows, int cols)
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
		for (int j = 0; j < cols; j++)
		{
			mat[i][j] = mat[i][j] / sum;
		}
	}
}

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

void printAllMaxStates(STATE *mat[], MAX_STATE *arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		STATE max_state = arr[i].state;
		printf("\nMax State #%d - Observation %d:\n", i + 1, arr[i].obsrv);
		if (WITH_LOGS)
		{
			//printf("State %d >> Final Prob = %e\n", max_indx, exp(max_state.prob));
			printf("State %d >> Final Prob = %e\n", arr[i].state_num, max_state.prob);
		}
		else
			printf("State %d >> Final Prob = %e\n", arr[i].state_num, max_state.prob);


		int *path = getPath(arr, i, mat);
		printPath(path, getPathLength(arr, i) );
		free(path);
	}
}



/***************************************************************************
****************************		MAIN		****************************
***************************************************************************/


int main(int argc, char* argv[])
{
	int			range[2], state_calc_num;
	int			rank, mpi_proc_num, max_idx = 0, *max_states_idx;
	float		**trans, **ab, *obsrv, *emission;
	float		*cuda_emission, *cuda_a, *cuda_b;		// pointers to memory in GPU
	STATE		**mat, *current, *next;
	MAX_STATE	*max_states_arr;

	MPI_Datatype	STATE_MPI_TYPE;
	MPI_Datatype	type[2] = { MPI_FLOAT, MPI_INT };
	MPI_Aint		offset[2];
	int				blocklen[2] = { 1, 1 };

	MPI_Status	status;
	cudaError_t cudaStatus;


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

	offset[0] = offsetof(STATE, prob);
	offset[1] = offsetof(STATE, parent);
	MPI_Type_create_struct(2, blocklen, offset, type, &STATE_MPI_TYPE);
	MPI_Type_commit(&STATE_MPI_TYPE);


	/* initialize random seed: */
	srand((unsigned int)time(NULL));

	trans			= allocateDoubleMatrix(NUM_OF_STATES, NUM_OF_STATES);
	ab				= allocateDoubleMatrix(2, NUM_OF_STATES);
	obsrv			= (float*)calloc(NUM_OF_OBSRV, sizeof(float));
	emission		= (float*)calloc(NUM_OF_STATES, sizeof(float));
	max_states_idx	= (int*)malloc(mpi_proc_num * sizeof(int));


	///////////////////////////////////////////////////////////


	if (rank == 0)
	{
		if (TEST_VALUES)
			testValues(trans, ab, obsrv);
		else
		{
			generateMatrix(trans, NUM_OF_STATES, NUM_OF_STATES);
			normalizeMatrix(trans, NUM_OF_STATES, NUM_OF_STATES);
			generateMatrix(ab, 2, NUM_OF_STATES);
			generateArray(obsrv, NUM_OF_OBSRV);
		}

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

	for (int i = 0; i < NUM_OF_STATES; i++)
		MPI_Bcast(trans[i], NUM_OF_STATES, MPI_FLOAT, 0, MPI_COMM_WORLD);



	///////////////////////////////////////////////////////////



	if (rank == 0)		///////////////////////		master
	{
		mat				= allocateStateMatrix(NUM_OF_OBSRV, NUM_OF_STATES);
		max_states_arr	= (MAX_STATE*)malloc(sizeof(MAX_STATE));

		int max_states_num = 0;

		bool zero_flag = true;

		double startTime = MPI_Wtime();

		for (int i = 0; i < NUM_OF_OBSRV; i++)		// loop on time slices (observations)
		{
			if (zero_flag)
			{
				for (int j = 0; j < NUM_OF_STATES; j++)
				{
					if (WITH_LOGS)
						mat[i][j].prob = 0;		// log(1) = 0
					else
						mat[i][j].prob = 1;

					mat[i][j].parent = -1;		// no parent
				}

				zero_flag = false;
			}

			MPI_Bcast(mat[i], NUM_OF_STATES, STATE_MPI_TYPE, 0, MPI_COMM_WORLD);

			if ( (obsrv[i] != 0) && (i < NUM_OF_OBSRV - 1) )
			{

				// calculate i emission
				if (WITH_CUDA)
				{
					cudaStatus = emissionWithCuda(emission, cuda_emission, cuda_a, cuda_b, obsrv[i], NUM_OF_STATES);
					if (cudaStatus != cudaSuccess)
					{
						fprintf(stderr, "ERROR: rank %d , observation %d >> Calculate emission failed!", rank, i);
						return 1;
					}
				}
				else
				{
					for (int j = 0; j < NUM_OF_STATES; j++)
					{
						emission[j] = emissionCalc(ab[0][j], ab[1][j], obsrv[i]);
					}
				}

				MPI_Bcast(emission, NUM_OF_STATES, MPI_FLOAT, 0, MPI_COMM_WORLD);


				for (int j = 1; j < mpi_proc_num; j++)
				{
					getRange(range, j, mpi_proc_num);
					state_calc_num = range[1] - range[0];
					MPI_Recv(&mat[i + 1][range[0]], state_calc_num, STATE_MPI_TYPE, j, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				}

				//state_calc_num = NUM_OF_STATES / mpi_proc_num;
				//MPI_Gather(next, state_calc_num, STATE_MPI_TYPE, mat[i + 1], state_calc_num, STATE_MPI_TYPE, 0, MPI_COMM_WORLD);



			}
			else		// finding current max state
			{

				if (i != NUM_OF_OBSRV - 1) 
					emission[0] = -1;	// signal that obsrvation is zero
				else
					emission[0] = -2;	// signal that's the last obsrvation

				MPI_Bcast(emission, NUM_OF_STATES, MPI_FLOAT, 0, MPI_COMM_WORLD);

				MPI_Gather(&max_idx, 1, MPI_INT, max_states_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);

				//printf("\nrank %d observation %d >> max_states_idx gathered:", rank, i);
				//printArray(max_states_idx, mpi_proc_num);


				// finding the maximum from the maximums the slaves found
				int tmp_idx;
				max_idx = max_states_idx[1];
				for (int j = 2; j < mpi_proc_num; j++)
				{
					tmp_idx = max_states_idx[j];
					if (mat[i][tmp_idx].prob > mat[i][max_idx].prob)
						max_idx = tmp_idx;
				}

				max_states_num++;
				max_states_arr = (MAX_STATE*)realloc(max_states_arr, (max_states_num + 1) * sizeof(MAX_STATE));
				max_states_arr[max_states_num - 1].obsrv = i;
				max_states_arr[max_states_num - 1].state_num = max_idx;
				max_states_arr[max_states_num - 1].state = mat[i][max_idx];

				//printf("\nrank %d observation %d >> max_state #%d , prob = %f\n", rank, i, 
				//	max_states_arr[max_states_num - 1].state_num, max_states_arr[max_states_num - 1].state.prob);

				zero_flag = true;
			}


			if (NUM_OF_STATES <= 10 && NUM_OF_OBSRV <= 10)
			{
				printf("\nMatrix Observation %d:\n", i);
				printMatrix(mat, i + 1, NUM_OF_STATES);
			}

			if (PRINT_OBSRV_STATUS)
			{
				cout << "Finished observation " << i << "...\n";
				fflush(stdout);
			}

		}

		printAllMaxStates(mat, max_states_arr, max_states_num);

		double endTime = MPI_Wtime();
		printf("\n\nMPI measured time: %lf\n\n", endTime - startTime);



	}
	else	///////////////////////		slaves		///////////////////////
	{
		getRange(range, rank, mpi_proc_num);
		state_calc_num = range[1] - range[0];

		current = (STATE*)malloc(NUM_OF_STATES * sizeof(STATE));
		next = (STATE*)malloc(state_calc_num * sizeof(STATE));

		bool more_calc = true;

		while (more_calc)
		{
			MPI_Bcast(current, NUM_OF_STATES, STATE_MPI_TYPE, 0, MPI_COMM_WORLD);
			MPI_Bcast(emission, NUM_OF_STATES, MPI_FLOAT, 0, MPI_COMM_WORLD);

			if (emission[0] != -1 && emission[0] != -2)		// normal observation
			{

				for (int m = 0; m < state_calc_num; m++)
				{
					if (WITH_LOGS)
						next[m].prob = current[0].prob + log(trans[0][m + range[0]]) + log(emission[0]);
					else
						next[m].prob = current[0].prob * trans[0][m + range[0]] * emission[0];

					next[m].parent = 0;

					for (int j = 1; j < NUM_OF_STATES; j++)
					{
						float tmp_calc;

						if (WITH_LOGS)
							tmp_calc = current[j].prob + log(trans[j][m + range[0]]) + log(emission[j]);
						else
							tmp_calc = current[j].prob * trans[j][m + range[0]] * emission[j];

						if (tmp_calc > next[m].prob)
						{
							next[m].prob = tmp_calc;
							next[m].parent = j;
						}
					}

				}

				MPI_Send(next, state_calc_num, STATE_MPI_TYPE, 0, 0, MPI_COMM_WORLD);

				//MPI_Gather(next, state_calc_num, STATE_MPI_TYPE, mat[i + 1], state_calc_num, STATE_MPI_TYPE, 0, MPI_COMM_WORLD);
				
				//printf("\nrank %d observation %d >> next array\n", rank, i);
				//printArray(next, state_calc_num);

			}
			else
			{
				max_idx = range[0];

				for (int j = range[0] + 1; j < range[1]; j++)
				{
					if (current[j].prob > current[max_idx].prob)
						max_idx = j;
				}

				//printf("\nrank %d Observation %d >> sending max: %f , index: %d\n", rank, i, current[max_idx].prob, max_idx);

				MPI_Gather(&max_idx, 1, MPI_INT, max_states_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);

				if (emission[0] == -2)
					more_calc = false;
			}


		}


	}



	///////////////////////////////////////////////////////////



	freeMatrix(trans, NUM_OF_STATES);
	freeMatrix(ab, 2);
	free(obsrv);
	free(emission);
	free(max_states_idx);

	if (rank == 0)
	{
		freeMatrix(mat, NUM_OF_OBSRV);
		free(max_states_arr);
		
		if (WITH_CUDA)
		{
			freeCuda(cuda_emission, cuda_a, cuda_b);

			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "ERROR: rank %d >> cudaDeviceReset failed!", rank);
				return 1;
			}
		}

	}
	else
	{
		free(current);
		free(next);
	}

	printf("\nrank %d >> DONE\n\n", rank);
	MPI_Finalize();

	return 0;

}

