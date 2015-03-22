#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "iostream"

using namespace std;


#define NUM_OF_STATES 1000			// number of states, default: 1000
#define NUM_OF_OBSRV 1000			// number of time slices, default: 30000


const bool GENERATE_ZEROES = false;
const bool TEST_VALUES = false;
const bool WITH_LOGS = true;
const bool PRINT_STATUS = true;


typedef struct STATE
{
	float	prob;
	int		parent;

} STATE;


//cudaError_t getHistogram(int arr[], int arrSize, int hist[], int histSize, int num_of_threads);



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

void printPath(int path[], int size)
{
	printf("Path: ");
	for (int i = 0; i < size; i++)
	{
		printf("%d", path[i]);
		if (i < size - 1)
			printf(" -> ");
	}
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

/* generate random array values between min and max */
void generateArray(float arr[], int size, int min, int max)
{
	int i;
	for (i = 0; i < size; i++)
	{
		if (GENERATE_ZEROES && i % 10 == 0)
			arr[i] = 0;
		else
			arr[i] = (float)min + ((float)rand() / RAND_MAX)*((float)(max - min));
	}
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

/* generate random matrix values between min and max, with number of rows and cols received */
void generateMatrix(float *mat[], int rows, int cols, int min, int max)
{
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			mat[i][j] = (float)min + ((float)rand() / RAND_MAX)*((float)(max - min));
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

	ab[0][0] = 2;		ab[0][1] = 9;
	ab[1][0] = 50;		ab[1][1] = 5;
	ab[2][0] = 5;		ab[2][1] = 8;

	obsrv[0] = 5;	obsrv[1] = 5;	obsrv[2] = 10;	obsrv[3] = 4;	obsrv[4] = 5;
}



/***************************************************************************
****************************		MAIN		****************************
***************************************************************************/


int main(int argc, char* argv[])
{
	int			range[2], state_calc_num;
	int			rank, mpi_proc_num;
	float		**trans, **ab, *obsrv, *emission;
	STATE		**mat, *current, *next;

	MPI_Datatype	STATE_MPI_TYPE;
	MPI_Datatype	type[2] = { MPI_FLOAT, MPI_INT };
	MPI_Aint		offset[2];
	int				blocklen[2] = { 1, 1 };

	MPI_Status	status;


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

	trans = allocateDoubleMatrix(NUM_OF_STATES, NUM_OF_STATES);
	ab = allocateDoubleMatrix(NUM_OF_STATES, 2);
	obsrv = (float*)calloc(NUM_OF_OBSRV, sizeof(float));
	emission = (float*)calloc(NUM_OF_STATES, sizeof(float));


	///////////////////////////////////////////////////////////


	if (rank == 0)
	{
		if (TEST_VALUES)
			testValues(trans, ab, obsrv);
		else
		{
			generateMatrix(trans, NUM_OF_STATES, NUM_OF_STATES, 0, 1);
			normalizeMatrix(trans, NUM_OF_STATES, NUM_OF_STATES);
			generateMatrix(ab, NUM_OF_STATES, 2, 0, 1);
			generateArray(obsrv, NUM_OF_OBSRV, 0, 1);
		}

		if (NUM_OF_STATES <= 10 && NUM_OF_OBSRV <= 10)
		{
			printf("rank %d >> Transition Matrix:\n", rank);
			printMatrix(trans, NUM_OF_STATES, NUM_OF_STATES);
			printf("\n");
			printf("rank %d >> a,b Matrix:\n", rank);
			printMatrix(ab, NUM_OF_STATES, 2);
			printf("\n");
			printf("rank %d >> Observations Array:\n", rank);
			printArray(obsrv, NUM_OF_OBSRV);
		}

	}

	for (int i = 0; i < NUM_OF_STATES; i++)
		MPI_Bcast(trans[i], NUM_OF_STATES, MPI_FLOAT, 0, MPI_COMM_WORLD);



	///////////////////////////////////////////////////////////



	if (rank == 0)		///////////////////////		master
	{
		mat = allocateStateMatrix(NUM_OF_OBSRV, NUM_OF_STATES);

		for (int i = 0; i < NUM_OF_STATES; i++)
		{
			if (WITH_LOGS)
				mat[0][i].prob = 0;			// log(1) = 0
			else
				mat[0][i].prob = 1;

			mat[0][i].parent = -1;		// no parent
		}

		double startTime = MPI_Wtime();

		for (int i = 0; i < NUM_OF_OBSRV - 1; i++)		// loop on time slices (observations)
		{
			MPI_Bcast(mat[i], NUM_OF_STATES, STATE_MPI_TYPE, 0, MPI_COMM_WORLD);

			// calculate i emission
			for (int j = 0; j < NUM_OF_STATES; j++)
			{
				emission[j] = emissionCalc(ab[j][0], ab[j][1], obsrv[i]);
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

			if (NUM_OF_STATES <= 10 && NUM_OF_OBSRV <= 10)
			{
				printf("\nMatrix Observation %d:\n", i);
				printMatrix(mat, i + 1, NUM_OF_STATES);
			}

			if (PRINT_STATUS)
			{
				cout << "Finished observation " << i << "...\n";
				fflush(stdout);
			}

		}

		if (NUM_OF_STATES <= 10 && NUM_OF_OBSRV <= 10)
		{
			printf("\nMatrix Final:\n");
			printMatrix(mat, NUM_OF_OBSRV, NUM_OF_STATES);
		}

		int max_indx = getMaxStateIndex(mat[NUM_OF_OBSRV - 1], NUM_OF_STATES);
		STATE max_state = mat[NUM_OF_OBSRV - 1][max_indx];
		printf("\nFinal Max State:\n");
		if (WITH_LOGS)
		{
			//printf("State %d >> Final Prob = %e\n", max_indx, exp(max_state.prob));
			printf("State %d >> Final Prob = %e\n", max_indx, max_state.prob);
		}
		else
			printf("State %d >> Final Prob = %e\n", max_indx, max_state.prob);

		int path[NUM_OF_OBSRV];
		path[NUM_OF_OBSRV - 1] = max_indx;
		for (int i = 2; i <= NUM_OF_OBSRV; i++)
		{
			path[NUM_OF_OBSRV - i] = max_state.parent;
			max_state = mat[NUM_OF_OBSRV - i][max_state.parent];
		}
		if (NUM_OF_STATES <= 10 && NUM_OF_OBSRV <= 10)
			printPath(path, NUM_OF_OBSRV);


		double endTime = MPI_Wtime();
		printf("\nMPI measured time: %lf\n\n", endTime - startTime);

	}
	else	///////////////////////		slaves		///////////////////////
	{
		getRange(range, rank, mpi_proc_num);
		state_calc_num = range[1] - range[0];

		current = (STATE*)malloc(NUM_OF_STATES * sizeof(STATE));
		next = (STATE*)malloc(state_calc_num * sizeof(STATE));


		for (int i = 0; i < NUM_OF_OBSRV - 1; i++)
		{

			MPI_Bcast(current, NUM_OF_STATES, STATE_MPI_TYPE, 0, MPI_COMM_WORLD);
			MPI_Bcast(emission, NUM_OF_STATES, MPI_FLOAT, 0, MPI_COMM_WORLD);


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


	}



	///////////////////////////////////////////////////////////



	freeMatrix(trans, NUM_OF_STATES);
	freeMatrix(ab, NUM_OF_STATES);
	free(obsrv);
	free(emission);

	if (rank == 0)
	{
		freeMatrix(mat, NUM_OF_STATES);
	}
	else
	{
		free(current);
		free(next);
	}

	printf("\n\nrank %d >> DONE\n\n", rank);
	MPI_Finalize();

	return 0;

}

