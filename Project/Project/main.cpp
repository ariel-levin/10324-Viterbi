/******************************************
*******************************************
***		Ariel Levin						***
***		ariel.lvn89@gmail.com			***
***		http://about.me/ariel.levin		***
*******************************************
******************************************/

#include "global.h"

using namespace std;


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
	float			*cuda_emission = NULL, *cuda_a = NULL, *cuda_b = NULL;		// pointers to memory in GPU


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

	// creating MPI STATE TYPE
	STATE_MPI_TYPE = createMpiStateType();

	// initialize random seed
	srand((unsigned int)time(NULL));

	// allocate space
	allocateSpace(&trans, &ab, &obsrv, &emission, &max_states_idx, mpi_proc_num);

	// initialize values
	if (rank == 0)
		initializeValues(trans, ab, obsrv, &cuda_emission, &cuda_a, &cuda_b);


	///////////////////////////////////////////////////////////

	printf("\n");

	if (rank == 0)		///////////////////////		master		///////////////////////
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
			calcEmission(emission, ab, obsrv[next_index], cuda_emission, cuda_a, cuda_b);

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
					calcEmission(emission, ab, obsrv[next_index], cuda_emission, cuda_a, cuda_b);

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
	else		///////////////////////		slaves		///////////////////////
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


	freeAll(rank, trans, ab, obsrv, emission, max_states_idx, mat, max_states_arr,
		current, next, cuda_emission, cuda_a, cuda_b);

	printf("\nrank %d >> DONE\n\n", rank);
	MPI_Finalize();

	return 0;
}

