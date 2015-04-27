/******************************************
*******************************************
***		Ariel Levin						***
***		ariel.lvn89@gmail.com			***
***		http://about.me/ariel.levin		***
*******************************************
******************************************/

#include "global.h"


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

