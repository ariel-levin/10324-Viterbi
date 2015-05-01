/******************************************
*******************************************
***		Ariel Levin						***
***		ariel.lvn89@gmail.com			***
***		http://about.me/ariel.levin		***
*******************************************
******************************************/

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "global.h"
#include "array.h"


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

/* copy values from int array a to array b, from index 'start' to 'end' */
void copyArray(int a[], int b[], int start, int end)
{
	int i;
	for (i = start; i <= end; i++)
	{
		b[i] = a[i];
	}
}

