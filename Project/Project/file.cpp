/******************************************
*******************************************
***		Ariel Levin						***
***		ariel.lvn89@gmail.com			***
***		http://about.me/ariel.levin		***
*******************************************
******************************************/

#define _CRT_SECURE_NO_DEPRECATE

#include "global.h"


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

bool loadMatrixFromFile(float *mat[], int rows, int cols, char fpath[], bool load_transposed)
{
	FILE* f = fopen(fpath, "r+");
	if (f == NULL)
	{
		printf("\nFailed opening the file..\n");
		return false;
	}

	if (load_transposed)
	{
		for (int i = 0; i < cols; i++)
		{
			for (int j = 0; j < rows; j++)
			{
				fscanf(f, "%f", &mat[j][i]);
			}
		}
	}
	else
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				fscanf(f, "%f", &mat[i][j]);
			}
		}
	}

	fclose(f);
	return true;
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

