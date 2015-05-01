/******************************************
*******************************************
***		Ariel Levin						***
***		ariel.lvn89@gmail.com			***
***		http://about.me/ariel.levin		***
*******************************************
******************************************/

#include "global.h"


cudaError_t emissionWithCuda(float emission[], float cuda_emission[], float cuda_a[], float cuda_b[],
	float obsrv, unsigned int num_of_states);

cudaError_t initCuda(float **cuda_a, float **cuda_b, float **cuda_emission, float a[], float b[],
	unsigned int num_of_states);

void freeCuda(float cuda_emission[], float cuda_a[], float cuda_b[]);

