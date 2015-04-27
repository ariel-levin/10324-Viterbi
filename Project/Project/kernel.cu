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
#include <math.h>


cudaError_t initCuda(float **cuda_a, float **cuda_b, float **cuda_emission, float a[], float b[], unsigned int num_of_states);
cudaError_t emissionWithCuda(float emission[], float cuda_emission[], float cuda_a[], float cuda_b[], float obsrv, unsigned int num_of_states);
void freeCuda(float cuda_emission[], float cuda_a[], float cuda_b[]);

bool	WITH_LOGS;


__global__ void emissionKernel(float emission[], float a[], float b[], float obsrv, int N, bool withLog)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N)
	{
		if (withLog)
			emission[i] = log(a[i] * exp(-pow(obsrv - b[i], 2)));
		else
			emission[i] = a[i] * exp(-pow(obsrv - b[i], 2));
	}
}


// Helper function for using CUDA to calculate the emission function
cudaError_t emissionWithCuda(float emission[], float cuda_emission[], float cuda_a[], float cuda_b[], float obsrv, unsigned int num_of_states)
{
	cudaError_t cudaStatus;

	// Invoke kernel 
	int threadsPerBlock = 1024;
	int blocksPerGrid = (num_of_states + threadsPerBlock - 1) / threadsPerBlock;
	emissionKernel << < blocksPerGrid, threadsPerBlock >> >(cuda_emission, cuda_a, cuda_b, obsrv, num_of_states, WITH_LOGS);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "emissionKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		freeCuda(cuda_emission, cuda_a, cuda_b);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		freeCuda(cuda_emission, cuda_a, cuda_b);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(emission, cuda_emission, num_of_states * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeCuda(cuda_emission, cuda_a, cuda_b);
		return cudaStatus;
	}

	return cudaStatus;
}

cudaError_t initCuda(float **cuda_a, float **cuda_b, float **cuda_emission, float a[], float b[],
	unsigned int num_of_states, bool withLog)
{
	cudaError_t cudaStatus;
	*cuda_a = 0;
	*cuda_b = 0;
	*cuda_emission = 0;

	WITH_LOGS = withLog;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		freeCuda(*cuda_emission, *cuda_a, *cuda_b);
		return cudaStatus;
	}

	// Allocate GPU buffers
	cudaStatus = cudaMalloc((void**)cuda_a, num_of_states * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freeCuda(*cuda_emission, *cuda_a, *cuda_b);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)cuda_b, num_of_states * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freeCuda(*cuda_emission, *cuda_a, *cuda_b);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)cuda_emission, num_of_states * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freeCuda(*cuda_emission, *cuda_a, *cuda_b);
		return cudaStatus;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(*cuda_a, a, num_of_states * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeCuda(*cuda_emission, *cuda_a, *cuda_b);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(*cuda_b, b, num_of_states * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeCuda(*cuda_emission, *cuda_a, *cuda_b);
		return cudaStatus;
	}

	return cudaStatus;
}

void freeCuda(float cuda_emission[], float cuda_a[], float cuda_b[])
{
	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_emission);
}

