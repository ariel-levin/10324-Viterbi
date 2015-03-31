
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>


cudaError_t initCuda(float **cuda_a, float **cuda_b, float **cuda_emission, float a[], float b[], unsigned int num_of_states);
cudaError_t emissionWithCuda(float emission[], float cuda_emission[], float cuda_a[], float cuda_b[], float obsrv, unsigned int num_of_states);
void freeCuda(float cuda_emission[], float cuda_a[], float cuda_b[]);


__global__ void emissionKernel(float emission[], float a[], float b[], float obsrv)
{
	int i = threadIdx.x;
	emission[i] = a[i] * exp(-pow(obsrv - b[i], 2));
}


cudaError_t initCuda(float **cuda_a, float **cuda_b, float **cuda_emission, float a[], float b[], unsigned int num_of_states)
{
	float *tmp_a = 0;
	float *tmp_b = 0;
	float *tmp_emission = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		freeCuda(tmp_emission, tmp_a, tmp_b);
		return cudaStatus;
	}

	// Allocate GPU buffers
	cudaStatus = cudaMalloc((void**)&tmp_a, num_of_states * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freeCuda(tmp_emission, tmp_a, tmp_b);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&tmp_b, num_of_states * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freeCuda(tmp_emission, tmp_a, tmp_b);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&tmp_emission, num_of_states * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freeCuda(tmp_emission, tmp_a, tmp_b);
		return cudaStatus;
	}



	// Copy input vectors from host memory to GPU buffers.

	cudaStatus = cudaMemcpy(tmp_a, a, num_of_states * sizeof(float), cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpyToSymbol(cuda_a, a, num_of_states * sizeof(float), size_t(0), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeCuda(tmp_emission, tmp_a, tmp_b);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(tmp_b, b, num_of_states * sizeof(float), cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpyToSymbol(cuda_b, b, num_of_states * sizeof(float), size_t(0), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeCuda(tmp_emission, tmp_a, tmp_b);
		return cudaStatus;
	}

	*cuda_a = tmp_a;
	*cuda_b = tmp_b;
	*cuda_emission = tmp_emission;

	return cudaStatus;
}

// Helper function for using CUDA to calculate the emission function
cudaError_t emissionWithCuda(float emission[], float cuda_emission[], float cuda_a[], float cuda_b[], float obsrv, unsigned int num_of_states)
{
	cudaError_t cudaStatus;

	// Launch a kernel on the GPU with one thread for each element.
	emissionKernel << < 1, num_of_states >> >(cuda_emission, cuda_a, cuda_b, obsrv);

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

void freeCuda(float cuda_emission[], float cuda_a[], float cuda_b[])
{
	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_emission);
}

