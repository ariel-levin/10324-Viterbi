
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t getHistogram(int arr[], int arrSize, int hist[], int histSize, int num_of_threads);


__global__ void addKernel(int arr[], int arrSize, int hist[], int histSize)
{
	int i, start, num_of_threads, num_per_thread;
	int id = threadIdx.x;

	num_of_threads = blockDim.x;
	num_per_thread = arrSize / num_of_threads;
	start = num_per_thread * id;

	//printf("id = %d , shift = %d\n", id, shift);

	for (i = start; i < start + num_per_thread; i++) {

	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t getHistogram(int arr[], int arrSize, int hist[], int histSize, int num_of_threads)
{
	int *dev_arr = 0;
	int *dev_hist = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_arr, arrSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_hist, histSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_arr, arr, arrSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//dim3 gr(2, 2);
	//dim3 bl(1, 2, 4);
	//addKernel<<<gr, bl>>>(dev_a, dev_sum);

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, num_of_threads >> >(dev_arr, arrSize, dev_hist, histSize);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hist, dev_hist, histSize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_arr);
	cudaFree(dev_hist);

	return cudaStatus;
}

