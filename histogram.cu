//---------------------------------------------------------------------------------
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//---------------------------------------------------------------------------------
static const unsigned int WORK_SIZE = 10000001;
static const unsigned int BLOCK_SIZE = 128;
static const unsigned int ELEMENT_COUNT = 1024;
static const unsigned int NUM_BINS = 10;

using namespace std;
//---------------------------------------------------------------------------------
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
      fprintf(stderr, "Error %s at line %d in file %s\n",                      \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(1);                                                                 \
    }                                                                          \
  }
//---------------------------------------------------------------------------------
__global__ void simpleHistogramKernel(int *A, int *H, int inputSize) {

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// ****      Write Kernel Code       ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

}
//---------------------------------------------------------------------------------
__global__ void privateHistogramKernel(int *A, int *H, int inputSize) {
  
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// ****      Write Kernel Code       ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

}
//---------------------------------------------------------------------------------
__global__ void aggrHistogramKernel(int *A, int *H, int inputSize) {

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// ****      Write Kernel Code       ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

}
//---------------------------------------------------------------------------------
void kernelLaunch(float *elapsedTime, int *H_h, int *H_d, int *A_d, int *V,
                  string kernelType) {

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Define gridsize kernel parameter here  ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	for (int i = 0; i < NUM_BINS; ++i) {
		H_h[i] = 0;
	}

	CUDA_CHECK_RETURN(
		cudaMemcpy(H_d, H_h, sizeof(int) * NUM_BINS, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	if (kernelType == "simple")
		simpleHistogramKernel<<<gridSize, BLOCK_SIZE>>>(A_d, H_d, WORK_SIZE);
	else if (kernelType == "private")
		privateHistogramKernel<<<gridSize, BLOCK_SIZE>>>(A_d, H_d, WORK_SIZE);
	else if (kernelType == "aggregation")
		aggrHistogramKernel<<<gridSize, BLOCK_SIZE>>>(A_d, H_d, WORK_SIZE);
	else
		cout << "Unknown kernel kernelType: " << kernelType << endl;

	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(elapsedTime, start, stop));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));

	cout << "done.\nCopying results back to host ...... ";
	CUDA_CHECK_RETURN(
		cudaMemcpy(H_h, H_d, sizeof(int) * NUM_BINS, cudaMemcpyDeviceToHost));

	cout << "done.\nVerifying results on host ...... ";

	bool valid = true;
	for (int i = 0; i < NUM_BINS; i++) {
		if (H_h[i] != V[i]) {
			valid = false;
		break;
		}
	}

	cout << "done\n";

	if (valid)
		cout << "GPU results are valid.\n";
	else
		cout << "GPU results are invalid.\n";
}
//---------------------------------------------------------------------------------
int main(void) {
	int *A_h;
	int *H_h;
	int *V;
	int *A_d;
	int *H_d;

	// Set Device
	CUDA_CHECK_RETURN(cudaSetDevice(0));

	// See random number generator
	srand(time(NULL));

	// Clear command prompt
	cout << "\033[2J\033[1;1H";

	cout << "Allocating arrays on host ...... ";
	A_h = new int[WORK_SIZE];
	H_h = new int[NUM_BINS];
	V = new int[NUM_BINS];

	cout << "done.\nPopluating arrays on host ...... ";

	default_random_engine generator;
	normal_distribution<double> distribution(6.0, 2.5);

	for (int i = 0; i < WORK_SIZE; i++) {
		double temp = distribution(generator);
		A_h[i] = (temp < 0.0) ? 0 : ((temp < 10.0) ? ((int)temp) : 9);
	}
	for (int i = 0; i < NUM_BINS; ++i)
		V[i] = 0;

	cout << "done.\nAllocating arrays on device ...... ";
	CUDA_CHECK_RETURN(cudaMalloc((void **)&A_d, sizeof(int) * WORK_SIZE));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&H_d, sizeof(int) * NUM_BINS));

	cout << "done.\nCopying arrays from host to device ...... ";
	CUDA_CHECK_RETURN(
		cudaMemcpy(A_d, A_h, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));

	cout << "done.\n";
	cout << "-----------------------------------------------------------------------------\n";
	cout << "Calculating histogram on host for verification ...... ";
	clock_t st, ed;
	st = clock();

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Validate results here  ****
	// Use array V
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	ed = clock() - st;
	cout << "done.\nElapsed time on host: " << ((float)ed) / CLOCKS_PER_SEC * 1000
		<< " ms\n";

	float elapsedTime;

	cout << "-----------------------------------------------------------------------------\n";
	cout << "Launching simple histogram kernel ...... ";
	kernelLaunch(&elapsedTime, H_h, H_d, A_d, V, "simple");
	cout << "Elapsed time on device: " << elapsedTime << " ms\n";

	cout << "-----------------------------------------------------------------------------\n";
	cout << "Launching privatized histogram kernel ...... ";
	kernelLaunch(&elapsedTime, H_h, H_d, A_d, V, "private");
	cout << "Elapsed time on device: " << elapsedTime << " ms\n";

	cout << "-----------------------------------------------------------------------------\n";
	cout << "Launching aggregation histogram kernel ...... ";
	kernelLaunch(&elapsedTime, H_h, H_d, A_d, V, "aggregation");
	cout << "Elapsed time on device: " << elapsedTime << " ms\n";

	cout << "-----------------------------------------------------------------------------\n";
	cout << "Bins distribution percentage: ";
	for (int i = 0; i < NUM_BINS; ++i) {
		cout << round(V[i] * 100.0 / WORK_SIZE) << "%, ";
	}
	cout << "\n";
	cout << "-----------------------------------------------------------------------------\n";

	cout << "Freeing memory on device ... ";
	CUDA_CHECK_RETURN(cudaFree((void *)A_d));
	CUDA_CHECK_RETURN(cudaFree((void *)H_d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	cout << "done.\nFreeing memory on host ... ";

	delete[] A_h;
	delete[] H_h;
	delete[] V;

	cout << "done.\nExiting program.\n";

	return 0;
}
