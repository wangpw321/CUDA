//-----------------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
//-----------------------------------------------------------------------------------------
using namespace std;
//-----------------------------------------------------------------------------------------
static const long TILE_WIDTH = 16;
static const long DEVICE = 1;
static const long VEC_LEN = 24L;
static const long NUM_VEC = 20000L;
//-----------------------------------------------------------------------------------------
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
//-----------------------------------------------------------------------------------------
__global__ void calDist(float* A_d, float *B_d, float *C_d, const long NUM_VEC,
		const long VEC_LEN) {

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// ****      Write Kernel Code       ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
}
//-----------------------------------------------------------------------------------------
int main(void) {
	//Clear command prompt
	cout << "\033[2J\033[1;1H";

	//Set device
	CUDA_CHECK_RETURN(cudaSetDevice(DEVICE));

	//Seed random number generator
	srand(time(NULL));

	float *A_h;
	float *A_d;
	float *B_h;
	float *B_d;
	float *C_h;
	float *C_d;

	cout << "Allocating matrices on host ... ";
	A_h = new float[VEC_LEN * NUM_VEC];
	B_h = new float[VEC_LEN * NUM_VEC];
	C_h = new float[NUM_VEC * NUM_VEC];

	cout << "done.\nPopluating arrays on host ... ";
	for (int i = 0; i < VEC_LEN * NUM_VEC; i++) {
		A_h[i] = (float) rand() / (float) RAND_MAX * 100;
		B_h[i] = (float) rand() / (float) RAND_MAX * 100;
	}

	cout << "done.\nAllocating arrays on device ... ";
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &A_d, sizeof(float) * VEC_LEN * NUM_VEC));
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &B_d, sizeof(float) * VEC_LEN * NUM_VEC));
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &C_d, sizeof(float) * NUM_VEC * NUM_VEC));

	cout << "done.\nCopying arrays from host to device ... ";
	CUDA_CHECK_RETURN(
			cudaMemcpy(A_d, A_h, sizeof(float) * VEC_LEN * NUM_VEC,
					cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(B_d, B_h, sizeof(float) * VEC_LEN * NUM_VEC,
					cudaMemcpyHostToDevice));

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Define kernel parameters here  ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	//Time kernel launch
	//Time kernel launch
	cudaEvent_t start, stop;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
	float elapsedTime;

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Launch kernel here  ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));

	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError()); //Check if an error occurred in device code
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));
	cout << "done.\nElapsed kernel time: " << elapsedTime << " ms\n";

	cout << "Copying results back to host .... ";
	CUDA_CHECK_RETURN(
			cudaMemcpy(C_h, C_d, sizeof(float) * NUM_VEC * NUM_VEC,
					cudaMemcpyDeviceToHost));

	//Add code to time host calculations
	clock_t st, ed;

	st = clock();
	bool valid = true;
	
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Validate results here  ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	cout << "done\n";

	if (valid) {
		cout << "GPU results are valid.\n";
	}

	ed = clock() - st;
	cout << "Elapsed time on host: " << ((float) ed) / CLOCKS_PER_SEC * 1000
			<< " ms" << endl;

	cout << "Freeing memory on device ... ";
	CUDA_CHECK_RETURN(cudaFree((void* ) A_d));
	CUDA_CHECK_RETURN(cudaFree((void* ) B_d));
	CUDA_CHECK_RETURN(cudaFree((void* ) C_d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	cout << "done.\nFreeing memory on host ... ";
	delete[] A_h;
	delete[] B_h;
	delete[] C_h;

	cout << "done.\nExiting program.\n";

	return 0;
}
//-----------------------------------------------------------------------------------------
