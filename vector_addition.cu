//---------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
//---------------------------------------------------------------------------------
static const int WORK_SIZE = 200000000;
static const int BLK_SIZE = 256;

using namespace std;
//---------------------------------------------------------------------------------
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

//---------------------------------------------------------------------------------

__global__ void matrixTransposeShared(const int *A_d, int *T_d, const int rows, const int cols) {
	
	int tx=threadIdx.x; int ty=threadIdx.y;
	int row = blockIdx.y * TILE_SIZE + ty;
	int col = blockIdx.x * TILE_SIZE + tx;
	int rowt = blockIdx.x * TILE_SIZE + ty;
	int colt = blockIdx.y * TILE_SIZE + tx;

	__shared__ int A_ds[TILE_SIZE][TILE_SIZE];
	
    int i = by + threadIdx.y; int j = bx + threadIdx.x;
    int ti = bx + threadIdx.y; int tj = by + threadIdx.x;

    if (row<rows && col<cols)
	A_ds[tx][ty] = A_d[row*cols+col];

    __syncthreads();
    if (colt < cols && rowt<rows)
	T_d[rowt * rows+colt] = A_ds[ty][tx];
}
//---------------------------------------------------------------------------------
int main(void) {
	unsigned int *A_h;
	unsigned int *A_d;
	unsigned int *B_h;
	unsigned int *B_d;
	unsigned int *C_h;
	unsigned int *C_d;

	//Set Device
	CUDA_CHECK_RETURN(cudaSetDevice(0));

	//See random number generator
	srand(time(NULL));

	//Clear command prompt
	cout << "\033[2J\033[1;1H";

	cout << "Allocating arrays on host ... ";
	A_h = new unsigned int[WORK_SIZE];
	B_h = new unsigned int[WORK_SIZE];
	C_h = new unsigned int[WORK_SIZE];

	cout << "done.\nPopluating arrays on host ... ";
	for (int i = 0; i < WORK_SIZE; i++) {
		A_h[i] = rand();
		B_h[i] = rand();
	}

	cout << "done.\nAllocating arrays on device ... ";
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &A_d, sizeof(unsigned int) * WORK_SIZE));
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &B_d, sizeof(unsigned int) * WORK_SIZE));
	CUDA_CHECK_RETURN(
			cudaMalloc((void** ) &C_d, sizeof(unsigned int) * WORK_SIZE));

	cout << "done.\nCopying arrays from host to device ... ";
	CUDA_CHECK_RETURN(
			cudaMemcpy(A_d, A_h, sizeof(int) * WORK_SIZE,
					cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(B_d, B_h, sizeof(int) * WORK_SIZE,
					cudaMemcpyHostToDevice));

	cout << "done.\nLaunching kernel ... ";

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** define kernel launch parameters ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	//Time kernel launch
	//Time kernel launch
	cudaEvent_t start, stop;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
	float elapsedTime;

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Add kernel call here ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

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
			cudaMemcpy(C_h, C_d, sizeof(int) * WORK_SIZE,
					cudaMemcpyDeviceToHost));

	cout << "done.\nVerifying results on host ... ";

	//Add code to time host calculations
	clock_t st, ed;

	st = clock();

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Add validation code here ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	ed = clock() - st;
	
	cout << "done\n";
	
	cout << "Elapsed time on host: " << ((float) ed) / CLOCKS_PER_SEC * 1000
			<< " ms" << endl;	

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Output whether results are valid ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@			

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
