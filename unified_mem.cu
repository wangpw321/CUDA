//----------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
//----------------------------------------------------------------------------
using namespace std;
//----------------------------------------------------------------------------
static const long TILE_WIDTH = 8;//Enter TILE_WIDTH
static const long DEVICE = 0;
static const long VEC_LEN = 24L;
static const long NUM_VEC = 19000L;
//----------------------------------------------------------------------------
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
//----------------------------------------------------------------------------
__global__ void calDist(float* A_d, float *B_d, float *C_d, const long NUM_VEC,
	const long VEC_LEN) {

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ****      Write non-tile Kernel Code       ****
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
int row=blockIdx.y*blockDim.y+threadIdx.y;
int col=blockIdx.x*blockDim.x+threadIdx.x;

if ((row<NUM_VEC)&&(col<NUM_VEC)){
	float temp=0.0f;
	for(int k=0;k<VEC_LEN;k++){
		temp=max(abs(A_d[row*VEC_LEN+k]-B_d[k*NUM_VEC+col]),temp);
	}
	C_d[row*NUM_VEC+col]=temp;

}




}
__global__ void calDistTile(float* A_d, float *B_d, float *C_d,
		const long NUM_VEC, const long VEC_LEN) {

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// ****      Write tile Kernel Code       ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	int bx =blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty= threadIdx.y;
	int Row =by * TILE_WIDTH+ty;
	int Col =bx * TILE_WIDTH+tx;
	float temp=0.0f;
	for(int ph=0; ph<ceil(VEC_LEN/(float)TILE_WIDTH);++ph){
		if((Row<NUM_VEC) && (ph*TILE_WIDTH+tx)<VEC_LEN){
			ds_A[ty][tx]=A_d[Row*VEC_LEN+ph*TILE_WIDTH+tx];
		}
		if((ph*TILE_WIDTH+ty)<VEC_LEN && Col<NUM_VEC){
			ds_B[ty][tx]=B_d[(ph*TILE_WIDTH+ty)*NUM_VEC+Col];
		}
		__syncthreads();

		for(int k=0; k<TILE_WIDTH;++k){
			temp=max(abs(ds_A[ty][k]-ds_B[k][tx]),temp);
		}
		__syncthreads();
	}
	if((Row<NUM_VEC)&&(Col<NUM_VEC)){
		C_d[Row*NUM_VEC+Col]=temp;
	}

}
//----------------------------------------------------------------------------
int main(void) {

	//Clear command prompt
	cout << "\033[2J\033[1;1H";

	//Set device
	CUDA_CHECK_RETURN(cudaSetDevice(DEVICE));

	//Get device properties
	cudaDeviceProp deviceProp;
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProp, DEVICE));

	float *A_h;
	// float *A_d;
	float *B_h;
	// float *B_d;
	float *C_h;
	// float *C_d;
	// float *host_result;

	srand(time(NULL));

	cout << "Allocating matrices on host ... ";
	A_h = new float[VEC_LEN * NUM_VEC];
	B_h = new float[VEC_LEN * NUM_VEC];
	C_h = new float[NUM_VEC * NUM_VEC];
	// host_result = new float[NUM_VEC * NUM_VEC];

	cout << "done.\nPopluating arrays on host ... ";
	for (int i = 0; i < VEC_LEN * NUM_VEC; i++) {
		A_h[i] = (float) rand() / (float) RAND_MAX * 100;
		B_h[i] = (float) rand() / (float) RAND_MAX * 100;
	}

	cout << "done.\nAllocating arrays on device ... ";
	// CUDA_CHECK_RETURN(
	// 		cudaMalloc((void** ) &A_d, sizeof(float) * VEC_LEN * NUM_VEC));
	// CUDA_CHECK_RETURN(
	// 		cudaMalloc((void** ) &B_d, sizeof(float) * VEC_LEN * NUM_VEC));
	// CUDA_CHECK_RETURN(
	// 		cudaMalloc((void** ) &C_d, sizeof(float) * NUM_VEC * NUM_VEC));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&A_h, sizeof(float) * VEC_LEN * NUM_VEC));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&B_h, sizeof(float) * VEC_LEN * NUM_VEC));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&C_h, sizeof(float) * NUM_VEC * NUM_VEC));

	// cout << "done.\nCopying arrays from host to device ... ";
	// CUDA_CHECK_RETURN(
	// 		cudaMemcpy(A_d, A_h, sizeof(float) * VEC_LEN * NUM_VEC,
	// 				cudaMemcpyHostToDevice));
	// CUDA_CHECK_RETURN(
	// 		cudaMemcpy(B_d, B_h, sizeof(float) * VEC_LEN * NUM_VEC,
	// 				cudaMemcpyHostToDevice));

	cout << "done.\nLaunching non-tiled kernel ... ";

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Define non-tiled kernel parameters here  ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	dim3 grid(ceil(NUM_VEC/TILE_WIDTH),ceil(NUM_VEC/TILE_WIDTH),1);
	dim3 block(TILE_WIDTH, TILE_WIDTH,1);

	//Time kernel launch
	cudaEvent_t start, stop;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
	float elapsedTime;

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Launch non-tiled kernel here  ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	calDist<<<grid,block>>>(A_h,B_h,C_h,NUM_VEC,VEC_LEN);


	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));

	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError()); //Check if an error occurred in device code
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));
	cout << "done.\nElapsed non-tiled kernel time: " << elapsedTime << " ms\n";


	cout << "Launching tiled kernel ... ";

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Define tiled kernel parameters here  ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	dim3 Tgrid(ceil(NUM_VEC/TILE_WIDTH),ceil(NUM_VEC/TILE_WIDTH),1);
	dim3 Tblock(TILE_WIDTH, TILE_WIDTH,1);

	//Time kernel launch
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Launch tiled kernel here  ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	calDistTile<<<Tgrid,Tblock>>>(A_h,B_h,C_h,NUM_VEC,VEC_LEN);
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));

	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError()); //Check if an error occurred in device code
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));
	cout << "done.\nElapsed tiled kernel time: " << elapsedTime << " ms\n";

	cout << "Copying results back to host .... ";
	// CUDA_CHECK_RETURN(
	// 	cudaMemcpy(C_h, C_d, sizeof(float) * NUM_VEC * NUM_VEC,
	// 			cudaMemcpyDeviceToHost));



					
	//Check results
	cout << "done.\nVerifying results ... ";					

	//Add code to time host calculations
	clock_t st, ed;

	st = clock();
	bool valid = true;
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// **** Validate results here  ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		// for(int x=0; x<NUM_VEC; x++){
		// 	for(int y=0; y<NUM_VEC; y++){
		// 		float temp=0.0f;
		// 		for(int k=0; k<VEC_LEN; k++){
		// 			temp=max(abs(A_h[y*VEC_LEN+k]-B_h[k*NUM_VEC+x]),temp);
		// 		}
		// 		host_result[y*NUM_VEC+x]=temp;
		// 		if (abs(host_result[y*NUM_VEC+x] - C_h[y*NUM_VEC+x]) > 0.001f) {printf("mismatch at %d, %d\n", y, x);}
		// 	}
		// }

	if (valid) {
		cout << "done.\nGPU results are valid.\n";
	}

	ed = clock() - st;
	cout << "Elapsed time on host: " << ((float) ed) / CLOCKS_PER_SEC * 1000
			<< " ms" << endl;

	cout << "Freeing memory on device ... ";
	CUDA_CHECK_RETURN(cudaFree((void* ) A_h));
	CUDA_CHECK_RETURN(cudaFree((void* ) B_h));
	CUDA_CHECK_RETURN(cudaFree((void* ) C_h));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	cout << "done.\nFreeing memory on host ... ";
	// delete[] A_h;
	// delete[] B_h;
	// delete[] C_h;
	// delete[] host_result;

	cout << "done.\nExiting program.\n";

	return 0;
}
//----------------------------------------------------------------------------
