//-------------------------------------------------------------------------------------------
#include <ctime>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
//-------------------------------------------------------------------------------------------
using namespace std;
//-------------------------------------------------------------------------------------------
static const long ARRAY_LENGTH = 1024;
static const int BLK_SIZE = 512;
static const long DEVICE = 0;

//-------------------------------------------------------------------------------------------
#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
      fprintf(stderr, "Error %s at line %d in file %s\n",                      \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(1);                                                                 \
    }                                                                          \
  }                                                                            \

//-------------------------------------------------------------------------------------------
__global__ void count_if(int *A_d, int* count, int val, const long aLength) {

  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  // ****      Write Kernel Code       ****
  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  __shared__ int A_ds[ARRAY_LENGTH];

  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x*blockDim.x;
  A_ds[t]=(A_d[start+t]!=val) ? 0 : 1;
  A_ds[blockDim.x+t]=(A_d[blockDim.x+start+t]!=val) ? 0 : 1;


  for (unsigned int stride =blockDim.x; stride>0; stride/=2){
    __syncthreads();
    if(t<stride){
      A_ds[t]+=A_ds[t+stride];
    }
  }
  *count=A_ds[0];
}

//-------------------------------------------------------------------------------------------
int main(void) {
  // Clear command prompt
  cout << "\033[2J\033[1;1H";

  // Set device
  CUDA_CHECK_RETURN(cudaSetDevice(DEVICE));

  int *A_h;
  int *A_d;
  int count_h = 0;
  int *count_d;
  int val = 253;
  
  cout << "Allocating array on host ... ";
  A_h = new int[ARRAY_LENGTH];

  srand(time(NULL));

  cout << "done.\nPopluating array on host ... ";
  for (int i = 0; i < ARRAY_LENGTH; i++) {
	A_h[i] = rand() % 254;
  }

  cout << "done.\nAllocating array on device ... ";
  CUDA_CHECK_RETURN(cudaMalloc((void **)&A_d, sizeof(int) * ARRAY_LENGTH));

  CUDA_CHECK_RETURN(cudaMalloc((void **)&count_d, sizeof(int)));

  cout << "done.\nCopying arrays from host to device ... ";
  CUDA_CHECK_RETURN(cudaMemcpy(A_d, A_h, sizeof(int) * ARRAY_LENGTH,
                               cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(count_d, &count_h, sizeof(int), 
                               cudaMemcpyHostToDevice));

  cout << "done.\nLaunching kernel ... ";
  // Time kernel launch
  cudaEvent_t start, stop;
  CUDA_CHECK_RETURN(cudaEventCreate(&start));
  CUDA_CHECK_RETURN(cudaEventCreate(&stop));
  float elapsedTime;

  CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
  count_if<<<1, BLK_SIZE>>>(A_d, count_d, val, ARRAY_LENGTH);

  CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));

  CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
  CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
  CUDA_CHECK_RETURN(
      cudaThreadSynchronize()); // Wait for the GPU launched work to complete
  CUDA_CHECK_RETURN(cudaGetLastError());
  CUDA_CHECK_RETURN(cudaEventDestroy(start));
  CUDA_CHECK_RETURN(cudaEventDestroy(stop));
  cout << "done.\nElapsed kernel time: " << elapsedTime << " ms\n";

  cout << "Copying results back to host .... ";
  CUDA_CHECK_RETURN(
      cudaMemcpy(&count_h, count_d, sizeof(int), cudaMemcpyDeviceToHost));

  cout << "done.\nOccurence of "<< val <<" found by GPU is: " << count_h << endl;

  // Add code to time host calculations
  clock_t st, ed;

  st = clock();
  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  // **** Validate results here  ****
  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  count_h=0;
  for(int n =0; n<ARRAY_LENGTH; n++){
    if(A_h[n]==val){
      count_h+=1;
    }
  }


  ed = clock() - st;
  cout << "Elapsed time on host: " << ((float)ed) / CLOCKS_PER_SEC * 1000
       << " ms" << endl;
	   
  cout << "Occurence of "<< val <<" found by CPU is: " << count_h << endl;

  cout << "Freeing memory on device ... ";
  CUDA_CHECK_RETURN(cudaFree((void *)A_d));
  CUDA_CHECK_RETURN(cudaDeviceReset());

  cout << "done.\nFreeing memory on host ... ";
  delete[] A_h;
  
  cout << "done.\nExiting program.\n";
  cout << "Question:\n";
  cout << "  Answer: The GPU algorithm is slower because it does not fully utilize GPU resources, and the overhead required for sending data and launching the kernel .\n";

  cout << "  Judge: When you don't atomic add, you cannot control each thread to operate independently.\n";

  return 0;
}
//-------------------------------------------------------------------------------------------
