//------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
//------------------------------------------------------------------------
using namespace std;
//------------------------------------------------------------------------
static const int TILE_SIZE = 512;
static const int SHIFT_VALUE = 10;
static const int NUM_OF_FILES = 6;
static const int MAX_FILE_LENGTH = 60000; 
static const long DEVICE = 0;
__constant__  int  S;
//------------------------------------------------------------------------
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
 //------------------------------------------------------------------------
__global__ void decrypt_caesar_cipher(char* N, char* P, int length) {

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// ****      Write Kernel Code       ****
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

}
//---------------------------------------------------------------------------------------------------
int loadInputFile(string fName, char* inputArray) {
	ifstream inputFile;

	inputFile.open(fName.c_str());
	int cnt = 0;
	if (inputFile.is_open()) {
		char temp;
		while (inputFile.get(temp)) {
			inputArray[cnt++] = temp;
		}
		inputFile.close();
	}
	return cnt;
}
//---------------------------------------------------------------------------------------------------
void writeOutput(string oName, char* output, int size) {
	ofstream outputFile;

	outputFile.open(oName.c_str());
	if (outputFile.is_open()) {
		for (size_t i = 0; i < size; i++) {
			outputFile << output[i];
		}
		outputFile.close();
	}
}
//---------------------------------------------------------------------------------------------------
int main(void) {

	//Clear command prompt
	cout << "\033[2J\033[1;1H";

	//Initalize random number generator
	srand(time(NULL));

	//Set device
	CUDA_CHECK_RETURN(cudaSetDevice(DEVICE));

	//Create input array
	cout << "Allocating input array on host ... ";
	int* file_lengths = new int[NUM_OF_FILES];
	char** h_N = new char* [NUM_OF_FILES];
	char** h_P = new char* [NUM_OF_FILES];
	char* d_N;
	char* d_P;
	//Create input array on device
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_N, sizeof(char) * MAX_FILE_LENGTH));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_P, sizeof(char) * MAX_FILE_LENGTH));

	cout << "done.\nLoading input data ... ";
	for (int i = 0; i < NUM_OF_FILES; i++) {
		h_N[i] = new char[MAX_FILE_LENGTH];
		h_P[i] = new char[MAX_FILE_LENGTH];
		int temp = loadInputFile("./encrypted" + to_string(i) + ".txt", h_N[i]);
		file_lengths[i] = temp;
	}

	cout << "done.\nCopying shift to device ... ";
	CUDA_CHECK_RETURN(
		cudaMemcpyToSymbol(S, &SHIFT_VALUE, sizeof(int)));
	cout << "done." << endl;	

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// *** Define kernel parameters here ***
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	
	cout << "Launching " << NUM_OF_FILES << " kernels on default stream ... ";		

	cudaEvent_t start, stop;
	float elapsedTime;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
		
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// ** Make kernel calls on default stream here  **
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	

	cudaEventRecord(stop, 0);

	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));
	cout << "done.\nElapsed time on default stream: " << elapsedTime << " ms\n";

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// ***  Define & create streams objects here ***
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	cout << "\nLaunching " << NUM_OF_FILES << " kernels with " << NUM_OF_FILES << " streams ... ";

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
		
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	// ***  Make stream kernel calls here  ***
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	

    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    // ***  Free stream objects memory here  ***    
    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
	cudaEventRecord(stop, 0);

	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));
	cout << "done.\nElapsed time with " << NUM_OF_FILES << " kernels: " << elapsedTime << " ms\n\n";

	cout << "Writing output ... ";
	for (int i = 0; i < NUM_OF_FILES; i++)
		writeOutput("decrypted" + to_string(i) + ".txt", h_P[i], file_lengths[i]);

	cout << "done.\nFreeing memory ...";
	CUDA_CHECK_RETURN(cudaFree((void*)d_N));
	CUDA_CHECK_RETURN(cudaFree((void*)d_P));
	CUDA_CHECK_RETURN(cudaDeviceReset());
	
	for (int i = 0; i < NUM_OF_FILES; i++) {
		delete[] h_N[i];
		delete[] h_P[i];
	}
	delete[] h_P;
	delete[] h_N;
	delete[] file_lengths;

	cout << "done.\nExiting program\n";
	
	return 0;
}
//---------------------------------------------------------------------------------------------------