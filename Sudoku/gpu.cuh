#include <cuda_runtime.h>
#define ARRAY_SIZE 10'000'000
#define THREADS_PER_BLOCK 1024

struct BoardKernelData {
	char* gpuBoards1;
	char* gpuBoards2;
	char* originalBoard1;
	char* originalBoard2;
	char* possibilities;
};

int solveGpu(char* boards, int nBoards);
int runKernel(BoardKernelData data, int nBoards);
__global__ void boardKernel(BoardKernelData data, int threadCount);

int solveGpu(char* boards, int nBoards) {
	int result = 1;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	// Allocate arrays for boards
	char* gpuBoards1;
	char* gpuBoards2;

	cudaStatus = cudaMalloc((void**)&gpuBoards1, ARRAY_SIZE * 81 * sizeof(char));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&gpuBoards2, ARRAY_SIZE * 81 * sizeof(char));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate arrays for original array indices
	char* originalBoard1;
	char* originalBoard2;

	cudaStatus = cudaMalloc((void**)&originalBoard1, ARRAY_SIZE * sizeof(char));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&originalBoard2, ARRAY_SIZE * sizeof(char));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate array for next field possibilities
	char* possibilities;

	cudaStatus = cudaMalloc((void**)&possibilities, ARRAY_SIZE * 9 * sizeof(char));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy boards to GPU memory
	cudaStatus = cudaMemcpy(gpuBoards1, boards, nBoards * 81 * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Initialize original boards
	char* originalBoardCpu = new char[nBoards];
	for (int i = 0; i < nBoards; i++) originalBoardCpu[i] = i;

	cudaStatus = cudaMemcpy(originalBoard1, originalBoardCpu, nBoards * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	BoardKernelData boardKernelData;
	boardKernelData.gpuBoards1 = gpuBoards1;
	boardKernelData.gpuBoards2 = gpuBoards2;
	boardKernelData.originalBoard1 = originalBoard1;
	boardKernelData.originalBoard2 = originalBoard2;
	boardKernelData.possibilities = possibilities;

	result = runKernel(boardKernelData, nBoards);

Error:
	cudaFree(gpuBoards1);
	cudaFree(gpuBoards2);
	cudaFree(originalBoard1);
	cudaFree(originalBoard2);
	cudaFree(possibilities);
	delete[] originalBoardCpu;
	return result;
}

__device__ int tmpBoardCount;

int runKernel(BoardKernelData data, int nBoards) {
	cudaError_t cudaStatus;
	int currentBoardCount = nBoards;
	bool switchBoards = false;

	while (currentBoardCount > 0) {
		int blocks = (currentBoardCount - 1) / THREADS_PER_BLOCK + 1;

		boardKernel << <blocks, THREADS_PER_BLOCK >> > (data, currentBoardCount);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		cudaStatus = cudaMemcpyFromSymbol(&currentBoardCount, tmpBoardCount, sizeof(int), 0, cudaMemcpyDeviceToHost); // TODO DELETE PARAMETERS
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		switchBoards = !switchBoards;
		break;
	}

	return 0;
Error:
	cudaFree(data.gpuBoards1);
	cudaFree(data.gpuBoards2);
	cudaFree(data.originalBoard1);
	cudaFree(data.originalBoard2);
	cudaFree(data.possibilities);
	return 1;
}

__global__ void boardKernel(BoardKernelData data, int threadCount) {
	int boardId = blockIdx.x * blockDim.x + threadIdx.x;
	if (boardId >= threadCount) return;

	tmpBoardCount = 0;

	printf("siema z kernela %d\n", boardId);
}