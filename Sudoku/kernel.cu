
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include "utils.cuh"
#include "cpu.cuh"
#include "gpu.cuh"

const char* boardFilename = "sudoku.txt";

bool solveBoard(char* board, int index);

int main()
{
	std::ifstream inFS;
	inFS.open(boardFilename);
	clock_t cpuStart, cpuEnd, gpuStart, gpuEnd;


	std::string line;
	getline(inFS, line);
	int nBoards = stoi(line);

	char* boards = new char[nBoards * 9 * 9];
	int boardsArrayIndex = 0;

	for (int board = 0; board < nBoards; board++) {
		for (int row = 0; row < 9;) {
			getline(inFS, line);
			if (line.length() == 0) continue;
			for (int column = 0; column < 9; column++) {
				boards[boardsArrayIndex++] = line[column] - '0';
			}
			row++;
		}
	}

	cpuStart = clock();
	solveCpu(boards, nBoards);
	cpuEnd = clock();

	gpuStart = clock();
	int result = solveGpu(boards, nBoards);
	gpuEnd = clock();

	printf("CPU time: %f\nGPU time: %f\n", ((double)cpuEnd - cpuStart) / CLOCKS_PER_SEC, ((double)gpuEnd - cpuEnd) / CLOCKS_PER_SEC);

	delete[] boards;

	return result;

}


