
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
	std::string nBoardsString;

	std::cout << "Number of boards to solve: ";
	std::cin >> nBoardsString;

	int nBoards = stoi(nBoardsString);

	char* boards = new char[nBoards * 81];
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

	char* solutionCpu = 0;
	int* solutionGpu = 0;

	cpuStart = clock();
	solveCpu(boards, nBoards, &solutionCpu);
	cpuEnd = clock();

	gpuStart = clock();
	int result = solveGpu(boards, nBoards, &solutionGpu);
	gpuEnd = clock();

	bool allGood = true;
	for (int i = 0; i < nBoards; i++) {
		char* solCpu = solutionCpu + i * 81;
		int* solGpu = solutionGpu + i * 82;

		bool isCorrectOriginal = isCorrect(boards + i * 81);

		bool isCorrectCpu = isCorrectAndFilled(solCpu);
		if (!isCorrectCpu && isCorrectOriginal) {
			printf("Error: CPU solution for board %d is incorrect!\n", i);
			printBoard(solCpu);
			allGood = false;
		}

		bool isCorrectGpu = isCorrectAndFilledInt(solGpu);
		if (!isCorrectGpu && isCorrectOriginal) {
			printf("Error: GPU solution for board %d is incorrect!\n", i);
			printBoardInt(solGpu);
			allGood = false;
		}
	}

	if (allGood) printf("All solutions are correct!\n");

	printf("CPU time: %f\nGPU time: %f\n", ((double)cpuEnd - cpuStart) / CLOCKS_PER_SEC, ((double)gpuEnd - cpuEnd) / CLOCKS_PER_SEC);

	delete[] boards;
	delete[] solutionCpu;
	delete[] solutionGpu;

	return result;

}


