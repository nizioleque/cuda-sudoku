
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include "utils.cuh"

const char* boardFilename = "sudoku.txt";

int main()
{
    cudaError_t cudaStatus;
    // read text file
    std::ifstream inFS;
    inFS.open(boardFilename);

    std::string line;
    getline(inFS, line);
    int nBoards = stoi(line);

    char* originalBoards = new char[nBoards * 9 * 9];
    int originalBoardsIndex = 0;

    for (int board = 0; board < nBoards; board++) {
        for (int row = 0; row < 9; row++) {
            getline(inFS, line);
            if (line.length() == 0) continue;
            for (int column = 0; column < 9; column++) {
                originalBoards[originalBoardsIndex++] = line[column] - '0';
            }
        }
    }


    // run CPU
    for (int board = 0; board < nBoards; board++) {

    }

    //

    // run GPU


    return 0;
}

