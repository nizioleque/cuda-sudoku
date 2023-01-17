bool solveBoard(char* board, int index);

void solveCpu(char* originalBoards, int nBoards) {
    char* boards = new char[nBoards * 81];
    memcpy(boards, originalBoards, nBoards * 81 * sizeof(char));

    for (int boardIndex = 0; boardIndex < nBoards; boardIndex++) {
        char* board = boards + 81 * boardIndex;

        // check if the board is correct
        bool result = isCorrect(board);
        if (!result) {
            continue;
        }

        bool foundSolution = solveBoard(board, 0);
    }


    if (PRINT_SOLUTIONS_CPU) {
        for (int i = 0; i < nBoards; i++) {
            printf("Solution for board %d (CPU):\n", i);
            printBoard(boards + i * 81);
            printf("\n");
        }
    }
}

bool solveBoard(char* board, int index) {
    while (board[index] != 0 && index < 81) {
        index++;
        continue;
    }

    if (index == 81) return true;

    for (int newValue = 1; newValue <= 9; newValue++) {
        if (isSafe(board, index, newValue)) {
            board[index] = newValue;

            bool foundSolution = solveBoard(board, index + 1);
            if (foundSolution) return true;

            board[index] = 0;
        }
    }

    return false;
}
