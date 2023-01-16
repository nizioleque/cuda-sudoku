bool solveBoard(char* board, int index);

void solveCpu(char* boards, int nBoards) {
    for (int boardIndex = 0; boardIndex < nBoards; boardIndex++) {
        char* board = boards + 81 * boardIndex;

        // check if the board is correct
        bool result = isCorrect(board);
        if (!result) {
            printf("Board %d is incorrect\n", boardIndex + 1);
            continue;
        }

        bool foundSolution = solveBoard(board, 0);
        std::cout << "found solution: " << foundSolution << std::endl;
        printBoard(board);
        std::cout << "\n";
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
