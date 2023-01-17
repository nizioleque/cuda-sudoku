void solveCpu(char* originalBoards, int nBoards, char** solution, clock_t* timeCpu);
bool solveBoard(
	char* board,
	int index,
	bool* possibilitiesRow,
	bool* possibilitiesCol,
	bool* possibilitiesArea
);

void solveCpu(char* originalBoards, int nBoards, char** solution, clock_t* timeCpu) {
	clock_t cpuStart, cpuEnd;

	cpuStart = clock();

	char* boards = new char[nBoards * 81];
	bool possRow[81];
	bool possCol[81];
	bool possArea[81];

	memcpy(boards, originalBoards, nBoards * 81 * sizeof(char));

	for (int boardIndex = 0; boardIndex < nBoards; boardIndex++) {
		char* board = boards + 81 * boardIndex;

		// check if the board is correct
		bool result = isCorrect(board);
		if (!result) {
			continue;
		}

		// initialize possibilities
		for (int i = 0; i < 81; i++) {
			possRow[i] = true;
			possCol[i] = true;
			possArea[i] = true;
		}

		for (int row = 0; row < 9; row++) {
			for (int col = 0; col < 9; col++) {
				char value = board[row * 9 + col] - 1;
				if (value == -1) continue;

				possRow[row * 9 + value] = false;
				possCol[col * 9 + value] = false;

				int areaX = col / 3;
				int areaY = row / 3;
				int areaIndex = areaY * 3 + areaX;
				possArea[areaIndex * 9 + value] = false;
			}
		}

		bool foundSolution = solveBoard(
			board,
			0,
			possRow,
			possCol,
			possArea
		);
	}

	cpuEnd = clock();
	*timeCpu = cpuEnd - cpuStart;

	if (PRINT_SOLUTIONS_CPU) {
		for (int i = 0; i < nBoards; i++) {
			printf("Solution for board %d (CPU):\n", i);
			printBoard(boards + i * 81);
			printf("\n");
		}
	}

	*solution = boards;
}

bool solveBoard(
	char* board,
	int index,
	bool* possRow,
	bool* possCol,
	bool* possArea
) {
	while (board[index] != 0 && index < 81) {
		index++;
		continue;
	}

	if (index == 81) return true;

	int row = index / 9;
	int col = index % 9;

	int areaX = col / 3;
	int areaY = row / 3;
	int areaIndex = areaY * 3 + areaX;

	for (int newValue = 0; newValue < 9; newValue++) {
		if (possRow[row * 9 + newValue]
			&& possCol[col * 9 + newValue]
			&& possArea[areaIndex * 9 + newValue]
			) {
			board[index] = newValue + 1;
			possRow[row * 9 + newValue] = false;
			possCol[col * 9 + newValue] = false;
			possArea[areaIndex * 9 + newValue] = false;

			bool foundSolution = solveBoard(
				board, 
				index + 1,
				possRow,
				possCol,
				possArea
			);

			if (foundSolution) return true;

			board[index] = 0;
			possRow[row * 9 + newValue] = true;
			possCol[col * 9 + newValue] = true;
			possArea[areaIndex * 9 + newValue] = true;
		}
	}

	return false;
}
