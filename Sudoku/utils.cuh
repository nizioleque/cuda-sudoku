#include <iostream>

#define PRINT_SOLUTIONS_CPU false
#define PRINT_SOLUTIONS_GPU false

__host__ __device__ void printBoard(char* board) {
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			printf("%d", board[i * 9 + j]);
		}
		printf("\n");
	}
}

__host__ __device__ void printBoardInt(int* board) {
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			printf("%d", board[i * 9 + j]);
		}
		printf("\n");
	}
}

bool isCorrect(char* board) {
	bool exists[10];

	// check areas
	for (int areaX = 0; areaX < 3; areaX++) {
		for (int areaY = 0; areaY < 3; areaY++) {
			std::fill_n(exists, 10, false);
			int startX = 3 * areaX;
			int startY = 3 * areaY;
			for (int fieldX = 0; fieldX < 3; fieldX++) {
				for (int fieldY = 0; fieldY < 3;fieldY++) {
					int fieldIndex = (startY + fieldY) * 9 + (startX + fieldX);
					int fieldValue = board[fieldIndex];
					if (fieldValue > 0 && exists[fieldValue]) {
						return false;
					}
					exists[fieldValue] = true;
				}
			}
		}
	}

	// check rows
	for (int row = 0; row < 9; row++) {
		std::fill_n(exists, 10, false);
		for (int fieldX = 0; fieldX < 9; fieldX++) {
			int fieldValue = board[row * 9 + fieldX];
			if (fieldValue > 0 && exists[fieldValue]) {
				return false;
			}
			exists[fieldValue] = true;
		}
	}

	// check columns
	for (int column = 0; column < 9; column++) {
		std::fill_n(exists, 10, false);
		for (int fieldY = 0; fieldY < 9; fieldY++) {
			int fieldValue = board[fieldY * 9 + column];
			if (fieldValue > 0 && exists[fieldValue]) {
				return false;
			}
			exists[fieldValue] = true;
		}
	}

	return true;
}

bool isSafe(char* board, int index, int value) {
	// calculate x, y
	int x = index % 9;
	int y = index / 9;

	// check area
	int areaStartX = (x / 3) * 3;
	int areaStartY = (y / 3) * 3;
	for (int fieldX = 0; fieldX < 3; fieldX++) {
		for (int fieldY = 0; fieldY < 3;fieldY++) {
			int fieldIndex = (areaStartY + fieldY) * 9 + (areaStartX + fieldX);
			int fieldValue = board[fieldIndex];
			if (fieldValue == value) return false;
		}
	}

	// check row
	for (int fieldX = 0; fieldX < 9; fieldX++) {
		int fieldValue = board[y * 9 + fieldX];
		if (fieldValue == value) return false;
	}

	// check column
	for (int fieldY = 0; fieldY < 9; fieldY++) {
		int fieldValue = board[fieldY * 9 + x];
		if (fieldValue == value) return false;
	}

	return true;
}