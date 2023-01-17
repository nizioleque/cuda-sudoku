# CUDA Sudoku

University project at Warsaw University of Technology

## Instruction

1. Put sudoku boards in the `Sudoku/sudoku.txt` file
2. Run the program
3. Enter the number of boards you want to solve - the program will take the first `n` boards from the file, so remember to put enough boards there!

The program will solve the boards on CPU and GPU and print the time of calculation for both. It will also check if all the solutions are correct and display an error if it finds any incorrect solutions.

To print the solutions, change the `PRINT_SOLUTIONS_CPU` and `PRINT_SOLUTIONS_GPU` flags in `utils.cuh`.

## Adjusting GPU parameters

- If you run out of GPU memory, try lowering the size of allocated arrays - decrease the value of `ARRAY_SIZE` in `gpu.cuh`. The current value (17.5M) is the maximum for a GTX 1650 Ti (Mobile), which has 4 GB of dedicated memory
- If you get empty solutions or an error about kernel resources, try lowering the number of threads in a single block - decreate the value of `THREADS_PER_BLOCK` in `gpu.cuh`
- If you get an incorrect memory access error, try solving fewer boards
