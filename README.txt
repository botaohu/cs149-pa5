CS 149 Programming Assignment 5

Botao Hu (botaohu@stanford.edu)

[Optimization]

1. Implement 1D fast fourier transformation based on Cooley-Tukey's algorithm
First, we manually hard-code FFT network on single thread for N = 2, 4, 8, 16 as the builing block. 

For large N = 256, 512, 1024, we implement Cooley-Tukey framework 

	1. Lay out the data into N1 x N2 matrix in column-major order. [in local memory]
	2. Perform DFT on each row of the matrix [in local memory]
	3. Twiddle and transpose the matrix. [in shared memory]
	4. Scale the matrix component-wise. [in local memory]
	5. Twiddle and transpose the matrix. [in shared memory]
	6. Perform DFT on each row of the matrix. [in local memory]

This framework reduces one large DFT into many smaller DFTs. Also, it reduces the count of complex operations from N^2 down to N(N_1 + N_2). The technique can be applied recursively. The total operation 
count then can be reduced to N log N. 

Reference:
Volkov and Kazian. Fitting FFT onto the G80 Architecture, CS 258 final project report, University of California, Berkeley
http://www.cs.berkeley.edu/~kubitron/courses/cs258-S08/projects/reports/project6_report.pdf

2. Implement 2D FFT.
We distribute all rows (or columns) on GPU grid.
Each block represents one 1D FFT task.

[Performance]
noisy_01:

CUDA IMPLEMENTATION STATISTICS:
  Host to Device Transfer Time: 3.018848 ms
  Kernel(s) Execution Time: 0.687296 ms
  Device to Host Transfer Time: 3.167104 ms
  Total CUDA Execution Time: 6.873248 ms

TOTAL SPEEDUP: 9533.280273

noisy_02:
CUDA IMPLEMENTATION STATISTICS:
  Host to Device Transfer Time: 2.999072 ms
  Kernel(s) Execution Time: 0.675808 ms
  Device to Host Transfer Time: 3.151552 ms
  Total CUDA Execution Time: 6.826432 ms

TOTAL SPEEDUP: 9598.660156

noisy_03:

CUDA IMPLEMENTATION STATISTICS:
  Host to Device Transfer Time: 11.428384 ms
  Kernel(s) Execution Time: 2.846336 ms
  Device to Host Transfer Time: 12.319008 ms
  Total CUDA Execution Time: 26.593727 ms

TOTAL SPEEDUP: 24508.542969

[Performance of CUFFT]
We also implement CUFFT-based algorithm for comparison (Cuda build-in library),

noisy_01:
CUDA IMPLEMENTATION STATISTICS:
  Host to Device Transfer Time: 3.078112 ms
  Kernel(s) Execution Time: 0.376640 ms
  Device to Host Transfer Time: 2.867776 ms
  Total CUDA Execution Time: 6.322528 ms

TOTAL SPEEDUP: 10363.671875

noisy_02:
CUDA IMPLEMENTATION STATISTICS:
  Host to Device Transfer Time: 3.133632 ms
  Kernel(s) Execution Time: 0.374656 ms
  Device to Host Transfer Time: 2.910112 ms
  Total CUDA Execution Time: 6.418400 ms

TOTAL SPEEDUP: 10208.869141

noisy_03:
CUDA IMPLEMENTATION STATISTICS:
  Host to Device Transfer Time: 11.517792 ms
  Kernel(s) Execution Time: 1.379744 ms
  Device to Host Transfer Time: 11.277120 ms
  Total CUDA Execution Time: 24.174656 ms

TOTAL SPEEDUP: 26961.025391

[Correctness]
The result of CpuReference and our method are visually identical.
The difference between the results are acceptable due to the error of the numerical calculation. 

[Running]
First execute
	source run.sh 
to set up the environment variables.
Then execute
	qsub -d$(pwd) pa5.pbs
If you want to execute CUFFT-based algorithm for comparison (Cuda build-in library),
then execute
	qsub -d$(pwd) pa5-cufft.pbs
