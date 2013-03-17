CS 149 Programming Assignment 5
(USE LATE DAY)

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

3. We use AsyncMemcpy to boost the speed of copying data.

[Performance]
noisy_01:
  Host to Device Transfer Time: 0.002240 ms
  Kernel(s) Execution Time: 2.345440 ms
  Device to Host Transfer Time: 0.001984 ms
  Total CUDA Execution Time: 2.349664 ms

TOTAL SPEEDUP: 27886.798828

noisy_02:
  Host to Device Transfer Time: 0.002240 ms
  Kernel(s) Execution Time: 2.341824 ms
  Device to Host Transfer Time: 0.002016 ms
  Total CUDA Execution Time: 2.346080 ms

TOTAL SPEEDUP: 27929.396484

noisy_03:
  Host to Device Transfer Time: 0.002272 ms
  Kernel(s) Execution Time: 8.081056 ms
  Device to Host Transfer Time: 0.002048 ms
  Total CUDA Execution Time: 8.085375 ms

TOTAL SPEEDUP: 80611.414062

[Performance of CUFFT]
We also implement CUFFT-based algorithm for comparison (Cuda build-in library),

noisy_01:
  Host to Device Transfer Time: 0.002240 ms
  Kernel(s) Execution Time: 2.345440 ms
  Device to Host Transfer Time: 0.001984 ms
  Total CUDA Execution Time: 2.349664 ms

TOTAL SPEEDUP: 27886.798828

noisy_02:
  Host to Device Transfer Time: 0.002240 ms
  Kernel(s) Execution Time: 2.341824 ms
  Device to Host Transfer Time: 0.002016 ms
  Total CUDA Execution Time: 2.346080 ms

TOTAL SPEEDUP: 27929.396484

noisy_03
  Host to Device Transfer Time: 0.002272 ms
  Kernel(s) Execution Time: 8.081056 ms
  Device to Host Transfer Time: 0.002048 ms
  Total CUDA Execution Time: 8.085375 ms

TOTAL SPEEDUP: 80611.414062

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
