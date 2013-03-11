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
Reference Kernel Execution Time: 96293.445312 ms
Optimized Kernel Execution Time: 41.956001 ms
Speedup: 2295.11

noisy_02:
Reference Kernel Execution Time: 96401.265625 ms
Optimized Kernel Execution Time: 42.112999 ms
Speedup: 2289.11

noisy_03:
Reference Kernel Execution Time: 785200.187500 ms
Optimized Kernel Execution Time: 202.889008 ms
Speedup: 3870.1

[Correctness]
The result of CpuReference and our method are visually identical.
The difference between the results are acceptable due to the error of the numerical calculation. 

