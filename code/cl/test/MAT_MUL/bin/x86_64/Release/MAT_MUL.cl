// Kernel code
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
 int width;
 int height;
 int stride;
 __global float* elements;
} Matrix;
// Thread block size
#define BLOCK_SIZE 16
// Get a matrix element
float GetElement(const Matrix A, int row, int col)
{
 return A.elements[row * A.stride + col];
}

// Set a matrix element
void SetElement(Matrix A, int row, int col, float value)
{
 A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
Matrix GetSubMatrix(Matrix A, int row, int col)
{
 Matrix Asub;
 Asub.width = BLOCK_SIZE;
 Asub.height = BLOCK_SIZE;
 Asub.stride = A.stride;
 Asub.elements =
 &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
 return Asub;
}
// Matrix multiplication function called by MatMulKernel()
void MatMul(Matrix C, Matrix A, Matrix B,
 __local float As[BLOCK_SIZE][BLOCK_SIZE],
 __local float Bs[BLOCK_SIZE][BLOCK_SIZE])
{
 // Block row and column
 int blockRow = get_group_id(1);
 int blockCol = get_group_id(0);

 // Each thread block computes one sub-matrix Csub of C
 Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

 // Each thread computes one element of Csub
 // by accumulating results into Cvalue
 float Cvalue = 0;

 // Thread row and column within Csub
 int row = get_local_id(1);
 int col = get_local_id(0);

 // Loop over all the sub-matrices of A and B that are
 // required to compute Csub
 // Multiply each pair of sub-matrices together
 // and accumulate the results
 for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

 // Get sub-matrix Asub of A
 Matrix Asub = GetSubMatrix(A, blockRow, m);

 // Get sub-matrix Bsub of B
 Matrix Bsub = GetSubMatrix(B, m, blockCol);

 // Load Asub and Bsub from device memory to shared memory
 // Each thread loads one element of each sub-matrix
 As[row][col] = GetElement(Asub, row, col);
 Bs[row][col] = GetElement(Bsub, row, col); 
 // Synchronize to make sure the sub-matrices are loaded
 // before starting the computation
 barrier(CLK_LOCAL_MEM_FENCE);

 // Multiply Asub and Bsub together
 for (int e = 0; e < BLOCK_SIZE; ++e)
 Cvalue += As[row][e] * Bs[e][col];

 // Synchronize to make sure that the preceding
 // computation is done before loading two new
 // sub-matrices of A and B in the next iteration
 barrier(CLK_LOCAL_MEM_FENCE);
 }
 // Write Csub to device memory
 // Each thread writes one element
 SetElement(Csub, row, col, Cvalue);
}
// Matrix multiplication kernel called by MatMulHost()
__kernel void matrixMulKernel(
 int Cwidth, int Cheight, int Cstride, __global float* Celements,
 int Awidth, int Aheight, int Astride, __global float* Aelements,
 int Bwidth, int Bheight, int Bstride, __global float* Belements,
 __local float As[BLOCK_SIZE][BLOCK_SIZE],
 __local float Bs[BLOCK_SIZE][BLOCK_SIZE])
{
 Matrix C = { Cwidth, Cheight, Cstride, Celements };
 Matrix A = { Awidth, Aheight, Astride, Aelements };
 Matrix B = { Bwidth, Bheight, Bstride, Belements };
 MatMul(A, B, C, As, Bs);
} 
