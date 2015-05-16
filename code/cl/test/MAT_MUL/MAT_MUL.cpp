#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

// Host code
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
 int width;
 int height;
 int stride;
 cl_mem elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMulHost(const Matrix A, 
				const Matrix B, 
				Matrix C,
 				const cl_context context,
 				const cl_kernel matMulKernel,
 				const cl_command_queue queue)
{
 // Load A and B to device memory
 Matrix d_A;
 d_A.width = d_A.stride = A.width; d_A.height = A.height;
 size_t size = A.width * A.height * sizeof(float);
 d_A.elements = clCreateBuffer(context,
 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
 size, A.elements, 0);
 Matrix d_B;
 d_B.width = d_B.stride = B.width; d_B.height = B.height;
 size = B.width * B.height * sizeof(float);
 d_B.elements = clCreateBuffer(context,
 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
 size, B.elements, 0);
 // Allocate C in device memory
 Matrix d_C;
 d_C.width = d_C.stride = C.width; d_C.height = C.height;
 size = C.width * C.height * sizeof(float);
 d_C.elements = clCreateBuffer(context,
 CL_MEM_WRITE_ONLY, size, 0, 0);
 // Invoke kernel
 cl_uint i = 0;
 clSetKernelArg(matMulKernel, i++,
 sizeof(d_A.width), (void*)&d_A.width);
 clSetKernelArg(matMulKernel, i++,
 sizeof(d_A.height), (void*)&d_A.height);
 clSetKernelArg(matMulKernel, i++, 
 sizeof(d_A.stride), (void*)&d_A.stride);
 clSetKernelArg(matMulKernel, i++,
 sizeof(d_A.elements), (void*)&d_A.elements);
 clSetKernelArg(matMulKernel, i++,
 sizeof(d_B.width), (void*)&d_B.width);
 clSetKernelArg(matMulKernel, i++,
 sizeof(d_B.height), (void*)&d_B.height);
 clSetKernelArg(matMulKernel, i++,
 sizeof(d_B. stride), (void*)&d_B.stride);
 clSetKernelArg(matMulKernel, i++,
 sizeof(d_B.elements), (void*)&d_B.elements);
 clSetKernelArg(matMulKernel, i++,
 sizeof(d_C.width), (void*)&d_C.width);
 clSetKernelArg(matMulKernel, i++,
 sizeof(d_C.height), (void*)&d_C.height);
 clSetKernelArg(matMulKernel, i++,
 sizeof(d_C.stride), (void*)&d_C.stride);
 clSetKernelArg(matMulKernel, i++,
 sizeof(d_C.elements), (void*)&d_C.elements);
 size_t localWorkSize[] = { BLOCK_SIZE, BLOCK_SIZE };
 size_t globalWorkSize[] =  { B.width / BLOCK_SIZE, A.height / BLOCK_SIZE };
 //push kernel into a command queue.
 clEnqueueNDRangeKernel(queue, matMulKernel, 2, 0, globalWorkSize, localWorkSize, 0, 0, 0);
 // Read C from device memory
 clEnqueueReadBuffer(queue, d_C.elements, CL_TRUE, 0, size, C.elements, 0, 0, 0);

 // Free device memory
 clReleaseMemObject(d_A.elements);
 clReleaseMemObject(d_C.elements);
 clReleaseMemObject(d_B.elements);
}

int main(){
	//setup OpenCL environment.
	
	return 0;
}



