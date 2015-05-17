#ifndef KERNELS_HPP
#define KERNELS_HPP

#include "memory.h"
#include <vector>

void* thread_mat_mul (void* arg);

typedef struct FMATRIX{
	int h;
	int w;
	int stride;
	float* data;  //row major.
	void pad(int dim, int size);  //adjust the matrix by addining miminum ds (ds>=0) to stride such that mod(stride+ds,size) = 0 when dim = 0, or mod(h+ds,size) = 0 when dim = 1.
}MATRIX;

/*struct for kernel parameters*/
typedef struct K_PARAM{
	FMATRIX mat;  //Host data matrix.
	cl_mem device_buffer;  //It references the device data.
	bool releaseFromGPU; //True if the corresponding data in GPU need to be freed, false otherwise.
	int inout;
	/*	[1] For input:   mat.h, mat.w, and mat.stride should specify the size of input buffer.
			Case 1 -- mat.data  = 0, then the input data has already resided in the device. The 'device_buffer' parameter references the data. No need to do data transfer.
			Case 2 -- mat.data != 0, then we need to copy the data in 'mat' to the device. Memory flags: CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR. 
		[2] For output:  mat.h, mat.w, and mat.stride should specify the size of output buffer.
			Case 1 -- mat.data  = 0, no need to write back the data from device to host.
			Case 2 -- mat.data != 0, data need to write back from device to host. The data will be stored in 'mat' parameter by the thread.
	*/
	CL_ENV* env;  //OpenCL environment.
	DEVICE device; //device on which we run the kernel.
}K_PARAM;


/*Matrix multiplication.*/
class MAT_MUL{
public:
	/*Run the kernel*/
	void run(K_PARAM* A, K_PARAM* B, K_PARAM* C);  // Caculate C = A*B. Dimensions of A and B must be multiple of 16.
	
	/*Get the resource requirement to run this kernel*/
	inline void get_rsc_requirement(){
		
	}
};

#endif
