#include "main.hpp"
#include "kernels.hpp"
#include "math.h"
#include "SDKThread.hpp"
#include <iostream>
#include <fstream>

void FMATRIX::pad(int dim, int size){  //adjust the matrix by addining miminum ds (ds>=0) to stride such that mod(stride+ds,size) = 0 when dim = 0, or mod(h+ds,size) = 0 when dim = 1.
	//dim: 0 to padd along x-direction (horizontal), and 1 to padd along y-direction (vertical).
	if(size<=0){
		puts("Warning: ignored invalid size parameter to FMATRIX::pad.");
		return;
	}
	/*Need to put a lock here for thread safe!*/
	if(dim==0){
		int ds = stride - stride%size;
		if(ds==size) return;
		float* _data = data;
		int _stride = stride;
		stride += ds;
		data = new float [stride*h];
		memset(data,0,sizeof(float)*stride*h);
		for(int j = 0;j<h;j++)
			for(int i = 0;i<stride;i++)
				memcpy(data+j*stride*sizeof(float),_data+j*_stride*sizeof(float),_stride*sizeof(float));
		delete _data;
	}else{
		int ds = h - h%size;
		if(ds==size) return;
		float* _data = data;
		int _h = h;
		h += ds;
		data = new float [stride*h];
		memcpy(data,_data,stride*sizeof(float)*_h);
		memset(data+_h*stride*sizeof(float),0,sizeof(float)*stride*ds);
		delete _data;
	}
}


string read_file(const char* fn){
  streampos size;
  char * memblock;
  ifstream file (fn, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    memblock = new char [size];
    file.seekg (0, ios::beg);
    file.read (memblock, size);
    file.close();
    string ret(memblock);
    delete[] memblock;
    return ret;
  }
  else{
  	printf("Unable to open file: %s\n",fn);
  	exit(-1);
  }
}

void* thread_mat_mul (void* arg){
	//calculate: C = A*B.
	
	vector<K_PARAM>& params = *(vector<K_PARAM>*)arg;  //parameters input to the kernel.
	
	/*Creating command queue associate with the context.*/
	cl_context context = params[0].env->context;
	cl_device_id device_id = params[0].device.id;
	cl_command_queue commandQueue = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, NULL);
	int BLOCK_SIZE = sqrt(params[0].device.MaxWorkgroupSize);  
	if(BLOCK_SIZE>16)//the size of block of each sub matrix.
		BLOCK_SIZE = 16;
		
	/*Note: the number of rows for matrix A must be multiple of BLOCK_SIZE.*/
	if(params[0].mat.h%BLOCK_SIZE != 0){
		printf("thread_mat_mul: the height of matrix A (C=A*B) must be multiple of BLOCK_SIZE(%d).\n", BLOCK_SIZE);
		exit(-1);
	}
	params[0].mat.pad(0,BLOCK_SIZE);
	params[1].mat.pad(1,BLOCK_SIZE);

	/*Create program object */
	string sourceStr;
	sourceStr = read_file("mat_mul.cl");
	const char *source = sourceStr.c_str();
	size_t sourceSize[] = {strlen(source)};
	cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
	
	/*Build program. */
	clBuildProgram(program, 1,&device_id,NULL,NULL,NULL);
	
	/*Allocate memory on the device for input and output, and set kernel arguments*/
	cl_mem* device_mem = new cl_mem [params.size()];
	cl_int status = 0;
	cl_kernel kernel = clCreateKernel(program,"thread_mat_mul", NULL);
	for(int i = 0;i<params.size();i++){  //for each parameter.
		if(params[i].inout == 1){  //input
			if (params[i].mat.data!=0){ //copy from host memory as input, read only.
				device_mem[i] = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, params[i].mat.stride * params[i].mat.h * sizeof(float),(void *) params[i].mat.data, NULL);
			}else{  // directly use the data that has been reside in the GPU.
				device_mem[i] = params[i].device_buffer;
			}
		}else{  //output: create output buffer in the device, read and write access.
			device_mem[i] = clCreateBuffer(
		                          context,
		                          CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
		                          params[i].mat.stride * params[i].mat.h * sizeof(float),
		                          0,
		                          &status);
			if(status != CL_SUCCESS){
				printf("thread_mat_mul: unable to allocate output memory buffer of size %ld bytes.\n",params[i].mat.stride * params[i].mat.h * sizeof(float));
				exit(-1);
			}
		}
		clSetKernelArg(kernel, i, sizeof(cl_mem), (void *)&device_mem);  //set kernel arguments.
	}

	/*Run the kernel.*/
	cl_event ndrEvt;
	size_t global_work_size[2] = {params[0].mat.stride/BLOCK_SIZE,params[0].mat.h/BLOCK_SIZE};
	size_t local_work_size[2]  = {BLOCK_SIZE,BLOCK_SIZE};
	clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    status = clEnqueueNDRangeKernel(
                 commandQueue,
                 kernel,
                 2,  //work dimension.
                 NULL,  //global work offset.
                 global_work_size,
                 local_work_size,
                 0,  //number of events in wait list.
                 NULL,  //event wait list.
                 &ndrEvt);
	if(status != CL_SUCCESS){
		printf("thread_mat_mul: unable to run the kernel.");
		exit(-1);
	}
    
	/*Wait for completion.*/
	clFinish(commandQueue);
	
	return 0;
}

void MAT_MUL::run(std::vector<K_PARAM>* P){
	appsdk::SDKThread t;
	t.create(thread_mat_mul,P);
}
