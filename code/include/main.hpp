#ifndef _MAIN_HPP_
#define _MAIN_HPP_

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#define DEBUG0		//program correctness debugging.
#define DEBUG
#define DEBUG2
//#define DEBUG3		//
//#define DEBUG4		//dumping.

#define USE_DEVICE_GPU
//#define USE_DEVICE_CPU

#define TANH 1
#define RELU 2
#define UNIT_TYPE_INPUT 3
#define UNIT_TYPE_FULL 4
#define UNIT_TYPE_SOFTMAX 5
#define UNIT_TYPE_CONV 6
#define SOLVER_SGD 9
#define SOLVER_ADAGRAD 10
#define SOLVER_NESTEROV 811


#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include<CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "SDKThread.hpp"


#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif
#endif

#ifdef _WIN32
	#include <Windows.h>
#else
	#include <unistd.h>
#endif



typedef struct SOLVER_PARAM {
    int algorithm; 	/*SOLVER_SGD, SOLVER_ADAGRAD, and SOLVER_NESTEROV*/
    float base_lr; 	/*begin learning rate*/
    float gamma; 	/*drop the learning rate by a factor of 1/gama every stepsize.*/
    int stepsize; 	/*drop the learning rate by a factor of 1/gama every stepsize.*/
    int max_iter; 	/*maximum number of iterations*/
    float momentum; /*learning momentum, not valid for ADAGRAD solver*/
    SOLVER_PARAM(){
    	algorithm = -1;
    	base_lr = -1;
    	gamma = -1;
    	stepsize = -1;
    	max_iter = -1;
    	momentum = -1;
    }
} SOLVER_PARAM;



class DEVICE{
	size_t available_mem_size;
	appsdk::ThreadLock lck_mem;
public:
	cl_device_id id;
	char name[100];  //Device name.
	std::string type;  //GPU, CPU.
	cl_ulong LocalMemSize; //Total Local Memory size (in bytes). Note that some devices such as CPU or AMD 7xx GPU don't have local memory.
	cl_ulong GlobalMemSize; //size of global memory.
	cl_ulong ConstMemSize;  //size of constant memory.
	cl_ulong MaxMemAllocSize;  //maximum memory allocation size. The minimum value is max (1/4th of CL_DEVICE_GLOBAL_MEM_SIZE, 128*1024*1024).
	size_t   MaxWorkgroupSize;  //maximum number of work elements in a single work group.
	size_t   MaxItemDimensions[10]; //the maximum number of work items in each dimension (require CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>1).
	size_t NumComputeUnit;
	cl_command_queue buffer_q;   //command queue used to do data transfer between host and device.
	cl_command_queue compute_q;  //used for sending computing instructions.
	bool getDeviceInfo(cl_device_id device, std::string t);
	void showInfo();
	size_t get_total_mem_size(){
		return GlobalMemSize;
	}
	
	/*reserve memory on the device. block until memory becomes available.*/
	void reserve_mem(const size_t& size){
		while(1){
			lck_mem.lock();
			if(available_mem_size>size){
				available_mem_size -= size;
				lck_mem.unlock();
				return;
			}
			lck_mem.unlock();
#ifdef _WIN32
			Sleep(1);      //sleep for 1 ms.
#else
			usleep(1000);   //sleep for 1 ms.
#endif
			
		}
	}
	
	/*release memory on the device.*/
	void release_mem(const size_t& size){
		lck_mem.lock();
		available_mem_size += size;
		lck_mem.unlock();
	}
};

class CL_ENV{
public:
	cl_context context;					/*Context*/
	std::vector<DEVICE*> devices;		/*All the devices used for computing*/
	SOLVER_PARAM	sol_param;			/*solver parameters*/
	void init();
};

#endif
