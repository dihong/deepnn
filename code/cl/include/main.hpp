#ifndef _MAIN_HPP_
#define _MAIN_HPP_

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;

class DEVICE{
public:
	cl_device_id id;
	char name[100];  //Device name.
	string type;  //GPU, CPU.
	cl_ulong LocalMemSize; //Total Local Memory size (in bytes). Note that some devices such as CPU or AMD 7xx GPU don't have local memory.
	cl_ulong GlobalMemSize; //size of global memory.
	cl_ulong ConstMemSize;  //size of constant memory.
	cl_ulong MaxMemAllocSize;  //maximum memory allocation size. The minimum value is max (1/4th of CL_DEVICE_GLOBAL_MEM_SIZE, 128*1024*1024).
	size_t   MaxWorkgroupSize;  //maximum number of work elements in a single work group.
	size_t   MaxItemDimensions[10]; //the maximum number of work items in each dimension (require CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>1).
	size_t NumComputeUnit;
	bool getDeviceInfo(cl_device_id device, string t);
	void showInfo();
};

class CL_ENV{
public:
	cl_context context; //OpenCL context.
	DEVICE* devices;
	int init();
};

#endif
