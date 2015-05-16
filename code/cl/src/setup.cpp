#include "main.hpp"

bool DEVICE::getDeviceInfo(cl_device_id device, string t){
	type = t;
	if(CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL)){
		puts("DEVICE:: getDeviceInfo: CL_DEVICE_NAME.");
		return false;
	}
	if(CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(LocalMemSize), &LocalMemSize, NULL)){
		puts("DEVICE:: getDeviceInfo: CL_DEVICE_LOCAL_MEM_SIZE error.");
		return false;
	}
	if(CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(GlobalMemSize), &GlobalMemSize, NULL)){
		puts("DEVICE:: getDeviceInfo: CL_DEVICE_GLOBAL_MEM_SIZE error.");
		return false;
	}
	if(CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(ConstMemSize), &ConstMemSize, NULL)){
		puts("DEVICE:: getDeviceInfo: CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE.");
		return false;
	}
	if(CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(MaxMemAllocSize), &MaxMemAllocSize, NULL)){
		puts("DEVICE:: getDeviceInfo: CL_DEVICE_MAX_MEM_ALLOC_SIZE error.");
		return false;
	}
	if(CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(MaxWorkgroupSize), &MaxWorkgroupSize, NULL)){
		puts("DEVICE:: getDeviceInfo: CL_DEVICE_MAX_WORK_GROUP_SIZE error.");
		return false;
	}
	if(CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(MaxItemDimensions), &MaxItemDimensions, NULL)){
		puts("DEVICE:: getDeviceInfo: CL_DEVICE_MAX_WORK_ITEM_SIZES error.");
		return false;
	}
	if(CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(NumComputeUnit), &NumComputeUnit, NULL)){
		puts("DEVICE:: getDeviceInfo: CL_DEVICE_MAX_COMPUTE_UNITS error.");
		return false;
	}
	return true;
}

void DEVICE::showInfo(){
	printf("name: %s\n",name);
	printf("type: %s\n",type.c_str());
	printf("local memory size (bytes): %ld\n",LocalMemSize);
	printf("global memory size (bytes): %ld\n",GlobalMemSize);
	printf("constant memory size (bytes): %ld\n",ConstMemSize);
	printf("number of computer units: %ld\n",NumComputeUnit);
}

int CL_SETUP::init(){
	/*Getting platforms information. Note: there may be multiple platforms.*/
	cl_uint numPlatforms;
	cl_platform_id platform = NULL;	//the chosen platform
	cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
	{
		puts("CL_SETUP::init: Error getting number of platforms.");
		return -1;
	}
	/*Use the first platform.*/
	cl_platform_id* platforms = 0;
	if(numPlatforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[0];
		free(platforms);
	}else{
		puts("CL_SETUP::init: No platform was found. Have you installed OpenCL?");
		return -1;
	}

	/*Getting devices information*/
	cl_uint				num_gpu = 0, num_cpu = 0;
	cl_device_id        *gpu_devices, *cpu_devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_gpu);  //get number of GPU devices
	if (status != CL_SUCCESS)
	{
		puts("CL_SETUP::init: Error getting number of GPU devices.");
		return -1;
	}
	gpu_devices = (cl_device_id*)malloc(num_gpu * sizeof(cl_device_id));
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_gpu, gpu_devices, NULL);  //get GPU devices
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_cpu);  //get number of CPU devices
	if (status != CL_SUCCESS)
	{
		puts("CL_SETUP::init: Error getting number of CPU devices.");
		return -1;
	}
	cpu_devices = (cl_device_id*)malloc(num_cpu * sizeof(cl_device_id));
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_cpu, cpu_devices, NULL);  //get CPU devices
	
	if(num_cpu + num_cpu<=0){
		puts("CL_SETUP::init: No device was discovered.");
		return -1;
	}
	
	DEVICE* devices = new DEVICE [num_cpu + num_cpu];  //device informaion.
	puts("============Device information============");
	for(int i = 0;i<num_gpu;i++){
		if(devices[i].getDeviceInfo(gpu_devices[i],"GPU")==false)
			return -1;
		devices[i].showInfo();
		puts("---------------------------------------");
	}
	for(int i = 0;i<num_gpu;i++){
		if(devices[i+num_gpu].getDeviceInfo(cpu_devices[i],"CPU")==false)
			return -1;
		devices[i+num_gpu].showInfo();
		puts("---------------------------------------");
	}
	
	/*Step 3: Create context.*/
	context = clCreateContext(NULL,1, gpu_devices,NULL,NULL,NULL);
	
	return 0;
}



