#include "main.hpp"
#include <memory.h>

using namespace std;

bool DEVICE::getDeviceInfo(cl_device_id device, string t) {
    id = device;
    type = t;
    if (CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof (name), name, NULL)) {
        puts("DEVICE:: getDeviceInfo: CL_DEVICE_NAME.");
        return false;
    }
    if (CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof (LocalMemSize), &LocalMemSize, NULL)) {
        puts("DEVICE:: getDeviceInfo: CL_DEVICE_LOCAL_MEM_SIZE error.");
        return false;
    }
    if (CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof (GlobalMemSize), &GlobalMemSize, NULL)) {
        puts("DEVICE:: getDeviceInfo: CL_DEVICE_GLOBAL_MEM_SIZE error.");
        return false;
    } else
        available_mem_size = GlobalMemSize;
    if (CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof (ConstMemSize), &ConstMemSize, NULL)) {
        puts("DEVICE:: getDeviceInfo: CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE.");
        return false;
    }
    if (CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof (MaxMemAllocSize), &MaxMemAllocSize, NULL)) {
        puts("DEVICE:: getDeviceInfo: CL_DEVICE_MAX_MEM_ALLOC_SIZE error.");
        return false;
    }
    if (CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (MaxWorkgroupSize), &MaxWorkgroupSize, NULL)) {
        puts("DEVICE:: getDeviceInfo: CL_DEVICE_MAX_WORK_GROUP_SIZE error.");
        return false;
    }
    if (CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof (MaxItemDimensions), &MaxItemDimensions, NULL)) {
        puts("DEVICE:: getDeviceInfo: CL_DEVICE_MAX_WORK_ITEM_SIZES error.");
        return false;
    }
    if (CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof (NumComputeUnit), &NumComputeUnit, NULL)) {
        puts("DEVICE:: getDeviceInfo: CL_DEVICE_MAX_COMPUTE_UNITS error.");
        return false;
    }
    return true;
}

void DEVICE::showInfo() {
    printf("name: %s\n", name);
    printf("id: %ld\n", (long) id);
    printf("type: %s\n", type.c_str());
#ifdef	__APPLE__
    printf("local memory size (bytes): %lld\n", LocalMemSize);
    printf("global memory size (bytes): %lld\n", GlobalMemSize);
    printf("max memory allocation (bytes): %lld\n", MaxMemAllocSize);
    printf("constant memory size (bytes): %lld\n", ConstMemSize);
#else
    printf("local memory size (bytes): %ld\n", LocalMemSize);
    printf("global memory size (bytes): %ld\n", GlobalMemSize);
    printf("max memory allocation (bytes): %ld\n", MaxMemAllocSize);
    printf("constant memory size (bytes): %ld\n", ConstMemSize);
#endif
    printf("number of compute units: %ld\n", NumComputeUnit);
    printf("max work group size: %ld\n", MaxWorkgroupSize);
}

void CL_ENV::init() {
    /*Getting platforms information. Note: there may be multiple platforms.*/
    cl_uint numPlatforms;
    cl_platform_id platform = NULL; //the chosen platform
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS) {
        puts("CL_ENV::init: Error getting number of platforms.");
        exit(-1);
    }
    /*Use the first platform.*/
    cl_platform_id* platforms = 0;
    if (numPlatforms > 0) {
        cl_platform_id* platforms = (cl_platform_id*) malloc(numPlatforms * sizeof (cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (status != CL_SUCCESS) {
            puts("CL_ENV::init: Error getting platform id.");
            exit(-1);
        }
        platform = platforms[0];
        free(platforms);
    } else {
        puts("CL_ENV::init: No platform was found. Have you installed OpenCL?");
        exit(-1);
    }

    /*Getting devices information*/
    cl_uint num_gpu = 0, num_cpu = 0;
    cl_device_id *gpu_devices, *cpu_devices;
    bool flag_gpu_device = false;
    bool flag_cpu_device = false;
#ifdef USE_DEVICE_GPU
    flag_gpu_device = true;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_gpu); //get number of GPU devices
    if (status != CL_SUCCESS) {
        puts("CL_ENV::init: Error getting number of GPU devices.");
        exit(-1);
    }

    gpu_devices = (cl_device_id*) malloc(num_gpu * sizeof (cl_device_id));
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_gpu, gpu_devices, NULL); //get GPU devices
    if (status != CL_SUCCESS) {
        puts("CL_ENV::init: Error getting GPU devices.");
        flag_gpu_device = false;
    }
#endif
#ifdef USE_DEVICE_CPU
    flag_cpu_device = true;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &num_cpu); //get number of CPU devices
    if (status != CL_SUCCESS) {
        puts("CL_ENV::init: Error getting number of CPU devices.");
        exit(-1);
    }
    if (num_cpu == 0) {
        puts("Error getting number of CPUs.");
    }
    cpu_devices = (cl_device_id*) malloc(num_cpu * sizeof (cl_device_id));
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_cpu, cpu_devices, NULL); //get CPU devices

    if (status != CL_SUCCESS) {
        puts("CL_ENV::init: Error getting CPU devices.");
        flag_cpu_device = false;
    }
#endif

    if (num_gpu + num_cpu <= 0) {
        puts("CL_ENV::init: No device was discovered.");
        exit(-1);
    }

    DEVICE* devs = new DEVICE [num_gpu + num_cpu]; //device informaion.
    puts("============OpenCL Devices information============");
    printf("Number of gpu = %d, number of cpu = %d.\n", num_gpu, num_cpu);
    if (flag_gpu_device)
		for (int i = 0; i < num_gpu; i++) {
		    if (devs[i].getDeviceInfo(gpu_devices[i], "GPU") == false)
		        exit(-1);
		    devs[i].showInfo();
		    devices.push_back(devs + i);
		    puts("---------------------------------------");
		}
    if (flag_cpu_device)
        for (int i = 0; i < num_cpu; i++) {
            if (devs[i + num_gpu].getDeviceInfo(cpu_devices[i], "CPU") == false)
                exit(-1);
            devs[i + num_gpu].showInfo();
            devices.push_back(devs + i + num_gpu);
            puts("---------------------------------------");
        }

    /*Create context.*/
    if (flag_gpu_device == false && flag_cpu_device == false) {
        puts("Cannot get any device.");
        exit(-1);
    }
    cl_device_id* devices_all = (cl_device_id*) malloc((num_cpu + num_gpu) * sizeof (cl_device_id));
    int num_devices = 0;
    if (flag_gpu_device) {
        memcpy(devices_all, gpu_devices, sizeof (cl_device_id*) * num_gpu);
        num_devices += num_gpu;
    }
    if (flag_cpu_device) {
        memcpy(devices_all + num_devices, cpu_devices, sizeof (cl_device_id*) * num_cpu);
        num_devices += num_cpu;
    }
    context = clCreateContext(NULL, num_devices, devices_all, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
        puts("CL_ENV::init: Error creating context.");
        exit(-1);
    }
    delete devices_all;

    /*Create command queues for each device*/
    //for(int i = 0;i<devices.size();i++){
    for (int i = 0; i < devices.size(); i++) {
        devices[i]->buffer_q = clCreateCommandQueue(context, devices[i]->id, CL_QUEUE_PROFILING_ENABLE, &status);
        if (status != CL_SUCCESS) {
            puts("CL_ENV::init: Error creating command queue.");
            exit(-1);
        }
        devices[i]->compute_q = clCreateCommandQueue(context, devices[i]->id, CL_QUEUE_PROFILING_ENABLE, &status);
        if (status != CL_SUCCESS) {
            puts("CL_ENV::init: Error creating command queue.");
            exit(-1);
        }
    }
    printf("Total number of devices for computing: %ld\n", devices.size());
}
