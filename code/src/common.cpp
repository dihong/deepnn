#include "main.hpp"
#include "common.hpp"
#include "math.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>

using namespace std;

// Load an OpenCL kernel from file

std::string readKernelFile(std::string filename) {

    // Open the file
    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        printf("** Error opening file '%s' **\n", filename.c_str());
        exit(-1);
    }

    // Get its size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    // Read the kernel code as a string
    char* source = (char *) calloc((size + 1), sizeof (char));
    size_t len = fread(source, 1, size * sizeof (char), file);
    fclose(file);
    std::string ret = source;
    free(source);
    return ret;
}

bool mycompfunc_float(const pair<float, int>& l, const pair<float, int>& r) {
    return l.first > r.first;
}

void quick_sort(float* arr, int N, int* order, float* sorted, bool descend) {
    vector< pair<float, int> > WI;
    pair<float, int> val_ind;
    for (int i = 0; i < N; i++) {
        val_ind.first = arr[i]; //value.
        val_ind.second = i; //index.
        WI.push_back(val_ind);
    }
    sort(WI.begin(), WI.end(), mycompfunc_float);
    if (descend)
        for (int i = 0; i < N; i++) {
            if (sorted) sorted[i] = WI[i].first;
            order[i] = WI[i].second;
        } else
        for (int i = N - 1; i >= 0; i--) {
            if (sorted) sorted[i] = WI[i].first;
            order[i] = WI[i].second;
        }
}

int UNIT::global_numeric_identifier_cnt = 1; /*global numeric identifier begins from 1*/


/*Create a new copy of host memory for data1, and initialize data1 with data2.*/
void PMEM::set_data1(){
	if (data2 == NULL){
        printf("PMEM::set_data1(%s): unable to map data2 from reading. data2 is NULL.\n",id.c_str());
        exit(-1);
	}
	if(data1 == NULL){
		data1 = new float [n_stride*d_stride];
		memset(data1,0,sizeof(float)*n_stride*d_stride);
	}
    cl_int status;
    float* ptr = (float*) clEnqueueMapBuffer(
            env.devices[0]->buffer_q,		//It doesn't matter command queue from which device is used. It just need to be a valid one.
            data2,
            CL_TRUE,
            CL_MAP_READ,
            0,
            sizeof(float)*n_stride*d_stride,
            0,
            NULL,
            NULL,
            &status);
    if (status != CL_SUCCESS) {
        printf("PMEM::set_data1(%s): unable to map data2 for reading. %s.\n", id.c_str(), getErrorString(status));
        exit(-1);
    }
	
	/*Read*/
	memcpy(data1,ptr,sizeof(float)*n_stride*d_stride);
    
	/*Unmap buffer*/
    cl_event event;
    clEnqueueUnmapMemObject(env.devices[0]->buffer_q,
            data2,
            ptr,
            0,
            0,
            &event);
    clFlush(env.devices[0]->buffer_q);
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
}


/*Create a new copy of opencl memory object for data2, and initialize data2 with data1.*/
void PMEM::set_data2(){
	if (data1 == NULL){
        printf("PMEM::set_data2(%s): unable to set data2 because data1 is NULL.\n",id.c_str());
        exit(-1);
	}
	
	cl_int status;
	if(data2 == NULL){
		data2 = clCreateBuffer(env.context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(float)*n_stride*d_stride , 0, &status);				/*Pinned memory*/
		if(status != CL_SUCCESS){
			printf("PMEM::set_data2(%s): Unable to allocate pinned cl_mem of size %ld bytes.\n", id.c_str(), sizeof(float)*n_stride*d_stride);
			exit(-1);
		}
	}
    
    float* ptr = (float*) clEnqueueMapBuffer(
            env.devices[0]->buffer_q,		//It doesn't matter command queue from which device is used. It just need to be a valid one.
            data2,
            CL_TRUE,
            CL_MAP_WRITE,
            0,
            sizeof(float)*n_stride*d_stride,
            0,
            NULL,
            NULL,
            &status);
    if (status != CL_SUCCESS) {
        printf("PMEM::set_data2(%s): unable to map data2 for writing. %s.\n", id.c_str(), getErrorString(status));
        exit(-1);
    }
	
	/*Write*/
	memcpy(ptr,data1,sizeof(float)*n_stride*d_stride);
    
	/*Unmap buffer*/
    cl_event event;
    clEnqueueUnmapMemObject(env.devices[0]->buffer_q,
            data2,
            ptr,
            0,
            0,
            &event);
    clFlush(env.devices[0]->buffer_q);
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
}


/*delete data1*/
void PMEM::free_data1(){
	if (data1==NULL) return;
	delete data1;
	data1 = NULL;
}


/*delete data2*/
void PMEM::free_data2(){	
	if (data2==NULL) return;
	int status = clReleaseMemObject(data2);
	if (status != CL_SUCCESS){
        printf("PMEM::free_data2(%s): unable to free data2. %s.\n", id.c_str(), getErrorString(status));
        exit(-1);
	}
	data2 = NULL;
}


/*migrate data2 into specified device, no matter where the data2 resided originally. The data2 must be valid before calling the function.*/
void PMEM::migrate(const DEVICE & device){
	if (data2==NULL)
		this->set_data2();
    cl_event event;
    int status = clEnqueueMigrateMemObjects(device.buffer_q,
    		1,
            &data2,
            0,
            0,
            0,
            &event);
    clFlush(device.buffer_q);
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
	if (status != CL_SUCCESS){
        printf("PMEM::migrate(%s): unable to migrate data. %s.\n", id.c_str(), getErrorString(status));
        exit(-1);
	}
}


/*Decrease the reference count of data2 by 1. When reference count reaches zero, it is released.*/
void PMEM::dec_data2_ref_count(){
	lck_refc.lock();
	if (refc<1){
        printf("PMEM::dec_data2_ref_count(%s): try to decrease a reference count refc = %d.\n", id.c_str(), refc);
        exit(-1);
	}
	refc--;
	if (refc == 0){
		this->free_data2();
#ifdef DEBUG3
		printf("PMEM:: dec_data2_ref_count(%s): memory object has been released.\n",id.c_str());
#endif
	}
	lck_refc.unlock();
}

/*Set the reference count to a given number*/
void PMEM::set_data2_ref_count(int count){
	lck_refc.lock();
	refc = count;
	lck_refc.unlock();
}		


/*Create buffer for data2 of input*/
void PMEM::create_input_buffer(){
	lck_input.lock();
	if(data2 == NULL){
		/*Create buffer*/
		cl_int status;
		data2 = clCreateBuffer(env.context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(float)*n_stride*d_stride , 0, &status);				/*Pinned memory*/
		if(status != CL_SUCCESS){
			printf("PMEM::create_input_buffer(%s): Unable to allocate pinned cl_mem of size %ld bytes.\n", id.c_str(), sizeof(float)*n_stride*d_stride);
			exit(-1);
		}
		/*Initialize buffer content as zeros*/
		float* ptr = (float*) clEnqueueMapBuffer(
		        env.devices[0]->buffer_q,		//It doesn't matter command queue from which device is used. It just need to be a valid one.
		        data2,
		        CL_TRUE,
		        CL_MAP_WRITE,
		        0,
		        sizeof(float)*n_stride*d_stride,
		        0,
		        NULL,
		        NULL,
		        &status);
		if (status != CL_SUCCESS) {
		    printf("PMEM::create_input_buffer(%s): unable to map data2 for initialization. %s.\n", id.c_str(), getErrorString(status));
		    exit(-1);
		}
	
		/*Write zeros*/
		memset(ptr,0,sizeof(float)*n_stride*d_stride);
		
		/*Unmap buffer*/
		cl_event event;
		clEnqueueUnmapMemObject(env.devices[0]->buffer_q,
		        data2,
		        ptr,
		        0,
		        0,
		        &event);
		clFlush(env.devices[0]->buffer_q);
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
	}
	lck_input.unlock();
}

/*Enroll a parameter (matrix) to be updated in the backward pass*/
void SOLVER::enroll_param(float* _p, int h, int w, int stride) {
    UPDATE_PARAM* p = new UPDATE_PARAM;
    if (param.algorithm == SOLVER_ADAGRAD)
        p->V = 0;
    else {
        p->V = new float [h * stride];
        memset(p->V, 0, sizeof (float)*h * stride);
    }
    if (param.algorithm != SOLVER_ADAGRAD)
        p->dEdWss = 0;
    else {
        p->dEdWss = new float [h * stride];
        memset(p->dEdWss, 0, sizeof (float)*h * stride);
    }
    if (param.algorithm != SOLVER_NESTEROV)
        p->Wp = 0;
    else {
        p->Wp = new float [h * stride];
        memset(p->Wp, 0, sizeof (float)*h * stride);
    }
    p->h = h;
    p->w = w;
    p->stride = stride;
    mapper[_p] = p;
}

/*This function is necessary for SOLVER_NESTEROV. See caffe description for details.*/
float* SOLVER::transform(float* W) {
    if (param.algorithm == SOLVER_NESTEROV) {
        if (mapper.find(W) == mapper.end()) {
            printf("SOLVER::transform: try to transform unregistered parameter.\n");
            exit(-1);
        }
        UPDATE_PARAM* p = mapper[W];
        memcpy(p->Wp, W, p->h * p->stride * sizeof (float));
        for (int i = 0; i < p->h; i++)
            for (int j = 0; j < p->w; j++)
                p->Wp[i * p->stride + j] += (mu)*(p->V[i * p->stride + j]);
        return p->Wp;
    } else {
        return W;
    }
}

/*Update the parameter W by, dEdW is the derivative.*/
void SOLVER::update(float* W, float* dEdW) {
    if (mapper.find(W) == mapper.end()) {
        printf("SOLVER::update: try to update unregistered parameter.\n");
        exit(-1);
    }
    UPDATE_PARAM* p = mapper[W];
    if (param.algorithm == SOLVER_NESTEROV || param.algorithm == SOLVER_SGD) {
        for (int i = 0; i < p->h; i++) {
            for (int j = 0; j < p->w; j++) {
                p->V[i * p->stride + j] = mu * p->V[i * p->stride + j] - (alpha)*(dEdW[i * p->stride + j]); 		/*update V*/
                W[i * p->stride + j] += p->V[i * p->stride + j]; 													/*update W*/
                //W[i * p->stride + j] -= 0.05*dEdW[i * p->stride + j];
            }
        }
    } else if (param.algorithm == SOLVER_ADAGRAD) {
        for (int i = 0; i < p->h; i++) {
            for (int j = 0; j < p->w; j++) {
                p->dEdWss[i * p->stride + j] += pow(dEdW[i * p->stride + j], 2); 									/*update dEdWss*/
                W[i * p->stride + j] -= (alpha)*(dEdW[i * p->stride + j] / sqrt(p->dEdWss[i * p->stride + j])); 	/*update W*/
            }
        }
    } else {
        printf("SOLVER::transform: undefined solver [%d]. \n", param.algorithm);
    }
}

void SOLVER::inc_iter() {
    ++iter;
    if (iter % param.stepsize == 0)
        alpha *= param.gamma;
}

const char *getErrorString(cl_int error) {
    switch (error) {
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

            // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}


struct timeval start, _end;

long mtime, seconds, useconds;

void tic() {
    gettimeofday(&start, NULL);
}

void toc(const string tag) {
    gettimeofday(&_end, NULL);
    seconds = _end.tv_sec - start.tv_sec;
    useconds = _end.tv_usec - start.tv_usec;
    mtime = ((seconds) * 1000 + useconds / 1000.0) + 0.5;
    if (mtime == 0)
        printf("Elapsed time: %ld us.\t%s\n", useconds, tag.c_str());
    else
        printf("Elapsed time: %ld ms.\t%s\n", mtime, tag.c_str());

}

void mxshow(float* M, int h, int w, int s) {
    int stride = s;
    if (stride == 0)
        stride = w;

    FILE* fp = fopen("matrices.txt", "a+");
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            fprintf(fp, "%.8f ", M[i * stride + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    fclose(fp);
}

void mxshow(int* M, int h, int w, int s) {
    int stride = s;
    if (stride == 0)
        stride = w;
    FILE* fp = fopen("matrices.txt", "a+");
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            fprintf(fp, "%d ", M[i * stride + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    fclose(fp);
}


/*Print data1 into a file*/
void PMEM::print_data1(std::string varname){
    FILE* fp = fopen("dumps.txt", "a+");
    if (!fp) {
        puts("PMEM::print_data1: unable to open the file 'dumps.txt'.");
        exit(-1);
    }
    fprintf(fp, "%s\n", varname.c_str());
    if(data1 == NULL){
		fprintf(fp, "NULL\n");
	}
	else{
		for (int i = 0; i < n_stride; i++) {
		    for (int j = 0; j < d_stride; j++) {
		        fprintf(fp, "%.4f ", data1[i * d_stride + j]);
		    }
		    fprintf(fp, "\n");
		}
    }
    fprintf(fp, "\n");
    fclose(fp);
}


/*Read data2 (not into data1), and print the data into a file.*/
void PMEM::print_data2(std::string varname){
    FILE* fp = fopen("dumps.txt", "a+");
    if (!fp) {
        puts("PMEM::print_data2: unable to open the file 'dumps.txt'.");
        exit(-1);
    }
    fprintf(fp, "%s\n", varname.c_str());
	if(data2 == NULL){
		fprintf(fp, "NULL\n\n");
		fclose(fp);
		return;
	}
    float* data = new float [n_stride*d_stride];		//temporal buffer.
	cl_int status;
    float* ptr = (float*) clEnqueueMapBuffer(
            env.devices[0]->buffer_q,		//It doesn't matter command queue from which device is used. It just need to be a valid one.
            data2,
            CL_TRUE,
            CL_MAP_READ,
            0,
            sizeof(float)*n_stride*d_stride,
            0,
            NULL,
            NULL,
            &status);
    if (status != CL_SUCCESS) {
        printf("PMEM::print_data2(%s): unable to map data2 for reading. %s.\n", id.c_str(), getErrorString(status));
        exit(-1);
    }
	
	/*Read*/
	memcpy(data,ptr,sizeof(float)*n_stride*d_stride);
    
	/*Unmap buffer*/
    cl_event event;
    clEnqueueUnmapMemObject(env.devices[0]->buffer_q,
            data2,
            ptr,
            0,
            0,
            &event);
    clFlush(env.devices[0]->buffer_q);
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
    
    /*Write data into file*/
	for (int i = 0; i < n_stride; i++) {
		for (int j = 0; j < d_stride; j++) {
			fprintf(fp, "%.4f ", data[i * d_stride + j]);
		}
		fprintf(fp, "\n");
	}
    fprintf(fp, "\n");
    fclose(fp);
    
    //cleanup
	delete data;
}	
    
 
 
/*round x up to align with at least 4*/
int roundup(int x, int base) {
    if (x < 1) return -1; //failed.
    if (base > 0) {
        if (x % base == 0) return x;
        else return x + base - (x % base);
    }
    if (x > 64) {
        if (x % 128 == 0) return x;
        else return x + 128 - (x % 128);
    } else if (x > 32)
        return 64;
    else if (x > 16)
        return 32;
    else if (x > 8)
        return 16;
    else if (x > 4)
        return 8;
    else
        return 4;
}

