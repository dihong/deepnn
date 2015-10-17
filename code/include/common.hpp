#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include "math.h"
#include "memory.h"
#include <string>
#include "SDKThread.hpp"
#include "main.hpp"
#include <map>
#include <set>
#include <typeinfo>   // operator typeid


#define INF 9//999999999

std::string readKernelFile(std::string filename);

/*Show error message*/
const char *getErrorString(cl_int error);

/*print matrix*/
void mxshow(float* M, int h, int w, int s = 0);

void mxshow(float* M, int h, int w, int s, std::string varname);

void mxshow(int* M, int h, int w, int s = 0);

/*print pinned memory into file*/
void mxshow(const CL_ENV& env, size_t offset, size_t stride, size_t height, size_t width, std::string varname);

void tic();

void toc(const std::string tag = "");


int roundup(int x, int base = 0);

void quick_sort(float* arr, int N, int* order, float* sorted, bool descend);


class SOLVER {
	typedef struct UPDATE_PARAM {
		float* V; 		/*the amount of quantity need to update*/
		float* dEdWss; 	/*sum square derivatives along the iteration axis. This is used by AdaGrad solver.*/
		float* Wp; 		/*Wp = W+mu*V. This is used by NESTEROV solver.*/
		int h; 			/*height of the valid data region of the parameter*/
		int w; 			/*width of the valid data region of the parameter*/
		int stride; 	/*stride of the data region of the parameter*/
	} UPDATE_PARAM;
    SOLVER_PARAM param;	/*Solver configuration*/
    size_t iter; 		/*current number of iterations*/
    float alpha; 		/*current learning rate*/
    float mu; 			/*current momentum. This value is fixed because the update can be tuned by chaning (gamma,stepsize).*/
    std::map<float*, UPDATE_PARAM*> mapper;
public:

    SOLVER(SOLVER_PARAM p) {
        param = p;
        iter = 0;
        alpha = p.base_lr;
        mu = p.momentum;
    }

    /*enroll parameters to the solver. All parameters must be registered before using the solver.*/
    void enroll_param(float* _p, int h, int w, int stride);

    /*transform the paramters from W1 to W2 before calculating the gradients. This is necessary for the NESTEROV solver*/
    float* transform(float* W);

    /*update the parameter: The parameters are updated in place*/
    void update(float* W, float* dEdW);

    /*increase the iteration number by 1*/
    void inc_iter();
};


/*This class is for storing data objects passing through the network*/
class PMEM {
	appsdk::ThreadLock lck_input;
	appsdk::ThreadLock lck_refc;
	const CL_ENV& env;			/*OpenCL environment*/
	int refc;					/*reference count*/
	const std::string id;		/*identifier of the memory object*/
public:
    float* data1; 		/*C pointer to of the corresponding opencl memory object in data2 (vectorized data)*/
    cl_mem data2; 		/*opencl memory object. Note: The location (host, or device) of the object is undefined by default. To move this object in/out device, use clEnqueueMigrateMemObjects().*/
    int n; 				/*Input buffer of the unit: Actual Number of batch samples (Number)*/
    int d; 				/*Input buffer of the unit: Actual Number of feature maps (Depth)*/
    int h; 				/*Input buffer of the unit: Actual height of each feature map (Height)*/
    int w; 				/*Input buffer of the unit: Actual width  of each feature map (Width)*/
    int w_stride;		/*Input buffer of the unit: Number of elements(float) between each row of an image. Note: this is only valid for units of 2D (1D has various f_stride) feature maps, e.g. CONV.*/
    int f_stride;		/*Input buffer of the unit: Number of elements(float) between each feature map of one sample.*/
    int d_stride;		/*Input buffer of the unit: Number of elements(float) between each sample of a batch.*/
    int n_stride;		/*Input buffer of the unit: Number of extended batch size. For FULL unit, n_stride >= n; while for other units, n_stride = n.*/

    PMEM(const CL_ENV&_env, const std::string& _id):env(_env),id(_id) {
        data1 = NULL;
        data2 = NULL;
        refc = -1;
    }

    PMEM(const CL_ENV&_env, const std::string& _id, int _n, int _h, int _w, int _d, int _w_stride, int _f_stride, int _d_stride, int _n_stride):env(_env),id(_id) {
        data1 = NULL;
        data2 = NULL;
        n = _n;
        h = _h;
        w = _w;
        d = _d;
        w_stride = _w_stride;
        f_stride = _f_stride;
        d_stride = _d_stride;
        n_stride = _n_stride;
    }
    void print_data2(std::string varname);		/*Read data2 (not into data1), and print the data into a file.*/
    void print_data1(std::string varname);		/*Print data1 into a file*/
    void set_data1();							/*Create a new copy of host memory for data1, and initialize data1 with data2.*/
    void set_data2();							/*Create a new copy of opencl memory object for data2, and initialize data2 with data1.*/
    void free_data1();							/*delete data1*/
    void free_data2();							/*delete data2*/
    void migrate(const DEVICE & device);		/*migrate data2 into specified device, no matter where the data2 resided originally. The data2 must be valid before calling the function.*/
    void dec_data2_ref_count();					/*Decrease the reference count of data2 by 1. When reference count reaches zero, it is released.*/
    void set_data2_ref_count(int count);		/*Set the reference count to a given number*/
    void create_input_buffer();					/*Create buffer for data2 of input*/
};


/*This class is for storing matrix data objects (e.g. parameters of full unit)*/
template <class datatype>		//datatype need to be either int or float.
class MATRIX {
	
protected:
	const CL_ENV& env;			/*OpenCL environment*/
	const std::string id;		/*variable identifier*/
public:
    int h; //height of valid data area.
    int w; //width of valid data area.
    int H; //height of zero-padded data area. padding ensures: H % BLOCK_SIZE = 0.
    int W; //width of zero-padded data area. padding ensures: W % BLOCK_SIZE = 0.
    datatype* data1; //It references the host data.
    cl_mem data2; //It references the device data.

    MATRIX(const CL_ENV& _env, const std::string& _id, int _h, int _w, int _H, int _W, datatype* _data1 = 0):env(_env),id(_id) {
        h = _h;
        w = _w;
        H = _H;
        W = _W;
        data1 = new datatype [H * W];
        memset(data1, 0, sizeof (datatype)*H * W);
        if (_data1) {
            for (int i = 0; i < h; i++)
                memcpy(data1 + W * i, _data1 + w * i, sizeof (datatype)*w);
        }
        data2 = 0;
    }
    void print_data2(std::string varname);	/*Read data2 (not into data1), and print the data into a file.*/
    void print_data1(std::string varname);	/*Print data1 into a file*/   
    void set_data1();	/*Copy data2 into data1.*/
    void set_data2();	/*Create a new copy of opencl memory object for data2 (if not exists), and initialize data2 with data1.*/
    void free_data2();	/*delete data2*/
    void migrate(const DEVICE & device);		/*migrate data2 into specified device, no matter where the data2 resided originally. The data2 must be valid before calling the function.*/
};


class UNIT {
private:
    int type; 		/*Type of the unit, e.g. UNIT_TYPE_INPUT, UNIT_TYPE_FULL, UNIT_TYPE_SOFTMAX...*/
    static int global_numeric_identifier_cnt;
protected:
    const std::string id; /*The globally unique identifier for the unit*/
    int nid; /*The globally unique numeric identifier (0,1,...)*/
    const CL_ENV& env; /*OpenCL environment*/
    std::map<UNIT*, int> pred; /*The predecessors of this unit, and rank of this unit in its predecessor's output list*/
    std::map<UNIT*, int> succ; /*The successors of this unit, and rank of this unit in its successor's input list*/
    std::vector<cl_event> forward_event_wait_list; /*A list of event that this unit has to wait when it is doing forward operation,  so as to ensure input X is ready.*/
    std::vector<cl_event> backward_event_wait_list; /*A list of event that this unit has to wait when it is doing backward operation, so as to ensure input Y is ready.*/
    float walltime; 		/*Time to compute this unit*/
public:

    PMEM *input; 						/*The PMEM object of this unit. Written by predecessors. Wait input data ready signals from predecessors to proceed.*/
    std::vector<PMEM*> outputs;			/*The output buffers that have to write. Each buffer is for one successor.*/
    std::vector<size_t> offsets;		/*The offset of PMEM of outputs to write.*/
    int n,h,w,d;						/*Actual output dimension of this unit. Note: the input dimension inherits from the predecessors.*/


    virtual void init() = 0; /*Initialize the unit*/
    virtual void forward(DEVICE& device) = 0; /*Run the forward pass on given device*/
    virtual void backward(DEVICE& device) = 0; /*Run the backward pass on given device*/
    	
    UNIT(const std::string& _id, const CL_ENV& _env, const int t) : id(_id), env(_env), type(t) {
        nid = global_numeric_identifier_cnt++;
        input = 0;
        walltime = 0;
        n = h = w = d = 0;
    }
    
    const CL_ENV& get_env() {
        return env;
    }

    /*Get the wall time of the unit, in milli-seconds.*/
    float get_wall_time(const int pass) {
        if (pass == 1) { 			/*Return the forward-pass time*/
            return 1;
        } else if (pass == -1) { /*Return the backward-pass time*/
            return 1;
        }
    }
    

    /*signal all the successors that the ourput of this unit is ready, so that they can proceed (forward).*/
    void signal_forward_ready() {
        std::map<UNIT*, int>::iterator it;
		for (it = succ.begin(); it != succ.end(); it++){
			clSetUserEventStatus(it->first->forward_event_wait_list[it->second], CL_COMPLETE);		//signal the successors' forward_event_wait_list ready.
		}
    }

    /*signal all the predecessors that the input of this unit is ready, so that they can proceed (backward).
    */
    void signal_backward_ready() {
        std::map<UNIT*, int>::iterator it;
		for (it = pred.begin(); it != pred.end(); it++) 
		    clSetUserEventStatus(it->first->backward_event_wait_list[it->second], CL_COMPLETE);		//signal the predecessors' backward_event_wait_list ready.
    }

    /*Call this function whenever a forward pass of this unit completes, to reset the UNIT's forward_event_wait_list.*/
    void clear_forward_event_list() {
        for (int i = 0; i < forward_event_wait_list.size(); i++) {
            clReleaseEvent(forward_event_wait_list[i]);
            forward_event_wait_list[i] = clCreateUserEvent(env.context, NULL);
        }
    }

    /*Call this function whenever a backward pass of this unit completes, to reset backward_event_wait_list.*/
    void clear_backward_event_list() {
        for (int i = 0; i < backward_event_wait_list.size(); i++) {
            clReleaseEvent(backward_event_wait_list[i]);
            backward_event_wait_list[i] = clCreateUserEvent(env.context, NULL);
        }
    }

    virtual bool validate() /*Validate the unit by derivative implementations: return true if this unit is good, false otherwise.*/ {
        return false;
    }
	
	/*Add predecessor [u] for this unit, where [r] is the rank (0,1,...) of this unit in the successors of [u].*/
    void add_pred(UNIT* u, const int& r) {
        pred[u] = r;
        cl_event event = clCreateUserEvent(env.context, NULL);
        forward_event_wait_list.push_back(event);
    }
	
	/*Add successor [u] for this unit.*/
    void add_succ(UNIT* u, const int& r) {
        succ[u] = r;
        cl_event event = clCreateUserEvent(env.context, NULL);
        if (type == UNIT_TYPE_INPUT) {
            /*The INPUT unit initially doesn't need to wait anything*/
            clSetUserEventStatus(event, CL_COMPLETE);
        }
        backward_event_wait_list.push_back(event);
    }

	/*Get the string identifier of this unit*/
    std::string get_id() {
        return id;
    }

	/*Get the numeric identifier of this unit*/
    int get_nid() {
        return nid;
    }


	/*Get the type of this unit*/
    int get_type() {
        return type;
    }

    /*get the concatenated string from list of predecessor id*/
    std::string get_pred_str() {
        std::string ret;
        if (pred.size() == 0)
            return ret;
        std::map<UNIT*, int>::iterator it = pred.begin();
        ret = it->first->get_id();
        ++it;
        for (; it != pred.end(); it++) {
            ret += "+" + it->first->get_id();
        }
        return ret;
    }

    /*get read-only copy of the predecessors*/
    std::vector<UNIT*> get_pred() {
        std::vector<UNIT*> ret;
        std::map<UNIT*, int>::iterator it;
        for (it = pred.begin(); it != pred.end(); it++)
            ret.push_back(it->first);
        return ret;
    }

    /*get read-only copy of the successors*/
    std::vector<UNIT*> get_succ() {
        std::vector<UNIT*> ret;
        std::map<UNIT*, int>::iterator it;
        for (it = succ.begin(); it != succ.end(); it++)
            ret.push_back(it->first);
        return ret;
    }

};


/*Copy data2 into data1.*/
template <class datatype>
void MATRIX<datatype>::set_data1(){
	if (data2 == NULL){
        printf("MATRIX::set_data1(%s): unable to map data2 from reading. data2 is NULL.\n",id.c_str());
        exit(-1);
	}
    cl_int status;
    datatype* ptr = (datatype*) clEnqueueMapBuffer(
            env.devices[0]->buffer_q,		//It doesn't matter command queue from which device is used. It just need to be a valid one.
            data2,
            CL_TRUE,
            CL_MAP_READ,
            0,
            sizeof(datatype)*H*W,
            0,
            NULL,
            NULL,
            &status);
    if (status != CL_SUCCESS) {
        printf("MATRIX::set_data1(%s): unable to map data2 for reading. %s.\n", getErrorString(status), id.c_str());
        exit(-1);
    }
	
	/*Read*/
	memcpy(data1,ptr,sizeof(datatype)*H*W);

    
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
template <class datatype>
void MATRIX<datatype>::set_data2(){
	cl_int status;
	if(data2 == NULL){
		data2 = clCreateBuffer(env.context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(datatype)*H*W , 0, &status);				/*Pinned memory*/
		if(status != CL_SUCCESS){
			printf("MATRIX::set_data2(%s): Unable to allocate pinned cl_mem of size %ld bytes.\n", id.c_str(), sizeof(datatype)*H*W);
			exit(-1);
		}
	}
    
    datatype* ptr = (datatype*) clEnqueueMapBuffer(
            env.devices[0]->buffer_q,		//It doesn't matter command queue from which device is used. It just need to be a valid one.
            data2,
            CL_TRUE,
            CL_MAP_WRITE,
            0,
            sizeof(datatype)*H*W,
            0,
            NULL,
            NULL,
            &status);
    if (status != CL_SUCCESS) {
        printf("MATRIX::set_data2(%s): unable to map data2 for writing. %s.\n", id.c_str(), getErrorString(status));
        exit(-1);
    }
	
	/*Write*/
	memcpy(ptr,data1,sizeof(datatype)*H*W);
    
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


/*delete data2*/
template <class datatype>
void MATRIX<datatype>::free_data2(){
	if (data2==NULL) return;
	int status = clReleaseMemObject(data2);
	if (status != CL_SUCCESS){
        printf("MATRIX::free_data2(%s): unable to free data2. %s.\n", id.c_str(), getErrorString(status));
        exit(-1);
	}
	data2 = NULL;
}


/*migrate data2 into specified device, no matter where the data2 resided originally. The data2 must be valid before calling the function.*/
template <class datatype>
void MATRIX<datatype>::migrate(const DEVICE & device){
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
        printf("MATRIX::migrate(%s): unable to migrate data. %s.\n", id.c_str(), getErrorString(status));
        exit(-1);
	}
}


/*Print data1 into a file*/
template <class datatype>
void MATRIX<datatype>::print_data1(std::string varname){
    FILE* fp = fopen("dumps.txt", "a+");
    if (!fp) {
        printf("MATRIX::print_data1(%s): unable to open the file 'dumps.txt'.",id.c_str());
        exit(-1);
    }
    fprintf(fp, "%s\n", varname.c_str());
    if(data1 == NULL){
		fprintf(fp, "NULL\n");
	}
	else{
		for (int i = 0; i < H; i++) {
		    for (int j = 0; j < W; j++) {
		    	//if (typeid(float)==typeid(datatype))
		        	fprintf(fp, "%.4f ", (float)data1[i * W + j]);
		        //else if (typeid(int)==typeid(datatype))
		        //	fprintf(fp, "%d ", data1[i * W + j]);
		        /*else{
		        	printf("MATRIX::print_data1(%s): unsupported data type.", id.c_str());
		        	exit(-1);
		        }*/
		    }
		    fprintf(fp, "\n");
		}
    }
    fprintf(fp, "\n");
    fclose(fp);
}



/*Read data2 (not into data1), and print the data into a file.*/
template <class datatype>
void MATRIX<datatype>::print_data2(std::string varname){
    FILE* fp = fopen("dumps.txt", "a+");
    if (!fp) {
        printf("MATRIX::print_data2(%s): unable to open the file 'dumps.txt'.", id.c_str());
        exit(-1);
    }
    fprintf(fp, "%s\n", varname.c_str());
	if(data2 == NULL){
		fprintf(fp, "NULL\n\n");
		fclose(fp);
		return;
	}
    datatype* data = new datatype [H*W];		//temporal buffer.
	cl_int status;
    datatype* ptr = (datatype*) clEnqueueMapBuffer(
            env.devices[0]->buffer_q,		//It doesn't matter command queue from which device is used. It just need to be a valid one.
            data2,
            CL_TRUE,
            CL_MAP_READ,
            0,
            sizeof(datatype)*H*W,
            0,
            NULL,
            NULL,
            &status);
    if (status != CL_SUCCESS) {
        printf("MATRIX::print_data2(%s): unable to map data2 for reading. %s.\n", getErrorString(status), id.c_str());
        exit(-1);
    }
	
	/*Read*/
	memcpy(data,ptr,sizeof(datatype)*H*W);
    
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
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
		    	//if (typeid(float)==typeid(datatype))
		        	fprintf(fp, "%.4f ", (float)data[i * W + j]);
		        /*else if (typeid(int)==typeid(datatype))
		        	fprintf(fp, "%d ", data[i * W + j]);
		        else{
		        	printf("MATRIX::print_data2(%s): unsupported data type.", id.c_str());
		        	exit(-1);
		        }*/
		}
		fprintf(fp, "\n");
	}
    fprintf(fp, "\n");
    fclose(fp);
    
    //cleanup
	delete data;
}

#endif
